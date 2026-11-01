"""Stop before starting a replacement; fence duplicate pollers by token identity."""

import asyncio
import hashlib
import logging
import tempfile
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from sqlalchemy import text

from bot.telegram_bot import _TelegramPollingLock
from config.instance import enabled, get_setting
from db.postgres.db import engine

logger = logging.getLogger(__name__)


def channel_signature(channel: str) -> tuple[str, str, str]:
    telegram = channel == "telegram"
    token = get_setting("BOT_TOKEN" if telegram else "MAX_BOT_TOKEN", "")
    active = enabled("TELEGRAM_ENABLED" if telegram else "MAX_ENABLED")
    return (
        token if active else "",
        get_setting("TELEGRAM_SOCKS5_PROXY", "") if telegram else "",
        channel,
    )


async def supervise_channel(
    channel: str, runner: Callable[[], Coroutine[Any, Any, bool | None]]
) -> None:
    while True:
        signature = channel_signature(channel)
        token = signature[0]
        if not token:
            await asyncio.sleep(2)
            continue
        digest = hashlib.sha256((channel + token).encode()).hexdigest()
        lock = _TelegramPollingLock(
            str(Path(tempfile.gettempdir()) / ("abitur-channel-" + digest))
        )
        task = None
        try:
            if not lock.acquire():
                await asyncio.sleep(2)
                continue
            async with engine.connect() as connection:
                lock_id = int.from_bytes(bytes.fromhex(digest[:16]), signed=True)
                acquired = await connection.scalar(
                    text("SELECT pg_try_advisory_lock(:key)"), {"key": lock_id}
                )
                await connection.commit()
                if not acquired:
                    await asyncio.sleep(2)
                    continue
                try:
                    task = asyncio.create_task(runner())
                    while not task.done() and channel_signature(channel) == signature:
                        await asyncio.sleep(2)
                        # Loss of the fencing connection must stop this poller.
                        await connection.execute(text("SELECT 1"))
                        await connection.commit()
                finally:
                    if task is not None:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                    try:
                        await connection.rollback()
                        await connection.execute(
                            text("SELECT pg_advisory_unlock(:key)"), {"key": lock_id}
                        )
                        await connection.commit()
                    except Exception:
                        # Never return a possibly locked session to the pool.
                        await connection.invalidate()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error(
                "Ошибка подключения канала %s; повтор через несколько секунд", channel
            )
        finally:
            lock.release()
        await asyncio.sleep(float(get_setting("BOT_RESTART_DELAY_SECONDS", "5")))
