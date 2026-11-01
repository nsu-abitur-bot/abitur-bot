"""Keep the bot's FAQ index current after changes committed by the API."""

import asyncio
import logging
from contextlib import suppress

import asyncpg

from db.postgres.services.faq import FAQ_CHANGE_CHANNEL
from faq.matcher import FAQMatcher

logger = logging.getLogger(__name__)


async def watch_faq_updates(matcher: FAQMatcher, dsn: str) -> None:
    """Listen for FAQ changes; re-read the table after every reconnection."""
    while True:
        connection: asyncpg.Connection | None = None
        try:
            connection = await asyncpg.connect(dsn=dsn)
            changed = asyncio.Event()

            def on_change(
                _connection: asyncpg.Connection,
                _pid: int,
                _channel: str,
                _payload: str,
            ) -> None:
                changed.set()

            await connection.add_listener(FAQ_CHANGE_CHANNEL, on_change)
            await matcher.refresh_from_db()
            logger.info("FAQ: подписка на изменения БД активна")

            while True:
                try:
                    await asyncio.wait_for(changed.wait(), timeout=30)
                except TimeoutError:
                    await connection.execute("SELECT 1")
                    continue
                changed.clear()
                await matcher.refresh_from_db()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("FAQ: подписка на изменения прервана; повтор через 5 секунд")
            await asyncio.sleep(5)
        finally:
            if connection is not None:
                with suppress(Exception):
                    await connection.close()
