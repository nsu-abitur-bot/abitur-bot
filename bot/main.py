import asyncio
import logging
from collections.abc import Awaitable, Callable

from bootstrap.runtime_settings import runtime_settings
from bot.channel_supervisor import supervise_channel
from bot.max_bot import run_max_bot
from bot.telegram_bot import run_telegram_bot
from db.postgres.config import DATABASE_URL
from faq.matcher import get_faq_matcher
from faq.updates import watch_faq_updates
from liveness import LivenessServer
from pipeline.llm_client import cleanup_redis

logger = logging.getLogger(__name__)


async def _run_with_restart(
    name: str,
    runner: Callable[[], Awaitable[bool | None]],
    restart_delay_seconds: float,
) -> None:
    """Держит адаптер живым: перезапускает при выходе.
    Если runner() возвращает False, значит адаптер решил вообще не запускаться.
    """
    while True:
        try:
            result = await runner()
            if result is False:
                break
            logger.warning(
                "%s adapter stopped unexpectedly, restart in %.1f sec",
                name,
                restart_delay_seconds,
            )
        except asyncio.CancelledError:
            logger.info("%s adapter task cancelled", name)
            raise
        except Exception:
            logger.exception(
                "%s adapter crashed, restart in %.1f sec",
                name,
                restart_delay_seconds,
            )

        await asyncio.sleep(restart_delay_seconds)


async def main() -> None:
    async with runtime_settings():
        liveness = LivenessServer()
        await liveness.start()
        tasks = [
            asyncio.create_task(supervise_channel("telegram", run_telegram_bot)),
            asyncio.create_task(supervise_channel("max", run_max_bot)),
            asyncio.create_task(
                watch_faq_updates(
                    get_faq_matcher,
                    DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://", 1),
                )
            ),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await cleanup_redis()
            await liveness.aclose()


if __name__ == "__main__":
    asyncio.run(main())
