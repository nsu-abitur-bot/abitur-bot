import asyncio
import logging
from collections.abc import Awaitable, Callable
from os import getenv

from bot.max_bot import run_max_bot
from bot.telegram_bot import run_telegram_bot
from llm.llm_client import cleanup_redis

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
    """Запускает Telegram и MAX адаптеры параллельно."""
    logger.info("Запуск bot-core адаптеров: Telegram + MAX")

    restart_delay_seconds = float(getenv("BOT_RESTART_DELAY_SECONDS", "5"))

    telegram_task = asyncio.create_task(
        _run_with_restart(
            name="Telegram",
            runner=run_telegram_bot,
            restart_delay_seconds=restart_delay_seconds,
        )
    )
    max_task = asyncio.create_task(
        _run_with_restart(
            name="MAX",
            runner=run_max_bot,
            restart_delay_seconds=restart_delay_seconds,
        )
    )

    try:
        await asyncio.gather(telegram_task, max_task)
    finally:
        logger.info("Закрытие соединений...")
        await cleanup_redis()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Боты остановлены")
