import asyncio
import logging

from bot.max_bot import run_max_bot
from bot.telegram_bot import run_telegram_bot
from llm.llm_client import cleanup_redis

logger = logging.getLogger(__name__)


async def main() -> None:
    """Запускает Telegram и MAX адаптеры параллельно."""
    logger.info("Запуск bot-core адаптеров: Telegram + MAX")

    telegram_task = asyncio.create_task(run_telegram_bot())
    max_task = asyncio.create_task(run_max_bot())

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
