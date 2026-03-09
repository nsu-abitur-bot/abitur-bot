import asyncio
import logging

from bot.main import main as bot_main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Запускает парсинг данных и bot."""

    # TODO сделать миграции перед парсингом

    logger.info("=== Запуск бота ===")
    bot_task = asyncio.create_task(bot_main())

    await asyncio.gather(bot_task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Программа остановлена")
