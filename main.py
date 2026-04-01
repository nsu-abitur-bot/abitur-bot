import asyncio
import logging

from bot.main import main as bot_main
from db.postgres.init_db import main as init_db
from logging_config import setup_logging

# Настраиваем логирование в самом начале
setup_logging()

logger = logging.getLogger(__name__)


async def main():
    """Запускает парсинг данных и bot."""

    logger.info("=== Инициализация базы данных и миграции ===")
    await init_db()

    logger.info("=== Запуск бота ===")
    bot_task = asyncio.create_task(bot_main())

    await asyncio.gather(bot_task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Программа остановлена")
