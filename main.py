import asyncio
import logging

from bot.main import main as bot_main
from parser.baza_to_rag import parse_baza_and_save_to_vectorstore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Запускает парсинг данных и bot."""

    # TODO сделать миграции перед парсингом

    logger.info("=== Инициализация векторной базы данных ===")
    try:
        await asyncio.to_thread(parse_baza_and_save_to_vectorstore)
    except Exception as e:
        logger.error(f"Ошибка при парсинге и сохранении данных: {e}")
        logger.info("Продолжаем запуск бота...")

    logger.info("=== Запуск бота ===")
    bot_task = asyncio.create_task(bot_main())

    await asyncio.gather(bot_task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Программа остановлена")
