import asyncio
import logging

from bot.main import main as bot_main
from db.postgres.init_db import main as init_db
from logging_config import setup_logging

# Настраиваем логирование в самом начале
setup_logging()

logger = logging.getLogger(__name__)


async def _load_faq_and_abbrev() -> None:
    from abbrev.expander import get_abbrev_expander
    from db.postgres.db import AsyncSessionLocal
    from db.postgres.services.abbrev import AbbrevDbService
    from db.postgres.services.faq import FaqDbService
    from faq.matcher import get_faq_matcher

    async with AsyncSessionLocal() as session:
        try:
            faq_entries = await FaqDbService(session).get_all()
            get_faq_matcher().load_items(
                [{"question": e.question, "aliases": e.aliases, "answer": e.answer}
                 for e in faq_entries]
            )
            logger.info("FAQ загружен из БД: %d записей", len(faq_entries))

            abbrev_entries = await AbbrevDbService(session).get_all()
            get_abbrev_expander().load_items(
                [{"short": e.short, "full": e.full} for e in abbrev_entries]
            )
            logger.info("Аббревиатуры загружены из БД: %d записей", len(abbrev_entries))
        except Exception as e:
            logger.exception("Ошибка загрузки FAQ/аббревиатур из БД: %s", e)


async def main():
    """Запускает парсинг данных и bot."""

    logger.info("=== Инициализация базы данных и миграции ===")
    await init_db()

    logger.info("=== Загрузка FAQ и аббревиатур из БД ===")
    await _load_faq_and_abbrev()

    logger.info("=== Запуск бота ===")
    bot_task = asyncio.create_task(bot_main())

    await asyncio.gather(bot_task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Программа остановлена")
