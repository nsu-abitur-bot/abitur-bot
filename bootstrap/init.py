"""Полная подготовка к старту: база, миграции, справочные данные."""

import asyncio
import logging

from bootstrap.seed import seed_admission_scores, seed_reference_data
from db.postgres.init_db import create_database_if_not_exists, run_migrations

logger = logging.getLogger(__name__)


async def main() -> None:
    logger.info("Инициализация базы данных PostgreSQL...")

    await create_database_if_not_exists()
    run_migrations()

    await seed_reference_data()
    # После справочника: матчинг проходных баллов опирается на факультеты.
    await seed_admission_scores()

    logger.info("Инициализация завершена успешно!")


if __name__ == "__main__":
    asyncio.run(main())
