"""
Скрипт для инициализации базы данных PostgreSQL.
Создаёт БД если её нет и применяет миграции.
"""

import asyncio
import logging as std_logging
import os
import sys

import asyncpg
from alembic import command
from alembic.config import Config
from dotenv import load_dotenv

# Загружаем переменные окружения из корня проекта
load_dotenv()

logger = std_logging.getLogger(__name__)


async def create_database_if_not_exists():
    """Создать БД если она не существует."""
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD")
    db_name = os.getenv("DB_NAME", "abitur")

    if not db_password:
        logger.error("DB_PASSWORD не задан в .env файле")
        sys.exit(1)

    # Подключаемся к postgres (служебная БД)
    try:
        conn = await asyncpg.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database="postgres",
        )

        # Проверяем существует ли БД
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", db_name
        )

        if not exists:
            # Создаём БД
            await conn.execute(f'CREATE DATABASE "{db_name}"')
            logger.info(f"База данных '{db_name}' создана")
        else:
            logger.info(f"База данных '{db_name}' уже существует")

        await conn.close()

    except Exception as e:
        logger.error(f"Ошибка при создании БД: {e}")
        sys.exit(1)


def run_migrations():
    """Применить миграции."""
    try:
        # Путь к alembic.ini в корне проекта
        config_path = "alembic.ini"
        alembic_cfg = Config(config_path)
        # Отключаем настройку логирования внутри Alembic,
        # чтобы не переопределять настройки бота
        alembic_cfg.attributes["configure_logger"] = False
        command.upgrade(alembic_cfg, "head")
        logger.info("Миграции применены успешно")
    except Exception as e:
        logger.error(f"Ошибка при применении миграций: {e}")
        sys.exit(1)


def _force_seed_enabled() -> bool:
    return os.getenv("SEED_FACULTIES_FORCE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


async def seed_reference_data():
    """Заливает справочные данные (факультеты).

    Не критично для работы: при ошибке логируем и продолжаем старт. По умолчанию
    заливаем только на пустую таблицу (первичный бутстрап), чтобы не затирать
    правки из админки. SEED_FACULTIES_FORCE=1 — принудительно перезалить из сида.
    """
    try:
        from db.seed.load_faculties import seed_faculties, seed_faculties_if_empty

        if _force_seed_enabled():
            stats = await seed_faculties()
            logger.info("Справочник факультетов принудительно перезалит: %s", stats)
        else:
            stats = await seed_faculties_if_empty()
            logger.info("Автозаливка справочника факультетов: %s", stats)
    except Exception as e:
        logger.error(
            "Не удалось загрузить справочник факультетов (старт продолжается): %s", e
        )


async def main():
    """Главная функция инициализации."""
    logger.info("Инициализация базы данных PostgreSQL...")

    # Создаём БД если нужно
    await create_database_if_not_exists()

    # Применяем миграции
    run_migrations()

    # Заливаем справочные данные (факультеты) — идемпотентно, только если пусто
    await seed_reference_data()

    logger.info("Инициализация завершена успешно!")


if __name__ == "__main__":
    asyncio.run(main())
