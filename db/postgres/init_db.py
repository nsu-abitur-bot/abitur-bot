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
    режим МЯГКИЙ: доливаем только недостающее (новые факультеты, алиасы,
    направления), не затирая правки из админки. На пустой БД это равносильно
    первичному бутстрапу, а на существующей — подхватывает новые записи сида при
    каждом деплое. SEED_FACULTIES_FORCE=1 — жёстко перезалить из сида.
    """
    try:
        from db.seed.load_faculties import seed_faculties

        if _force_seed_enabled():
            stats = await seed_faculties(soft=False)
            logger.info("Справочник факультетов принудительно перезалит: %s", stats)
        else:
            stats = await seed_faculties(soft=True)
            logger.info("Мягкая доливка справочника факультетов: %s", stats)
    except Exception as e:
        logger.error(
            "Не удалось загрузить справочник факультетов (старт продолжается): %s", e
        )


# Ключ в таблице settings: когда последний раз заливали проходные баллы.
# Отдельная отметка нужна потому, что updated_at строк не обновляется, когда
# значения не изменились (SQLAlchemy не выпускает UPDATE) — по нему свежесть
# импорта определить нельзя.
LAST_SCORES_IMPORT_KEY = "admission_scores_last_import"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "да"}


def _scores_max_age_hours() -> float:
    try:
        return float(os.getenv("ADMISSION_SCORES_MAX_AGE_HOURS", "24"))
    except (TypeError, ValueError):
        return 24.0


async def seed_admission_scores():
    """Подтягивает проходные баллы прошлых лет со страницы итогов приёма НГУ.

    Не критично для работы: при любой ошибке логируем и продолжаем старт (сам
    парсер при сетевой ошибке возвращает пустой список). Upsert идемпотентен по
    (program_id, year, form), поэтому повторные запуски безопасны.

    Чтобы рестарты контейнера не дёргали сайт НГУ, заливка пропускается, если
    данные уже свежее ADMISSION_SCORES_MAX_AGE_HOURS (по умолчанию 24 ч).
    Отключить целиком: SEED_ADMISSION_SCORES=0. Игнорировать свежесть:
    SEED_ADMISSION_SCORES_FORCE=1.
    """
    if not _env_flag("SEED_ADMISSION_SCORES", True):
        logger.info("Автозаливка проходных баллов отключена (SEED_ADMISSION_SCORES=0)")
        return

    try:
        from datetime import UTC, datetime, timedelta

        from db.postgres.db import AsyncSessionLocal
        from db.postgres.services.admission_score import AdmissionScoreService
        from db.postgres.services.settings import SettingsService
        from parser.scores import DEFAULT_SCORES_URL, parse_scores

        now = datetime.now(UTC).replace(tzinfo=None)

        if not _env_flag("SEED_ADMISSION_SCORES_FORCE", False):
            async with AsyncSessionLocal() as session:
                raw_last = await SettingsService(session).get_value(
                    LAST_SCORES_IMPORT_KEY
                )
            if raw_last:
                try:
                    age = now - datetime.fromisoformat(raw_last)
                except ValueError:
                    age = None
                if age is not None and age < timedelta(hours=_scores_max_age_hours()):
                    logger.info("Проходные баллы заливались %s назад — пропускаю", age)
                    return

        url = os.getenv("ADMISSION_SCORES_URL", DEFAULT_SCORES_URL)
        logger.info("Загружаю проходные баллы: %s", url)
        rows = await parse_scores(url)
        if not rows:
            logger.warning(
                "Страница итогов приёма не дала строк — проходные баллы не обновлены"
            )
            return

        async with AsyncSessionLocal() as session:
            stats = await AdmissionScoreService(session).upsert_from_rows(rows)
            await SettingsService(session).set_value(
                LAST_SCORES_IMPORT_KEY,
                now.isoformat(),
                "Время последней автозаливки проходных баллов",
            )
        logger.info("Проходные баллы обновлены: %s", stats)
    except Exception as e:
        logger.error("Не удалось обновить проходные баллы (старт продолжается): %s", e)


async def main():
    """Главная функция инициализации."""
    logger.info("Инициализация базы данных PostgreSQL...")

    # Создаём БД если нужно
    await create_database_if_not_exists()

    # Применяем миграции
    run_migrations()

    # Заливаем справочник факультетов — мягкая доливка недостающего
    await seed_reference_data()

    # Подтягиваем проходные баллы (после справочника — он нужен для матчинга)
    await seed_admission_scores()

    logger.info("Инициализация завершена успешно!")


if __name__ == "__main__":
    asyncio.run(main())
