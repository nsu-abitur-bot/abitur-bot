"""Заливка справочных данных при старте: факультеты и проходные баллы.

Ни один из сидов не критичен для работы: при ошибке пишем в лог и продолжаем
старт, иначе недоступный сайт НГУ означал бы, что бот вообще не поднимется.
"""

import logging
import os
from datetime import UTC, datetime, timedelta

from db.postgres.db import AsyncSessionLocal
from db.postgres.services.admission_score import AdmissionScoreService
from db.postgres.services.settings import SettingsService
from db.seed.load_faculties import seed_faculties
from parser.scores import DEFAULT_SCORES_URL, parse_scores

logger = logging.getLogger(__name__)

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


async def seed_reference_data() -> None:
    """Заливает справочник факультетов.

    По умолчанию режим МЯГКИЙ: доливаем только недостающее (новые факультеты,
    алиасы, направления), не затирая правки из админки. На пустой БД это
    равносильно первичному бутстрапу, а на существующей — подхватывает новые
    записи сида при каждом деплое. SEED_FACULTIES_FORCE=1 — жёстко перезалить.
    """
    try:
        if _env_flag("SEED_FACULTIES_FORCE", False):
            stats = await seed_faculties(soft=False)
            logger.info("Справочник факультетов принудительно перезалит: %s", stats)
        else:
            stats = await seed_faculties(soft=True)
            logger.info("Мягкая доливка справочника факультетов: %s", stats)
    except Exception as e:
        logger.error(
            "Не удалось загрузить справочник факультетов (старт продолжается): %s", e
        )


async def _scores_are_fresh(now: datetime) -> bool:
    """Свежие ли данные, чтобы не дёргать сайт НГУ на каждом рестарте."""
    async with AsyncSessionLocal() as session:
        raw_last = await SettingsService(session).get_value(LAST_SCORES_IMPORT_KEY)
    if not raw_last:
        return False

    try:
        age = now - datetime.fromisoformat(raw_last)
    except ValueError:
        return False

    if age < timedelta(hours=_scores_max_age_hours()):
        logger.info("Проходные баллы заливались %s назад — пропускаю", age)
        return True
    return False


async def seed_admission_scores() -> None:
    """Подтягивает проходные баллы прошлых лет со страницы итогов приёма НГУ.

    Upsert идемпотентен по (program_id, year, form), поэтому повторные запуски
    безопасны. Отключить целиком: SEED_ADMISSION_SCORES=0. Игнорировать
    свежесть: SEED_ADMISSION_SCORES_FORCE=1.
    """
    if not _env_flag("SEED_ADMISSION_SCORES", True):
        logger.info("Автозаливка проходных баллов отключена (SEED_ADMISSION_SCORES=0)")
        return

    try:
        now = datetime.now(UTC).replace(tzinfo=None)

        if not _env_flag("SEED_ADMISSION_SCORES_FORCE", False) and (
            await _scores_are_fresh(now)
        ):
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
