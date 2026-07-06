"""Загрузчик справочника факультетов и направлений из JSON-сида.

Запуск (через uv, как требует CLAUDE.md)::

    uv run python -m db.seed.load_faculties
    uv run python -m db.seed.load_faculties path/to/faculties.json

Скрипт идемпотентен: повторный запуск обновляет существующие записи, а не
создаёт дубликаты (см. FacultyService.upsert_from_seed).
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any

from db.postgres.db import AsyncSessionLocal
from db.postgres.services.faculty import FacultyService

logger = logging.getLogger(__name__)

DEFAULT_SEED_PATH = os.path.join(os.path.dirname(__file__), "faculties.json")


def load_seed_file(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    faculties = data.get("faculties")
    if not isinstance(faculties, list):
        raise ValueError(f"Сид-файл {path} должен содержать ключ 'faculties' со списком.")
    return faculties


async def seed_faculties(path: str = DEFAULT_SEED_PATH) -> dict[str, int]:
    faculties = load_seed_file(path)
    async with AsyncSessionLocal() as session:
        service = FacultyService(session)
        stats = await service.upsert_from_seed(faculties)
    return stats


async def seed_faculties_if_empty(path: str = DEFAULT_SEED_PATH) -> dict[str, int]:
    """Заливает справочник ТОЛЬКО если таблица факультетов пуста.

    Нужна для автозаливки при старте контейнера: на свежей БД справочник
    подхватится сам, а на существующей — не затрёт правки, сделанные через
    админку (справочником владеет админ, сид — лишь первичный бутстрап).
    """
    async with AsyncSessionLocal() as session:
        existing = await FacultyService(session).get_all_faculties(only_active=False)
    if existing:
        logger.info(
            "Справочник факультетов уже заполнен (%d) — автозаливка пропущена",
            len(existing),
        )
        return {"skipped": len(existing)}
    logger.info("Справочник факультетов пуст — выполняю автозаливку из сида")
    return await seed_faculties(path)


async def _main() -> None:
    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SEED_PATH
    logger.info("Загрузка справочника факультетов из %s", path)
    stats = await seed_faculties(path)
    logger.info("Готово: %s", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
