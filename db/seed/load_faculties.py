"""Загрузчик справочника факультетов и направлений из JSON-сида.

Запуск (через uv, как требует AGENTS.md)::

    uv run python -m db.seed.load_faculties                  # мягкий режим
    uv run python -m db.seed.load_faculties --force          # жёсткая перезапись
    uv run python -m db.seed.load_faculties path/to/faculties.json

По умолчанию режим МЯГКИЙ: добавляется только недостающее (новые факультеты,
алиасы, направления, код у направления без кода), существующие данные и правки
из админки не затираются. ``--force`` перезаписывает алиасы/коды значениями из
сида и включает is_active. См. FacultyService.upsert_from_seed.
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


async def seed_faculties(
    path: str = DEFAULT_SEED_PATH, soft: bool = False
) -> dict[str, int]:
    """Заливает справочник из сида. ``soft=True`` — только доливка недостающего."""
    faculties = load_seed_file(path)
    async with AsyncSessionLocal() as session:
        service = FacultyService(session)
        stats = await service.upsert_from_seed(faculties, soft=soft)
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
    args = [a for a in sys.argv[1:] if a != "--force"]
    force = "--force" in sys.argv[1:]
    path = args[0] if args else DEFAULT_SEED_PATH
    mode = "жёсткий (перезапись)" if force else "мягкий (только доливка)"
    logger.info("Загрузка справочника факультетов из %s, режим: %s", path, mode)
    stats = await seed_faculties(path, soft=not force)
    logger.info("Готово: %s", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
