"""Тесты заливки справочника факультетов (реальная тестовая БД).

Проверяем рабочий путь — `seed_faculties(soft=True)`, который зовётся при
старте из `bootstrap/seed.py`. Раньше здесь тестировалась
`seed_faculties_if_empty`: она заливала только в пустую базу и в проде не
вызывалась, а мягкий режим умеет больше — доливает недостающее, не затирая
правки из админки.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

import db.seed.load_faculties as seed_module
from db.postgres.services.faculty import FacultyService


@pytest.fixture
def maker(test_engine: AsyncEngine, monkeypatch):
    factory = async_sessionmaker(
        bind=test_engine, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr(seed_module, "AsyncSessionLocal", factory)
    return factory


@pytest.mark.asyncio
async def test_soft_seed_bootstraps_empty_db(maker):
    stats = await seed_module.seed_faculties(soft=True)

    assert stats.get("faculties_created", 0) > 0
    async with maker() as session:
        loaded = await FacultyService(session).get_all_faculties(only_active=False)
    assert len(loaded) > 0


@pytest.mark.asyncio
async def test_repeated_soft_seed_does_not_duplicate(maker):
    """Мягкая заливка идёт при каждом старте — дубликатов быть не должно."""
    await seed_module.seed_faculties(soft=True)
    async with maker() as session:
        after_first = await FacultyService(session).get_all_faculties(only_active=False)

    stats = await seed_module.seed_faculties(soft=True)

    async with maker() as session:
        after_second = await FacultyService(session).get_all_faculties(only_active=False)
    assert len(after_second) == len(after_first)
    assert stats.get("faculties_created", 0) == 0
