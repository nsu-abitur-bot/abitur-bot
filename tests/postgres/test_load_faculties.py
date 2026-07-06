"""Тесты автозаливки справочника факультетов (реальная тестовая БД)."""

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

import db.seed.load_faculties as seed_module
from db.postgres.services.faculty import FacultyService


@pytest.mark.asyncio
async def test_seed_if_empty_bootstraps_then_skips(test_engine: AsyncEngine, monkeypatch):
    maker = async_sessionmaker(
        bind=test_engine, class_=AsyncSession, expire_on_commit=False
    )
    monkeypatch.setattr(seed_module, "AsyncSessionLocal", maker)

    # На пустой БД — выполняется загрузка из сида.
    stats = await seed_module.seed_faculties_if_empty()
    assert "skipped" not in stats
    assert stats.get("faculties_created", 0) > 0

    async with maker() as session:
        loaded = await FacultyService(session).get_all_faculties(only_active=False)
    assert len(loaded) > 0

    # Повторный вызов на непустой БД — пропуск, данные не меняются.
    stats_again = await seed_module.seed_faculties_if_empty()
    assert "skipped" in stats_again

    async with maker() as session:
        after = await FacultyService(session).get_all_faculties(only_active=False)
    assert len(after) == len(loaded)
