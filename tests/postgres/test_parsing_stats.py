"""Интеграционные тесты статистики парсинга списков на реальной PostgreSQL (#487)."""

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import select, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

import scheduler.main as scheduler_main
from api.auth.dependencies import get_current_admin
from api.main import app
from db.postgres.db import get_db
from db.postgres.dto import RatingEntry
from db.postgres.models import Leaderboard
from db.postgres.services.rating import RatingService
from parser.rating import ListFetchResult

MOCK_URL = "http://mock.local/api/rating"


def _run_migration(connection: Connection, direction: str) -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "db/alembic/versions"
        / "2026_11_28_1200-c4e8f1a2b3d4_leaderboard_parsing_timestamps.py"
    )
    spec = importlib.util.spec_from_file_location("leaderboard_parsing_migration", path)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    with Operations.context(MigrationContext.configure(connection)):
        getattr(migration, direction)()


@pytest.fixture
def parsing_env(test_engine: AsyncEngine, monkeypatch):
    maker = async_sessionmaker(
        bind=test_engine, class_=AsyncSession, expire_on_commit=False
    )
    settings = {"RATING_USE_MOCK": "1", "MOCK_RATING_URL": MOCK_URL}
    monkeypatch.setattr(scheduler_main, "AsyncSessionLocal", maker)
    monkeypatch.setattr(
        scheduler_main, "get_setting", lambda key, default="": settings.get(key, default)
    )
    monkeypatch.setattr(scheduler_main, "is_nsu", lambda: True)
    monkeypatch.setattr(scheduler_main, "send_pending_notifications", AsyncMock())
    monkeypatch.setitem(app.dependency_overrides, get_current_admin, lambda: None)

    async def _override_get_db():
        async with maker() as s:
            yield s

    monkeypatch.setitem(app.dependency_overrides, get_db, _override_get_db)
    return maker


@pytest.mark.asyncio
async def test_parsing_stats_tracks_real_parse_and_check_lifecycle(
    parsing_env, monkeypatch
):
    """Непарсенный список, разбор, неизменившийся список, ошибка и метаданные."""
    maker = parsing_env
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # 0. Пустая БД.
        empty = await client.get("/api/v1/stats/parsing")
        assert empty.status_code == 200
        assert empty.json() == {
            "last_parsed_at": None,
            "data_updated_at": None,
            "total_lists": 0,
        }

        # 1. Новый активный список без единого парсинга — даты должны быть None,
        # хотя updated_at уже заполнен default-значением.
        async with maker() as s:
            created_lb = await RatingService(s).get_or_create_leaderboard(
                MOCK_URL, direction_name="Физика"
            )
            assert created_lb.updated_at is not None
        unparsed = await client.get("/api/v1/stats/parsing")
        assert unparsed.status_code == 200
        assert unparsed.json() == {
            "last_parsed_at": None,
            "data_updated_at": None,
            "total_lists": 1,
        }

        # 2. Переименование метаданных без парсинга не создаёт дату парсинга.
        async with maker() as s:
            await RatingService(s).get_or_create_leaderboard(
                MOCK_URL, direction_name="Физика (обновлено)"
            )
        after_meta_before_parse = await client.get("/api/v1/stats/parsing")
        assert after_meta_before_parse.json() == {
            "last_parsed_at": None,
            "data_updated_at": None,
            "total_lists": 1,
        }

        # 3. Успешный разбор с новыми данными обновляет обе даты.
        t_parse = datetime(2026, 11, 20, 10, 0, 0)
        monkeypatch.setattr("db.postgres.services.rating.timestamp", lambda: t_parse)
        monkeypatch.setattr(
            scheduler_main,
            "parse_mock_rating_page",
            lambda url: ListFetchResult(
                entries=[RatingEntry("1111111", 3, "Подано", "Общий конкурс")],
                page_hash="h1",
                direction_name="Физика (обновлено)",
            ),
        )
        res = await scheduler_main.run_sync_job(bot=None)
        assert res.changed is True
        after_parse = (await client.get("/api/v1/stats/parsing")).json()
        assert after_parse == {
            "last_parsed_at": t_parse.isoformat(),
            "data_updated_at": t_parse.isoformat(),
            "total_lists": 1,
        }

        # 4. Повторное изменение метаданных не меняет даты парсинга.
        t_meta = t_parse + timedelta(hours=3)
        async with maker() as s:
            lb = await RatingService(s).get_or_create_leaderboard(
                MOCK_URL, direction_name="Физика переименованная"
            )
            lb.updated_at = t_meta
            await s.commit()
        after_meta = (await client.get("/api/v1/stats/parsing")).json()
        assert after_meta == after_parse

        # 5. Успешная проверка неизменившегося списка: обновляет last_parsed_at,
        # но сохраняет data_updated_at на времени реального изменения данных.
        t_check = t_parse + timedelta(minutes=30)
        monkeypatch.setattr("db.postgres.services.rating.timestamp", lambda: t_check)
        monkeypatch.setattr(
            scheduler_main,
            "parse_mock_rating_page",
            lambda url: ListFetchResult(
                entries=[RatingEntry("1111111", 3, "Подано", "Общий конкурс")],
                page_hash="h1",
                direction_name="Физика переименованная",
            ),
        )
        res_same = await scheduler_main.run_sync_job(bot=None)
        assert res_same.changed is False
        after_check = (await client.get("/api/v1/stats/parsing")).json()
        assert after_check == {
            "last_parsed_at": t_check.isoformat(),
            "data_updated_at": t_parse.isoformat(),
            "total_lists": 1,
        }

        # 6. Ошибка загрузки не меняет ни одну дату.
        t_err = t_parse + timedelta(minutes=45)
        monkeypatch.setattr("db.postgres.services.rating.timestamp", lambda: t_err)
        monkeypatch.setattr(
            scheduler_main,
            "parse_mock_rating_page",
            lambda url: ListFetchResult.failure("таймаут"),
        )
        res_err = await scheduler_main.run_sync_job(bot=None)
        assert res_err.failed is True
        after_err = (await client.get("/api/v1/stats/parsing")).json()
        assert after_err == after_check

        # 7. Неактивные списки не участвуют ни в count, ни в агрегации дат.
        future = t_err + timedelta(hours=1)
        async with maker() as s:
            inactive = Leaderboard(
                url="https://example.test/inactive",
                direction="Архив",
                is_active=False,
                last_parsed_at=future,
                data_updated_at=future,
                updated_at=future,
            )
            s.add(inactive)
            await s.commit()
            row = (
                await s.execute(select(Leaderboard).where(Leaderboard.url == MOCK_URL))
            ).scalar_one()
            assert row.direction == "Физика переименованная"
        after_inactive = (await client.get("/api/v1/stats/parsing")).json()
        assert after_inactive == after_check


@pytest.mark.asyncio
async def test_leaderboard_parsing_timestamps_migration_ignores_legacy_updated_at(
    test_engine: AsyncEngine,
):
    """Старая дата updated_at не считается достоверной историей парсинга при миграции."""
    legacy_updated = datetime(2026, 11, 20, 9, 0, 0)
    async with test_engine.begin() as connection:
        await connection.run_sync(_run_migration, "downgrade")
        await connection.execute(
            text(
                """
                INSERT INTO leaderboard (
                    id, url, direction, content_hash, is_active, updated_at
                )
                VALUES (
                    'legacy-lb', 'https://example.test/legacy',
                    'Математика', 'h0', true, :dt
                )
                """
            ),
            {"dt": legacy_updated},
        )
        await connection.run_sync(_run_migration, "upgrade")
        row = (
            (
                await connection.execute(
                    text(
                        "SELECT last_parsed_at, data_updated_at, updated_at "
                        "FROM leaderboard WHERE id = 'legacy-lb'"
                    )
                )
            )
            .mappings()
            .one()
        )
        assert row["last_parsed_at"] is None
        assert row["data_updated_at"] is None
        assert row["updated_at"] == legacy_updated
        await connection.run_sync(_run_migration, "downgrade")
        await connection.run_sync(_run_migration, "upgrade")
