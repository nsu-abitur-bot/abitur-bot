"""Обход конкурсных списков планировщиком: хэши, ошибки, уведомления (#307)."""

from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

import scheduler.main as scheduler_main
from db.postgres.dto import RatingEntry
from db.postgres.models import (
    Leaderboard,
    NotificationOutbox,
    NotificationStatus,
    User,
    UserRating,
)
from db.postgres.services.rating import RatingService
from parser.rating import ListFetchResult

MOCK_URL = "http://mock.local/api/rating"


@pytest.fixture
def maker(test_engine: AsyncEngine, monkeypatch) -> async_sessionmaker[AsyncSession]:
    m = async_sessionmaker(bind=test_engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(scheduler_main, "AsyncSessionLocal", m)
    return m


@pytest.fixture
def mock_mode(monkeypatch):
    """run_sync_job в mock-режиме: без таксономии и HTTP-сессии, парсер подменяем."""
    settings = {"RATING_USE_MOCK": "1", "MOCK_RATING_URL": MOCK_URL}
    monkeypatch.setattr(
        scheduler_main, "get_setting", lambda key, default="": settings.get(key, default)
    )
    monkeypatch.setattr(scheduler_main, "is_nsu", lambda: True)
    delivered = AsyncMock()
    monkeypatch.setattr(scheduler_main, "send_pending_notifications", delivered)
    return delivered


async def _seed_user(maker, telegram_id: int = 42, applicant_id: str = "1111111") -> None:
    async with maker() as s:
        s.add(User(telegram_id=telegram_id, applicant_id=applicant_id))
        await s.commit()


async def _seed_leaderboard(maker, content_hash: str | None = None) -> str:
    async with maker() as s:
        service = RatingService(s)
        lb = await service.get_or_create_leaderboard(MOCK_URL, direction_name="Физика")
        if content_hash is not None:
            lb.content_hash = content_hash
            await s.commit()
        return lb.id


async def _ratings(maker) -> list[UserRating]:
    async with maker() as s:
        return list((await s.execute(select(UserRating))).scalars())


async def test_unchanged_hash_skips_write(maker, mock_mode, monkeypatch):
    await _seed_user(maker)
    await _seed_leaderboard(maker, content_hash="h1")
    monkeypatch.setattr(
        scheduler_main,
        "parse_mock_rating_page",
        lambda url: ListFetchResult(
            entries=[RatingEntry("1111111", 3, "Подано", "Общий конкурс")],
            page_hash="h1",
            direction_name="Физика",
        ),
    )

    result = await scheduler_main.run_sync_job(bot=None)

    assert result.changed is False
    assert await _ratings(maker) == []
    # Хвостовой вызов в начале цикла; после пропуска рассылка не дёргается.
    assert mock_mode.await_count == 1


async def test_failed_list_keeps_hash(maker, mock_mode, monkeypatch):
    await _seed_leaderboard(maker, content_hash="h1")
    monkeypatch.setattr(
        scheduler_main,
        "parse_mock_rating_page",
        lambda url: ListFetchResult.failure("сайт недоступен"),
    )

    result = await scheduler_main.run_sync_job(bot=None)

    # 1 список из 1 упал — доля 1.0 выше порога 0.2, цикл неудачный.
    assert result.failed_lists == 1
    assert result.failed is True
    async with maker() as s:
        lb = (await s.execute(select(Leaderboard))).scalar_one()
        assert lb.content_hash == "h1"
    # Хэш не обновился — следующий цикл увидит список заново.
    assert await _ratings(maker) == []


async def test_changed_list_writes_entries_and_delivers(maker, mock_mode, monkeypatch):
    await _seed_user(maker)
    monkeypatch.setattr(
        scheduler_main,
        "parse_mock_rating_page",
        lambda url: ListFetchResult(
            entries=[RatingEntry("1111111", 3, "Подано", "Общий конкурс")],
            page_hash="h2",
            direction_name="Физика",
        ),
    )

    result = await scheduler_main.run_sync_job(bot=None)

    assert result.changed is True
    [rating] = await _ratings(maker)
    assert rating.place == 3
    async with maker() as s:
        [row] = (await s.execute(select(NotificationOutbox))).scalars()
        assert row.status == NotificationStatus.pending
        assert row.payload["new_place"] == 3
        assert row.payload["is_new"] is True
        lb = (await s.execute(select(Leaderboard))).scalar_one()
        assert lb.content_hash == "h2"
    # Хвостовой вызов + вызов после изменившегося списка.
    assert mock_mode.await_count >= 2


async def test_leaderboard_dedup_keeps_best_place(maker, mock_mode, monkeypatch):
    await _seed_user(maker)
    monkeypatch.setattr(
        scheduler_main,
        "parse_mock_rating_page",
        lambda url: ListFetchResult(
            entries=[
                RatingEntry("1111111", 5, "Подано", "Общий конкурс"),
                RatingEntry("1111111", 3, "Подано", "Отдельная квота"),
            ],
            page_hash="h3",
            direction_name="Физика",
        ),
    )

    result = await scheduler_main.run_sync_job(bot=None)

    assert result.changed is True
    ratings = await _ratings(maker)
    assert len(ratings) == 1
    # Дедуп по applicant_id: из двух номинаций остаётся лучшее (меньшее) место.
    assert ratings[0].place == 3
    assert ratings[0].competition_type == "Отдельная квота"
