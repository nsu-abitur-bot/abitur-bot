"""Очередь уведомлений: запись вместе с местами и правила повторов."""

from typing import Any, cast

import pytest
from aiogram import Bot
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

import bot.notifications as notifications_module
from db.postgres.dto import RatingEntry
from db.postgres.models import Leaderboard, NotificationOutbox, NotificationStatus, User
from db.postgres.services.notification_outbox import (
    MAX_DELIVERY_ATTEMPTS,
    NotificationOutboxService,
)
from db.postgres.services.rating import RatingService


def _as_bot(fake: Any) -> Bot:
    """Фейк повторяет только send_message — этого рассылке достаточно."""
    return cast(Bot, fake)


class FailingBot:
    async def send_message(self, chat_id: int, text: str) -> None:
        raise RuntimeError("timeout")


@pytest.fixture
def maker(test_engine: AsyncEngine, monkeypatch) -> async_sessionmaker[AsyncSession]:
    m = async_sessionmaker(bind=test_engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(notifications_module, "AsyncSessionLocal", m)
    monkeypatch.setattr(notifications_module, "SEND_DELAY_SECONDS", 0)
    return m


async def _seed(maker, telegram_id: int | None = 42) -> None:
    async with maker() as s:
        s.add(User(telegram_id=telegram_id, applicant_id="1111111"))
        await s.commit()
        service = RatingService(s)
        lb = await service.get_or_create_leaderboard("https://x/lb", "Физика")
        await service.update_ratings_from_entries(
            lb.id,
            [RatingEntry("1111111", 3, "Подано", "Общий конкурс")],
            content_hash="h1",
            enqueue_notifications=True,
        )


async def _rows(maker) -> list[NotificationOutbox]:
    async with maker() as s:
        return list((await s.execute(select(NotificationOutbox))).scalars())


@pytest.mark.asyncio
async def test_update_writes_outbox_and_hash_in_one_commit(maker):
    await _seed(maker)
    rows = await _rows(maker)
    assert len(rows) == 1
    assert rows[0].status == NotificationStatus.pending
    assert rows[0].payload["new_place"] == 3
    assert rows[0].payload["is_new"] is True
    async with maker() as s:
        lb = (await s.execute(select(Leaderboard))).scalar_one()
        assert lb.content_hash == "h1"


@pytest.mark.asyncio
async def test_without_enqueue_flag_outbox_untouched(maker):
    async with maker() as s:
        s.add(User(telegram_id=1, applicant_id="2222222"))
        await s.commit()
        service = RatingService(s)
        lb = await service.get_or_create_leaderboard("https://x/lb2")
        stats = await service.update_ratings_from_entries(
            lb.id, [RatingEntry("2222222", 1)]
        )
    assert len(stats["notifications"]) == 1
    assert await _rows(maker) == []


@pytest.mark.asyncio
async def test_gives_up_after_max_attempts(maker):
    await _seed(maker)
    for _ in range(MAX_DELIVERY_ATTEMPTS):
        assert (
            await notifications_module.send_pending_notifications(_as_bot(FailingBot()))
            == 0
        )

    [row] = await _rows(maker)
    assert row.status == NotificationStatus.failed
    assert row.attempts == MAX_DELIVERY_ATTEMPTS
    async with maker() as s:
        assert await NotificationOutboxService(s).count_pending() == 0


@pytest.mark.asyncio
async def test_user_without_channel_is_not_retried(maker):
    await _seed(maker, telegram_id=None)
    await notifications_module.send_pending_notifications(_as_bot(FailingBot()))
    [row] = await _rows(maker)
    assert row.status == NotificationStatus.failed
    assert row.attempts == 0
