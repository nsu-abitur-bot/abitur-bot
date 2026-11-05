"""Отправка уведомлений в Telegram: повтор при flood wait (429, #292)."""

from typing import Any, cast

import pytest
from aiogram import Bot
from aiogram.exceptions import TelegramRetryAfter

import bot.notifications as notifications_module
from bot.notifications import DeliveryStatus, send_to_user
from db.postgres.models import User


def _retry_after(seconds: int) -> TelegramRetryAfter:
    return TelegramRetryAfter(
        method=cast(Any, None), message="Too Many Requests", retry_after=seconds
    )


class FloodWaitBot:
    """Поднимает 429 первые ``fails`` вызовов, дальше отправляет успешно."""

    def __init__(self, fails: int):
        self.fails = fails
        self.calls = 0

    async def send_message(self, chat_id: int, text: str) -> None:
        self.calls += 1
        if self.calls <= self.fails:
            raise _retry_after(0)


class AlwaysFailingBot:
    def __init__(self, error: Exception):
        self.error = error
        self.calls = 0

    async def send_message(self, chat_id: int, text: str) -> None:
        self.calls += 1
        raise self.error


@pytest.fixture(autouse=True)
def _no_flood_margin(monkeypatch) -> None:
    # Чтобы тесты не ждали реальный запас к retry_after.
    monkeypatch.setattr(notifications_module, "FLOOD_WAIT_MARGIN_SECONDS", 0.0)


def _user() -> User:
    return User(user_id=1, telegram_id=42, applicant_id="1111111")


async def test_flood_wait_is_retried_and_delivered():
    bot = FloodWaitBot(fails=1)
    result = await send_to_user(cast(Bot, bot), _user(), "текст")
    assert result.status is DeliveryStatus.DELIVERED
    assert bot.calls == 2


async def test_flood_wait_gives_up_after_attempt_limit():
    bot = FloodWaitBot(fails=notifications_module.TELEGRAM_FLOOD_ATTEMPTS)
    result = await send_to_user(cast(Bot, bot), _user(), "текст")
    assert result.status is DeliveryStatus.FAILED
    assert bot.calls == notifications_module.TELEGRAM_FLOOD_ATTEMPTS
    assert "telegram" in (result.error or "")


async def test_other_errors_are_not_retried():
    bot = AlwaysFailingBot(RuntimeError("timeout"))
    result = await send_to_user(cast(Bot, bot), _user(), "текст")
    assert result.status is DeliveryStatus.FAILED
    assert bot.calls == 1
