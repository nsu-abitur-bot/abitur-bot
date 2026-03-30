"""Тесты для FeedbackService — сервис хранения обратной связи в PostgreSQL."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.services.feedback import FeedbackService
from db.postgres.services.user import UserService


@pytest.mark.asyncio
async def test_create_feedback(session: AsyncSession):
    """Тест сохранения обратной связи."""
    user_service = UserService(session)
    await user_service.create_user(user_id=200001)

    feedback_service = FeedbackService(session)
    feedback = await feedback_service.create_feedback(
        user_id=200001,
        session_id="200001",
        text="Очень полезный бот, добавьте больше примеров по документам.",
    )

    assert feedback is not None
    assert feedback.user_id == 200001
    assert feedback.session_id == "200001"
    assert (
        feedback.text == "Очень полезный бот, добавьте больше примеров по документам."
    )
    assert feedback.created_at is not None
