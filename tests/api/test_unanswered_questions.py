"""Эндпоинт очереди вопросов без ответа (#258)."""

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from api.main import app
from api.routes.message_log import get_message_log_service


def test_unanswered_returns_grouped_questions():
    """Роут отдаёт группы с частотой и ограничивает список limit."""
    service = AsyncMock()
    service.get_question_groups = AsyncMock(
        return_value=[
            {
                "canonical": "стипендия для аспирантов",
                "question": "Есть ли стипендия для аспирантов?",
                "count": 4,
                "variants": ["Есть ли стипендия для аспирантов?"],
            },
            {
                "canonical": "апелляция",
                "question": "Как проходит апелляция?",
                "count": 1,
                "variants": ["Как проходит апелляция?"],
            },
        ]
    )
    app.dependency_overrides[get_message_log_service] = lambda: service

    try:
        response = TestClient(app).get("/api/v1/logs/unanswered?limit=1")
    finally:
        app.dependency_overrides.pop(get_message_log_service, None)

    assert response.status_code == 200
    assert response.json() == {
        "questions": [
            {
                "question": "Есть ли стипендия для аспирантов?",
                "count": 4,
                "variants": ["Есть ли стипендия для аспирантов?"],
            }
        ]
    }
    service.get_question_groups.assert_awaited_once_with(
        raw_limit=500,
        message_type="no_answer",
    )


def test_unanswered_empty_queue():
    service = AsyncMock()
    service.get_question_groups = AsyncMock(return_value=[])
    app.dependency_overrides[get_message_log_service] = lambda: service

    try:
        response = TestClient(app).get("/api/v1/logs/unanswered")
    finally:
        app.dependency_overrides.pop(get_message_log_service, None)

    assert response.status_code == 200
    assert response.json() == {"questions": []}
