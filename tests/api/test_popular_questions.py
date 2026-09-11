from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes import message_log as message_log_route
from api.routes.message_log import get_message_log_service


@pytest.fixture
def clustered(monkeypatch):
    """Подменяет кластеризацию: роут проверяем без эмбеддингов и без БД."""
    fake = AsyncMock(
        return_value=[
            {
                "question": "Как поступить в НГУ?",
                "count": 3,
                "variants": [
                    "Как поступить в НГУ?",
                    "Что нужно для поступления в НГУ?",
                ],
            }
        ]
    )
    monkeypatch.setattr(
        message_log_route.popular_questions_service, "get_popular_questions", fake
    )
    return fake


def test_get_popular_questions_returns_semantic_clusters(clustered):
    service = AsyncMock()
    app.dependency_overrides[get_message_log_service] = lambda: service

    try:
        response = TestClient(app).get(
            "/api/v1/logs/popular?limit=5&raw_limit=50&similarity_threshold=0.9"
        )
    finally:
        app.dependency_overrides.pop(get_message_log_service, None)

    assert response.status_code == 200
    assert response.json() == {
        "questions": [
            {
                "question": "Как поступить в НГУ?",
                "count": 3,
                "variants": [
                    "Как поступить в НГУ?",
                    "Что нужно для поступления в НГУ?",
                ],
            }
        ]
    }
    # Сервис БД передаётся первым позиционным аргументом, параметры — по имени.
    clustered.assert_awaited_once_with(
        service,
        limit=5,
        raw_limit=50,
        similarity_threshold=0.9,
    )
