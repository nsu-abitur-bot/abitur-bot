from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from api.main import app
from api.routes.message_log import get_message_log_service


def test_get_popular_questions_returns_semantic_clusters():
    service = AsyncMock()
    service.get_popular_questions.return_value = [
        {
            "question": "Как поступить в НГУ?",
            "count": 3,
            "variants": [
                "Как поступить в НГУ?",
                "Что нужно для поступления в НГУ?",
            ],
        }
    ]
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
    service.get_popular_questions.assert_awaited_once_with(
        limit=5,
        raw_limit=50,
        similarity_threshold=0.9,
    )
