from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from api.main import app
from api.routes.message_log import get_message_log_service


def _stats_payload(**overrides):
    payload = {
        "total_questions": 4,
        "total_hits": 1,
        "hit_rate": 0.25,
        "buckets": [
            {
                "period": "2026-09-24T00:00:00",
                "questions": 4,
                "hits": 1,
                "hit_rate": 0.25,
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_faq_stats_returns_hit_rate():
    service = AsyncMock()
    service.get_faq_hit_stats.return_value = _stats_payload()
    app.dependency_overrides[get_message_log_service] = lambda: service

    try:
        response = TestClient(app).get("/api/v1/logs/faq-stats")
    finally:
        app.dependency_overrides.pop(get_message_log_service, None)

    assert response.status_code == 200
    data = response.json()
    assert data["total_questions"] == 4
    assert data["total_hits"] == 1
    assert data["hit_rate"] == 0.25
    assert data["buckets"][0]["questions"] == 4
    service.get_faq_hit_stats.assert_awaited_once()


def test_faq_stats_no_questions_hit_rate_is_null():
    service = AsyncMock()
    service.get_faq_hit_stats.return_value = _stats_payload(
        total_questions=0, total_hits=0, hit_rate=None, buckets=[]
    )
    app.dependency_overrides[get_message_log_service] = lambda: service

    try:
        response = TestClient(app).get("/api/v1/logs/faq-stats")
    finally:
        app.dependency_overrides.pop(get_message_log_service, None)

    assert response.status_code == 200
    assert response.json()["hit_rate"] is None


def test_faq_stats_rejects_unknown_group_by():
    response = TestClient(app).get("/api/v1/logs/faq-stats?group_by=century")

    assert response.status_code == 400
