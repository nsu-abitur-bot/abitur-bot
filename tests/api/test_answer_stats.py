"""Эндпоинт метрик ответов для дашборда наблюдаемости (#242)."""

from datetime import datetime
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from api.main import app
from api.routes.message_log import get_message_log_service


def _stats() -> dict:
    return {
        "total_questions": 10,
        "total_answers": 8,
        "total_no_answer": 2,
        "no_answer_share": 0.2,
        "total_errors": 1,
        "error_share": 0.1,
        "buckets": [
            {
                "period": datetime(2026, 11, 9),
                "questions": 10,
                "answers": 8,
                "no_answer": 2,
                "no_answer_share": 0.2,
                "errors": 1,
                "error_share": 0.1,
                "latency_ms_p50": 1500.0,
                "latency_ms_p95": 4200.0,
            }
        ],
    }


def test_answer_stats_returns_buckets():
    service = AsyncMock()
    service.get_answer_stats = AsyncMock(return_value=_stats())
    app.dependency_overrides[get_message_log_service] = lambda: service

    try:
        response = TestClient(app).get(
            "/api/v1/logs/answer-stats",
            params={"start": "2026-11-09T00:00:00", "group_by": "day"},
        )
    finally:
        app.dependency_overrides.pop(get_message_log_service, None)

    assert response.status_code == 200
    body = response.json()
    assert body["total_questions"] == 10
    assert body["no_answer_share"] == 0.2
    assert body["buckets"][0]["latency_ms_p95"] == 4200.0
    service.get_answer_stats.assert_awaited_once_with(
        start=datetime(2026, 11, 9, 0, 0),
        end=None,
        group_by="day",
    )


def test_answer_stats_rejects_unknown_group_by():
    service = AsyncMock()
    app.dependency_overrides[get_message_log_service] = lambda: service

    try:
        response = TestClient(app).get("/api/v1/logs/answer-stats?group_by=minute")
    finally:
        app.dependency_overrides.pop(get_message_log_service, None)

    assert response.status_code == 400
    service.get_answer_stats.assert_not_awaited()
