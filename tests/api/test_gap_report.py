"""Отчёт о пробелах наполнения (#262)."""

from datetime import datetime
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

import api.routes.gap_report as gap_report_route
from api.main import app


def _fake_services(monkeypatch) -> AsyncMock:
    """Подменяет сборку отчёта целиком: роут проверяет контракт ответа."""
    fake = AsyncMock(
        return_value={
            "period_days": 30,
            "generated_at": datetime(2026, 11, 11, 12, 0, 0),
            "topics": [{"topic": "Документы", "count": 42}],
            "popular": [
                {
                    "question": "Когда подать документы?",
                    "count": 25,
                    "variants": ["Когда подать документы?"],
                }
            ],
            "unanswered": [
                {
                    "question": "Есть ли стипендия для аспирантов?",
                    "count": 7,
                    "variants": ["Есть ли стипендия для аспирантов?"],
                }
            ],
            "quality": {
                "questions": 100,
                "answers": 90,
                "no_answer": 8,
                "no_answer_share": 0.08,
                "errors": 2,
                "error_share": 0.02,
            },
            "disliked": [
                {"question": "Как проходит апелляция?", "negative": 3, "rated": 4}
            ],
        }
    )
    monkeypatch.setattr(gap_report_route, "build_gap_report", fake)
    return fake


def test_gap_report_returns_all_sections(monkeypatch):
    fake = _fake_services(monkeypatch)

    response = TestClient(app).get("/api/v1/reports/gaps?days=14&limit=5")

    assert response.status_code == 200
    body = response.json()
    # Тело — из фикстуры; параметры запроса проверены ниже через kwargs.
    assert body["period_days"] == 30
    assert body["topics"] == [{"topic": "Документы", "count": 42}]
    assert body["popular"][0]["count"] == 25
    assert body["unanswered"][0]["question"] == "Есть ли стипендия для аспирантов?"
    assert body["quality"]["no_answer_share"] == 0.08
    assert body["disliked"][0]["negative"] == 3
    fake.assert_awaited_once()
    # Параметры запроса дошли до сборки отчёта.
    assert fake.await_args is not None
    assert fake.await_args.kwargs == {"days": 14, "limit": 5}


def test_gap_report_rejects_bad_params(monkeypatch):
    _fake_services(monkeypatch)

    for query in ("days=0", "days=400", "limit=0", "limit=99"):
        response = TestClient(app).get(f"/api/v1/reports/gaps?{query}")
        assert response.status_code == 422, query
