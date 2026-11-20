"""Тесты API статистики парсинга рейтинговых списков (#4)."""

from collections import namedtuple
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from api.main import app
from db.postgres.db import get_db

Row = namedtuple("Row", ["last_parsed_at", "total_lists"])


def test_parsing_stats_empty():
    db = AsyncMock()
    result = MagicMock()
    result.one_or_none.return_value = Row(last_parsed_at=None, total_lists=0)
    db.execute.return_value = result

    app.dependency_overrides[get_db] = lambda: db
    try:
        response = TestClient(app).get("/api/v1/stats/parsing")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    assert response.json() == {"last_parsed_at": None, "total_lists": 0}


def test_parsing_stats_with_data():
    db = AsyncMock()
    result = MagicMock()
    dt = datetime(2026, 11, 20, 14, 30, 0)
    result.one_or_none.return_value = Row(last_parsed_at=dt, total_lists=15)
    db.execute.return_value = result

    app.dependency_overrides[get_db] = lambda: db
    try:
        response = TestClient(app).get("/api/v1/stats/parsing")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    assert response.json() == {
        "last_parsed_at": "2026-11-20T14:30:00",
        "total_lists": 15,
    }
