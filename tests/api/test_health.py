"""Эндпоинт живости API (#241)."""

from fastapi.testclient import TestClient

from api.main import app


def test_health_returns_ok_without_auth():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
