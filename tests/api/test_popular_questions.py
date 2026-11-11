from unittest.mock import AsyncMock

import fakeredis.aioredis
import httpx
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.services.popular_statistics import PopularStatistics, get_popular_statistics


def test_get_popular_questions_reads_prepared_snapshot():
    statistics = AsyncMock()
    statistics.read.return_value = {
        "questions": [{"question": "Как поступить?", "count": 3, "variants": []}],
        "status": "ready",
        "updated_at": "2026-11-11T07:00:00Z",
    }
    app.dependency_overrides[get_popular_statistics] = lambda: statistics
    try:
        response = TestClient(app).get(
            "/api/v1/logs/popular?limit=5&raw_limit=50&similarity_threshold=0.9"
        )
    finally:
        app.dependency_overrides.pop(get_popular_statistics, None)
    assert response.status_code == 200
    assert response.json()["questions"][0]["count"] == 3
    assert response.json()["status"] == "ready"
    statistics.read.assert_awaited_once_with(
        limit=5, raw_limit=50, similarity_threshold=0.9
    )


@pytest.mark.asyncio
async def test_uncached_http_request_never_opens_database_or_calls_models():
    def forbidden():
        raise AssertionError("GET tried to prepare statistics")

    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    statistics = PopularStatistics(redis, forbidden)
    app.dependency_overrides[get_popular_statistics] = lambda: statistics
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/logs/popular?limit=5")
        assert response.status_code == 200
        assert response.json() == {
            "questions": [],
            "status": "pending",
            "updated_at": None,
        }
    finally:
        app.dependency_overrides.pop(get_popular_statistics, None)
        await redis.aclose()
