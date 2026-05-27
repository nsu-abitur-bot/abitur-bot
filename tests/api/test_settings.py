from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from api.main import app
from api.routes.settings import get_settings_service
from api.schemas.rate_limit import RateLimitSettings


class TestRateLimitSettings:
    def setup_method(self):
        self.client = TestClient(app)

    def teardown_method(self):
        app.dependency_overrides.pop(get_settings_service, None)

    def test_get_rate_limit_settings(self):
        service = AsyncMock()
        service.get_rate_limit_settings.return_value = RateLimitSettings(
            system_requests_per_day=10000,
            user_requests_per_day=100,
        )
        app.dependency_overrides[get_settings_service] = lambda: service

        response = self.client.get("/api/v1/settings/rate-limit")

        assert response.status_code == 200
        assert response.json() == {
            "system_requests_per_day": 10000,
            "user_requests_per_day": 100,
        }
        service.get_rate_limit_settings.assert_awaited_once()

    def test_update_rate_limit_settings(self):
        service = AsyncMock()
        service.update_rate_limit_settings.return_value = RateLimitSettings(
            system_requests_per_day=5000,
            user_requests_per_day=50,
        )
        app.dependency_overrides[get_settings_service] = lambda: service

        response = self.client.put(
            "/api/v1/settings/rate-limit",
            json={
                "system_requests_per_day": 5000,
                "user_requests_per_day": 50,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["system_requests_per_day"] == 5000
        assert data["user_requests_per_day"] == 50
        service.update_rate_limit_settings.assert_awaited_once()

    def test_update_rate_limit_settings_invalid_value(self):
        response = self.client.put(
            "/api/v1/settings/rate-limit",
            json={
                "system_requests_per_day": 0,
                "user_requests_per_day": 100,
            },
        )

        assert response.status_code == 422
