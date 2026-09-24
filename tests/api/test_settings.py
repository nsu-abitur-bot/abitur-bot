from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from api.main import app
from api.routes.settings import get_settings_service
from api.schemas.rate_limit import RateLimitSettings
from rag.crag import CragConfig


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


class TestCragSettings:
    def setup_method(self):
        self.client = TestClient(app)

    def teardown_method(self):
        app.dependency_overrides.pop(get_settings_service, None)

    def test_get_crag_settings(self):
        cfg = CragConfig(
            enabled=False,
            relevance_threshold=0.7,
            min_chunks=3,
            allow_refine=False,
            use_faculty_table=True,
            max_graded_chunks=8,
        )
        with patch("api.routes.settings.load_crag_config", AsyncMock(return_value=cfg)):
            response = self.client.get("/api/v1/settings/crag")

        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is False
        assert data["relevance_threshold"] == 0.7
        assert data["min_chunks"] == 3
        assert data["max_graded_chunks"] == 8

    def test_update_crag_settings(self):
        service = AsyncMock()
        app.dependency_overrides[get_settings_service] = lambda: service
        updated = CragConfig(
            enabled=True,
            relevance_threshold=0.6,
            min_chunks=1,
            allow_refine=True,
            use_faculty_table=False,
            max_graded_chunks=10,
        )
        with patch(
            "api.routes.settings.load_crag_config", AsyncMock(return_value=updated)
        ):
            response = self.client.put(
                "/api/v1/settings/crag",
                json={
                    "enabled": True,
                    "relevance_threshold": 0.6,
                    "min_chunks": 1,
                    "allow_refine": True,
                    "use_faculty_table": False,
                    "max_graded_chunks": 10,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["use_faculty_table"] is False
        assert data["relevance_threshold"] == 0.6
        service.update_crag_settings.assert_awaited_once()

    def test_update_crag_settings_invalid_threshold(self):
        response = self.client.put(
            "/api/v1/settings/crag",
            json={
                "enabled": True,
                "relevance_threshold": 2,
                "min_chunks": 1,
                "allow_refine": True,
                "use_faculty_table": True,
                "max_graded_chunks": 10,
            },
        )

        assert response.status_code == 422


class TestFaqSettings:
    def setup_method(self):
        self.client = TestClient(app)

    def teardown_method(self):
        app.dependency_overrides.pop(get_settings_service, None)

    def test_get_faq_settings(self):
        with patch("api.routes.settings.load_faq_threshold", AsyncMock(return_value=0.8)):
            response = self.client.get("/api/v1/settings/faq")

        assert response.status_code == 200
        assert response.json() == {"similarity_threshold": 0.8}

    def test_update_faq_settings(self):
        service = AsyncMock()
        app.dependency_overrides[get_settings_service] = lambda: service
        with (
            patch(
                "api.routes.settings.load_faq_threshold",
                AsyncMock(return_value=0.75),
            ),
            patch("api.routes.settings.get_faq_matcher") as get_matcher,
        ):
            response = self.client.put(
                "/api/v1/settings/faq",
                json={"similarity_threshold": 0.75},
            )

        assert response.status_code == 200
        assert response.json() == {"similarity_threshold": 0.75}
        service.update_faq_settings.assert_awaited_once_with(similarity_threshold=0.75)
        # Порог применяется к матчеру этого процесса сразу.
        assert get_matcher.return_value.threshold == 0.75

    def test_update_faq_settings_invalid_threshold(self):
        response = self.client.put(
            "/api/v1/settings/faq",
            json={"similarity_threshold": 2},
        )

        assert response.status_code == 422
