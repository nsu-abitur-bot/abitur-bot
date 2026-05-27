from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from api.main import app


def _make_report(report_id: int = 1, status: str = "open"):
    return SimpleNamespace(
        id=report_id,
        user_id=123,
        session_id="session-123",
        channel="telegram",
        comment="Ответ был неверным",
        question="Вопрос пользователя",
        bot_response="Ответ бота",
        logs_snapshot=[{"message_type": "llm_response", "content": "Ответ бота"}],
        status=status,
        created_at=datetime(2026, 5, 26, 12, 0, 0),
        reviewed_at=None,
    )


class TestFeedbackReportsEndpoint:
    def setup_method(self):
        self.client = TestClient(app)

    @patch("api.routes.feedback_report.FeedbackReportService")
    def test_list_feedback_reports(self, mock_service_cls):
        mock_service = AsyncMock()
        mock_service.list_reports.return_value = ([_make_report()], 1)
        mock_service_cls.return_value = mock_service

        response = self.client.get("/api/v1/feedback")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["reports"][0]["comment"] == "Ответ был неверным"
        assert "logs_snapshot" not in data["reports"][0]

    @patch("api.routes.feedback_report.FeedbackReportService")
    def test_get_feedback_report(self, mock_service_cls):
        mock_service = AsyncMock()
        mock_service.get_report.return_value = _make_report()
        mock_service_cls.return_value = mock_service

        response = self.client.get("/api/v1/feedback/1")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 1
        assert data["logs_snapshot"][0]["message_type"] == "llm_response"

    @patch("api.routes.feedback_report.FeedbackReportService")
    def test_get_feedback_report_not_found(self, mock_service_cls):
        mock_service = AsyncMock()
        mock_service.get_report.return_value = None
        mock_service_cls.return_value = mock_service

        response = self.client.get("/api/v1/feedback/999")

        assert response.status_code == 404

    @patch("api.routes.feedback_report.FeedbackReportService")
    def test_update_feedback_report_status(self, mock_service_cls):
        mock_service = AsyncMock()
        mock_service.update_status.return_value = _make_report(status="reviewed")
        mock_service_cls.return_value = mock_service

        response = self.client.patch(
            "/api/v1/feedback/1",
            json={"status": "reviewed"},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "reviewed"
        mock_service.update_status.assert_awaited_once_with(1, "reviewed")

    def test_update_feedback_report_invalid_status(self):
        response = self.client.patch(
            "/api/v1/feedback/1",
            json={"status": "closed"},
        )

        assert response.status_code == 422

    def test_list_feedback_reports_invalid_limit(self):
        response = self.client.get("/api/v1/feedback?limit=0")

        assert response.status_code == 422
