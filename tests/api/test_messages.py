"""
Тесты для API эндпоинта /api/v1/messages.
"""

from datetime import datetime
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from api.main import app


def _make_message(
    id: str,
    user_id: int,
    session_id: str,
    user_text: str,
    bot_response: str,
    created_at: datetime | None = None,
):
    """Создаёт мок объекта Message."""
    msg = AsyncMock()
    msg.id = id
    msg.user_id = user_id
    msg.session_id = session_id
    msg.user_text = user_text
    msg.bot_response = bot_response
    msg.created_at = created_at or datetime(2026, 3, 8, 12, 0, 0)
    return msg


class TestMessagesEndpoint:
    """Тесты для эндпоинта /api/v1/messages."""

    def setup_method(self):
        self.client = TestClient(app)

    @patch("api.main.MessageService")
    def test_get_messages_empty(self, mock_service_cls):
        """Тест получения пустого списка сообщений."""
        mock_service = AsyncMock()
        mock_service.get_all_messages.return_value = []
        mock_service_cls.return_value = mock_service

        response = self.client.get("/api/v1/messages")
        assert response.status_code == 200
        assert response.json() == []

    @patch("api.main.MessageService")
    def test_get_messages_returns_data(self, mock_service_cls):
        """Тест получения списка сообщений."""
        mock_service = AsyncMock()
        mock_service.get_all_messages.return_value = [
            _make_message(
                id="msg-1",
                user_id=123,
                session_id="123",
                user_text="Вопрос 1",
                bot_response="Ответ 1",
            ),
            _make_message(
                id="msg-2",
                user_id=456,
                session_id="456",
                user_text="Вопрос 2",
                bot_response="Ответ 2",
            ),
        ]
        mock_service_cls.return_value = mock_service

        response = self.client.get("/api/v1/messages")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["user_id"] == 123
        assert data[0]["user_text"] == "Вопрос 1"
        assert data[0]["bot_response"] == "Ответ 1"
        assert data[1]["user_id"] == 456

    @patch("api.main.MessageService")
    def test_get_messages_by_user_id(self, mock_service_cls):
        """Тест фильтрации сообщений по user_id."""
        mock_service = AsyncMock()
        mock_service.get_messages_by_user.return_value = [
            _make_message(
                id="msg-1",
                user_id=123,
                session_id="123",
                user_text="Вопрос",
                bot_response="Ответ",
            ),
        ]
        mock_service_cls.return_value = mock_service

        response = self.client.get("/api/v1/messages?user_id=123")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["user_id"] == 123

    @patch("api.main.MessageService")
    def test_get_messages_pagination(self, mock_service_cls):
        """Тест пагинации."""
        mock_service = AsyncMock()
        mock_service.get_all_messages.return_value = [
            _make_message(
                id="msg-3",
                user_id=100,
                session_id="100",
                user_text="Вопрос 3",
                bot_response="Ответ 3",
            ),
        ]
        mock_service_cls.return_value = mock_service

        response = self.client.get("/api/v1/messages?limit=1&offset=2")
        assert response.status_code == 200

    @patch("api.main.MessageService")
    def test_get_messages_db_error(self, mock_service_cls):
        """Тест обработки ошибки БД."""
        mock_service = AsyncMock()
        mock_service.get_all_messages.side_effect = Exception("DB down")
        mock_service_cls.return_value = mock_service

        response = self.client.get("/api/v1/messages")
        assert response.status_code == 503

    def test_get_messages_invalid_limit(self):
        """Тест невалидного значения limit."""
        response = self.client.get("/api/v1/messages?limit=-1")
        assert response.status_code == 422

    def test_get_messages_invalid_offset(self):
        """Тест невалидного значения offset."""
        response = self.client.get("/api/v1/messages?offset=-5")
        assert response.status_code == 422
