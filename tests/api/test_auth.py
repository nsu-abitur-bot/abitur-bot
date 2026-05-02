"""
Тесты для auth эндпоинтов (/api/v1/auth/*).
Мокает AdminService чтобы не требовалась реальная БД.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from api.auth import security
from api.auth.dependencies import get_current_admin, require_admin, require_superadmin
from api.main import app
from db.postgres.models import AdminRole

from .conftest import FakeAdmin


def _make_db_admin(
    admin_id: str = "admin-123",
    username: str = "admin",
    role: str = AdminRole.superadmin,
    is_active: bool = True,
    password: str = "password123",
) -> MagicMock:
    admin = MagicMock()
    admin.id = admin_id
    admin.username = username
    admin.role = role
    admin.is_active = is_active
    admin.created_at = datetime(2026, 1, 1)
    admin.created_by_id = None
    admin.password_hash = security.hash_password(password)
    return admin


class TestLoginEndpoint:
    def setup_method(self):
        self.client = TestClient(app)

    @patch("api.auth.router.AdminService")
    def test_login_success(self, mock_cls):
        mock_service = AsyncMock()
        mock_service.get_by_username.return_value = _make_db_admin(password="secret")
        mock_cls.return_value = mock_service

        response = self.client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "secret"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    @patch("api.auth.router.AdminService")
    def test_login_wrong_password(self, mock_cls):
        mock_service = AsyncMock()
        mock_service.get_by_username.return_value = _make_db_admin(password="correct")
        mock_cls.return_value = mock_service

        response = self.client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "wrong"},
        )
        assert response.status_code == 401

    @patch("api.auth.router.AdminService")
    def test_login_nonexistent_user(self, mock_cls):
        mock_service = AsyncMock()
        mock_service.get_by_username.return_value = None
        mock_cls.return_value = mock_service

        response = self.client.post(
            "/api/v1/auth/login",
            json={"username": "nobody", "password": "secret"},
        )
        assert response.status_code == 401

    @patch("api.auth.router.AdminService")
    def test_login_inactive_user(self, mock_cls):
        mock_service = AsyncMock()
        mock_service.get_by_username.return_value = _make_db_admin(
            password="secret", is_active=False
        )
        mock_cls.return_value = mock_service

        response = self.client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "secret"},
        )
        assert response.status_code == 401


class TestRegisterEndpoint:
    def setup_method(self):
        self.client = TestClient(app)

    @patch("api.auth.router.AdminService")
    def test_register_first_admin_becomes_superadmin(self, mock_cls):
        """Первый зарегистрировавшийся получает роль superadmin."""
        mock_service = AsyncMock()
        mock_service.count.return_value = 0
        mock_service.get_by_username.return_value = None
        new_admin = _make_db_admin(role=AdminRole.superadmin)
        mock_service.create.return_value = new_admin
        mock_cls.return_value = mock_service

        response = self.client.post(
            "/api/v1/auth/register",
            json={"username": "admin", "password": "password123"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["role"] == AdminRole.superadmin

        # Проверяем что create вызван с ролью superadmin
        call_kwargs = mock_service.create.call_args.kwargs
        assert call_kwargs["role"] == AdminRole.superadmin

    @patch("api.auth.router.AdminService")
    def test_register_without_invite_code_fails(self, mock_cls):
        """Если уже есть админы, регистрация без invite_code отклоняется."""
        mock_service = AsyncMock()
        mock_service.count.return_value = 1
        mock_cls.return_value = mock_service

        response = self.client.post(
            "/api/v1/auth/register",
            json={"username": "new_admin", "password": "password123"},
        )
        assert response.status_code == 400
        assert "invite" in response.json()["detail"].lower()

    @patch("api.auth.router.AdminService")
    def test_register_with_valid_invite_code(self, mock_cls):
        mock_service = AsyncMock()
        mock_service.count.return_value = 1
        mock_service.get_by_username.return_value = None

        invite = MagicMock()
        invite.used_at = None
        invite.expires_at = None
        invite.role = AdminRole.admin
        invite.created_by_id = "creator-id"
        mock_service.get_invite_code.return_value = invite

        new_admin = _make_db_admin(role=AdminRole.admin)
        mock_service.create.return_value = new_admin
        mock_cls.return_value = mock_service

        response = self.client.post(
            "/api/v1/auth/register",
            json={
                "username": "new_admin",
                "password": "password123",
                "invite_code": "valid-code",
            },
        )
        assert response.status_code == 201
        mock_service.use_invite_code.assert_called_once()

    @patch("api.auth.router.AdminService")
    def test_register_with_used_invite_code_fails(self, mock_cls):
        mock_service = AsyncMock()
        mock_service.count.return_value = 1

        invite = MagicMock()
        invite.used_at = datetime(2026, 1, 1)  # уже использован
        mock_service.get_invite_code.return_value = invite
        mock_cls.return_value = mock_service

        response = self.client.post(
            "/api/v1/auth/register",
            json={
                "username": "new_admin",
                "password": "password123",
                "invite_code": "used-code",
            },
        )
        assert response.status_code == 400

    @patch("api.auth.router.AdminService")
    def test_register_with_invalid_invite_code_fails(self, mock_cls):
        mock_service = AsyncMock()
        mock_service.count.return_value = 1
        mock_service.get_invite_code.return_value = None
        mock_cls.return_value = mock_service

        response = self.client.post(
            "/api/v1/auth/register",
            json={
                "username": "new_admin",
                "password": "password123",
                "invite_code": "bad-code",
            },
        )
        assert response.status_code == 400

    @patch("api.auth.router.AdminService")
    def test_register_duplicate_username_fails(self, mock_cls):
        mock_service = AsyncMock()
        mock_service.count.return_value = 0
        mock_service.get_by_username.return_value = _make_db_admin()  # уже существует
        mock_cls.return_value = mock_service

        response = self.client.post(
            "/api/v1/auth/register",
            json={"username": "admin", "password": "password123"},
        )
        assert response.status_code == 409


class TestProtectedEndpoints:
    def setup_method(self):
        self.client = TestClient(app)

    def test_get_me_returns_current_admin(self):
        """GET /auth/me возвращает данные текущего (фиктивного) админа."""
        response = self.client.get("/api/v1/auth/me")
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "test_admin"
        assert data["role"] == AdminRole.superadmin

    @patch("api.auth.router.AdminService")
    def test_create_invite_code(self, mock_cls):
        mock_service = AsyncMock()
        invite = MagicMock()
        invite.code = "abc123"
        invite.role = AdminRole.admin
        invite.expires_at = None
        mock_service.create_invite_code.return_value = invite
        mock_cls.return_value = mock_service

        response = self.client.post(
            "/api/v1/auth/invite",
            json={"role": "admin"},
        )
        assert response.status_code == 201
        data = response.json()
        assert "code" in data
        assert data["role"] == AdminRole.admin

    @patch("api.auth.router.AdminService")
    def test_list_admins(self, mock_cls):
        mock_service = AsyncMock()
        mock_service.list_all.return_value = [_make_db_admin()]
        mock_cls.return_value = mock_service

        response = self.client.get("/api/v1/auth/admins")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_protected_route_without_token_returns_401(self):
        """Без токена все защищённые маршруты возвращают 401."""
        # Убираем фиктивный override чтобы проверить реальное поведение
        app.dependency_overrides.pop(get_current_admin, None)
        app.dependency_overrides.pop(require_admin, None)
        app.dependency_overrides.pop(require_superadmin, None)

        response = self.client.get("/api/v1/faq")
        assert response.status_code == 401

    def test_viewer_cannot_call_write_endpoints(self):
        """Роль viewer не может вызывать мутирующие эндпоинты."""
        # Подменяем get_current_admin на viewer, убираем override require_admin
        fake_viewer = FakeAdmin(role=AdminRole.viewer)
        app.dependency_overrides[get_current_admin] = lambda: fake_viewer
        app.dependency_overrides.pop(require_admin, None)

        response = self.client.post(
            "/api/v1/faq",
            json={"question": "q", "aliases": [], "answer": "a"},
        )
        assert response.status_code == 403

    def test_admin_cannot_access_superadmin_routes(self):
        """Роль admin не может обращаться к superadmin-маршрутам."""
        fake_admin = FakeAdmin(role=AdminRole.admin)
        app.dependency_overrides[get_current_admin] = lambda: fake_admin
        app.dependency_overrides.pop(require_superadmin, None)

        response = self.client.get("/api/v1/auth/admins")
        assert response.status_code == 403
