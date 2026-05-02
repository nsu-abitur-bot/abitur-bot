"""
Общие фикстуры для тестов API.
Автоматически обходит авторизацию, подставляя фиктивного суперадмина.
"""

from dataclasses import dataclass, field
from datetime import datetime

import pytest

from api.auth.dependencies import get_current_admin, require_admin, require_superadmin
from api.main import app
from db.postgres.models import AdminRole


@dataclass
class FakeAdmin:
    id: str = "fake-admin-id"
    username: str = "test_admin"
    role: str = field(default_factory=lambda: AdminRole.superadmin)
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime(2026, 1, 1))
    created_by_id: str | None = None


@pytest.fixture(autouse=True)
def override_auth():
    fake = FakeAdmin()
    app.dependency_overrides[get_current_admin] = lambda: fake
    app.dependency_overrides[require_admin] = lambda: fake
    app.dependency_overrides[require_superadmin] = lambda: fake
    yield
    app.dependency_overrides.pop(get_current_admin, None)
    app.dependency_overrides.pop(require_admin, None)
    app.dependency_overrides.pop(require_superadmin, None)
