"""
Тесты для AdminService — используют реальную тестовую БД.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.models import AdminRole
from db.postgres.services.admin import AdminService


@pytest.mark.asyncio
async def test_count_empty(session: AsyncSession):
    service = AdminService(session)
    assert await service.count() == 0


@pytest.mark.asyncio
async def test_create_admin(session: AsyncSession):
    service = AdminService(session)
    admin = await service.create(
        username="alice",
        password_hash="hash",
        role=AdminRole.superadmin,
    )
    assert admin.id is not None
    assert admin.username == "alice"
    assert admin.role == AdminRole.superadmin
    assert admin.is_active is True
    assert admin.created_by_id is None


@pytest.mark.asyncio
async def test_count_after_create(session: AsyncSession):
    service = AdminService(session)
    await service.create(username="bob", password_hash="h", role=AdminRole.admin)
    assert await service.count() == 1


@pytest.mark.asyncio
async def test_get_by_id_found(session: AsyncSession):
    service = AdminService(session)
    created = await service.create(
        username="charlie", password_hash="h", role=AdminRole.admin
    )
    found = await service.get_by_id(created.id)
    assert found is not None
    assert found.username == "charlie"


@pytest.mark.asyncio
async def test_get_by_id_not_found(session: AsyncSession):
    service = AdminService(session)
    assert await service.get_by_id("nonexistent-id") is None


@pytest.mark.asyncio
async def test_get_by_username_found(session: AsyncSession):
    service = AdminService(session)
    await service.create(username="dave", password_hash="h", role=AdminRole.viewer)
    found = await service.get_by_username("dave")
    assert found is not None
    assert found.role == AdminRole.viewer


@pytest.mark.asyncio
async def test_get_by_username_not_found(session: AsyncSession):
    service = AdminService(session)
    assert await service.get_by_username("nobody") is None


@pytest.mark.asyncio
async def test_list_all(session: AsyncSession):
    service = AdminService(session)
    await service.create(username="eve", password_hash="h", role=AdminRole.admin)
    await service.create(username="frank", password_hash="h", role=AdminRole.viewer)
    admins = await service.list_all()
    assert len(admins) == 2
    usernames = {a.username for a in admins}
    assert usernames == {"eve", "frank"}


@pytest.mark.asyncio
async def test_set_active_deactivate(session: AsyncSession):
    service = AdminService(session)
    admin = await service.create(
        username="grace", password_hash="h", role=AdminRole.admin
    )
    updated = await service.set_active(admin.id, is_active=False)
    assert updated is not None
    assert updated.is_active is False


@pytest.mark.asyncio
async def test_set_active_not_found(session: AsyncSession):
    service = AdminService(session)
    result = await service.set_active("nonexistent", is_active=False)
    assert result is None


@pytest.mark.asyncio
async def test_update_role(session: AsyncSession):
    service = AdminService(session)
    admin = await service.create(username="mia", password_hash="h", role=AdminRole.viewer)
    updated = await service.update_role(admin.id, AdminRole.admin)
    assert updated is not None
    assert AdminRole(updated.role) == AdminRole.admin


@pytest.mark.asyncio
async def test_update_role_not_found(session: AsyncSession):
    service = AdminService(session)
    result = await service.update_role("nonexistent", AdminRole.admin)
    assert result is None


@pytest.mark.asyncio
async def test_count_superadmins(session: AsyncSession):
    service = AdminService(session)
    assert await service.count_superadmins() == 0
    await service.create(username="nora", password_hash="h", role=AdminRole.superadmin)
    await service.create(username="omar", password_hash="h", role=AdminRole.admin)
    await service.create(username="pia", password_hash="h", role=AdminRole.superadmin)
    assert await service.count_superadmins() == 2


@pytest.mark.asyncio
async def test_create_invite_code(session: AsyncSession):
    service = AdminService(session)
    admin = await service.create(
        username="henry", password_hash="h", role=AdminRole.superadmin
    )
    invite = await service.create_invite_code(
        code="code-abc",
        created_by_id=admin.id,
        role=AdminRole.admin,
    )
    assert invite.code == "code-abc"
    assert invite.role == AdminRole.admin
    assert invite.used_at is None
    assert invite.expires_at is None


@pytest.mark.asyncio
async def test_create_invite_code_with_expiry(session: AsyncSession):
    service = AdminService(session)
    admin = await service.create(
        username="iris", password_hash="h", role=AdminRole.superadmin
    )
    expires = (datetime.now(UTC) + timedelta(hours=24)).replace(tzinfo=None)
    invite = await service.create_invite_code(
        code="exp-code",
        created_by_id=admin.id,
        role=AdminRole.viewer,
        expires_at=expires,
    )
    assert invite.expires_at is not None


@pytest.mark.asyncio
async def test_get_invite_code_found(session: AsyncSession):
    service = AdminService(session)
    admin = await service.create(
        username="jack", password_hash="h", role=AdminRole.superadmin
    )
    await service.create_invite_code(
        code="findme", created_by_id=admin.id, role=AdminRole.admin
    )
    found = await service.get_invite_code("findme")
    assert found is not None
    assert found.code == "findme"


@pytest.mark.asyncio
async def test_get_invite_code_not_found(session: AsyncSession):
    service = AdminService(session)
    assert await service.get_invite_code("nocode") is None


@pytest.mark.asyncio
async def test_use_invite_code(session: AsyncSession):
    service = AdminService(session)
    creator = await service.create(
        username="kate", password_hash="h", role=AdminRole.superadmin
    )
    user = await service.create(username="liam", password_hash="h", role=AdminRole.admin)
    await service.create_invite_code(
        code="use-me", created_by_id=creator.id, role=AdminRole.admin
    )
    await service.use_invite_code("use-me", user.id)

    invite = await service.get_invite_code("use-me")
    assert invite is not None
    assert invite.used_by_id == user.id
    assert invite.used_at is not None
