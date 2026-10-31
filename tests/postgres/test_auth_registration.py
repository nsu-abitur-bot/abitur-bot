"""Registration security regressions against independent PostgreSQL connections."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from api.auth.router import router
from db.postgres.db import get_async_session
from db.postgres.models import AdminRole, InviteCode
from db.postgres.services.admin import AdminService
from tests.postgres.conftest import TEST_DATABASE_URL, TEST_DB_NAME


@pytest.fixture
async def auth_client(test_engine):
    engine = create_async_engine(TEST_DATABASE_URL, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    app = FastAPI()
    app.include_router(router)

    async def sessions():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_async_session] = sessions
    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        yield client, factory
    await engine.dispose()


async def seed(factory, codes):
    async with factory() as session:
        service = AdminService(session)
        root = await service.create("root", "hash", AdminRole.superadmin)
        for code in codes:
            await service.create_invite_code(code, root.id, AdminRole.viewer)
        await session.commit()


async def register(client, username, code=None):
    return await client.post(
        "/auth/register",
        json={"username": username, "password": "test-password", "invite_code": code},
    )


@pytest.mark.asyncio
async def test_empty_database_requires_invite(auth_client):
    client, factory = auth_client
    for code in (None, "unknown"):
        assert (await register(client, "intruder", code)).status_code == 400
    async with factory() as session:
        assert await AdminService(session).count() == 0


@pytest.mark.asyncio
async def test_concurrent_same_invite_creates_exactly_one_admin(auth_client):
    client, factory = auth_client
    await seed(factory, ["shared"])
    results = await asyncio.gather(
        register(client, "alice", "shared"), register(client, "bobby", "shared")
    )
    assert sorted(r.status_code for r in results) == [201, 400]
    async with factory() as session:
        assert await AdminService(session).count() == 2
        invite = await session.get(InviteCode, "shared")
        assert invite.used_by_id == next(
            r.json()["id"] for r in results if r.status_code == 201
        )


@pytest.mark.asyncio
async def test_same_username_race_rolls_back_losing_invite(auth_client):
    client, factory = auth_client
    await seed(factory, ["first", "second"])
    results = await asyncio.gather(
        register(client, "alice", "first"), register(client, "alice", "second")
    )
    assert sorted(r.status_code for r in results) == [201, 409]
    async with factory() as session:
        invites = list((await session.scalars(select(InviteCode))).all())
        unused = [i.code for i in invites if i.used_at is None]
        assert len(unused) == 1
        assert await AdminService(session).count() == 2
    response = await register(client, "bobby", unused[0])
    assert response.status_code == 201
    assert response.json()["role"] == "viewer"


@pytest.mark.asyncio
async def test_expired_invite_is_not_consumed(auth_client):
    client, factory = auth_client
    await seed(factory, ["expired"])
    async with factory() as session:
        invite = await session.get(InviteCode, "expired")
        invite.expires_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(seconds=1)
        await session.commit()
    assert (await register(client, "alice", "expired")).status_code == 400
    async with factory() as session:
        assert (await session.get(InviteCode, "expired")).used_at is None
        assert await AdminService(session).count() == 1


@pytest.mark.asyncio
async def test_bootstrap_concurrent_and_duplicate_refused(auth_client):
    _, factory = auth_client

    async def bootstrap(username):
        try:
            async with factory.begin() as session:
                await AdminService(session).bootstrap_superadmin(username, "hash")
            return "created"
        except ValueError:
            return "refused"

    assert sorted(await asyncio.gather(bootstrap("first"), bootstrap("second"))) == [
        "created",
        "refused",
    ]
    assert await bootstrap("third") == "refused"
    async with factory() as session:
        admins = await AdminService(session).list_all()
        assert len(admins) == 1
        assert admins[0].role == AdminRole.superadmin
        assert admins[0].created_by_id is None


@pytest.mark.asyncio
async def test_bootstrap_refuses_existing_non_superadmin(auth_client):
    _, factory = auth_client
    async with factory.begin() as session:
        await AdminService(session).create("existing", "original", AdminRole.viewer)
    async with factory.begin() as session:
        with pytest.raises(ValueError, match="empty admins"):
            await AdminService(session).bootstrap_superadmin("root", "replacement")
    async with factory() as session:
        admins = await AdminService(session).list_all()
        assert [(a.username, a.password_hash, a.role) for a in admins] == [
            ("existing", "original", AdminRole.viewer)
        ]


@pytest.mark.asyncio
async def test_cli_creates_then_refuses_without_disclosing_password(
    auth_client, monkeypatch, capsys
):
    from api.auth import bootstrap
    from api.auth.security import verify_password

    _, factory = auth_client
    monkeypatch.setenv("DB_NAME", TEST_DB_NAME)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr("sys.argv", ["bootstrap", "--username", "local-root"])
    monkeypatch.setattr(bootstrap.getpass, "getpass", lambda _: "local-test-password")
    assert await asyncio.to_thread(bootstrap.main) == 0
    assert await asyncio.to_thread(bootstrap.main) == 1
    output = capsys.readouterr()
    assert "local-test-password" not in output.out + output.err
    async with factory() as session:
        admins = await AdminService(session).list_all()
        assert len(admins) == 1
        assert admins[0].role == AdminRole.superadmin
        assert verify_password("local-test-password", admins[0].password_hash)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "passwords", [("short", "short"), ("long-password", "different")]
)
async def test_cli_invalid_password_does_not_create(
    auth_client, monkeypatch, capsys, passwords
):
    from api.auth import bootstrap

    _, factory = auth_client
    answers = iter(passwords)
    monkeypatch.setattr("sys.argv", ["bootstrap", "--username", "local-root"])
    monkeypatch.setattr(bootstrap.getpass, "getpass", lambda _: next(answers))
    assert bootstrap.main() == 1
    output = capsys.readouterr()
    assert all(p not in output.out + output.err for p in passwords)
    async with factory() as session:
        assert await AdminService(session).count() == 0


@pytest.mark.asyncio
async def test_cli_requires_hidden_input(auth_client, monkeypatch, capsys):
    import getpass
    import warnings

    from api.auth import bootstrap

    _, factory = auth_client
    monkeypatch.setattr("sys.argv", ["bootstrap", "--username", "local-root"])

    def insecure_input(_):
        warnings.warn("unavailable", getpass.GetPassWarning)
        pytest.fail("must not read echoed input")

    monkeypatch.setattr(bootstrap.getpass, "getpass", insecure_input)
    assert bootstrap.main() == 1
    async with factory() as session:
        assert await AdminService(session).count() == 0


@pytest.mark.asyncio
async def test_other_integrity_error_is_not_username_conflict(auth_client):
    from sqlalchemy import text

    client, factory = auth_client
    await seed(factory, ["retry"])
    async with factory.begin() as session:
        await session.execute(
            text(
                "ALTER TABLE admins ADD CONSTRAINT reject_alice "
                "CHECK (username <> 'alice')"
            )
        )
    assert (await register(client, "alice", "retry")).status_code == 500
    async with factory() as session:
        assert (await session.get(InviteCode, "retry")).used_at is None
        assert await AdminService(session).count() == 1
    assert (await register(client, "bobby", "retry")).status_code == 201
