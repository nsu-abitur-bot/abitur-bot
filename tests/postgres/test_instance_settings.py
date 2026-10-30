import json

import pytest
from cryptography.fernet import Fernet
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from api.auth.dependencies import get_current_admin
from api.main import app
from api.services.instance_settings import (
    ConfigurationError,
    guard_embeddings,
    merge_changes,
)
from config import instance
from db.postgres.db import get_async_session
from db.postgres.models import Admin, AdminRole, Settings
from db.postgres.services.instance_settings import (
    SETTINGS_KEY,
    InstanceSettingsService,
    InstanceSnapshot,
    SettingsConflict,
)


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("SETTINGS_ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("LIGHTRAG_WORKSPACE_BASE", str(tmp_path))
    monkeypatch.setattr(instance, "_values", None)
    monkeypatch.setattr(instance, "_revision", 0)


async def test_encrypted_roundtrip_conflict_and_secret_removal(session):
    service = InstanceSettingsService(session)
    initial = await service.read()
    assert initial.revision == 0
    saved = await service.write({"BOT_TOKEN": "secret-token"}, 0)
    raw = await session.scalar(select(Settings.value).where(Settings.key == SETTINGS_KEY))
    assert "secret-token" not in raw
    assert json.loads(raw)["revision"] == 1
    assert (await service.read()).values["BOT_TOKEN"] == "secret-token"
    with pytest.raises(SettingsConflict):
        await service.write({}, 0)
    await session.rollback()
    await service.write({"BOT_TOKEN": ""}, saved.revision)
    assert not (await service.read()).values["BOT_TOKEN"]


async def test_wrong_master_key_fails_closed(session, monkeypatch):
    service = InstanceSettingsService(session)
    await service.write({"OPENAI_API_KEY": "secret"}, 0)
    monkeypatch.setenv("SETTINGS_ENCRYPTION_KEY", Fernet.generate_key().decode())
    with pytest.raises(ValueError, match="расшифровать"):
        await service.read()


async def test_reject_embedding_change_for_existing_index(session, tmp_path):
    graph = tmp_path / "abitur_kb"
    graph.mkdir()
    (graph / "kv_store_full_docs.json").write_text('{"doc": {"content": "old"}}')
    before = InstanceSnapshot(1, {})
    candidate = merge_changes(before, {"OPENAI_EMBEDDING_MODEL": "new-model"})
    with pytest.raises(ConfigurationError, match="переиндексации"):
        await guard_embeddings(session, before, candidate)


async def test_settings_api_permissions_masking_and_concurrent_edit(session):
    async def db_override():
        yield session

    admin = Admin(
        id="test-admin",
        username="test",
        password_hash="not-used",
        role=AdminRole.superadmin,
        is_active=True,
    )
    app.dependency_overrides[get_async_session] = db_override
    app.dependency_overrides[get_current_admin] = lambda: admin
    try:
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://test"
        ) as client:
            service = InstanceSettingsService(session)
            await service.write({"BOT_TOKEN": "never-return-this"}, 0)
            response = await client.get("/api/v1/settings/instance")
            assert response.status_code == 200
            assert "never-return-this" not in response.text
            token = next(
                x for x in response.json()["settings"] if x["key"] == "BOT_TOKEN"
            )
            assert token["value"] is None and token["configured"]
            response = await client.put(
                "/api/v1/settings/instance", json={"revision": 0, "changes": {}}
            )
            assert response.status_code == 409
            response = await client.put(
                "/api/v1/settings/instance",
                json={"revision": 1, "changes": {"SCHEDULER_ENABLED": "false"}},
            )
            assert response.status_code == 200
            assert instance.get_setting("SCHEDULER_ENABLED") == "false"
            admin.role = AdminRole.admin
            assert (await client.get("/api/v1/settings/instance")).status_code == 403
            admin.role = AdminRole.viewer
            assert (
                await client.put(
                    "/api/v1/settings/instance", json={"revision": 2, "changes": {}}
                )
            ).status_code == 403
    finally:
        app.dependency_overrides.pop(get_async_session, None)
        app.dependency_overrides.pop(get_current_admin, None)


async def test_failed_connection_never_changes_saved_configuration(session, monkeypatch):
    from api.routes import instance_settings as routes

    async def fail(*args):
        raise ConfigurationError("Подключение не удалось")

    async def db_override():
        yield session

    monkeypatch.setattr(routes, "check_connection", fail)
    admin = Admin(
        id="test",
        username="test",
        password_hash="unused",
        role=AdminRole.superadmin,
        is_active=True,
    )
    app.dependency_overrides[get_async_session] = db_override
    app.dependency_overrides[get_current_admin] = lambda: admin
    try:
        service = InstanceSettingsService(session)
        await service.write({"BOT_TOKEN": "old", "TELEGRAM_ENABLED": "true"}, 0)
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://test"
        ) as client:
            response = await client.put(
                "/api/v1/settings/instance",
                json={"revision": 1, "changes": {"BOT_TOKEN": "bad"}},
            )
        assert response.status_code == 400
        snapshot = await service.read()
        assert snapshot.revision == 1 and snapshot.values["BOT_TOKEN"] == "old"
    finally:
        app.dependency_overrides.pop(get_async_session, None)
        app.dependency_overrides.pop(get_current_admin, None)


async def test_real_sdk_http_validation_and_two_process_snapshots(
    session, test_engine, monkeypatch
):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from sqlalchemy.ext.asyncio import async_sessionmaker

    from bootstrap import runtime_settings

    class ProviderHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            valid = self.headers.get("Authorization") == "Bearer test-working-key"
            self.send_response(200 if valid else 401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if not valid:
                payload = {
                    "error": {
                        "message": "Invalid API key",
                        "type": "invalid_request_error",
                    }
                }
            elif self.path.endswith("/embeddings"):
                payload = {
                    "object": "list",
                    "data": [{"index": 0, "embedding": [1.0]}],
                    "model": "test",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            else:
                payload = {
                    "id": "test-response",
                    "object": "response",
                    "created_at": 1,
                    "model": "test",
                    "status": "completed",
                    "output": [],
                }
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, format: str, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), ProviderHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    async def db_override():
        yield session

    admin = Admin(
        id="test",
        username="test",
        password_hash="unused",
        role=AdminRole.superadmin,
        is_active=True,
    )
    app.dependency_overrides[get_async_session] = db_override
    app.dependency_overrides[get_current_admin] = lambda: admin
    changes = {
        "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
        "OPENAI_API_KEY": "test-working-key",
        **{
            k: "openai"
            for k in (
                "LLM_PROVIDER",
                "FAQ_EMBEDDING_PROVIDER",
                "LIGHTRAG_LLM_PROVIDER",
                "LIGHTRAG_EMBEDDING_PROVIDER",
                "PDF_PARSER_PROVIDER",
            )
        },
    }
    try:
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://test"
        ) as client:
            response = await client.put(
                "/api/v1/settings/instance", json={"revision": 0, "changes": changes}
            )
            assert response.status_code == 200, response.text
            assert "test-working-key" not in response.text
            await session.commit()  # Release the one-connection test pool.
            instance.install({}, 0)  # Simulate another process with an old snapshot.
            monkeypatch.setattr(
                runtime_settings,
                "AsyncSessionLocal",
                async_sessionmaker(test_engine, expire_on_commit=False),
            )
            await runtime_settings.refresh_settings()
            assert instance.get_setting("OPENAI_API_KEY") == "test-working-key"
            assert instance.revision() == 1
            response = await client.put(
                "/api/v1/settings/instance",
                json={"revision": 1, "changes": {"OPENAI_API_KEY": "rejected-key"}},
            )
            assert response.status_code == 400
            assert "отклонён" in response.json()["detail"]
            assert (await InstanceSettingsService(session).read()).revision == 1
            assert instance.get_setting("OPENAI_API_KEY") == "test-working-key"
    finally:
        app.dependency_overrides.pop(get_async_session, None)
        app.dependency_overrides.pop(get_current_admin, None)
        server.shutdown()
        server.server_close()


async def test_environment_migration_preserves_settings_and_does_not_overwrite(
    session, test_engine, monkeypatch
):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from bootstrap import runtime_settings

    monkeypatch.setattr(
        runtime_settings,
        "AsyncSessionLocal",
        async_sessionmaker(test_engine, expire_on_commit=False),
    )
    monkeypatch.setenv("BOT_TOKEN", "old-env-token")
    monkeypatch.setenv("MAX_BOT_TOKEN", "")
    monkeypatch.setenv("GEMINI_API_KEY", "old-env-key")
    monkeypatch.setenv("SEED_ADMISSION_SCORES", "0")
    monkeypatch.setenv("CRAG_ENABLED", "false")
    session.add(Settings(key="max_document_pages", value="7"))
    await session.commit()
    await runtime_settings.import_environment()
    saved = await InstanceSettingsService(session).read()
    assert saved.values["BOT_TOKEN"] == "old-env-token"
    assert saved.values["TELEGRAM_ENABLED"] == "true"
    assert saved.values["MAX_ENABLED"] == "false"
    assert saved.values["SEED_ADMISSION_SCORES"] == "false"
    assert saved.values["MAX_DOCUMENT_PAGES"] == "7"
    assert (
        await session.scalar(select(Settings.value).where(Settings.key == "crag_enabled"))
        == "false"
    )
    await session.commit()
    with pytest.raises(ValueError, match="уже сохранены"):
        await runtime_settings.import_environment()


async def test_concurrent_writers_cannot_lose_updates(test_engine):
    import asyncio

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    concurrent_engine = create_async_engine(test_engine.url, pool_size=2)
    maker = async_sessionmaker(concurrent_engine, expire_on_commit=False)

    async def write(value):
        async with maker() as connection:
            try:
                return await InstanceSettingsService(connection).write(
                    {"BOT_TOKEN": value}, 0
                )
            except SettingsConflict as exc:
                return exc

    try:
        results = await asyncio.gather(write("first"), write("second"))
        assert sum(isinstance(result, SettingsConflict) for result in results) == 1
        assert sum(isinstance(result, InstanceSnapshot) for result in results) == 1
    finally:
        await concurrent_engine.dispose()
