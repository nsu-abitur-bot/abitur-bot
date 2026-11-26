import asyncio
import json
import pickle
from collections import Counter
from contextlib import suppress
from typing import Any

import numpy as np
import pytest
import pytest_asyncio
from cryptography.fernet import Fernet
from nano_vectordb import NanoVectorDB
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from api.services import model_migration as migration
from config.content_gate import ModelMigrationBusy, content_write, migration_marker
from config.instance import generation_id, get_setting, install, settings_scope
from db.postgres.models import Document, FaqEntry, Settings
from db.postgres.services.faq import FaqDbService
from db.postgres.services.instance_settings import InstanceSettingsService
from db.postgres.services.model_migration import ModelMigrationService
from tests.rag.test_reembed import fixture as build_graph


class Embeddings:
    def __init__(self):
        self.calls = []
        self.fail_on = None

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        if len(self.calls) == self.fail_on:
            raise RuntimeError("provider unavailable; must not be published")
        return [[1.0, 2.0, 3.0] for _ in texts]


@pytest_asyncio.fixture
async def migration_state(test_engine, monkeypatch, tmp_path):
    sessions = async_sessionmaker(test_engine, expire_on_commit=False)
    monkeypatch.setattr(migration, "AsyncSessionLocal", sessions)
    monkeypatch.setenv("LIGHTRAG_WORKSPACE_BASE", str(tmp_path))
    monkeypatch.setenv("LIGHTRAG_STORAGE", "json")
    monkeypatch.setenv("SETTINGS_ENCRYPTION_KEY", Fernet.generate_key().decode())
    source = tmp_path / "abitur_kb"
    build_graph(source)
    (source / "kv_store_full_docs.json").write_text(
        json.dumps({"doc": {"content": "Документ вуза"}})
    )
    (source / "kv_store_doc_status.json").write_text(
        json.dumps({"doc": {"status": "processed", "chunks_count": 1}})
    )
    embeddings = Embeddings()

    class Provider:
        def get_embeddings_model(self):
            return embeddings

    monkeypatch.setattr(migration, "get_faq_provider", lambda: Provider())

    async def check(*args):
        return "OK"

    monkeypatch.setattr(migration, "check_connection", check)
    async with sessions() as session:
        snapshot = await InstanceSettingsService(session).write(
            {
                "LLM_PROVIDER": "gemini",
                "FAQ_EMBEDDING_PROVIDER": "gemini",
                "LIGHTRAG_EMBEDDING_PROVIDER": "gemini",
                "GEMINI_EMBEDDING_MODEL": "old",
                "GEMINI_GRAPH_EMBEDDING_MODEL": "old",
            },
            0,
        )
        session.add(
            FaqEntry(question="Документы?", answer="Паспорт", aliases=["Что принести?"])
        )
        session.add(
            Document(title="Документ вуза", graph_id="abitur_kb", rag_doc_id="doc")
        )
        await session.commit()
    install(snapshot.values, snapshot.revision)
    yield sessions, source, embeddings
    install({}, 0)


def changes() -> dict[str, str | None]:
    return {
        "TEXT_PROVIDER": "deepseek",
        "TEXT_MODEL": "deepseek-flash",
        "DEEPSEEK_API_KEY": "synthetic-new-secret",
        "EMBEDDING_PROVIDER": "gigachat",
        "EMBEDDING_MODEL": "EmbeddingsGigaR",
        "GIGACHAT_API_KEY": "synthetic-auth-secret",
    }


async def wait_status(sessions, job_id, status) -> dict[str, Any]:
    async with asyncio.timeout(10):
        while True:
            async with sessions() as session:
                job = await ModelMigrationService(session).get(job_id)
            assert job is not None
            if job["status"] == status:
                return job
            await asyncio.sleep(0.03)


@pytest.mark.asyncio
async def test_populated_generation_switch_rebuilds_all_stores_and_faq(migration_state):
    sessions, source, embedder = migration_state
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    result = await migration.start_migration(1, changes(), confirm_reindex=True)
    job_id = result["job"]["id"]
    assert (await migration.start_migration(1, changes(), confirm_reindex=True))["job"][
        "id"
    ] == job_id
    assert len(embedder.calls) == 1  # idempotent save does not recheck the provider
    async with sessions() as session:
        assert (await InstanceSettingsService(session).read()).revision == 1
        with pytest.raises(ModelMigrationBusy):
            await FaqDbService(session).create(
                question="Новый вопрос", answer="Новый ответ", aliases=[]
            )
        await session.rollback()
        stored = (await session.scalars(select(Settings.value))).all()
        assert all(
            "synthetic-new-secret" not in value and "synthetic-auth-secret" not in value
            for value in stored
        )
    worker = asyncio.create_task(migration.run_migrations())
    try:
        job = await wait_status(sessions, job_id, "completed")
        assert job["completed_revision"] == 2
    finally:
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker
    assert not migration_marker().exists()
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    async with sessions() as session:
        snapshot = await InstanceSettingsService(session).read()
        assert await session.scalar(select(FaqEntry.answer)) == "Паспорт"
        assert await session.scalar(select(Document.rag_doc_id)) == "doc"
    with settings_scope(snapshot.values, snapshot.revision):
        assert generation_id() == job_id
        root = migration.Path(get_setting("LIGHTRAG_WORKSPACE_BASE", "./data/lightrag"))
        for name in ("chunks", "entities", "relationships"):
            data = json.loads((root / "abitur_kb" / f"vdb_{name}.json").read_text())
            assert data["embedding_dim"] == 3
        cache = pickle.loads((root / "faq_cache.pkl").read_bytes())
        assert set(cache["vectors"]) == {"Документы?", "Что принести?"}
        assert all(len(v) == 3 for v in cache["vectors"].values())
        index = NanoVectorDB(3, storage_file=str(root / "abitur_kb" / "vdb_chunks.json"))
        assert index.query(np.array([1.0, 2.0, 3.0]), top_k=1)[0]["__id__"] == "chunk"
        async with content_write():
            pass


@pytest.mark.asyncio
async def test_failure_and_worker_restart_preserve_old_index_and_resume_batches(
    migration_state,
):
    from api.routes.model_settings import control_job

    sessions, source, embedder = migration_state
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    result = await migration.start_migration(1, changes(), confirm_reindex=True)
    job_id = result["job"]["id"]
    # Preflight + chunks succeed, then entity embeddings fail.
    embedder.fail_on = 3
    worker = asyncio.create_task(migration.run_migrations())
    try:
        job = await wait_status(sessions, job_id, "failed")
        assert "synthetic" not in json.dumps(ModelMigrationService.public(job))
    finally:
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    async with sessions() as session:
        assert (await InstanceSettingsService(session).read()).revision == 1
    # Retry through durable state, in a fresh worker; no snapshot/index reset.
    # The route normally uses the same factory from its own module.
    import api.routes.model_settings as routes

    original = routes.AsyncSessionLocal
    routes.AsyncSessionLocal = sessions
    try:
        await control_job(job_id, "retry")
    finally:
        routes.AsyncSessionLocal = original
    embedder.fail_on = None
    worker = asyncio.create_task(migration.run_migrations())
    try:
        await wait_status(sessions, job_id, "completed")
    finally:
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker
    counts = Counter(text for batch in embedder.calls for text in batch)
    assert counts["Документ вуза"] == 1  # completed batch is reused after failure
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


@pytest.mark.asyncio
async def test_pg_generation_clones_canonical_stores_and_isolates_query_cache(
    migration_state, monkeypatch
):
    from sqlalchemy import text

    from rag import pg_storage

    sessions, source, embedder = migration_state
    monkeypatch.setenv("LIGHTRAG_STORAGE", "postgres")
    monkeypatch.setattr(pg_storage, "AsyncSessionLocal", sessions)
    await pg_storage.ensure_lightrag_tables()
    try:
        async with sessions() as session:
            await session.execute(
                text(
                    "INSERT INTO lightrag_doc_full(id,workspace,content) "
                    "VALUES('doc','abitur_kb','Документ вуза')"
                )
            )
            await session.execute(
                text(
                    "INSERT INTO lightrag_doc_chunks"
                    "(id,workspace,full_doc_id,content,tokens,chunk_order_index) "
                    "VALUES('chunk','abitur_kb','doc','Документ вуза',2,0)"
                )
            )
            await session.execute(
                text(
                    "INSERT INTO lightrag_doc_status(id,workspace,status,chunks_count) "
                    "VALUES('doc','abitur_kb','processed',1)"
                )
            )
            await session.execute(
                text(
                    "INSERT INTO lightrag_llm_cache(id,workspace,return_value) "
                    "VALUES('old-answer','abitur_kb','old-cache')"
                )
            )
            await session.commit()
        result = await migration.start_migration(1, changes(), confirm_reindex=True)
        job_id = result["job"]["id"]
        worker = asyncio.create_task(migration.run_migrations())
        try:
            await wait_status(sessions, job_id, "completed")
        finally:
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker
        async with sessions() as session:
            snapshot = await InstanceSettingsService(session).read()
            rows = (
                await session.execute(
                    text(
                        "SELECT workspace,content FROM lightrag_doc_full "
                        "ORDER BY workspace"
                    )
                )
            ).all()
            assert len(rows) == 2
            assert all(r.content == "Документ вуза" for r in rows)
            assert (
                await session.execute(text("SELECT count(*) FROM lightrag_llm_cache"))
            ).scalar_one() == 1
        with settings_scope(snapshot.values, snapshot.revision):
            assert await pg_storage.get_full_doc("abitur_kb", "doc") == "Документ вуза"
            assert pg_storage.physical_workspace("abitur_kb") == "abitur_kb__" + job_id
    finally:
        async with sessions() as session:
            for table in pg_storage._TABLES_DDL:
                await session.execute(text("DROP TABLE IF EXISTS " + table))
            await session.commit()


@pytest.mark.asyncio
async def test_shutdown_drains_failed_embedding_thread_and_preserves_cancellation(
    migration_state, monkeypatch
):
    import threading

    sessions, source, embedder = migration_state
    entered, release = threading.Event(), threading.Event()
    original = embedder.embed_documents

    def blocked(texts):
        if texts == ["Документ вуза"]:
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test release timeout")
            raise RuntimeError("provider failed during shutdown")
        return original(texts)

    monkeypatch.setattr(embedder, "embed_documents", blocked)
    await migration.start_migration(1, changes(), confirm_reindex=True)
    worker = asyncio.create_task(migration.run_migrations())
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        worker.cancel()
        await asyncio.sleep(0.02)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(worker, timeout=2)
    finally:
        release.set()
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
    async with sessions() as session:
        assert (await InstanceSettingsService(session).read()).revision == 1


@pytest.mark.asyncio
async def test_cancel_before_build_keeps_original_configuration(migration_state):
    sessions, source, embedder = migration_state
    result = await migration.start_migration(1, changes(), confirm_reindex=True)
    job_id = result["job"]["id"]
    async with sessions() as session:
        service = ModelMigrationService(session)
        job = await service.get(job_id)
        assert job is not None
        job["status"] = "cancelling"
        await service.save(job)
        await session.commit()
    worker = asyncio.create_task(migration.run_migrations())
    try:
        await wait_status(sessions, job_id, "cancelled")
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
    assert len(embedder.calls) == 1
    assert not migration_marker().exists()
    async with sessions() as session:
        assert (await InstanceSettingsService(session).read()).revision == 1


@pytest.mark.asyncio
async def test_stale_job_never_activates_over_a_newer_settings_revision(migration_state):
    sessions, source, embedder = migration_state
    result = await migration.start_migration(1, changes(), confirm_reindex=True)
    job_id = result["job"]["id"]
    # Simulate an external settings writer, bypassing the API's file fence.
    async with sessions() as session:
        service = InstanceSettingsService(session)
        current = await service.read()
        await service.write({**current.values, "UNIVERSITY_SHORT_NAME": "Updated"}, 1)
    worker = asyncio.create_task(migration.run_migrations())
    try:
        await wait_status(sessions, job_id, "failed")
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
    assert len(embedder.calls) == 1
    assert not migration_marker().exists()
    async with sessions() as session:
        current = await InstanceSettingsService(session).read()
        assert current.revision == 2
        assert "_embedding_generation" not in current.values
        assert current.values["UNIVERSITY_SHORT_NAME"] == "Updated"


@pytest.mark.asyncio
async def test_legacy_same_provider_model_and_dimension_still_requires_contract_migration(
    migration_state,
):
    import shutil

    from rag.reembed import rebuild_workspace
    from tests.rag.test_reembed import Embeddings as ThreeDimensional

    sessions, source, embedder = migration_state
    replacement = source.parent / "replacement"
    rebuild_workspace(source, replacement, ThreeDimensional())
    shutil.rmtree(source)
    replacement.rename(source)
    async with sessions() as session:
        service = InstanceSettingsService(session)
        snapshot = await service.read()
        await service.write(
            {
                **snapshot.values,
                "GEMINI_EMBEDDING_MODEL": "same-model",
                "GEMINI_GRAPH_EMBEDDING_MODEL": "same-model",
            },
            1,
        )
    desired = {
        **changes(),
        "EMBEDDING_PROVIDER": "gemini",
        "EMBEDDING_MODEL": "same-model",
    }
    result = await migration.start_migration(2, desired, confirm_reindex=True)
    assert result["job"] is not None
    async with sessions() as session:
        assert (await InstanceSettingsService(session).read()).revision == 2


@pytest.mark.asyncio
async def test_cancel_serializes_with_worker_claim(migration_state, monkeypatch):
    from api.routes import model_settings

    sessions, source, embedder = migration_state
    monkeypatch.setattr(model_settings, "AsyncSessionLocal", sessions)
    result = await migration.start_migration(1, changes(), confirm_reindex=True)
    job_id = result["job"]["id"]
    fetched, release, cancelling = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_active = ModelMigrationService.active
    original_lock = ModelMigrationService.lock
    paused = False

    async def active(service):
        nonlocal paused
        job = await original_active(service)
        task = asyncio.current_task()
        assert task is not None
        if task.get_name() == "claim-worker" and not paused:
            paused = True
            fetched.set()
            await release.wait()
        return job

    async def lock(service):
        task = asyncio.current_task()
        assert task is not None
        if task.get_name() == "cancel-request":
            cancelling.set()
        await original_lock(service)

    monkeypatch.setattr(ModelMigrationService, "active", active)
    monkeypatch.setattr(ModelMigrationService, "lock", lock)
    worker = asyncio.create_task(migration.run_migrations(), name="claim-worker")
    cancellation = None
    try:
        await asyncio.wait_for(fetched.wait(), 3)
        cancellation = asyncio.create_task(
            model_settings.control_job(job_id, "cancel"), name="cancel-request"
        )
        await asyncio.wait_for(cancelling.wait(), 3)
        await asyncio.sleep(0.05)
        assert not cancellation.done(), "Cancellation must serialize with the claim"
        release.set()
        assert (await asyncio.wait_for(cancellation, 3))["status"] == "cancelling"
        await wait_status(sessions, job_id, "cancelled")
        assert len(embedder.calls) == 1
        async with sessions() as session:
            assert (await InstanceSettingsService(session).read()).revision == 1
    finally:
        release.set()
        worker.cancel()
        if cancellation is not None:
            cancellation.cancel()
        await asyncio.gather(
            worker, *([cancellation] if cancellation else []), return_exceptions=True
        )
