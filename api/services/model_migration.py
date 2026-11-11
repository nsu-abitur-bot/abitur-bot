"""Build immutable embedding generations; keep the active generation readable."""

import asyncio
import hashlib
import json
import logging
import os
import pickle
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select

from api.services.instance_settings import check_connection, merge_changes
from config.content_gate import (
    ModelMigrationBusy,
    active_migration,
    index_fence,
    migration_marker,
    storage_root,
)
from config.instance import SPECS, get_setting, install, settings_scope
from config.model_settings import GENERATION, materialize_models
from db.postgres.db import AsyncSessionLocal
from db.postgres.models import ContentProposal, Document, FaqEntry
from db.postgres.services.instance_settings import (
    InstanceSettingsService,
    SettingsConflict,
)
from db.postgres.services.model_migration import ACTIVE_STATES, ModelMigrationService
from llm.factory import get_faq_provider
from rag import pg_storage
from rag.reembed import rebuild_workspace

logger = logging.getLogger(__name__)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(value, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def space(values: dict[str, str], *, graph: bool) -> tuple[str, str, str]:
    selector = "LIGHTRAG_EMBEDDING_PROVIDER" if graph else "FAQ_EMBEDDING_PROVIDER"
    provider = values.get(selector, SPECS[selector].default)
    suffix = (
        "_GRAPH_EMBEDDING_MODEL"
        if graph and provider != "gigachat"
        else "_EMBEDDING_MODEL"
    )
    key = provider.upper() + suffix
    endpoint = values.get(provider.upper() + "_BASE_URL", "").rstrip("/")
    return provider, values.get(key, SPECS[key].default), endpoint


async def prepare(values: dict[str, str], revision: int) -> int:
    """Only explicit checks; never install proposed values as the live snapshot."""
    await check_connection(values["TEXT_PROVIDER"] + ":chat", values)
    with settings_scope(values, revision):
        embedder = get_faq_provider().get_embeddings_model()
        if embedder is None:
            raise ValueError("Модель не поддерживает эмбеддинги")
        vectors = await asyncio.to_thread(
            embedder.embed_documents, ["Проверка модели эмбеддингов"]
        )
    if len(vectors) != 1:
        raise ValueError("Получен неполный ответ модели эмбеддингов")
    vector = np.asarray(vectors[0], dtype=np.float32)
    if (
        vector.ndim != 1
        or not len(vector)
        or not np.isfinite(vector).all()
        or np.linalg.norm(vector) <= 0
    ):
        raise ValueError("Модель вернула некорректный вектор")
    return len(vector)


async def start_migration(
    revision: int, changes: dict[str, str | None], *, confirm_reindex: bool
) -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        current = await InstanceSettingsService(session).read()
        if current.revision != revision:
            raise SettingsConflict("Настройки изменились; обновите страницу")
        candidate = materialize_models(merge_changes(current, changes))
    # Duplicate save returns the existing job before any additional paid checks.
    async with index_fence(exclusive=True):
        async with AsyncSessionLocal() as session:
            jobs = ModelMigrationService(session)
            existing = await jobs.active()
            if existing and existing["status"] in ACTIVE_STATES:
                if {
                    k: v
                    for k, v in ModelMigrationService.target(existing).items()
                    if k != "_embedding_dim"
                } == {
                    k: v for k, v in candidate.items() if k != "_embedding_dim"
                } and existing["revision"] == revision:
                    return {"job": jobs.public(existing)}
                raise ModelMigrationBusy("Уже выполняется другой пересчёт")
    dimension = await prepare(candidate, revision)
    candidate["_embedding_dim"] = str(dimension)
    async with index_fence(exclusive=True):
        async with AsyncSessionLocal() as session:
            jobs = ModelMigrationService(session)
            await jobs.lock()
            current = await InstanceSettingsService(session).read()
            if current.revision != revision:
                raise SettingsConflict("Настройки изменились; обновите страницу")
            existing = await jobs.active()
            if existing and existing["status"] in ACTIVE_STATES:
                raise ModelMigrationBusy("Уже выполняется пересчёт")
            if await session.scalar(
                select(ContentProposal.id)
                .where(ContentProposal.status.in_(["queued", "processing"]))
                .limit(1)
            ):
                raise ModelMigrationBusy(
                    "Дождитесь завершения очереди загрузки материалов"
                )
            with settings_scope(current.values, current.revision):
                root = Path(get_setting("LIGHTRAG_WORKSPACE_BASE", "./data/lightrag"))
            populated = (
                bool(await session.scalar(select(FaqEntry.id).limit(1)))
                or bool(
                    await session.scalar(
                        select(Document.id)
                        .where(Document.status.notin_(["deleted", "удалён"]))
                        .limit(1)
                    )
                )
                or any(root.glob("*/vdb_*.json"))
            )
            changed = any(
                space(current.values, graph=g) != space(candidate, graph=g)
                for g in (False, True)
            )
            changed = changed or current.values.get("_model_settings_version") != "2"
            changed = changed or bool(
                current.values.get("_embedding_dim")
                and int(current.values["_embedding_dim"]) != dimension
            )
            for path in root.glob("*/vdb_*.json"):
                stored_dim = json.loads(path.read_text()).get("embedding_dim")
                changed = changed or (stored_dim is not None and stored_dim != dimension)
            if not changed or not populated:
                updated = await InstanceSettingsService(session).write(
                    candidate, revision
                )
                install(updated.values, updated.revision)
                return {"revision": updated.revision, "job": None}
            if not confirm_reindex:
                raise ValueError(
                    "Нужен пересчёт наполненной базы. Подтвердите платные вызовы "
                    "и временную блокировку правок"
                )
            job = {
                "id": uuid.uuid4().hex,
                "status": "queued",
                "revision": revision,
                "progress": {"stage": "queued"},
                "error": None,
            }
            await jobs.create(job, candidate)
            await session.commit()
            atomic_json(migration_marker(), {"id": job["id"]})
            return {"job": jobs.public(job)}


class CheckpointEmbedder:
    """Persist completed batches so a restart does not rebill them."""

    def __init__(self, embedder: Any, dimension: int, cache: Path, progress: Any) -> None:
        self.embedder, self.embedding_dim = embedder, dimension
        self.cache_path, self.progress = cache, progress
        self.cache = json.loads(cache.read_text()) if cache.exists() else {}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.progress(len(self.cache))
        keys = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        missing = dict(zip(keys, texts))
        missing = {k: v for k, v in missing.items() if k not in self.cache}
        if missing:
            vectors = self.embedder.embed_documents(list(missing.values()))
            if len(vectors) != len(missing):
                raise ValueError("Получен неполный набор векторов")
            for key, vector in zip(missing, vectors):
                array = np.asarray(vector, dtype=np.float32)
                if (
                    array.shape != (self.embedding_dim,)
                    or not np.isfinite(array).all()
                    or np.linalg.norm(array) <= 0
                ):
                    raise ValueError("Получен несовместимый вектор")
                self.cache[key] = array.tolist()
            atomic_json(self.cache_path, self.cache)
            self.progress(len(self.cache))
        return [self.cache[k] for k in keys]


async def update_progress(job_id: str, progress: dict[str, Any]) -> None:
    async with AsyncSessionLocal() as session:
        service = ModelMigrationService(session)
        await service.lock()
        job = await service.get(job_id)
        if job is None or job["status"] == "cancelling":
            raise ModelMigrationBusy("Пересчёт отменён")
        job["progress"] = progress
        await service.save(job)
        await session.commit()


async def execute_job(job: dict[str, Any]) -> None:
    values = ModelMigrationService.target(job)
    job_id = job["id"]
    base = storage_root()
    folder = base / ".migration-jobs" / job_id
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    source, destination = folder / "source", base / ".generations" / job_id
    async with index_fence(exclusive=True):
        atomic_json(migration_marker(), {"id": job_id})
        async with AsyncSessionLocal() as session:
            current = await InstanceSettingsService(session).read()
            if current.revision != job["revision"]:
                raise SettingsConflict("Настройки изменились; задание устарело")
            faq = (await session.scalars(select(FaqEntry))).all()
            items = [
                {"question": f.question, "aliases": f.aliases or [], "answer": f.answer}
                for f in faq
            ]
            documents = (
                await session.scalars(
                    select(Document).where(Document.status.notin_(["deleted", "удалён"]))
                )
            ).all()
            graph_ids = {doc.graph_id for doc in documents}
            registered: dict[str, list[str]] = {}
            for doc in documents:
                registered.setdefault(doc.graph_id, []).append(doc.rag_doc_id)
        with settings_scope(current.values, current.revision):
            active_root = Path(get_setting("LIGHTRAG_WORKSPACE_BASE", "./data/lightrag"))
            graph_ids.update(
                p.name
                for p in active_root.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            )
            if source.exists():
                shutil.rmtree(source)
            if not source.exists():
                source.mkdir(mode=0o700)
                for graph_id in sorted(graph_ids):
                    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", graph_id):
                        raise ValueError("Недопустимое имя базы знаний")
                    original = active_root / graph_id
                    if original.is_symlink() or any(
                        p.is_symlink() for p in original.rglob("*")
                    ):
                        raise ValueError("Индекс содержит символические ссылки")
                    if original.exists():
                        shutil.copytree(original, source / graph_id)
                    else:
                        (source / graph_id).mkdir()
                    if pg_storage.is_pg_storage_enabled():
                        for name, data in (
                            await pg_storage.dump_kv_stores(graph_id)
                        ).items():
                            (source / graph_id / name).write_text(data)
                for graph_id, ids in registered.items():
                    full_path = source / graph_id / "kv_store_full_docs.json"
                    status_path = source / graph_id / "kv_store_doc_status.json"
                    full = json.loads(full_path.read_text()) if full_path.exists() else {}
                    statuses = (
                        json.loads(status_path.read_text())
                        if status_path.exists()
                        else {}
                    )
                    if any(
                        doc_id not in full
                        or statuses.get(doc_id, {}).get("status")
                        not in {"processed", "preprocessed"}
                        for doc_id in ids
                    ):
                        raise ModelMigrationBusy(
                            "Не все материалы готовы к пересчёту. Проверьте их индексацию"
                        )
                for status_path in source.glob("*/kv_store_doc_status.json"):
                    statuses = json.loads(status_path.read_text())
                    if any(
                        row.get("status") in {"pending", "processing"}
                        for row in statuses.values()
                    ):
                        raise ModelMigrationBusy("Дождитесь завершения индексации")
                atomic_json(folder / "faq-items.json", items)
                atomic_json(folder / "source-ready.json", {"graphs": sorted(graph_ids)})
    ready = json.loads((folder / "source-ready.json").read_text())
    items = json.loads((folder / "faq-items.json").read_text())
    candidate = {**values, GENERATION: job_id}
    loop = asyncio.get_running_loop()

    def build() -> None:
        with settings_scope(candidate, current.revision):
            raw = get_faq_provider().get_embeddings_model()
            if raw is None:
                raise ValueError("Подключение эмбеддингов недоступно")

            def progress(count: int) -> None:
                future = asyncio.run_coroutine_threadsafe(
                    update_progress(
                        job_id, {"stage": "embedding", "completed_vectors": count}
                    ),
                    loop,
                )
                future.result(timeout=30)

            embedder = CheckpointEmbedder(
                raw, int(values["_embedding_dim"]), folder / "vectors.json", progress
            )
            destination.mkdir(parents=True, exist_ok=True)
            for graph_id in ready["graphs"]:
                target = destination / graph_id
                # Only this job owns the inactive generation. Canonical source and
                # completed embedding batches remain intact for a resumed attempt.
                if target.exists():
                    shutil.rmtree(target)
                rebuild_workspace(source / graph_id, target, embedder)
            questions: list[str] = []
            for item in items:
                questions.extend(
                    [str(item["question"]), *(str(alias) for alias in item["aliases"])]
                )
            vectors = {}
            for offset in range(0, len(questions), 16):
                batch = questions[offset : offset + 16]
                vectors.update(zip(batch, embedder.embed_documents(batch)))
            from faq.matcher import FAQMatcher

            signature = FAQMatcher._embedding_configuration()
            fd, temporary = tempfile.mkstemp(dir=destination)
            with os.fdopen(fd, "wb") as stream:
                pickle.dump({"model": signature, "vectors": vectors}, stream)
            os.replace(temporary, destination / "faq_cache.pkl")
            atomic_json(destination / "faq-items.json", items)
            atomic_json(
                destination / "generation-ready.json",
                {
                    "id": job_id,
                    "dimension": embedder.embedding_dim,
                    "graphs": ready["graphs"],
                },
            )

    task = asyncio.create_task(asyncio.to_thread(build))
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError as cancellation:
        # A cancelled await does not stop the thread. Keep ownership of the job
        # until no embedding call can still write its checkpoint/generation.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if task.done() and not task.cancelled():
            task.exception()
        raise cancellation
    async with index_fence(exclusive=True):
        async with AsyncSessionLocal() as session:
            service = ModelMigrationService(session)
            await service.lock()
            latest = await service.get(job_id)
            if latest is None or latest["status"] == "cancelling":
                raise ModelMigrationBusy("Пересчёт отменён")
            current = await InstanceSettingsService(session).read()
            if current.revision != job["revision"]:
                raise SettingsConflict("Настройки изменились; задание устарело")
            if pg_storage.is_pg_storage_enabled():
                await pg_storage.clone_generation(
                    session, ready["graphs"], current.values.get(GENERATION, ""), job_id
                )
            updated = await InstanceSettingsService(session).write(
                candidate, current.revision
            )
            # write() commits the atomic settings+namespace transition. A crash
            # before the following journal update is reconciled by the worker.
            latest["status"] = "completed"
            latest["completed_revision"] = updated.revision
            latest["progress"] = {"stage": "completed"}
            await service.save(latest)
            await session.commit()
            atomic_json(base / ".model-generation.json", {"id": job_id})
            if active_migration() == job_id:
                migration_marker().unlink()
            install(updated.values, updated.revision)


async def run_migrations() -> None:
    import fcntl

    base = storage_root()
    base.mkdir(parents=True, exist_ok=True)
    while True:
        with (base / ".model-worker.lock").open("a+") as worker_lock:
            try:
                fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                await asyncio.sleep(2)
                continue
            try:
                async with AsyncSessionLocal() as session:
                    service = ModelMigrationService(session)
                    job = await service.active()
                    current = await InstanceSettingsService(session).read()
                    if job and current.values.get(GENERATION) == job["id"]:
                        job["status"], job["completed_revision"] = (
                            "completed",
                            current.revision,
                        )
                        await service.save(job)
                        await session.commit()
                        atomic_json(base / ".model-generation.json", {"id": job["id"]})
                        if active_migration() == job["id"]:
                            migration_marker().unlink()
                    elif job and job["status"] in ACTIVE_STATES:
                        if job["status"] != "cancelling":
                            job["status"] = "running"
                            await service.save(job)
                            await session.commit()
                if job and job["status"] in ACTIVE_STATES:
                    try:
                        if job["status"] == "cancelling":
                            raise ModelMigrationBusy("Пересчёт отменён")
                        await execute_job(job)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        async with index_fence(exclusive=True):
                            async with AsyncSessionLocal() as session:
                                service = ModelMigrationService(session)
                                latest = await service.get(job["id"])
                                if latest is None:
                                    raise RuntimeError("Задание потеряно")
                                latest["status"] = (
                                    "cancelled"
                                    if latest["status"] == "cancelling"
                                    else "failed"
                                )
                                latest["error"] = (
                                    "Пересчёт не завершён. Прежний индекс сохранён; "
                                    "проверьте подключение и повторите задание"
                                )
                                await service.save(latest)
                                await session.commit()
                            if active_migration() == job["id"]:
                                migration_marker().unlink()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Keep polling across database/startup outages. Never log an
                # upstream exception containing decrypted settings or URLs.
                logger.warning("Задания пересчёта временно недоступны")
            finally:
                fcntl.flock(worker_lock, fcntl.LOCK_UN)
        await asyncio.sleep(2)
