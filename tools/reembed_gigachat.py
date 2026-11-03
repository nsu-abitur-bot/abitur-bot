"""Offline migration of one instance; input is a JSON object on stdin, never argv."""

import argparse
import asyncio
import fcntl
import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import select

from api.services.instance_settings import merge_changes
from config.instance import install
from db.postgres.db import AsyncSessionLocal
from db.postgres.models import Document, FaqEntry, Settings
from db.postgres.services.document import ACTIVE_DOCUMENT_STATUSES
from db.postgres.services.instance_settings import SETTINGS_KEY, InstanceSettingsService
from faq.matcher import FAQMatcher
from llm.providers.gigachat import GigaChatEmbedding
from rag.reembed import rebuild_workspace


def private_json(path: Path, data: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as output:
        json.dump(data, output, ensure_ascii=False)


@contextmanager
def _migration_lock(root: Path) -> Iterator[None]:
    root.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        root.parent / ".embedding-migration.lock",
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
    )
    with os.fdopen(descriptor, "w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


async def migrate(changes: dict[str, str]) -> dict:
    root = Path(os.environ.get("LIGHTRAG_WORKSPACE_BASE", "./data/lightrag"))
    if root.is_symlink():
        raise ValueError("Символическая ссылка вместо индекса")
    root = root.resolve()
    with _migration_lock(root):
        return await _migrate(changes, root)


async def _migrate(changes: dict[str, str], root: Path) -> dict:
    if os.getenv("LIGHTRAG_STORAGE", "json") != "json":
        raise ValueError("Эта процедура поддерживает файловое JSON-хранилище")
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists() and any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Символические ссылки не поддерживаются")
    directory = Path(tempfile.mkdtemp(prefix="embedding-migration-", dir=root.parent))
    stage = directory / "new-index"
    stage.mkdir()
    async with AsyncSessionLocal() as session:
        service = InstanceSettingsService(session)
        snapshot = await service.read()
        proposed = merge_changes(
            snapshot,
            {
                **changes,
                "FAQ_EMBEDDING_PROVIDER": "gigachat",
                "LIGHTRAG_EMBEDDING_PROVIDER": "gigachat",
            },
        )
        if await session.scalar(
            select(Document.id)
            .where(Document.status.in_(ACTIVE_DOCUMENT_STATUSES))
            .limit(1)
        ):
            raise ValueError("Сначала завершите обработку документов")
        documents = (await session.scalars(select(Document))).all()
        for document in documents:
            if document.status != "проиндексирован":
                continue
            workspace = root / document.graph_id
            if workspace.parent != root or not (workspace / "vdb_chunks.json").is_file():
                raise ValueError("Для документа отсутствует исходный векторный индекс")
            canonical = json.loads((workspace / "kv_store_full_docs.json").read_text())
            if document.rag_doc_id not in canonical:
                raise ValueError("Для документа отсутствует исходный текст")
        raw = await session.scalar(
            select(Settings.value).where(Settings.key == SETTINGS_KEY)
        )
        private_json(
            directory / "settings-before.json",
            {"revision": snapshot.revision, "encrypted_snapshot": raw},
        )
        adapter = GigaChatEmbedding(
            auth_key=proposed["GIGACHAT_API_KEY"],
            model=proposed["GIGACHAT_EMBEDDING_MODEL"],
            scope=proposed["GIGACHAT_SCOPE"],
        )
        # Validate the key/model before preparing the replacement index.
        await adapter(["Проверка модели эмбеддингов"])
        counts = {}
        if root.exists():
            for child in sorted(root.iterdir()):
                if child.is_symlink():
                    raise ValueError("Символические ссылки не поддерживаются")
                if child.is_dir() and list(child.glob("vdb_*.json")):
                    counts[child.name] = await asyncio.to_thread(
                        rebuild_workspace, child, stage / child.name, adapter
                    )
                elif child.is_dir():
                    # A populated workspace without vectors needs a full document rebuild.
                    docs = child / "kv_store_full_docs.json"
                    if docs.exists() and json.loads(docs.read_text()):
                        raise ValueError(
                            "В наполненном индексе отсутствуют векторные файлы"
                        )
                    shutil.copytree(child, stage / child.name)
                else:
                    shutil.copy2(child, stage / child.name)
        entries = (await session.scalars(select(FaqEntry))).all()
        items = [
            {"question": entry.question, "answer": entry.answer, "aliases": entry.aliases}
            for entry in entries
        ]
        # A disposable cache is built from scratch, never the old provider's vectors.
        install(proposed, snapshot.revision)
        matcher = FAQMatcher(
            faq_path=directory / "faq.yaml",
            embedder=adapter,
            cache_path=directory / "faq_cache.pkl",
        )
        await asyncio.to_thread(matcher.load_items, items)
        private_json(
            directory / "prepared.json",
            {
                "model": adapter.model,
                "dimensions": adapter.embedding_dim,
                "workspaces": counts,
                "faq_entries": len(items),
            },
        )
        # Check the snapshot again before publishing. API and bot must remain stopped.
        if (await service.read()).revision != snapshot.revision:
            raise ValueError("Настройки изменились во время пересчёта")
        old = directory / "old-index"
        if root.exists():
            os.replace(root, old)
        os.replace(stage, root)
        cache = Path(
            os.getenv("FAQ_EMBEDDINGS_CACHE_PATH", str(Path("faq/faq_cache.pkl")))
        )
        if cache.exists():
            shutil.copy2(cache, directory / "old-faq-cache.pkl")
            cache.unlink()
        if matcher.size:
            cache.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(directory / "faq_cache.pkl", cache)
        try:
            updated = await service.write(proposed, snapshot.revision)
        except Exception:
            # A lost commit acknowledgement is ambiguous: leave the instance offline.
            # The journal and both indexes allow an operator to reconcile it safely.
            private_json(
                directory / "needs-recovery.json",
                {"expected_revision": snapshot.revision + 1},
            )
            raise RuntimeError(
                "Не удалось подтвердить смену настроек; "
                "оставьте экземпляр остановленным и проверьте журнал миграции"
            ) from None
        private_json(directory / "completed.json", {"revision": updated.revision})
        return {
            "revision": updated.revision,
            "model": adapter.model,
            "dimensions": adapter.embedding_dim,
            "workspaces": counts,
            "faq_entries": len(items),
            "backup_directory": str(directory),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Confirm all instance API/bot/scheduler processes are stopped",
    )
    args = parser.parse_args()
    if not args.offline:
        parser.error(
            "Перед запуском остановите API, bot и scheduler экземпляра; "
            "требуется --offline"
        )
    changes = json.load(sys.stdin)
    if not isinstance(changes, dict) or set(changes) - {
        "GIGACHAT_API_KEY",
        "GIGACHAT_SCOPE",
        "GIGACHAT_EMBEDDING_MODEL",
    }:
        parser.error("Ожидаются только настройки подключения GigaChat")
    try:
        print(json.dumps(asyncio.run(migrate(changes)), ensure_ascii=False))
    except Exception as exc:
        # SDK/SQL exceptions must never expose credentials or document content.
        print(
            "Пересчёт не завершён: "
            + type(exc).__name__
            + ". Экземпляр оставьте остановленным; "
            "проверьте каталог embedding-migration-*.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
