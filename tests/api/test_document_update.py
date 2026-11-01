import asyncio
from typing import cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from api.services import document_update


class FakeDocument:
    def __init__(
        self,
        document_id: str,
        content_hash: str | None = None,
        rag_doc_id: str = "old-rag-id",
    ):
        self.id = document_id
        self.title = f"Document {document_id}"
        self.source_url = f"https://example.test/{document_id}"
        self.content_hash = content_hash
        self.rag_doc_id = rag_doc_id
        self.status = "проиндексирован"


@pytest.mark.asyncio
async def test_check_documents_runs_fetches_in_parallel_and_saves(monkeypatch):
    active_fetches = 0
    max_active_fetches = 0
    saved = {}
    docs = [FakeDocument(str(index), "old") for index in range(4)]

    class FakeDocumentService:
        def __init__(self, session):  # noqa: ANN001
            pass

        async def list_checkable(self, graph_id, document_ids):  # noqa: ANN001
            saved["list_args"] = (graph_id, document_ids)
            return docs

        async def save_check_results(self, results):  # noqa: ANN001
            saved["results"] = results

    async def fake_fetch_url_bytes(url: str) -> bytes:
        nonlocal active_fetches, max_active_fetches
        active_fetches += 1
        max_active_fetches = max(max_active_fetches, active_fetches)
        await asyncio.sleep(0.01)
        active_fetches -= 1
        return url.encode()

    monkeypatch.setattr(document_update, "DocumentService", FakeDocumentService)
    monkeypatch.setattr(document_update, "fetch_url_bytes", fake_fetch_url_bytes)

    service = document_update.DocumentUpdateService(
        session=cast(AsyncSession, object()), graph_id="graph", check_concurrency=4
    )
    results = await service.check_documents(["1", "2"])

    assert len(results) == 4
    assert max_active_fetches > 1
    assert saved["list_args"] == ("graph", ["1", "2"])
    assert saved["results"] == results


@pytest.mark.asyncio
async def test_check_documents_marks_source_unavailable(monkeypatch):
    saved = {}
    docs = [FakeDocument("1", "old")]

    class FakeDocumentService:
        def __init__(self, session):  # noqa: ANN001
            pass

        async def list_checkable(self, graph_id, document_ids):  # noqa: ANN001
            return docs

        async def save_check_results(self, results):  # noqa: ANN001
            saved["results"] = results

    async def fake_fetch_url_bytes(url: str) -> bytes:
        raise ValueError("unreachable")

    monkeypatch.setattr(document_update, "DocumentService", FakeDocumentService)
    monkeypatch.setattr(document_update, "fetch_url_bytes", fake_fetch_url_bytes)

    service = document_update.DocumentUpdateService(session=cast(AsyncSession, object()))
    results = await service.check_documents()

    assert results[0]["status"] == "источник недоступен"
    assert results[0]["message"] == "Не удалось проверить источник"
    assert saved["results"] == results


@pytest.mark.asyncio
async def test_update_changed_documents_preserves_old_doc_when_indexing_fails(
    monkeypatch,
):
    """Сбой индексации оставляет старую версию в RAG и не удаляет её."""
    doc = FakeDocument("1", content_hash="old-hash", rag_doc_id="rag-v1")
    deleted_calls = []

    class FakeDocumentService:
        def __init__(self, session):  # noqa: ANN001
            pass

        async def list_checkable(self, graph_id, document_ids):  # noqa: ANN001
            return [doc]

        async def get_by_id(self, document_id):  # noqa: ANN001
            return doc if doc.id == document_id else None

        async def save_check_results(self, results):  # noqa: ANN001
            pass

        async def mark_error(self, document_id):  # noqa: ANN001
            doc.status = "ошибка индексации"

    async def fake_fetch_url_bytes(url: str) -> bytes:
        return b"new-content"

    async def fake_process_url(url: str) -> str:
        return "Новый текст"

    async def failing_add_texts_async(texts, *args, **kwargs):  # noqa: ANN001
        raise RuntimeError("Embedding timeout")

    class FakeMemory:
        async def delete_doc(self, graph_id, doc_id):  # noqa: ANN001
            deleted_calls.append((graph_id, doc_id))
            return True

    from api.services import document_indexer

    monkeypatch.setattr(document_update, "DocumentService", FakeDocumentService)
    monkeypatch.setattr(document_update, "fetch_url_bytes", fake_fetch_url_bytes)
    monkeypatch.setattr(document_update, "process_url", fake_process_url)
    monkeypatch.setattr(document_indexer, "add_texts_async", failing_add_texts_async)
    monkeypatch.setattr(document_indexer, "get_graph_memory", lambda: FakeMemory())

    service = document_update.DocumentUpdateService(session=cast(AsyncSession, object()))
    results = await service.update_changed_documents(["1"])

    assert len(results) == 1
    assert results[0]["status"] == "ошибка индексации"
    # Старый документ НЕ должен быть удалён!
    assert ("abitur_kb", "rag-v1") not in deleted_calls
    assert doc.rag_doc_id == "rag-v1"


@pytest.mark.asyncio
async def test_update_changed_documents_success_switches_rag_doc_id(monkeypatch):
    """Успешное обновление переключает rag_doc_id на новый и удаляет старый."""
    doc = FakeDocument("1", content_hash="old-hash", rag_doc_id="rag-v1")
    deleted_calls = []

    class FakeDocumentService:
        def __init__(self, session):  # noqa: ANN001
            pass

        async def list_checkable(self, graph_id, document_ids):  # noqa: ANN001
            return [doc]

        async def get_by_id(self, document_id):  # noqa: ANN001
            return doc if doc.id == document_id else None

        async def save_check_results(self, results):  # noqa: ANN001
            pass

        async def mark_indexed(self, doc_id, **kwargs):  # noqa: ANN001
            doc.rag_doc_id = str(kwargs.get("rag_doc_id") or "")
            doc.status = "проиндексирован"
            doc.content_hash = kwargs.get("content_hash")
            return doc

    async def fake_fetch_url_bytes(url: str) -> bytes:
        return b"new-content"

    async def fake_process_url(url: str) -> str:
        return "Новый текст"

    async def fake_add_texts_async(texts, *args, **kwargs):  # noqa: ANN001
        return 1

    class FakeMemory:
        async def delete_doc(self, graph_id, doc_id):  # noqa: ANN001
            deleted_calls.append((graph_id, doc_id))
            return True

    from api.services import document_indexer

    monkeypatch.setattr(document_update, "DocumentService", FakeDocumentService)
    monkeypatch.setattr(document_update, "fetch_url_bytes", fake_fetch_url_bytes)
    monkeypatch.setattr(document_update, "process_url", fake_process_url)
    monkeypatch.setattr(document_indexer, "add_texts_async", fake_add_texts_async)
    monkeypatch.setattr(document_indexer, "get_graph_memory", lambda: FakeMemory())

    service = document_update.DocumentUpdateService(session=cast(AsyncSession, object()))
    results = await service.update_changed_documents(["1"])

    assert len(results) == 1
    assert results[0]["status"] == "обновлён"
    # Старый документ был удалён ПОСЛЕ переключения
    assert ("abitur_kb", "rag-v1") in deleted_calls
    # Новый rag_doc_id отличается от старого
    assert doc.rag_doc_id != "rag-v1"
