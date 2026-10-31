from typing import Any, cast

import pytest

from api.services import document_indexer
from db.postgres.models import Document
from db.postgres.services.document import DocumentService


class FakeDocument:
    def __init__(self, doc_id: str = "doc-1", rag_doc_id: str = "rag-v1"):
        self.id = doc_id
        self.title = "Test Doc"
        self.source_url = "https://example.test/doc"
        self.rag_doc_id = rag_doc_id
        self.content_hash: str | None = "hash-v1"
        self.content_length: int | None = 100
        self.status = "проиндексирован"


class FakeDocumentService:
    def __init__(self, session=None, document=None):  # noqa: ANN001
        self.session = session
        self.document = document or FakeDocument()
        self.calls: list[tuple[str, Any]] = []

    async def create_or_update_for_source(self, **kwargs):  # noqa: ANN001
        self.calls.append(("create_or_update_for_source", kwargs))
        return self.document

    async def mark_indexed(self, doc_id, **kwargs):  # noqa: ANN001
        self.calls.append(("mark_indexed", (doc_id, kwargs)))
        self.document.rag_doc_id = str(kwargs.get("rag_doc_id") or "")
        self.document.status = "проиндексирован"
        self.document.content_hash = kwargs.get("content_hash")
        return self.document

    async def mark_error(self, doc_id):  # noqa: ANN001
        self.calls.append(("mark_error", doc_id))
        self.document.status = "ошибка индексации"
        return True


class BaseFakeMemory:
    def __init__(self, status: str = "processed", chunks: list[str] | None = None):
        self.status = status
        self.chunks = chunks if chunks is not None else ["chunk-1"]
        self.delete_doc_calls: list[tuple[str, str]] = []
        self.delete_doc_safe_calls: list[tuple[str, str, set[str] | None]] = []

    async def get_doc_status(self, graph_id: str, doc_id: str) -> str | None:
        return self.status

    async def get_chunks_list(self, graph_id: str, doc_id: str) -> list[str]:
        return list(self.chunks)

    async def delete_doc(self, graph_id: str, doc_id: str) -> bool:
        self.delete_doc_calls.append((graph_id, doc_id))
        return True

    async def delete_doc_safe(
        self,
        graph_id: str,
        old_doc_id: str,
        preserve_chunk_ids: set[str] | None = None,
    ) -> bool:
        self.delete_doc_safe_calls.append((graph_id, old_doc_id, preserve_chunk_ids))
        return True


@pytest.mark.asyncio
async def test_index_document_preserves_old_version_when_indexing_fails(monkeypatch):
    """При сбое индексации старый документ не удаляется, а rag_doc_id не меняется."""
    doc = FakeDocument(doc_id="doc-123", rag_doc_id="old-rag-doc-v1")
    service = FakeDocumentService(document=doc)
    events: list[object] = []

    async def failing_add_texts_async(texts, *args, **kwargs):  # noqa: ANN001
        events.append("add_failed")
        raise RuntimeError("LLM embedding timeout / API key expired")

    fake_memory = BaseFakeMemory()

    monkeypatch.setattr(document_indexer, "add_texts_async", failing_add_texts_async)
    monkeypatch.setattr(document_indexer, "get_graph_memory", lambda: fake_memory)

    result = await document_indexer.index_document(
        document=cast(Document, doc),
        text="Новый текст документа",
        content_hash="new-hash",
        document_service=cast(DocumentService, service),
    )

    assert result is None
    # Старый документ НЕ должен быть удален
    assert len(fake_memory.delete_doc_calls) == 0
    assert len(fake_memory.delete_doc_safe_calls) == 0

    # Document.rag_doc_id в БД по-прежнему указывает на старую версию
    assert doc.rag_doc_id == "old-rag-doc-v1"
    # Статус зафиксирован как ошибка
    assert doc.status == "ошибка индексации"


@pytest.mark.asyncio
async def test_index_document_preserves_old_version_when_saved_count_zero(monkeypatch):
    """Если add_texts_async вернул 0 (ничего не сохранилось),
    старая версия не удаляется.
    """
    doc = FakeDocument(doc_id="doc-123", rag_doc_id="old-rag-doc-v1")
    service = FakeDocumentService(document=doc)
    fake_memory = BaseFakeMemory()

    async def zero_add_texts_async(texts, *args, **kwargs):  # noqa: ANN001
        return 0

    monkeypatch.setattr(document_indexer, "add_texts_async", zero_add_texts_async)
    monkeypatch.setattr(document_indexer, "get_graph_memory", lambda: fake_memory)

    result = await document_indexer.index_document(
        document=cast(Document, doc),
        text="Новый текст",
        content_hash="new-hash",
        document_service=cast(DocumentService, service),
    )

    assert result is None
    assert len(fake_memory.delete_doc_calls) == 0
    assert len(fake_memory.delete_doc_safe_calls) == 0
    assert doc.rag_doc_id == "old-rag-doc-v1"
    assert doc.status == "ошибка индексации"


@pytest.mark.asyncio
async def test_index_document_preserves_old_on_internal_failed_status(
    monkeypatch,
):
    """[P1] LightRAG перехватил ошибку extraction/merge и сохранил DocStatus.FAILED без
    исключения из ainsert(). saved_count > 0, но переключение в Postgres и удаление
    старой версии должны быть заблокированы.
    """
    doc = FakeDocument(doc_id="doc-fail", rag_doc_id="old-rag-doc-v1")
    service = FakeDocumentService(document=doc)
    # add_texts_async вернул 1, но статус документа в LightRAG — failed
    fake_memory = BaseFakeMemory(status="failed")

    async def fake_add_texts_async(texts, *args, **kwargs):  # noqa: ANN001
        return 1

    monkeypatch.setattr(document_indexer, "add_texts_async", fake_add_texts_async)
    monkeypatch.setattr(document_indexer, "get_graph_memory", lambda: fake_memory)

    result = await document_indexer.index_document(
        document=cast(Document, doc),
        text="Текст с упавшей обработкой",
        content_hash="hash-failed",
        document_service=cast(DocumentService, service),
    )

    assert result is None
    # Старый документ НЕ должен быть удален!
    assert len(fake_memory.delete_doc_safe_calls) == 0
    assert len(fake_memory.delete_doc_calls) == 0

    # Document.rag_doc_id остался неизменным
    assert doc.rag_doc_id == "old-rag-doc-v1"
    assert doc.status == "ошибка индексации"


@pytest.mark.asyncio
async def test_index_document_preserves_old_version_on_busy_queue(monkeypatch):
    """[P1] При занятом pipeline ainsert() ставит задачу в очередь (PENDING) и
    возвращается немедленно. Документ не успевает стать PROCESSED за время таймаута;
    старая версия должна быть сохранена.
    """
    doc = FakeDocument(doc_id="doc-busy", rag_doc_id="old-rag-doc-v1")
    service = FakeDocumentService(document=doc)
    # Статус остаётся pending
    fake_memory = BaseFakeMemory(status="pending")

    async def fake_add_texts_async(texts, *args, **kwargs):  # noqa: ANN001
        return 1

    monkeypatch.setattr(document_indexer, "add_texts_async", fake_add_texts_async)
    monkeypatch.setattr(document_indexer, "get_graph_memory", lambda: fake_memory)

    result = await document_indexer.index_document(
        document=cast(Document, doc),
        text="Текст в очередь",
        content_hash="hash-busy",
        document_service=cast(DocumentService, service),
        wait_timeout=0.02,
    )

    assert result is None
    assert len(fake_memory.delete_doc_safe_calls) == 0
    assert doc.rag_doc_id == "old-rag-doc-v1"
    assert doc.status == "ошибка индексации"


@pytest.mark.asyncio
async def test_index_document_order_add_switch_delete(monkeypatch):
    """Проверяет строгий порядок add -> switch -> delete при успешной индексации."""
    doc = FakeDocument(doc_id="doc-456", rag_doc_id="old-rag-doc-v1")
    service = FakeDocumentService(document=doc)
    operation_log = []

    async def fake_add_texts_async(texts, *args, **kwargs):  # noqa: ANN001
        source_ids = kwargs.get("source_ids") or []
        new_rag_doc_id = source_ids[0] if source_ids else ""
        operation_log.append(("add", new_rag_doc_id))
        return 1

    orig_mark_indexed = service.mark_indexed

    async def tracked_mark_indexed(doc_id, **kwargs):  # noqa: ANN001
        operation_log.append(("switch", kwargs.get("rag_doc_id")))
        return await orig_mark_indexed(doc_id, **kwargs)

    service.mark_indexed = tracked_mark_indexed

    class TrackedFakeMemory(BaseFakeMemory):
        async def delete_doc_safe(
            self,
            graph_id: str,
            old_doc_id: str,
            preserve_chunk_ids: set[str] | None = None,
        ) -> bool:
            operation_log.append(("delete", old_doc_id, preserve_chunk_ids))
            return True

    fake_memory = TrackedFakeMemory(status="processed", chunks=["chunk-x1", "chunk-x2"])

    monkeypatch.setattr(document_indexer, "add_texts_async", fake_add_texts_async)
    monkeypatch.setattr(document_indexer, "get_graph_memory", lambda: fake_memory)

    result = await document_indexer.index_document(
        document=cast(Document, doc),
        text="Успешный текст документа",
        content_hash="hash-v2",
        document_service=cast(DocumentService, service),
    )

    assert result is not None
    assert len(operation_log) == 3

    add_op, new_id = operation_log[0]
    switch_op, switch_id = operation_log[1]
    delete_op, delete_id, preserved = operation_log[2]

    assert add_op == "add"
    assert switch_op == "switch"
    assert delete_op == "delete"

    # Новый rag_doc_id не совпадает со старым
    assert new_id != "old-rag-doc-v1"
    # Переключение переключило ровно на новый rag_doc_id
    assert switch_id == new_id
    # Удаление удалило именно старый rag_doc_id с защитой чанков новой версии
    assert delete_id == "old-rag-doc-v1"
    assert preserved == {"chunk-x1", "chunk-x2"}

    # В объекте документа теперь новый rag_doc_id
    assert result.rag_doc_id == new_id


@pytest.mark.asyncio
async def test_index_document_cleans_up_if_delete_fails(monkeypatch):
    """Если удаление старой версии упало, операция всё равно считается успешной."""
    doc = FakeDocument(doc_id="doc-789", rag_doc_id="old-rag-doc-v1")
    service = FakeDocumentService(document=doc)

    async def fake_add_texts_async(texts, *args, **kwargs):  # noqa: ANN001
        return 1

    class FailingDeleteMemory(BaseFakeMemory):
        async def delete_doc_safe(
            self,
            graph_id: str,
            old_doc_id: str,
            preserve_chunk_ids: set[str] | None = None,
        ) -> bool:
            raise RuntimeError("Delete error from LightRAG")

    monkeypatch.setattr(document_indexer, "add_texts_async", fake_add_texts_async)
    monkeypatch.setattr(
        document_indexer, "get_graph_memory", lambda: FailingDeleteMemory()
    )

    result = await document_indexer.index_document(
        document=cast(Document, doc),
        text="Текст",
        content_hash="hash-v2",
        document_service=cast(DocumentService, service),
    )

    # Успешно, так как новая версия уже проиндексирована и переключена
    assert result is not None
    assert result.rag_doc_id != "old-rag-doc-v1"
