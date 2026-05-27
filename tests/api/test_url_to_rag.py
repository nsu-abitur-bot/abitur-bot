import pytest

from api.services import url_to_rag


@pytest.mark.asyncio
async def test_parse_and_save_url_expands_before_saving(monkeypatch):
    saved_payload = {}

    async def fake_process_url(url: str) -> str:
        return "Документы принимает НГУ."

    async def fake_add_texts_async(texts, source_ids, file_paths):  # noqa: ANN001
        saved_payload["texts"] = texts
        saved_payload["source_ids"] = source_ids
        saved_payload["file_paths"] = file_paths
        return 1

    class FakeSessionContext:
        async def __aenter__(self):  # noqa: ANN001
            return object()

        async def __aexit__(self, exc_type, exc, traceback):  # noqa: ANN001
            return None

    class FakeDocument:
        id = "document-id"
        rag_doc_id = "old-document-id"

    class FakeDocumentService:
        def __init__(self, session):  # noqa: ANN001
            pass

        async def create_or_update_for_source(self, **kwargs):  # noqa: ANN001
            saved_payload["document_kwargs"] = kwargs
            return FakeDocument()

        async def mark_indexed(self, *args, **kwargs):  # noqa: ANN001
            saved_payload["mark_indexed"] = (args, kwargs)
            return FakeDocument()

    class FakeMemory:
        async def delete_doc(self, graph_id, doc_id):  # noqa: ANN001
            saved_payload["deleted"] = (graph_id, doc_id)
            return True

    monkeypatch.setattr(url_to_rag, "process_url", fake_process_url)
    monkeypatch.setattr(url_to_rag, "add_texts_async", fake_add_texts_async)
    monkeypatch.setattr(url_to_rag, "AsyncSessionLocal", lambda: FakeSessionContext())
    monkeypatch.setattr(url_to_rag, "DocumentService", FakeDocumentService)
    monkeypatch.setattr(url_to_rag, "get_graph_memory", lambda: FakeMemory())

    result = await url_to_rag.parse_and_save_url("https://example.test/doc", "Документ")

    assert result is True
    assert saved_payload["texts"] == [
        "Документы принимает НГУ (Новосибирский государственный университет)."
    ]
    assert saved_payload["source_ids"] == ["document-id"]
    assert saved_payload["file_paths"] == ["https://example.test/doc"]
