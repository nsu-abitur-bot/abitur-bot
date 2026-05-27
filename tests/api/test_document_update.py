import asyncio
from typing import cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from api.services import document_update


class FakeDocument:
    def __init__(self, document_id: str, content_hash: str | None = None):
        self.id = document_id
        self.title = f"Document {document_id}"
        self.source_url = f"https://example.test/{document_id}"
        self.content_hash = content_hash


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
