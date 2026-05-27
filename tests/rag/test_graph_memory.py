from contextlib import asynccontextmanager

import pytest

from rag.graph_memory import GraphMemory


class _FakeRag:
    def __init__(self) -> None:
        self.deleted_doc_ids: list[str] = []

    async def adelete_by_doc_id(self, doc_id: str) -> None:
        self.deleted_doc_ids.append(doc_id)


async def test_delete_doc_clears_cache_after_success(monkeypatch, tmp_path):
    memory = GraphMemory()
    memory.workspace_path = str(tmp_path)
    rag = _FakeRag()
    clear_cache_calls: list[str] = []

    @asynccontextmanager
    async def fake_use_graph(graph_id: str):
        assert graph_id == "test_graph"
        yield rag

    async def fake_clear_cache(graph_id: str) -> None:
        clear_cache_calls.append(graph_id)

    monkeypatch.setattr(memory, "_use_graph", fake_use_graph)
    monkeypatch.setattr(memory, "clear_cache", fake_clear_cache)

    deleted = await memory.delete_doc("test_graph", "doc-1")

    assert deleted is True
    assert rag.deleted_doc_ids == ["doc-1"]
    assert clear_cache_calls == ["test_graph"]


@pytest.mark.asyncio
async def test_query_with_sources_uses_document_title(monkeypatch):
    memory = GraphMemory()
    source_url = "https://example.test/rules.pdf"

    class FakeRag:
        async def aquery_llm(self, question, param):
            return {
                "llm_response": {"content": "Ответ"},
                "data": {
                    "chunks": [
                        {
                            "file_path": source_url,
                            "content": "Фрагмент правил приема",
                        }
                    ]
                },
            }

    @asynccontextmanager
    async def fake_use_graph(graph_id: str):
        assert graph_id == "test_graph"
        yield FakeRag()

    async def fake_get_list_docs(graph_id: str):
        assert graph_id == "test_graph"
        return [{"id": "doc-uuid", "url": source_url}]

    async def fake_get_source_titles(graph_id: str):
        assert graph_id == "test_graph"
        return {source_url: "Правила приема"}

    async def fake_rerank_sources_with_llm(**kwargs):
        return [
            {"url": source["url"], "title": source["title"]}
            for source in kwargs["sources"]
        ]

    monkeypatch.setattr(memory, "_use_graph", fake_use_graph)
    monkeypatch.setattr(memory, "get_list_docs", fake_get_list_docs)
    monkeypatch.setattr(memory, "_get_source_titles", fake_get_source_titles)
    monkeypatch.setattr(
        memory, "_rerank_sources_with_llm", fake_rerank_sources_with_llm
    )

    answer, sources = await memory.query_with_sources("test_graph", "Вопрос")

    assert answer == "Ответ"
    assert sources == [{"url": source_url, "title": "Правила приема"}]
