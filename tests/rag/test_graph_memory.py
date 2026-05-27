from contextlib import asynccontextmanager

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
