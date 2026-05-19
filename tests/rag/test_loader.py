from rag import loader


class _FakeGraphMemory:
    def __init__(self, save_result: bool) -> None:
        self.save_result = save_result
        self.saved: list[tuple[str, str, str | None, str | None]] = []
        self.cleanup_calls: list[str] = []
        self.clear_cache_calls: list[str] = []

    async def save(
        self,
        graph_id: str,
        text: str,
        source_id: str | None = None,
        file_paths_str: str | None = None,
    ) -> bool:
        self.saved.append((graph_id, text, source_id, file_paths_str))
        return self.save_result

    async def cleanup(self, graph_id: str | None = None) -> None:
        if graph_id is not None:
            self.cleanup_calls.append(graph_id)

    async def clear_cache(self, graph_id: str) -> None:
        self.clear_cache_calls.append(graph_id)


async def test_add_texts_async_clears_cache_after_success(monkeypatch):
    graph_memory = _FakeGraphMemory(save_result=True)
    monkeypatch.setattr(loader, "get_graph_memory", lambda: graph_memory)

    saved_count = await loader.add_texts_async(
        texts=["Документ"],
        graph_id="test_graph",
        source_ids=["Источник"],
        file_paths=["https://example.test/doc"],
    )

    assert saved_count == 1
    assert graph_memory.cleanup_calls == ["test_graph"]
    assert graph_memory.clear_cache_calls == ["test_graph"]


async def test_add_texts_async_does_not_clear_cache_without_saved_texts(monkeypatch):
    graph_memory = _FakeGraphMemory(save_result=False)
    monkeypatch.setattr(loader, "get_graph_memory", lambda: graph_memory)

    saved_count = await loader.add_texts_async(
        texts=["Документ"],
        graph_id="test_graph",
    )

    assert saved_count == 0
    assert graph_memory.cleanup_calls == ["test_graph"]
    assert graph_memory.clear_cache_calls == []
