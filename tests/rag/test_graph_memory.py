import json
import logging
from contextlib import asynccontextmanager

import pytest

from rag.graph_memory import GraphMemory


@pytest.mark.asyncio
async def test_crag_search_uses_history_for_follow_up_and_refinement(monkeypatch):
    from rag.crag import CragConfig

    memory = GraphMemory()
    searches: list[str] = []

    @asynccontextmanager
    async def fake_use_graph(graph_id: str):
        yield object()

    async def fake_retrieve(rag, question: str, mode: str):
        searches.append(question)
        if "ММФ" in question:
            return [
                {
                    "content": (
                        "Для направления подготовки «Математика и компьютерные науки "
                        "(01.03.02, бакалавр)» на ММФ бюджетных мест: 80."
                    ),
                    "file_path": "https://mmf.nsu.ru",
                }
            ]
        return [
            {
                "content": (
                    "Для направления подготовки «Программная инженерия "
                    "(09.03.04, бакалавр)» на ФИТ бюджетных мест: 125."
                ),
                "file_path": "https://nsu.ru/fit",
            }
        ]

    async def fake_config():
        return CragConfig(
            enabled=True,
            relevance_threshold=0.5,
            min_chunks=1,
            allow_refine=False,
            use_faculty_table=True,
        )

    class _FakeProvider:
        async def generate(self, messages, profile=None, **kwargs):
            return '{"score": 0.9, "relevant": true}'

    monkeypatch.setattr("rag.crag.get_llm_provider", lambda: _FakeProvider())
    monkeypatch.setattr(memory, "_use_graph", fake_use_graph)
    monkeypatch.setattr(memory, "_retrieve_raw_chunks", fake_retrieve)
    monkeypatch.setattr("rag.graph_memory.load_crag_config", fake_config)

    history_text = (
        "user: какие условия поступления на ФИТ?\n"
        "assistant: Факультет информационных технологий "
        "ведет прием на программы бакалавриата."
    )
    retrieval = await memory.query_with_crag(
        "test_graph",
        "а на ММФ?",
        conversation_history=history_text,
    )

    assert searches == [f"{history_text}\n\nа на ММФ?"]
    assert "математика и компьютерные науки" in retrieval.context.lower()
    assert any("https://mmf.nsu.ru" in s.get("url", "") for s in retrieval.sources)


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
    monkeypatch.setattr(memory, "_rerank_sources_with_llm", fake_rerank_sources_with_llm)

    retrieval = await memory.query_with_sources("test_graph", "Вопрос")

    assert retrieval.context == "Ответ"
    assert retrieval.evidence == ["Фрагмент правил приема"]
    assert retrieval.sources == [{"url": source_url, "title": "Правила приема"}]


@pytest.mark.asyncio
async def test_query_with_sources_logs_rag_query_and_response(monkeypatch, caplog):
    memory = GraphMemory()
    source_url = "https://example.test/rules.pdf"

    class FakeRag:
        async def aquery_llm(self, question, param):
            return {
                "llm_response": {"content": "Ответ из RAG"},
                "data": {
                    "chunks": [
                        {
                            "file_path": source_url,
                            "content": "Фрагмент правил приема",
                            "score": 0.91,
                        }
                    ],
                    "entities": [{"entity_name": "НГУ"}],
                    "relations": [{"src_id": "НГУ", "tgt_id": "прием"}],
                },
            }

    @asynccontextmanager
    async def fake_use_graph(graph_id: str):
        assert graph_id == "test_graph"
        yield FakeRag()

    async def fake_get_source_titles(graph_id: str):
        assert graph_id == "test_graph"
        return {source_url: "Правила приема"}

    async def fake_rerank_sources_with_llm(**kwargs):
        return [{"url": source_url, "title": "Правила приема"}]

    monkeypatch.setattr(memory, "_use_graph", fake_use_graph)
    monkeypatch.setattr(memory, "_get_source_titles", fake_get_source_titles)
    monkeypatch.setattr(memory, "_rerank_sources_with_llm", fake_rerank_sources_with_llm)

    with caplog.at_level(logging.DEBUG, logger="rag.graph_memory"):
        retrieval = await memory.query_with_sources(
            "test_graph",
            "Какие документы нужны?",
            conversation_history="user: До этого спрашивал про сроки",
        )

    assert retrieval.context == "Ответ из RAG"
    assert retrieval.sources == [{"url": source_url, "title": "Правила приема"}]
    log_text = caplog.text
    assert "RAG retrieval query" in log_text
    assert "Какие документы нужны?" in log_text
    assert "RAG retrieval response" in log_text
    assert "Ответ из RAG" in log_text
    assert "RAG retrieval data" in log_text
    assert "chunks=1 entities=1 relations=1" in log_text
    assert "RAG retrieval sources" in log_text


@pytest.mark.parametrize("failure", [TimeoutError("timeout"), RuntimeError("429")])
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("factory_failure", [False, True])
async def test_source_reranker_failure_preserves_retrieved_context(
    monkeypatch, failure, direct, factory_failure
):
    memory = GraphMemory()
    candidates = [
        {"url": f"https://example.test/{i}", "title": f"Title {i}", "snippet": "text"}
        for i in range(3)
    ]

    class Provider:
        async def generate(self, messages, profile):
            raise failure

    class Rag:
        async def aquery_llm(self, question, param):
            return {
                "llm_response": {"content": "Retrieved answer"},
                "data": {
                    "chunks": [
                        {"file_path": "local.txt", "content": "ignored"},
                        *[
                            {"file_path": s["url"], "content": s["snippet"]}
                            for s in candidates
                        ],
                        {"file_path": candidates[0]["url"], "content": "duplicate"},
                    ]
                },
            }

    @asynccontextmanager
    async def use_graph(graph_id):
        yield Rag()

    async def titles(graph_id):
        return {s["url"]: s["title"] for s in candidates}

    def get_provider():
        if factory_failure:
            raise failure
        return Provider()

    monkeypatch.setattr("rag.graph_memory.get_llm_provider", get_provider)
    monkeypatch.setattr(memory, "_use_graph", use_graph)
    monkeypatch.setattr(memory, "_get_source_titles", titles)
    expected = [{"url": s["url"], "title": s["title"]} for s in candidates[:2]]
    if direct:
        assert (
            await memory._rerank_sources_with_llm("question", None, candidates, 2)
            == expected
        )
    else:
        retrieval = await memory.query_with_sources("test", "question", max_sources=2)
        assert (retrieval.context, retrieval.sources) == ("Retrieved answer", expected)


async def test_source_reranker_propagates_cancellation(monkeypatch):
    import asyncio

    class Provider:
        async def generate(self, messages, profile):
            raise asyncio.CancelledError

    monkeypatch.setattr("rag.graph_memory.get_llm_provider", lambda: Provider())
    with pytest.raises(asyncio.CancelledError):
        await GraphMemory()._rerank_sources_with_llm(
            "question",
            None,
            [{"url": "https://example.test", "title": "T", "snippet": "S"}],
            1,
        )


async def test_source_reranker_preserves_ranking_filtering_and_limit(monkeypatch):
    class Provider:
        async def generate(self, messages, profile):
            return json.dumps(
                [
                    "https://example.test/2",
                    "unknown",
                    None,
                    "https://example.test/2",
                    "https://example.test/1",
                    "https://example.test/0",
                ]
            )

    monkeypatch.setattr("rag.graph_memory.get_llm_provider", lambda: Provider())
    candidates = [
        {"url": f"https://example.test/{i}", "title": f"Title {i}", "snippet": "text"}
        for i in range(3)
    ]
    assert await GraphMemory()._rerank_sources_with_llm(
        "question", "previous question", candidates, 2
    ) == [{"url": s["url"], "title": s["title"]} for s in [candidates[2], candidates[1]]]


@pytest.mark.asyncio
async def test_crag_separates_search_instructions_from_filter_question(monkeypatch):
    from pipeline.prompts import LIGHTRAG_LEVEL_HINT
    from rag.crag import CragConfig, FacultyHint

    memory = GraphMemory()
    question = "Какие документы нужны?"
    search_query = f"{question}\n\n{LIGHTRAG_LEVEL_HINT}"
    searches = []
    filtered = []

    @asynccontextmanager
    async def use_graph(graph_id):
        yield object()

    async def retrieve(rag, query, mode):
        searches.append(query)
        return []

    async def config():
        return CragConfig(enabled=True)

    async def crag(question, chunks, **kwargs):
        filtered.append((question, kwargs.get("conversation_history")))
        return [], FacultyHint(faculty=None, level="master")

    monkeypatch.setattr(memory, "_use_graph", use_graph)
    monkeypatch.setattr(memory, "_retrieve_raw_chunks", retrieve)
    monkeypatch.setattr("rag.graph_memory.load_crag_config", config)
    monkeypatch.setattr("rag.graph_memory.run_crag", crag)
    await memory.query_with_crag(
        "test",
        question,
        search_query=search_query,
        conversation_history="user: магистратура",
    )
    assert searches == [f"user: магистратура\n\n{search_query}"]
    assert filtered == [(question, "user: магистратура")]
