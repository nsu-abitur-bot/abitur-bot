"""Сбои эмбеддинга и поиска не должны выглядеть как данные (#282, #283).

Здесь настоящий LightRAG с файловыми сторами во временной папке: отравление
стора нулевым вектором и «полупроиндексированный» документ — свойства
LightRAG и nano_vectordb, на подделке их не воспроизвести. Сеть не нужна:
эмбеддинг, модель и токенизатор подменены.
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import Tokenizer, wrap_embedding_func_with_attrs

from rag import graph_memory as graph_memory_module
from rag.graph_memory import GraphMemory, RetrievalError

EMBEDDING_DIM = 8
DOC_TEXT = "Приём документов в НГУ начинается двадцатого июня. " * 20


class _WordTokenizer:
    """Токенизатор без сети: tiktoken при первом запуске качает словарь."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._words: list[str] = []

    def encode(self, content: str) -> list[int]:
        tokens = []
        for word in content.split(" "):
            if word not in self._vocab:
                self._vocab[word] = len(self._words)
                self._words.append(word)
            tokens.append(self._vocab[word])
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return " ".join(self._words[t] for t in tokens)


class _EmbeddingSwitch:
    """Эмбеддинг, который по флагу падает, как провайдер при 429."""

    def __init__(self) -> None:
        self.failing = False

    async def __call__(self, texts: list[str]) -> np.ndarray:
        if self.failing:
            raise RuntimeError("RateLimitError 429")
        return np.array(
            [[float(len(t) % 7 + 1)] + [1.0] * (EMBEDDING_DIM - 1) for t in texts]
        )


async def _fake_llm(prompt: str, **kwargs) -> str:
    # На извлечение ключевых слов отвечаем ими, чтобы hybrid-поиск дошёл до
    # векторного стора; на остальное (извлечение графа) — пусто.
    if "keywords" in prompt or "keywords" in str(kwargs.get("system_prompt", "")):
        return '{"high_level_keywords": ["приём"], "low_level_keywords": ["НГУ"]}'
    return ""


async def _make_memory(tmp_path, embedding: _EmbeddingSwitch):
    @wrap_embedding_func_with_attrs(embedding_dim=EMBEDDING_DIM)
    async def embedding_func(texts: list[str]) -> np.ndarray:
        return await embedding(texts)

    rag = LightRAG(
        working_dir=str(tmp_path),
        workspace="test_graph",
        llm_model_func=_fake_llm,
        embedding_func=embedding_func,
        tokenizer=Tokenizer("words", _WordTokenizer()),
        chunk_token_size=40,
        chunk_overlap_token_size=5,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    memory = GraphMemory()
    memory.workspace_path = str(tmp_path)

    @asynccontextmanager
    async def use_graph(graph_id: str):
        yield rag

    async def no_clear_cache(graph_id: str) -> None:
        return None

    memory._use_graph = use_graph  # type: ignore[method-assign]
    memory.clear_cache = no_clear_cache  # type: ignore[method-assign]
    return memory, rag


async def _stored_vectors(rag: LightRAG) -> int:
    client = await rag.chunks_vdb._get_client()  # type: ignore[attr-defined]
    return len(client)


@pytest.mark.asyncio
async def test_save_reports_failure_and_leaves_nothing_when_embedding_fails(tmp_path):
    embedding = _EmbeddingSwitch()
    memory, rag = await _make_memory(tmp_path, embedding)
    embedding.failing = True

    saved = await memory.save("test_graph", DOC_TEXT, source_id="doc-1")

    assert saved is False
    assert await _stored_vectors(rag) == 0
    assert await rag.doc_status.get_by_id("doc-1") is None
    assert await rag.full_docs.get_by_id("doc-1") is None
    # Чанки удаляются по chunks_list, который LightRAG затирает при FAILED.
    assert await rag.text_chunks.is_empty()

    await rag.finalize_storages()


@pytest.mark.asyncio
async def test_save_succeeds_when_embedding_works(tmp_path):
    embedding = _EmbeddingSwitch()
    memory, rag = await _make_memory(tmp_path, embedding)

    saved = await memory.save("test_graph", DOC_TEXT, source_id="doc-1")

    assert saved is True
    assert await _stored_vectors(rag) > 0
    status = await rag.doc_status.get_by_id("doc-1")
    assert status is not None and status["status"] == "processed"

    await rag.finalize_storages()


@pytest.mark.asyncio
async def test_query_raises_when_query_embedding_fails(tmp_path):
    embedding = _EmbeddingSwitch()
    memory, rag = await _make_memory(tmp_path, embedding)
    assert await memory.save("test_graph", DOC_TEXT, source_id="doc-1")
    embedding.failing = True

    with pytest.raises(RetrievalError, match="429"):
        await memory.query("test_graph", "Когда приём документов?", mode="hybrid")

    await rag.finalize_storages()


# Ниже — разбор ответа LightRAG без настоящего стора: aquery_llm ловит
# исключения запроса сам и возвращает status=failure, и это надо отличать от
# честного «ничего не нашлось».

QUERY_FAILED = {
    "status": "failure",
    "message": "Query failed: RateLimitError 429",
    "data": {},
    "metadata": {},
    "llm_response": {"content": None},
}
NO_RESULTS = {
    "status": "failure",
    "message": "Query returned no results",
    "data": {},
    "metadata": {"failure_reason": "no_results", "mode": "hybrid"},
    "llm_response": {"content": "Sorry, I'm not able to provide an answer."},
}


class _RagReturning:
    def __init__(self, response: dict) -> None:
        self.response = response

    async def aquery_llm(self, question, param):
        return self.response


def _memory_with_rag(monkeypatch, fake_rag) -> GraphMemory:
    memory = GraphMemory()

    @asynccontextmanager
    async def use_graph(graph_id: str):
        yield fake_rag

    async def no_titles(graph_id: str) -> dict:
        return {}

    async def crag_config():
        return SimpleNamespace()

    monkeypatch.setattr(memory, "_use_graph", use_graph)
    monkeypatch.setattr(memory, "_get_source_titles", no_titles)
    monkeypatch.setattr(graph_memory_module, "load_crag_config", crag_config)
    return memory


@pytest.mark.asyncio
async def test_query_with_sources_raises_on_swallowed_failure(monkeypatch):
    memory = _memory_with_rag(monkeypatch, _RagReturning(QUERY_FAILED))

    with pytest.raises(RetrievalError, match="429"):
        await memory.query_with_sources("test_graph", "Вопрос")


@pytest.mark.asyncio
async def test_query_with_sources_no_results_is_empty_not_error(monkeypatch):
    memory = _memory_with_rag(monkeypatch, _RagReturning(NO_RESULTS))

    # Текст fail_response LightRAG не должен уйти модели как «контекст».
    assert await memory.query_with_sources("test_graph", "Вопрос") == ("", [])


@pytest.mark.asyncio
async def test_query_with_sources_wraps_unexpected_exception(monkeypatch):
    class BrokenRag:
        async def aquery_llm(self, question, param):
            raise OSError("vdb file is corrupted")

    memory = _memory_with_rag(monkeypatch, BrokenRag())

    with pytest.raises(RetrievalError, match="corrupted"):
        await memory.query_with_sources("test_graph", "Вопрос")


@pytest.mark.asyncio
async def test_query_with_crag_raises_on_retrieval_failure(monkeypatch):
    memory = _memory_with_rag(monkeypatch, _RagReturning(QUERY_FAILED))

    with pytest.raises(RetrievalError, match="429"):
        await memory.query_with_crag("test_graph", "Вопрос")


@pytest.mark.asyncio
async def test_query_with_crag_nothing_kept_is_empty(monkeypatch):
    response = {"data": {"chunks": [{"content": "Чужой факультет", "file_path": ""}]}}
    memory = _memory_with_rag(monkeypatch, _RagReturning(response))

    async def drop_everything(question, candidates, retrieve_fn, config):
        return [], SimpleNamespace(faculty=None, level=None)

    monkeypatch.setattr(graph_memory_module, "run_crag", drop_everything)

    assert await memory.query_with_crag("test_graph", "Вопрос") == ("", [])
