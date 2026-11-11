"""Exercise the real Gemini embedding adapter across all unified consumers."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from lightrag.utils import Tokenizer

from config.instance import settings_scope
from config.model_settings import materialize_models
from faq.matcher import FAQMatcher
from llm import embedding as embedding_module
from llm.embedding import UnifiedEmbedding
from llm.providers.gemini import GeminiEmbeddings
from rag import graph_memory as graph_module
from rag.graph_memory import GraphMemory
from rag.reembed import rebuild_workspace
from tests.rag.test_reembed import fixture as build_graph
from tests.rag.test_retrieval_failures import _WordTokenizer


@pytest.mark.asyncio
async def test_gemini_staging_faq_and_live_graph_use_identical_sdk_parameters(
    tmp_path, monkeypatch
):
    calls = []

    class Models:
        def embed_content(self, **kwargs):
            calls.append(kwargs)
            dimension = getattr(kwargs.get("config"), "output_dimensionality", None) or 4
            return SimpleNamespace(
                embeddings=[
                    SimpleNamespace(values=[1.0] * dimension) for _ in kwargs["contents"]
                ]
            )

    class AsyncModels:
        async def embed_content(self, **kwargs):
            return Models().embed_content(**kwargs)

    class Client:
        models = Models()
        aio = SimpleNamespace(models=AsyncModels())

        def close(self):
            pass

    async def close():
        pass

    Client.aio.aclose = close

    class Provider:
        def get_embeddings_model(self):
            return GeminiEmbeddings(cast(Any, Client()))

    monkeypatch.setattr(embedding_module, "get_faq_provider", lambda: Provider())
    # A regressed legacy graph path still stays offline and exposes its 768D
    # request, rather than accidentally contacting a real provider in this test.
    monkeypatch.setattr(
        "llm.providers.gemini_graph_adapters.genai.Client", lambda **kwargs: Client()
    )
    monkeypatch.setenv("LIGHTRAG_STORAGE", "json")
    monkeypatch.setenv("LIGHTRAG_WORKSPACE_BASE", str(tmp_path / "root"))
    monkeypatch.setattr(
        graph_module, "TiktokenTokenizer", lambda: Tokenizer("words", _WordTokenizer())
    )
    values = materialize_models(
        {
            "TEXT_PROVIDER": "deepseek",
            "TEXT_MODEL": "deepseek-flash",
            "EMBEDDING_PROVIDER": "gemini",
            "EMBEDDING_MODEL": "mock-gemini-embedding-v2",
            "GEMINI_API_KEY": "synthetic-key",
        }
    )
    values.update({"_embedding_dim": "4", "_embedding_generation": "f" * 32})
    source = tmp_path / "source"
    build_graph(source)
    with settings_scope(values, 2):
        target = tmp_path / "root" / ".generations" / ("f" * 32) / "abitur_kb"
        rebuild_workspace(source, target, UnifiedEmbedding())
        matcher = FAQMatcher(
            load_yaml=False, embedder=UnifiedEmbedding(), cache_path=tmp_path / "faq.pkl"
        )
        matcher.load_items([{"question": "Question", "aliases": [], "answer": "Answer"}])
        assert await matcher.match_async("Question") == "Answer"
        memory = GraphMemory()
        async with memory._use_graph("abitur_kb") as graph:
            graph = cast(Any, graph)
            try:
                vectors = await graph.embedding_func(["Question"])
                assert graph.embedding_func.embedding_dim == 4
                assert len(vectors[0]) == 4
            finally:
                await graph.embedding_func.shutdown()
                await graph.llm_model_func.shutdown()
                await graph.finalize_storages()
    assert len(calls) == 6
    assert all(call["model"] == "mock-gemini-embedding-v2" for call in calls)
    assert all("config" not in call for call in calls)


@pytest.mark.asyncio
async def test_persistent_queue_uses_each_questions_settings_and_masks_binding():
    from lightrag.utils import priority_limit_async_func_call

    from config.instance import capture_settings, get_setting
    from rag.trace import trace_queued_callback, trace_worker_callback

    @trace_worker_callback
    async def model(_prompt):
        return get_setting("TEXT_MODEL")

    queue = priority_limit_async_func_call(1, queue_name="settings-test", llm_timeout=1)(
        model
    )
    call = trace_queued_callback(queue, "model")
    try:
        for name in ("first-model", "second-model"):
            with settings_scope(
                {"TEXT_MODEL": name, "DEEPSEEK_API_KEY": "secret-marker"}, 1
            ):
                assert await call("prompt") == name
                assert "secret-marker" not in repr(capture_settings())
    finally:
        await cast(Any, queue).shutdown()
