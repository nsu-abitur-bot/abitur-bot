"""Request traces through the actual persistent LightRAG callback queues (#303)."""

import asyncio
import logging
from typing import Any, cast

import pytest
import pytest_asyncio
from lightrag.utils import Tokenizer

from rag import graph_memory as graph_memory_module
from rag.graph_memory import GraphMemory
from rag.trace import RetrievalResult, capture_lightrag_trace
from tests.rag.test_retrieval_failures import EMBEDDING_DIM, _WordTokenizer


@pytest_asyncio.fixture
async def queued_graph(monkeypatch, tmp_path):
    class NoNetworkEmbedding:
        embedding_dim = EMBEDDING_DIM

        def __init__(self, **kwargs):
            pass

        async def __call__(self, texts):
            marker = texts[0]
            logging.getLogger("lightrag").info("callback %s", marker)
            if marker.endswith("ERROR"):
                raise ValueError(marker)
            return [[1.0] * EMBEDDING_DIM for _ in texts]

        async def aclose(self):
            pass

    class NoNetworkModel(NoNetworkEmbedding):
        async def __call__(self, prompt, **kwargs):
            await super().__call__([prompt])
            return prompt

    settings = {
        "LIGHTRAG_WORKSPACE_BASE": str(tmp_path),
        "GEMINI_API_KEY": "synthetic-key",
        "LIGHTRAG_STORAGE": "json",
        "LIGHTRAG_EMBEDDING_WORKERS": "1",
        "LIGHTRAG_LLM_WORKERS": "1",
    }
    monkeypatch.setattr(
        graph_memory_module,
        "get_setting",
        lambda name, default=None: settings.get(name, default),
    )
    monkeypatch.setattr(graph_memory_module, "GeminiEmbedding", NoNetworkEmbedding)
    monkeypatch.setattr(graph_memory_module, "GeminiLLM", NoNetworkModel)
    monkeypatch.setattr(
        graph_memory_module,
        "TiktokenTokenizer",
        lambda: Tokenizer("words", _WordTokenizer()),
    )
    memory = GraphMemory()
    async with memory._use_graph("trace-queue") as graph:
        rag = cast(Any, graph)
        try:
            yield rag
        finally:
            await rag.llm_model_func.shutdown()
            await rag.embedding_func.shutdown()
            await rag.finalize_storages()


async def _call(graph, callback: str, marker: str):
    if callback == "embedding":
        return await graph.chunks_vdb.embedding_func([marker])
    return await graph.llm_model_func(marker)


@pytest.mark.parametrize("callback", ["embedding", "llm"])
async def test_reused_queue_traces_each_job_and_freezes_returned_trace(
    queued_graph, callback
):
    with capture_lightrag_trace() as first:
        await _call(queued_graph, callback, "FIRST")
        result = RetrievalResult(context="first", trace=first.get_lines())
    snapshot = list(result.trace)
    with capture_lightrag_trace() as second:
        await _call(queued_graph, callback, "SECOND")
    assert any("callback FIRST" in line for line in result.trace)
    assert any("callback SECOND" in line for line in second.get_lines())
    assert not any("SECOND" in line for line in first.get_lines())
    assert result.trace == snapshot


@pytest.mark.parametrize("callback", ["embedding", "llm"])
async def test_queue_job_failure_belongs_to_awaiting_request(queued_graph, callback):
    first_ready = asyncio.Event()
    second_done = asyncio.Event()

    async def first_request():
        with capture_lightrag_trace() as trace:
            await _call(queued_graph, callback, "FIRST")
            first_ready.set()
            await second_done.wait()
        return trace.get_lines()

    async def second_request():
        await first_ready.wait()
        try:
            with capture_lightrag_trace() as trace:
                with pytest.raises(ValueError, match="SECOND_ERROR"):
                    await _call(queued_graph, callback, "SECOND_ERROR")
            return trace.get_lines()
        finally:
            second_done.set()

    first, second = await asyncio.gather(first_request(), second_request())
    assert not any("SECOND_ERROR" in line for line in first), first
    assert any("callback SECOND_ERROR" in line for line in second), second
    assert any(line.startswith("ERROR:") and "SECOND_ERROR" in line for line in second)


@pytest.mark.parametrize("timeout_kind", ["caller", "worker"])
async def test_real_queue_timeout_keeps_ownership_and_remains_reusable(timeout_kind):
    from lightrag.utils import priority_limit_async_func_call

    from rag.trace import trace_queued_callback, trace_worker_callback

    release = asyncio.Event()
    finished = asyncio.Event()

    @trace_worker_callback
    async def callback(marker):
        if marker == "TIMED_OUT":
            try:
                await release.wait()
                logging.getLogger("lightrag").info("late TIMED_OUT")
            finally:
                finished.set()
        logging.getLogger("lightrag").info("callback %s", marker)
        return marker

    limit = (
        priority_limit_async_func_call(max_size=1, max_execution_timeout=0.02)
        if timeout_kind == "worker"
        else priority_limit_async_func_call(max_size=1)
    )
    queued = limit(callback)
    call = cast(Any, trace_queued_callback(queued, "test queue"))
    try:
        with capture_lightrag_trace() as first:
            with pytest.raises(TimeoutError):
                await call(
                    "TIMED_OUT", _timeout=0.02 if timeout_kind == "caller" else None
                )
        snapshot = first.get_lines()
        assert any(line.startswith("ERROR:") for line in snapshot)
        release.set()
        await asyncio.wait_for(finished.wait(), 1)
        with capture_lightrag_trace() as second:
            assert await call("SECOND") == "SECOND"
        assert first.get_lines() == snapshot
        assert second.get_lines() == ["INFO: callback SECOND"]
    finally:
        release.set()
        await call.shutdown()


async def test_cancelled_await_does_not_capture_later_worker_or_next_job():
    from lightrag.utils import priority_limit_async_func_call

    from rag.trace import trace_queued_callback, trace_worker_callback

    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    traces = []

    @trace_worker_callback
    async def callback(marker):
        if marker == "CANCELLED":
            entered.set()
            await release.wait()
            logging.getLogger("lightrag").info("late CANCELLED")
            finished.set()
        logging.getLogger("lightrag").info("callback %s", marker)
        return marker

    call = cast(
        Any,
        trace_queued_callback(
            priority_limit_async_func_call(max_size=1)(callback), "queue"
        ),
    )

    async def request():
        with capture_lightrag_trace() as trace:
            traces.append(trace)
            await call("CANCELLED")

    task = asyncio.create_task(request())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        snapshot = traces[0].get_lines()
        with capture_lightrag_trace() as second:
            release.set()
            await asyncio.wait_for(finished.wait(), 1)
            assert await call("SECOND") == "SECOND"
        assert traces[0].get_lines() == snapshot
        assert second.get_lines() == ["INFO: callback SECOND"]
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await call.shutdown()


async def test_queue_capacity_priority_and_cancelled_pending_job_are_preserved():
    from lightrag.utils import QueueFullError, priority_limit_async_func_call

    from rag.trace import trace_queued_callback, trace_worker_callback

    entered = asyncio.Event()
    release = asyncio.Event()
    order = []

    @trace_worker_callback
    async def callback(marker):
        order.append(marker)
        if marker == "BLOCK":
            entered.set()
            await release.wait()
        return marker

    call = cast(
        Any,
        trace_queued_callback(
            priority_limit_async_func_call(max_size=1, max_queue_size=2)(callback),
            "queue",
        ),
    )
    tasks = []
    try:
        tasks.append(asyncio.create_task(call("BLOCK")))
        await asyncio.wait_for(entered.wait(), 1)
        tasks.append(asyncio.create_task(call("LOW", _priority=20)))
        tasks.append(asyncio.create_task(call("HIGH", _priority=1)))
        # Both enqueue operations reach their future await without yielding on
        # an unfilled queue. Only the already executing BLOCK holds the worker.
        await asyncio.sleep(0)
        with capture_lightrag_trace() as full:
            with pytest.raises(QueueFullError):
                await call("OVERFLOW", _queue_timeout=0.02)
        assert any("QueueFullError" in line for line in full.get_lines())
        release.set()
        assert await asyncio.gather(*tasks) == ["BLOCK", "LOW", "HIGH"]
        assert order == ["BLOCK", "HIGH", "LOW"]

        entered.clear()
        release.clear()
        tasks = [asyncio.create_task(call("BLOCK"))]
        await asyncio.wait_for(entered.wait(), 1)
        pending = asyncio.create_task(call("CANCEL_PENDING"))
        tasks.append(pending)
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        release.set()
        await tasks[0]
        assert await call("AFTER") == "AFTER"
        assert "CANCEL_PENDING" not in order
    finally:
        release.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await call.shutdown()
