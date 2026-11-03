import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from llm.providers.gigachat import GigaChatEmbedding, GigaChatError


def adapter(handler, **kwargs):
    return GigaChatEmbedding(
        auth_key="test-auth-key",
        model="EmbeddingsGigaR",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def vector(index=0):
    return {"index": index, "embedding": [0.1] * 2560}


def test_oauth_is_cached_and_vectors_follow_input_order():
    oauth = []

    def handler(request):
        if request.url.path.endswith("/oauth"):
            oauth.append(request)
            assert request.headers["Authorization"] == "Basic test-auth-key"
            assert b"scope=GIGACHAT_API_PERS" == request.content
            return httpx.Response(
                200,
                json={
                    "access_token": "token",
                    "expires_at": int((time.time() + 1800) * 1000),
                },
            )
        assert request.headers["Authorization"] == "Bearer token"
        return httpx.Response(200, json={"data": [vector(1), vector(0)]})

    embedder = adapter(handler)
    assert len(embedder.embed_documents(["a", "b"])) == 2
    assert len(embedder.embed_documents(["c", "d"])) == 2
    assert len(oauth) == 1


def test_concurrent_requests_refresh_token_once():
    calls = []

    def handler(request):
        if request.url.path.endswith("/oauth"):
            calls.append(True)
            time.sleep(0.02)
            return httpx.Response(
                200, json={"access_token": "concurrent", "expires_at": time.time() + 1800}
            )
        return httpx.Response(200, json={"data": [vector()]})

    embedder = adapter(handler)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(embedder.embed_query, ["a", "b", "c", "d"]))
    assert len(calls) == 1


@pytest.mark.parametrize(
    "data",
    [
        [],
        [vector(1)],
        [vector(), vector()],
        [{"index": 0, "embedding": [0.1]}],
        [{"index": 0, "embedding": [0] * 2560}],
        [{"index": 0, "embedding": [float("nan")] * 2560}],
    ],
)
def test_invalid_vectors_fail_closed(data):
    def handler(request):
        if request.url.path.endswith("/oauth"):
            return httpx.Response(
                200, json={"access_token": "token", "expires_at": time.time() + 1800}
            )
        # Serialize NaN explicitly; upstream JSON parsers may accept it.
        import json

        return httpx.Response(200, text=json.dumps({"data": data}))

    with pytest.raises(GigaChatError):
        adapter(handler).embed_documents(["test"])


def test_unauthorized_refreshes_once_and_sanitizes_errors():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path.endswith("/oauth"):
            return httpx.Response(
                200,
                json={"access_token": "secret-token", "expires_at": time.time() + 1800},
            )
        return httpx.Response(401, text="test-auth-key secret-token")

    with pytest.raises(GigaChatError) as caught:
        adapter(handler).embed_query("test")
    assert caught.value.status_code == 401
    assert "test-auth-key" not in str(caught.value)
    assert "secret-token" not in str(caught.value)
    assert len(calls) == 4


async def test_async_adapter_and_empty_input():
    def handler(request):
        if request.url.path.endswith("/oauth"):
            return httpx.Response(
                200, json={"access_token": "token", "expires_at": time.time() + 1800}
            )
        return httpx.Response(200, json={"data": [vector()]})

    embedder = adapter(handler)
    assert await embedder([]) == []
    assert len((await embedder(["test"]))[0]) == 2560
    await embedder.aclose()


def test_separate_adapters_serialize_the_same_credential():
    import threading

    active = 0
    peak = 0
    guard = threading.Lock()

    def handler(request):
        nonlocal active, peak
        if request.url.path.endswith("/oauth"):
            return httpx.Response(
                200, json={"access_token": "token", "expires_at": time.time() + 1800}
            )
        with guard:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.03)
            return httpx.Response(200, json={"data": [vector()]})
        finally:
            with guard:
                active -= 1

    def run(_):
        return adapter(handler).embed_query("question")

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(run, range(4)))
    assert peak == 1


@pytest.mark.parametrize("status", [429, 503])
def test_transient_embedding_failure_is_retried(status, monkeypatch):
    calls = 0
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)

    def handler(request):
        nonlocal calls
        if request.url.path.endswith("/oauth"):
            return httpx.Response(
                200, json={"access_token": "token", "expires_at": time.time() + 1800}
            )
        calls += 1
        if calls <= 2:
            return httpx.Response(
                status, headers={"Retry-After": "1"}, text="secret upstream body"
            )
        return httpx.Response(200, json={"data": [vector()]})

    assert len(adapter(handler).embed_query("question")) == 2560
    assert calls == 3
    assert len(sleeps) == 2


def test_error_identifies_status_without_upstream_body(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)

    def handler(request):
        if request.url.path.endswith("/oauth"):
            return httpx.Response(
                200, json={"access_token": "token", "expires_at": time.time() + 1800}
            )
        return httpx.Response(429, text="test-auth-key secret-token document text")

    with pytest.raises(GigaChatError) as caught:
        adapter(handler).embed_query("question")
    assert "HTTP 429" in str(caught.value)
    assert "test-auth-key" not in str(caught.value)
    assert "secret-token" not in str(caught.value)
    assert "document text" not in str(caught.value)


def _embed_in_process(directory, barrier, active, peak):
    import os

    os.environ["GIGACHAT_LOCK_DIRECTORY"] = directory

    def handler(request):
        if request.url.path.endswith("/oauth"):
            return httpx.Response(
                200, json={"access_token": "token", "expires_at": time.time() + 1800}
            )
        with active.get_lock():
            active.value += 1
            with peak.get_lock():
                peak.value = max(peak.value, active.value)
        time.sleep(0.05)
        with active.get_lock():
            active.value -= 1
        return httpx.Response(200, json={"data": [vector()]})

    embedder = adapter(handler)
    barrier.wait(timeout=10)
    embedder.embed_query("question")


def test_shared_directory_serializes_independent_processes(tmp_path):
    import multiprocessing
    import os

    if os.name != "posix":
        pytest.skip("Production shared lock uses POSIX flock")
    context = multiprocessing.get_context("spawn")
    active = context.Value("i", 0)
    peak = context.Value("i", 0)
    barrier = context.Barrier(3)
    workers = [
        context.Process(
            target=_embed_in_process, args=(str(tmp_path), barrier, active, peak)
        )
        for _ in range(3)
    ]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=20)
            assert worker.exitcode == 0
        assert peak.value == 1
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)


@pytest.mark.parametrize(
    "statuses", [[503, 503, 503, 401, 200], [401, 503, 503, 503, 200]]
)
def test_auth_refresh_does_not_consume_transient_retry_budget(statuses, monkeypatch):
    sequence = iter(statuses)
    oauth = []
    embeddings = []
    monkeypatch.setattr(time, "sleep", lambda _: None)

    def handler(request):
        if request.url.path.endswith("/oauth"):
            oauth.append(True)
            return httpx.Response(
                200,
                json={
                    "access_token": f"token-{len(oauth)}",
                    "expires_at": time.time() + 1800,
                },
            )
        status = next(sequence)
        embeddings.append(status)
        return httpx.Response(status, json={"data": [vector()]} if status == 200 else {})

    assert len(adapter(handler).embed_query("question")) == 2560
    assert len(oauth) == 2
    assert embeddings == statuses
