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
