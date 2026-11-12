import json

import httpx
import pytest

from tools.admission_usage import IndexingLimit, UsageLedger, usage_counts


def test_sync_embedding_usage_does_not_disclose_text_headers_or_vectors(tmp_path):
    path = tmp_path / "usage.jsonl"
    ledger = UsageLedger(path)
    response = {"data": [{"embedding": [0.42], "usage": {"prompt_tokens": 11}}]}
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=response))
    with ledger.capture(), httpx.Client(transport=transport) as client:
        client.post(
            "https://api.giga.chat/v1/embeddings",
            headers={"Authorization": "Bearer never-store-this"},
            json={"model": "EmbeddingsGigaR", "input": ["private-input-text"]},
        )
    result = path.read_text()
    assert "never-store-this" not in result and "private-input-text" not in result
    assert '"embedding":' not in result and "0.42" not in result
    assert json.loads(result)["usage"] == {"prompt_tokens": 11, "total_tokens": 11}


@pytest.mark.asyncio
async def test_async_deepseek_preserves_cache_usage(tmp_path):
    path = tmp_path / "usage.jsonl"
    ledger = UsageLedger(path)
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "prompt_cache_hit_tokens": 80,
        "prompt_cache_miss_tokens": 20,
    }
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"model": "deepseek-flash", "usage": usage})
    )
    with ledger.capture():
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post(
                "https://api.deepseek.com/chat/completions",
                json={"model": "deepseek-flash", "max_tokens": 128, "messages": []},
            )
    assert json.loads(path.read_text())["usage"] == usage


def test_error_is_recorded_and_next_network_attempt_is_blocked(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    received = []

    def reply(request):
        received.append(request)
        return httpx.Response(429, json={"error": "do-not-store-upstream-body"})

    with ledger.capture(), httpx.Client(transport=httpx.MockTransport(reply)) as client:
        arguments = {"json": {"model": "deepseek-flash", "max_tokens": 128}}
        client.post("https://api.deepseek.com/chat/completions", **arguments)
        with pytest.raises(IndexingLimit):
            client.post("https://api.deepseek.com/chat/completions", **arguments)
    assert len(received) == 1
    row = json.loads(ledger.path.read_text())
    assert row["http_status"] == 429 and row["usage"] == {}
    assert "do-not-store-upstream-body" not in ledger.path.read_text()


def test_missing_usage_is_unknown_and_blocks_following_spending(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"data": []}))
    with ledger.capture(), httpx.Client(transport=transport) as client:
        arguments = {"json": {"model": "EmbeddingsGigaR", "input": ["example"]}}
        client.post("https://api.giga.chat/v1/embeddings", **arguments)
        with pytest.raises(IndexingLimit):
            client.post("https://api.giga.chat/v1/embeddings", **arguments)


def test_does_not_double_count_top_level_and_item_usage():
    assert usage_counts(
        {"usage": {"prompt_tokens": 10}, "data": [{"usage": {"prompt_tokens": 10}}]},
        "gigachat",
    ) == {"prompt_tokens": 10}


def test_integral_gigachat_number_is_accepted_without_rounding_fraction():
    assert usage_counts({"data": [{"usage": {"prompt_tokens": 10.0}}]}, "gigachat") == {
        "prompt_tokens": 10,
        "total_tokens": 10,
    }
    assert usage_counts({"data": [{"usage": {"prompt_tokens": 10.5}}]}, "gigachat") == {}


def test_limit_refuses_before_request_is_sent(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.jsonl", max_attempts=0)
    received = []

    def reply(request):
        received.append(request)
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(reply)
    with ledger.capture(), httpx.Client(transport=transport) as client:
        with pytest.raises(IndexingLimit):
            client.post(
                "https://api.deepseek.com/chat/completions",
                json={"model": "deepseek-flash", "max_tokens": 128},
            )
    assert not received and not ledger.path.exists()


@pytest.mark.parametrize(
    "url",
    [
        "https://unapproved.invalid/chat/completions",
        "http://api.deepseek.com/chat/completions",
        "https://api.deepseek.com:444/chat/completions",
        "https://api.deepseek.com/other/chat/completions",
        "https://user:secret@api.deepseek.com/chat/completions",
        "http://eval388/api/v1/auth/login",
    ],
)
def test_unapproved_destination_never_reaches_transport(tmp_path, url):
    received = []

    def reply(request):
        received.append(request)
        return httpx.Response(200, json={})

    ledger = UsageLedger(tmp_path / "usage.jsonl")
    with ledger.capture(), httpx.Client(transport=httpx.MockTransport(reply)) as client:
        with pytest.raises(IndexingLimit):
            client.post(url, json={"model": "deepseek-flash", "max_tokens": 128})
    assert not received


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 17},
        {"prompt_tokens": 17, "completion_tokens": 3, "total_tokens": 21},
        {
            "prompt_tokens": 17,
            "completion_tokens": 3,
            "total_tokens": 20,
            "prompt_cache_hit_tokens": 30,
            "prompt_cache_miss_tokens": 0,
        },
    ],
)
def test_incomplete_or_inconsistent_usage_stops_spending(tmp_path, usage):
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"usage": usage})
    )
    with ledger.capture(), httpx.Client(transport=transport) as client:
        client.post(
            "https://api.deepseek.com/chat/completions",
            json={"model": "deepseek-flash", "max_tokens": 128},
        )
        assert ledger.uncertain_failure
        with pytest.raises(IndexingLimit):
            client.post(
                "https://api.deepseek.com/chat/completions",
                json={"model": "deepseek-flash", "max_tokens": 128},
            )


def test_provider_redirect_is_not_followed(tmp_path):
    received = []

    def reply(request):
        received.append(request.url.host)
        return httpx.Response(302, headers={"Location": "https://unapproved.invalid/"})

    ledger = UsageLedger(tmp_path / "usage.jsonl")
    with (
        ledger.capture(),
        httpx.Client(
            transport=httpx.MockTransport(reply), follow_redirects=True
        ) as client,
    ):
        response = client.post(
            "https://api.deepseek.com/chat/completions",
            json={"model": "deepseek-flash", "max_tokens": 128},
        )
    assert response.status_code == 302
    assert received == ["api.deepseek.com"]


@pytest.mark.asyncio
async def test_local_api_is_allowed_only_with_asgi_transport(tmp_path):
    async def application(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    ledger = UsageLedger(tmp_path / "usage.jsonl")
    with ledger.capture():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application)
        ) as client:
            response = await client.post("http://eval388/api/v1/auth/login", json={})
    assert response.status_code == 200 and ledger.sequence == 0


@pytest.mark.asyncio
async def test_asgi_transport_does_not_allow_unapproved_route(tmp_path):
    received = []

    async def application(scope, receive, send):
        received.append(scope)

    ledger = UsageLedger(tmp_path / "usage.jsonl")
    with ledger.capture():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application)
        ) as client:
            with pytest.raises(IndexingLimit):
                await client.post("http://eval388/api/v1/other-action", json={})
    assert not received


def test_incomplete_embedding_batch_cannot_appear_fully_accounted(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"data": [{"usage": {"prompt_tokens": 10}}]})
    )
    with ledger.capture(), httpx.Client(transport=transport) as client:
        client.post(
            "https://api.giga.chat/v1/embeddings",
            json={"model": "EmbeddingsGigaR", "input": ["first", "second"]},
        )
    assert ledger.uncertain_failure


def test_embedding_count_does_not_exhaust_generation_budget(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    for _ in range(65):
        ledger.begin(
            httpx.Request(
                "POST",
                "https://api.giga.chat/v1/embeddings",
                json={"model": "EmbeddingsGigaR", "input": ["text"]},
            )
        )
    assert ledger.operation_counts["embedding"] == 65
    assert ledger.operation_counts.get("generation", 0) == 0


def test_generation_remains_bounded_separately(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    request = httpx.Request(
        "POST",
        "https://api.deepseek.com/chat/completions",
        json={"model": "deepseek-flash", "max_tokens": 128},
    )
    for _ in range(32):
        ledger.begin(request)
    with pytest.raises(IndexingLimit):
        ledger.begin(request)
