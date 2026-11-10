import json

import httpx
import pytest

from tools.admission_evaluation_usage import EvaluationUsageLedger
from tools.admission_usage import IndexingLimit


class Chunks(httpx.AsyncByteStream):
    def __init__(self, body: bytes):
        self.body = body

    async def __aiter__(self):
        for offset in range(0, len(self.body), 7):
            yield self.body[offset : offset + 7]


def stream_body(usage=True, done=True):
    body = b'data: {"choices":[{"delta":{"content":"private-answer"}}]}\r\n\r\n'
    if usage:
        body += (
            b'data: {"model":"deepseek-flash","usage":{"prompt_tokens":100,'
            b'"completion_tokens":20,"total_tokens":120,"prompt_cache_hit_tokens":80,'
            b'"prompt_cache_miss_tokens":20}}\n\n'
        )
    if done:
        body += b"data: [DONE]\n\n"
    return body


@pytest.mark.asyncio
@pytest.mark.parametrize("usage,done", [(True, True), (False, True), (True, False)])
async def test_stream_preserved_and_incomplete_usage_blocks_spending(
    tmp_path, usage, done
):
    body = stream_body(usage, done)
    received = []

    def reply(request):
        received.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream; charset=utf-8"},
            stream=Chunks(body),
        )

    ledger = EvaluationUsageLedger(tmp_path / "usage.jsonl")
    payload = {
        "model": "deepseek-flash",
        "max_tokens": 1500,
        "messages": [{"role": "user", "content": "private-question"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    with ledger.capture():
        async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as c:
            async with c.stream(
                "POST", "https://api.deepseek.com/chat/completions", json=payload
            ) as response:
                assert b"".join([chunk async for chunk in response.aiter_bytes()]) == body
            if not (usage and done):
                with pytest.raises(IndexingLimit):
                    await c.post(
                        "https://api.deepseek.com/chat/completions", json=payload
                    )

    assert received == [payload]
    saved = ledger.path.read_text()
    assert "private-answer" not in saved and "private-question" not in saved
    row = json.loads(saved)
    assert row["usage_complete"] is usage
    assert row["stream_evidence"]["done_seen"] is done
    assert row["stream_evidence"]["usage_seen"] is usage
    assert ledger.uncertain_failure is not (usage and done)
    if usage:
        assert row["usage"]["prompt_cache_hit_tokens"] == 80


@pytest.mark.asyncio
async def test_usage_required_before_network_and_unconsumed_stream_is_unknown(tmp_path):
    calls = []

    def reply(request):
        calls.append(1)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Chunks(stream_body()),
        )

    ledger = EvaluationUsageLedger(tmp_path / "usage.jsonl")
    payload = {"model": "deepseek-flash", "max_tokens": 100, "stream": True}
    with ledger.capture():
        async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as c:
            with pytest.raises(IndexingLimit):
                await c.post("https://api.deepseek.com/chat/completions", json=payload)
            assert not calls
            async with c.stream(
                "POST",
                "https://api.deepseek.com/chat/completions",
                json={**payload, "stream_options": {"include_usage": True}},
            ):
                pass

    assert calls == [1]
    assert ledger.uncertain_failure
    assert (
        json.loads(ledger.path.read_text())["error_type"] == "IncompleteEvaluationStream"
    )


@pytest.mark.asyncio
async def test_real_sdk_consumes_stream_and_records_usage_on_done(tmp_path):
    from openai import AsyncOpenAI

    frames = [
        {
            "id": "test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "deepseek-flash",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "private-answer"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "deepseek-flash",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "deepseek-flash",
            "choices": [],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        },
    ]
    body = b"".join(b"data: " + json.dumps(f).encode() + b"\n\n" for f in frames)
    body += b"data: [DONE]\n\n"
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Chunks(body)
        )
    )
    ledger = EvaluationUsageLedger(tmp_path / "usage.jsonl")
    with ledger.capture():
        async with AsyncOpenAI(
            api_key="synthetic-test-only",
            base_url="https://api.deepseek.com",
            http_client=httpx.AsyncClient(transport=transport),
        ) as client:
            stream = await client.chat.completions.create(
                model="deepseek-flash",
                messages=[{"role": "user", "content": "test"}],
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=128,
            )
            text = []
            async with stream:
                async for frame in stream:
                    if frame.choices:
                        text.append(frame.choices[0].delta.content or "")
    assert "".join(text) == "private-answer"
    assert not ledger.uncertain_failure
    assert json.loads(ledger.path.read_text())["usage_complete"]
    assert "private-answer" not in ledger.path.read_text()


def test_embedding_oauth_and_destination_guards_remain_available(tmp_path):
    ledger = EvaluationUsageLedger(tmp_path / "usage.jsonl")
    seen = []

    def reply(request):
        seen.append(request.url.host)
        if request.url.host == "api.giga.chat":
            return httpx.Response(200, json={"data": [{"usage": {"prompt_tokens": 5}}]})
        return httpx.Response(200, json={"access_token": "synthetic-never-save"})

    with ledger.capture(), httpx.Client(transport=httpx.MockTransport(reply)) as c:
        c.post(
            "https://ngw.devices.sberbank.ru:9443/api/v2/oauth", data={"scope": "test"}
        )
        c.post(
            "https://api.giga.chat/v1/embeddings",
            json={"model": "EmbeddingsGigaR", "input": ["private-input"]},
        )
        with pytest.raises(IndexingLimit):
            c.post("https://unapproved.invalid/chat/completions", json={})
    assert seen == ["ngw.devices.sberbank.ru", "api.giga.chat"]
    saved = ledger.path.read_text()
    assert "synthetic-never-save" not in saved and "private-input" not in saved
    assert all(r["usage_complete"] for r in map(json.loads, saved.splitlines()))


@pytest.mark.asyncio
async def test_unexpected_unread_stream_is_recorded_before_failing(tmp_path):
    from tools.admission_evaluation_usage import IncompleteEvaluationStream

    ledger = EvaluationUsageLedger(tmp_path / "usage.jsonl")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "application/octet-stream"},
            stream=Chunks(b"private"),
        )
    )
    with ledger.capture():
        async with httpx.AsyncClient(transport=transport) as c:
            with pytest.raises(IncompleteEvaluationStream):
                async with c.stream(
                    "POST",
                    "https://api.deepseek.com/chat/completions",
                    json={
                        "model": "deepseek-flash",
                        "max_tokens": 100,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    },
                ):
                    pass
    assert ledger.uncertain_failure
    saved = ledger.path.read_text()
    assert "private" not in saved
    assert not json.loads(saved)["usage_complete"]


@pytest.mark.asyncio
async def test_provider_error_frame_records_only_fixed_flags(tmp_path):
    from tools.admission_evaluation_usage import IncompleteEvaluationStream

    body = b'data: {"error":{"message":"private-provider-message"}}\n\n'
    ledger = EvaluationUsageLedger(tmp_path / "usage.jsonl")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Chunks(body)
        )
    )
    with ledger.capture():
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(IncompleteEvaluationStream):
                async with client.stream(
                    "POST",
                    "https://api.deepseek.com/chat/completions",
                    json={
                        "model": "deepseek-flash",
                        "max_tokens": 100,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    },
                ) as response:
                    await response.aread()
    saved = ledger.path.read_text()
    assert "private-provider-message" not in saved
    row = json.loads(saved)
    assert row["stream_evidence"]["provider_error_seen"]
    assert row["stream_evidence"]["json_events_seen"] == 1
    assert not row["usage_complete"]
    assert ledger.uncertain_failure
