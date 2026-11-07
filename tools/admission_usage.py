"""Measure non-streaming indexing HTTP attempts without persisting request bodies."""

import json
import re
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Generator
from unittest.mock import patch

import httpx


class IndexingLimit(RuntimeError):
    """A local pilot limit was reached; no additional request was sent."""


def integer(value: Any) -> int | None:
    if type(value) is float and value.is_integer() and value >= 0:
        return int(value)
    return value if type(value) is int and value >= 0 else None


def usage_counts(data: dict, provider: str) -> dict:
    """Keep only numeric usage; GigaChat may report usage separately per input."""
    raw = data.get("usage")
    if not isinstance(raw, dict) and provider == "gigachat":
        rows = data.get("data", [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            return {}
        parts = [row.get("usage") for row in rows]
        if parts and all(isinstance(part, dict) for part in parts):
            values = [
                integer(part.get("prompt_tokens"))
                for part in parts
                if isinstance(part, dict)
            ]
            if all(value is not None for value in values):
                total = sum(value for value in values if value is not None)
                return {"prompt_tokens": total, "total_tokens": total}
    if not isinstance(raw, dict):
        return {}
    result = {}
    for name in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    ):
        value = integer(raw.get(name))
        if value is not None:
            result[name] = value
    return result


def complete_usage(usage: dict, provider: str) -> bool:
    """Require a complete token count, not merely a nonempty usage object."""
    prompt = usage.get("prompt_tokens")
    if prompt is None:
        return False
    if provider == "gigachat":
        return (
            usage.get("total_tokens", prompt) == prompt
            and usage.get("completion_tokens", 0) == 0
        )
    output, total = usage.get("completion_tokens"), usage.get("total_tokens")
    if output is None or total != prompt + output:
        return False
    hit, miss = (
        usage.get("prompt_cache_hit_tokens"),
        usage.get("prompt_cache_miss_tokens"),
    )
    return (hit is None and miss is None) or (
        hit is not None and miss is not None and hit + miss == prompt
    )


def local_api_call(client: httpx.AsyncClient, request: httpx.Request) -> bool:
    """The only unmetered calls are these in-process authenticated API routes."""
    return (
        isinstance(client._transport, httpx.ASGITransport)
        and (
            request.method == "POST"
            or (
                request.method == "GET"
                and request.url.path == "/api/v1/content-proposals"
            )
        )
        and request.url.scheme == "http"
        and request.url.host == "eval388"
        and request.url.port in (None, 80)
        and not request.url.userinfo
        and not request.url.query
        and bool(
            re.fullmatch(
                r"/api/v1/(auth/login|content-proposals|"
                r"content-proposals/[0-9a-f-]{36}/approve)",
                request.url.path,
            )
        )
    )


class UsageLedger:
    def __init__(self, path: Path, max_attempts: int = 292) -> None:
        self.path = path
        self.max_attempts = max_attempts
        self.lock = threading.Lock()
        self.sequence = 0
        self.operation_counts: dict[str, int] = {}
        self.reserved_deepseek_micro_cny = 0
        self.reserved_embedding_bytes = 0
        self.uncertain_failure = False

    def begin(self, request: httpx.Request) -> dict:
        host, endpoint = request.url.host, request.url.path
        if (
            request.method != "POST"
            or request.url.scheme != "https"
            or request.url.userinfo
            or request.url.query
        ):
            raise IndexingLimit("Unapproved provider request")
        if (
            host == "api.deepseek.com"
            and endpoint in ("/chat/completions", "/v1/chat/completions")
            and request.url.port in (None, 443)
        ):
            provider, operation = "deepseek", "generation"
        elif (
            host == "api.giga.chat"
            and endpoint == "/v1/embeddings"
            and request.url.port in (None, 443)
        ):
            provider, operation = "gigachat", "embedding"
        elif (
            host == "ngw.devices.sberbank.ru"
            and endpoint == "/api/v2/oauth"
            and request.url.port == 9443
        ):
            provider, operation = "gigachat", "oauth"
        else:
            raise IndexingLimit("Unapproved provider destination")
        payload = {} if operation == "oauth" else json.loads(request.content)
        model = payload.get("model")
        if model is not None and not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,96}", model):
            raise IndexingLimit("Unexpected model identifier")
        if payload.get("stream"):
            raise IndexingLimit("This indexing meter supports non-streaming calls")
        if provider == "deepseek" and model != "deepseek-flash":
            raise IndexingLimit("Pilot price guard requires deepseek-flash")
        if operation == "embedding" and (
            model != "EmbeddingsGigaR"
            or not isinstance(payload.get("input"), list)
            or not payload["input"]
        ):
            raise IndexingLimit("Pilot requires a nonempty EmbeddingsGigaR batch")
        with self.lock:
            if self.uncertain_failure:
                raise IndexingLimit("Inspect the failed attempt before spending again")
            if self.sequence >= self.max_attempts:
                raise IndexingLimit("Pilot HTTP attempt limit reached")
            limits = {"generation": 32, "embedding": 256, "oauth": 4}
            if self.operation_counts.get(operation, 0) >= limits[operation]:
                raise IndexingLimit("Pilot operation attempt limit reached")
            if provider == "deepseek":
                maximum = integer(payload.get("max_tokens"))
                if maximum is None or maximum > 8192:
                    raise IndexingLimit("Explicit output limit up to 8192 is required")
                # Conservative Flash peak rate, cache treated as a miss. JSON bytes
                # plus framing are a planning bound, not a provider token count.
                reserve = (len(request.content) + 512) * 2 + maximum * 8
                if self.reserved_deepseek_micro_cny + reserve > 5_000_000:
                    raise IndexingLimit("Pilot planning reserve exceeds 5 CNY")
                self.reserved_deepseek_micro_cny += reserve
            if operation == "embedding":
                size = len(request.content)
                if self.reserved_embedding_bytes + size > 200_000:
                    raise IndexingLimit("Pilot embedding input byte limit reached")
                self.reserved_embedding_bytes += size
            self.sequence += 1
            self.operation_counts[operation] = self.operation_counts.get(operation, 0) + 1
            return {
                "attempt": self.sequence,
                "provider": provider,
                "operation": operation,
                "request_model": model,
                "input_count": len(payload["input"])
                if operation == "embedding"
                else None,
                "started_at": datetime.now(UTC).isoformat(),
                "started_monotonic": time.monotonic(),
            }

    def finish(
        self, row: dict, response: httpx.Response | None, error: BaseException | None
    ) -> None:
        row["elapsed_seconds"] = round(time.monotonic() - row.pop("started_monotonic"), 4)
        row["http_status"] = response.status_code if response is not None else None
        row["error_type"] = type(error).__name__ if error is not None else None
        row["usage"] = {}
        row["usage_complete"] = row["operation"] == "oauth"
        if response is not None and row["operation"] != "oauth":
            try:
                body = response.json()
                row["usage"] = usage_counts(body, row["provider"])
                row["usage_complete"] = complete_usage(row["usage"], row["provider"])
                raw = body.get("usage")
                if isinstance(raw, dict) and any(
                    name in raw and integer(raw[name]) is None
                    for name in (
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "prompt_cache_hit_tokens",
                        "prompt_cache_miss_tokens",
                    )
                ):
                    row["usage_complete"] = False
                if row["operation"] == "embedding":
                    rows = body.get("data")
                    row["usage_complete"] = row["usage_complete"] and (
                        isinstance(rows, list) and len(rows) == row["input_count"]
                    )
                model = body.get("model")
                if isinstance(model, str) and re.fullmatch(
                    r"[a-zA-Z0-9_.:-]{1,96}", model
                ):
                    row["response_model"] = model
            except (ValueError, TypeError, AttributeError):
                pass
        with self.lock:
            failed = error is not None or response is None or not response.is_success
            if failed or not row["usage_complete"]:
                self.uncertain_failure = True
            with self.path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()

    @contextmanager
    def capture(self) -> Generator[None, None, None]:
        original_sync = httpx.Client.send
        original_async = httpx.AsyncClient.send
        ledger = self

        def send(client, request, **kwargs):
            row = ledger.begin(request)
            kwargs["follow_redirects"] = False
            response = None
            try:
                response = original_sync(client, request, **kwargs)
                response.read()
            except BaseException as exc:
                ledger.finish(row, response, exc)
                raise
            ledger.finish(row, response, None)
            return response

        async def async_send(client, request, **kwargs):
            if local_api_call(client, request):
                kwargs["follow_redirects"] = False
                return await original_async(client, request, **kwargs)
            row = ledger.begin(request)
            kwargs["follow_redirects"] = False
            response = None
            try:
                response = await original_async(client, request, **kwargs)
                await response.aread()
            except BaseException as exc:
                ledger.finish(row, response, exc)
                raise
            ledger.finish(row, response, None)
            return response

        with (
            patch.object(httpx.Client, "send", send),
            patch.object(httpx.AsyncClient, "send", async_send),
        ):
            yield
