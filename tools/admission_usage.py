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
        parts = [row.get("usage") for row in rows if isinstance(row, dict)]
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


class UsageLedger:
    def __init__(self, path: Path, max_attempts: int = 64) -> None:
        self.path = path
        self.max_attempts = max_attempts
        self.lock = threading.Lock()
        self.sequence = 0
        self.reserved_deepseek_micro_cny = 0
        self.reserved_embedding_bytes = 0
        self.uncertain_failure = False

    def begin(self, request: httpx.Request) -> dict | None:
        host, endpoint = request.url.host, request.url.path
        if host == "api.deepseek.com" and endpoint.endswith("/chat/completions"):
            provider, operation = "deepseek", "generation"
        elif host == "api.giga.chat" and endpoint.endswith("/embeddings"):
            provider, operation = "gigachat", "embedding"
        elif host == "ngw.devices.sberbank.ru" and endpoint.endswith("/oauth"):
            provider, operation = "gigachat", "oauth"
        else:
            return None
        payload = {} if operation == "oauth" else json.loads(request.content)
        model = payload.get("model")
        if model is not None and not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,96}", model):
            raise IndexingLimit("Unexpected model identifier")
        if payload.get("stream"):
            raise IndexingLimit("This indexing meter supports non-streaming calls")
        if provider == "deepseek" and model != "deepseek-flash":
            raise IndexingLimit("Pilot price guard requires deepseek-flash")
        with self.lock:
            if self.uncertain_failure:
                raise IndexingLimit("Inspect the failed attempt before spending again")
            if self.sequence >= self.max_attempts:
                raise IndexingLimit("Pilot HTTP attempt limit reached")
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
            return {
                "attempt": self.sequence,
                "provider": provider,
                "operation": operation,
                "request_model": model,
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
        if response is not None and row["operation"] != "oauth":
            try:
                body = response.json()
                row["usage"] = usage_counts(body, row["provider"])
                model = body.get("model")
                if isinstance(model, str) and re.fullmatch(
                    r"[a-zA-Z0-9_.:-]{1,96}", model
                ):
                    row["response_model"] = model
            except (ValueError, TypeError, AttributeError):
                pass
        with self.lock:
            failed = error is not None or response is None or not response.is_success
            if failed or (row["operation"] != "oauth" and not row["usage"]):
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
            if row is None:
                return original_sync(client, request, **kwargs)
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
            row = ledger.begin(request)
            if row is None:
                return await original_async(client, request, **kwargs)
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
