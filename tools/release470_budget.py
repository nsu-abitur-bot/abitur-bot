"""Durable pilot accounting; installing hooks does not authorize spending."""

import contextvars
import hashlib
import json
import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch
from urllib.parse import urlsplit

import httpx

from tools.admission_evaluation_usage import _MeteredStream
from tools.admission_usage import IndexingLimit, complete_usage, usage_counts
from tools.release470_cloudru import (
    HOST as CLOUD_HOST,
)
from tools.release470_cloudru import (
    MODEL as CLOUD_MODEL,
)
from tools.release470_cloudru import (
    CloudCounter,
    cost_micro_rub,
    final_usage,
    validate_request,
)
from tools.release470_gate import TransportGate
from tools.release470_store import BudgetStore
from tools.release470_tokens import OfficialCounter

CASES = {"setup", "faq-1", "faq-2", "rag-1", "rag-2", "tools-1", "tools-2"}
MODELS = {"deepseek-flash", "DeepSeek-V4.1-Flash-1017"}


class Release470Ledger:
    """Every process attaches the same explicitly initialized local budget file."""

    def __init__(
        self,
        directory: Path,
        *,
        budget_path: Path,
        counter=None,
        embedding_counter=None,
        cloud_tokenizer_path: Path | None = None,
    ) -> None:
        self.store = BudgetStore(budget_path)
        self.directory = directory  # Report directories cannot create/reset budgets.
        if (
            counter is not None or embedding_counter is not None
        ) and not self.store.policy["offline_only"]:
            raise IndexingLimit("Injected counters are only for offline tests")
        self.counter = counter
        self.embedding_counter = embedding_counter
        self.cloud_tokenizer_path = cloud_tokenizer_path or Path(
            "/evaluation/cloudru-tokenizer.json"
        )
        self._case = contextvars.ContextVar("release470_case", default="unassigned")

    def context(self, phase: str, case_id: str) -> None:
        if phase not in {"setup", "pilot"} or case_id not in CASES:
            raise IndexingLimit("Unknown pilot context")
        self._case.set(case_id)

    @contextmanager
    def question(self, case_id: str):
        if case_id not in CASES:
            raise IndexingLimit("Unknown question")
        token = self._case.set(case_id)
        try:
            yield
        finally:
            self._case.reset(token)

    def output_limit(self, requested: int) -> int:
        if self._case.get() not in CASES or type(requested) is not int:
            raise IndexingLimit("Explicit bounded context/output is required")
        return self.store.output_limit(self._case.get(), requested)

    def begin(self, request: httpx.Request) -> dict:
        if self._case.get() not in CASES:
            raise IndexingLimit("Unlabelled background or provider call refused")
        if (
            request.method != "POST"
            or request.url.scheme != "https"
            or request.url.userinfo
            or request.url.query
            or request.url.port not in (None, 443)
        ):
            raise IndexingLimit("Unapproved provider request")
        if request.url.host == CLOUD_HOST and request.url.path == "/v1/embeddings":
            return self._begin_cloud(request)
        # The old GigaChat package is never an implicit Cloud.ru permission.
        if request.url.host != "api.deepseek.com" or request.url.path not in {
            "/chat/completions",
            "/v1/chat/completions",
        }:
            raise IndexingLimit("Provider outside this initialized CNY policy")
        try:
            payload = json.loads(request.content)
        except (ValueError, TypeError):
            raise IndexingLimit("Invalid provider JSON") from None
        if (
            not isinstance(payload, dict)
            or payload.get("model") != "deepseek-flash"
            or payload.get("thinking") != {"type": "disabled"}
            or type(payload.get("max_tokens")) is not int
            or not 1 <= payload["max_tokens"] <= 1024
        ):
            raise IndexingLimit("Unsupported bounded model contract")
        streaming = payload.get("stream", False)
        if type(streaming) is not bool or (
            streaming and payload.get("stream_options") != {"include_usage": True}
        ):
            raise IndexingLimit("Streaming requires explicit final usage")
        if self.counter is None:
            self.counter = OfficialCounter()
        prompt = self.counter(payload)
        if type(prompt) is not int or prompt < 1:
            raise IndexingLimit("No verified prompt count")
        held = {
            "input": prompt,
            "output": payload["max_tokens"],
            "embedding": 0,
            "micro_cny": prompt * 2 + payload["max_tokens"] * 8,
        }
        fingerprint = hashlib.sha256(request.content).hexdigest()
        attempt = self.store.reserve(
            self._case.get(), "generation", held, fingerprint, self.counter.evidence
        )
        return {
            "attempt": attempt,
            "fingerprint": fingerprint,
            "case_id": self._case.get(),
            "streaming": streaming,
            "started_monotonic": time.monotonic(),
            "provider": "deepseek",
            "operation": "generation",
        }

    def _begin_cloud(self, request: httpx.Request) -> dict:
        settings = self.store.policy["cloudru"]
        if not settings["approved"]:
            raise IndexingLimit("Cloud.ru budget is not explicitly approved")
        try:
            payload = json.loads(request.content)
        except ValueError:
            raise IndexingLimit("Invalid Cloud.ru JSON") from None
        validate_request(payload)
        if self.embedding_counter is None:
            self.embedding_counter = CloudCounter(self.cloud_tokenizer_path, settings)
        tokens = self.embedding_counter(payload)
        if type(tokens) is not int or tokens < 1:
            raise IndexingLimit("Unknown Cloud.ru token bound")
        held = {
            "input": 0,
            "output": 0,
            "embedding": tokens,
            "micro_cny": 0,
            "micro_rub": cost_micro_rub(tokens, settings["rate_hundredths"]),
        }
        fingerprint = hashlib.sha256(request.content).hexdigest()
        attempt = self.store.reserve(
            self._case.get(),
            "embedding",
            held,
            fingerprint,
            self.embedding_counter.evidence,
        )
        return {
            "attempt": attempt,
            "fingerprint": fingerprint,
            "case_id": self._case.get(),
            "streaming": False,
            "started_monotonic": time.monotonic(),
            "provider": "cloudru",
            "operation": "embedding",
            "input_count": len(payload["input"]),
        }

    def finish(self, row, response, error) -> None:
        if row["provider"] == "cloudru":
            self._finish_cloud(row, response, error)
            return
        usage, model, body = {}, None, {}
        if response is not None:
            try:
                body = response.json()
                usage = usage_counts(body, "deepseek")
                raw_model = body.get("model")
                model = (
                    raw_model
                    if isinstance(raw_model, str) and raw_model in MODELS
                    else None
                )
            except (ValueError, TypeError, AttributeError):
                pass
        complete = complete_usage(usage, "deepseek")
        if response is not None:
            raw = body.get("usage", {}) if isinstance(body, dict) else {}
            if not isinstance(raw, dict) or any(
                k in raw and (type(raw[k]) is not int or raw[k] < 0)
                for k in (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "prompt_cache_hit_tokens",
                    "prompt_cache_miss_tokens",
                )
            ):
                complete = False
        uncertain = (
            error is not None
            or response is None
            or not response.is_success
            or not complete
            or model is None
        )
        actual = (
            None
            if not complete
            else {
                "input": usage["prompt_tokens"],
                "output": usage["completion_tokens"],
                "embedding": 0,
                "micro_cny": usage["prompt_tokens"] * 2 + usage["completion_tokens"] * 8,
            }
        )
        receipt = {
            "provider": "deepseek",
            "currency": "CNY",
            "status": response.status_code if response is not None else None,
            "usage": usage,
            "response_model": model,
            "usage_complete": complete,
            "error": error is not None,
            "elapsed_seconds": round(time.monotonic() - row["started_monotonic"], 4),
        }
        if "stream_evidence" in row:
            receipt["stream_evidence"] = row["stream_evidence"]
        self.store.settle(row["attempt"], row["fingerprint"], actual, receipt, uncertain)
        if self.store.snapshot()["halted"]:
            raise IndexingLimit("Uncertain provider outcome; shared budget halted")

    def _finish_cloud(self, row, response, error) -> None:
        body = {}
        if response is not None:
            try:
                body = response.json()
            except ValueError:
                pass
        usage, complete, valid_vectors = final_usage(body, row["input_count"])
        uncertain = (
            error is not None
            or response is None
            or not response.is_success
            or not complete
            or not valid_vectors
        )
        actual = (
            None
            if not complete
            else {
                "input": 0,
                "output": 0,
                "embedding": usage["prompt_tokens"],
                "micro_cny": 0,
                "micro_rub": cost_micro_rub(
                    usage["prompt_tokens"],
                    self.store.policy["cloudru"]["rate_hundredths"],
                ),
            }
        )
        receipt = {
            "provider": "cloudru",
            "currency": "RUB",
            "status": response.status_code if response is not None else None,
            "usage": usage,
            "usage_complete": complete,
            "vectors_valid": valid_vectors,
            "response_model": CLOUD_MODEL
            if isinstance(body, dict) and body.get("model") == CLOUD_MODEL
            else None,
            "error": error is not None,
            "elapsed_seconds": round(time.monotonic() - row["started_monotonic"], 4),
        }
        self.store.settle(row["attempt"], row["fingerprint"], actual, receipt, uncertain)
        if self.store.snapshot()["halted"]:
            raise IndexingLimit("Uncertain Cloud.ru outcome; shared budget halted")

    def accounting(self) -> dict:
        state = self.store.snapshot()
        return {
            **state,
            "token_totals": {
                k: state["totals"][k] for k in ("input", "output", "embedding")
            },
            "live_integration_verified": False,
        }

    def _bounded_request(self, request: httpx.Request) -> httpx.Request:
        if request.url.host != "api.deepseek.com":
            return request
        try:
            body = json.loads(request.content)
            body["max_tokens"] = self.output_limit(body["max_tokens"])
        except (KeyError, ValueError, TypeError):
            raise IndexingLimit("Explicit provider output allowance required") from None
        return httpx.Request(
            request.method,
            request.url,
            headers={
                k: v for k, v in request.headers.items() if k.lower() != "content-length"
            },
            content=json.dumps(body, ensure_ascii=False).encode(),
            extensions=dict(request.extensions),
        )

    @contextmanager
    def capture(self, *, patch_provider: bool = True) -> Generator[None, None, None]:
        """Install before importing/starting API, bot or their background tasks."""
        if os.environ.get("PYTHON_DOTENV_DISABLED") != "1":
            raise IndexingLimit("Disable dotenv before importing application code")
        original_sync, original_async = httpx.Client.send, httpx.AsyncClient.send
        ledger = self

        def send(client, request, **kwargs: Any):
            gate = TransportGate(ledger.store.path)
            gate.acquire()
            row, response = None, None
            try:
                request = ledger._bounded_request(request)
                if request.url.host == "api.deepseek.com" and json.loads(
                    request.content
                ).get("stream"):
                    raise IndexingLimit("SSE requires the async transport")
                row = ledger.begin(request)
                kwargs["follow_redirects"] = False
                response = original_sync(client, request, **kwargs)
                response.read()
            except BaseException as exc:
                if row is not None:
                    ledger.finish(row, response, exc)
                raise
            else:
                ledger.finish(row, response, None)
                return response
            finally:
                gate.close()

        async def async_send(client, request, **kwargs: Any):
            gate = TransportGate(ledger.store.path)
            await gate.acquire_async()
            row, response, handed_off = None, None, False
            try:
                request = ledger._bounded_request(request)
                row = ledger.begin(request)
                kwargs["follow_redirects"] = False
                response = await original_async(client, request, **kwargs)
                if row["streaming"] and response.is_success:
                    if response.is_stream_consumed:
                        raise IndexingLimit(
                            "Pre-consumed SSE cannot establish wire usage"
                        )
                    if (
                        not kwargs.get("stream")
                        or response.headers.get("content-type", "").split(";")[0]
                        != "text/event-stream"
                    ):
                        raise IndexingLimit("Unexpected SSE transport")
                    decoder = httpx.Response(
                        200,
                        headers={
                            "content-encoding": str(
                                response.headers.get("content-encoding", "identity")
                            )
                        },
                    )._get_content_decoder()
                    response.stream = _HeldStream(
                        _MeteredStream(
                            response.stream, cast(Any, ledger), row, decoder=decoder
                        ),
                        gate,
                    )
                    handed_off = True
                    return response
                await response.aread()
            except BaseException as exc:
                try:
                    if row is not None:
                        ledger.finish(row, None if row["streaming"] else response, exc)
                finally:
                    if response is not None:
                        await response.aclose()
                raise
            else:
                ledger.finish(row, response, None)
                return response
            finally:
                if not handed_off:
                    gate.close()

        import urllib.request

        import aiohttp
        import requests

        original_aiohttp = aiohttp.ClientSession._request
        original_urlopen = urllib.request.OpenerDirector.open

        async def aio_request(client, method, url, **kwargs: Any):
            dest = urlsplit(str(url))
            if (
                dest.scheme != "https"
                or dest.hostname != "api.telegram.org"
                or dest.port not in (None, 443)
            ):
                raise IndexingLimit("Non-HTTPX provider transport refused")
            kwargs["allow_redirects"] = False
            return await original_aiohttp(client, method, url, **kwargs)

        def requests_send(*args, **kwargs):
            raise IndexingLimit("Unmetered requests transport refused")

        def urlopen(opener, fullurl, *args, **kwargs):
            url = (
                fullurl.full_url
                if isinstance(fullurl, urllib.request.Request)
                else str(fullurl)
            )
            dest = urlsplit(url)
            if (
                dest.scheme != "https"
                or dest.hostname != "api.telegram.org"
                or dest.port not in (None, 443)
            ):
                raise IndexingLimit("Unmetered urllib provider transport refused")
            return original_urlopen(opener, fullurl, *args, **kwargs)

        with (
            patch.object(httpx.Client, "send", send),
            patch.object(httpx.AsyncClient, "send", async_send),
            patch.object(requests.Session, "send", requests_send),
            patch.object(aiohttp.ClientSession, "_request", aio_request),
            patch.object(urllib.request.OpenerDirector, "open", urlopen),
        ):
            if not patch_provider:
                yield
            else:
                from llm.providers.deepseek import DeepSeekProvider

                original_params = DeepSeekProvider._params

                def parameters(provider, profile):
                    params = original_params(provider, profile)
                    state = ledger.store.snapshot()
                    # An in-flight call can still refund unused output. Do not
                    # block the event loop here: the transport gate will wait and
                    # atomically clamp the actual wire payload afterwards.
                    params["max_tokens"] = (
                        min(params["max_tokens"], 1024)
                        if state["pending_or_uncertain"] and not state["halted"]
                        else ledger.output_limit(params["max_tokens"])
                    )
                    return params

                with patch.object(DeepSeekProvider, "_params", parameters):
                    yield


class _HeldStream(httpx.AsyncByteStream):
    def __init__(self, stream, gate) -> None:
        self.stream, self.gate = stream, gate

    async def __aiter__(self):
        try:
            async for chunk in self.stream:
                yield chunk
        finally:
            try:
                await self.stream.aclose()
            finally:
                self.gate.close()

    async def aclose(self) -> None:
        try:
            await self.stream.aclose()
        finally:
            self.gate.close()
