"""Meter evaluation SSE responses without changing or persisting answer bytes."""

import json
from collections.abc import AsyncIterator, Generator
from contextlib import contextmanager
from unittest.mock import patch

import httpx

from tools.admission_usage import IndexingLimit, UsageLedger


class IncompleteEvaluationStream(RuntimeError):
    """The consumed stream did not establish a complete provider usage record."""


class EvaluationUsageLedger(UsageLedger):
    """Retain the indexing destination, price and completeness guards for SSE."""

    def begin(self, request: httpx.Request) -> dict:
        if request.url.host != "api.deepseek.com":
            return super().begin(request)
        payload = json.loads(request.content)
        if not payload.get("stream"):
            return super().begin(request)
        if (
            request.url.host != "api.deepseek.com"
            or payload.get("stream") is not True
            or payload.get("stream_options") != {"include_usage": True}
        ):
            raise IndexingLimit("Evaluation stream requires explicit final usage")
        # Validate/reserve a copy. The real request and generation settings stay intact.
        reservation = httpx.Request(
            request.method, request.url, json={**payload, "stream": False}
        )
        return {**super().begin(reservation), "streaming": True}

    @contextmanager
    def capture(self) -> Generator[None, None, None]:
        original_sync, original_async = httpx.Client.send, httpx.AsyncClient.send
        ledger = self

        def send(client, request, **kwargs):
            if request.url.host == "api.deepseek.com" and json.loads(request.content).get(
                "stream"
            ):
                raise IndexingLimit("Evaluation streams require the async transport")
            row = ledger.begin(request)
            response = None
            kwargs["follow_redirects"] = False
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
            response = None
            kwargs["follow_redirects"] = False
            try:
                response = await original_async(client, request, **kwargs)
                if row.get("streaming") and response.is_success:
                    if (
                        not kwargs.get("stream")
                        or response.headers.get("content-type", "").split(";")[0]
                        != "text/event-stream"
                    ):
                        raise IncompleteEvaluationStream("Unexpected stream transport")
                    encoding = str(response.headers.get("content-encoding", "identity"))
                    row["content_encoding"] = (
                        encoding
                        if encoding in ("identity", "gzip", "deflate", "br", "zstd")
                        else "other"
                    )
                    # Transport bytes may be compressed. Decode a separate copy
                    # for accounting, leaving the SDK's stream/decoder unchanged.
                    decoder = httpx.Response(
                        200, headers={"content-encoding": encoding}
                    )._get_content_decoder()
                    response.stream = _MeteredStream(
                        response.stream, ledger, row, decoder=decoder
                    )
                    return response
                await response.aread()
            except BaseException as exc:
                evidence = (
                    httpx.Response(response.status_code, json={})
                    if response is not None and row.get("streaming")
                    else response
                )
                ledger.finish(row, evidence, exc)
                if response is not None:
                    await response.aclose()
                raise
            ledger.finish(row, response, None)
            return response

        with (
            patch.object(httpx.Client, "send", send),
            patch.object(httpx.AsyncClient, "send", async_send),
        ):
            yield


class _MeteredStream(httpx.AsyncByteStream):
    def __init__(self, stream, ledger: UsageLedger, row: dict, *, decoder) -> None:
        self.stream = stream
        self.ledger = ledger
        self.row = row
        self.buffer = b""
        self.event: list[bytes] = []
        self.event_size = 0
        self.usage_body: dict = {}
        self.done = False
        self.recorded = False
        self.failed: BaseException | None = None
        self.total_bytes = 0
        self.wire_bytes = 0
        self.decoder = decoder
        self.events_seen = 0
        self.provider_error_seen = False

    def _event(self) -> None:
        if not self.event:
            return
        data = b"\n".join(self.event)
        self.event.clear()
        self.event_size = 0
        if self.done:
            raise IncompleteEvaluationStream("Data after stream terminator")
        if data == b"[DONE]":
            self.done = True
            return
        body = json.loads(data)
        self.events_seen += 1
        if not isinstance(body, dict):
            raise IncompleteEvaluationStream("Invalid SSE object")
        if body.get("error") is not None:
            self.provider_error_seen = True
            raise IncompleteEvaluationStream("Provider error frame")
        if body.get("usage") is not None:
            if self.usage_body:
                raise IncompleteEvaluationStream("Ambiguous repeated stream usage")
            # Full frame content remains transient. Only the existing numeric
            # allowlist in UsageLedger.finish is written to disk.
            self.usage_body = {"usage": body["usage"], "model": body.get("model")}

    def _consume(self, chunk: bytes) -> None:
        self.total_bytes += len(chunk)
        if self.total_bytes > 8_000_000:
            raise IncompleteEvaluationStream("Evaluation response size limit")
        self.buffer += chunk
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            line = line.removesuffix(b"\r")
            if not line:
                self._event()
            elif line.startswith(b"data:"):
                data = line[5:].removeprefix(b" ")
                self.event_size += len(data)
                if self.event_size > 262_144:
                    raise IncompleteEvaluationStream("Evaluation SSE event size limit")
                self.event.append(data)
        if len(self.buffer) > 262_144:
            raise IncompleteEvaluationStream("Evaluation SSE line size limit")

    def _finish(self) -> None:
        if self.recorded:
            return
        self.recorded = True
        error = self.failed
        if not self.done or self.buffer or self.event or not self.usage_body:
            error = error or IncompleteEvaluationStream()
        response = httpx.Response(200, json=self.usage_body)
        # Fixed counters/flags explain the next failure without retaining frames,
        # provider error messages, generated text or headers.
        self.row["stream_evidence"] = {
            "wire_bytes_seen": self.wire_bytes,
            "bytes_seen": self.total_bytes,
            "json_events_seen": self.events_seen,
            "done_seen": self.done,
            "usage_seen": bool(self.usage_body),
            "provider_error_seen": self.provider_error_seen,
            "pending_bytes": len(self.buffer),
            "pending_event_lines": len(self.event),
        }
        self.ledger.finish(self.row, response, error)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self.stream:
                self.wire_bytes += len(chunk)
                self._consume(self.decoder.decode(chunk))
                yield chunk
            self._consume(self.decoder.flush())
        except BaseException as exc:
            self.failed = exc
            raise
        finally:
            self._finish()

    async def aclose(self) -> None:
        try:
            await self.stream.aclose()
        except BaseException as exc:
            self.failed = exc
            raise
        finally:
            self._finish()
