"""Controlled HTTP boundaries: no network or LLM calls."""

import asyncio
import threading

import httpx
import pytest
import requests
from fastapi import HTTPException
from pydantic import HttpUrl

from api.routes import rag
from parser import nsu, scores


@pytest.mark.parametrize("error", [requests.ConnectTimeout, requests.ReadTimeout])
def test_page_timeout_is_finite_and_reported(monkeypatch, capsys, error):
    def fetch(url, **kwargs):
        assert kwargs["timeout"] == (5, 30)
        raise error("delayed source")

    monkeypatch.setattr(nsu.requests, "get", fetch)
    assert nsu.parse_page("https://example.test/", [], {}) == {}
    assert "Ошибка при запросе" in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [requests.ConnectTimeout, requests.ReadTimeout])
async def test_scores_timeout_is_finite_and_reported(monkeypatch, caplog, error):
    def fetch(url, **kwargs):
        assert kwargs["timeout"] == (5, 30)
        raise error("delayed source")

    monkeypatch.setattr(scores.requests, "get", fetch)
    assert await scores.parse_scores("https://example.test/") == []
    assert "Ошибка загрузки страницы" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "check"),
    [("scores", "responsive"), ("rag", "responsive"), ("rag", "single_fetch")],
)
async def test_delayed_fetch_keeps_loop_responsive(monkeypatch, target, check):
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    observations = []
    calls = []
    body = "Fallback source content. " * 10

    def fetch(url, **kwargs):
        calls.append(url)
        started.set()
        release.wait(5)
        finished.set()
        response = requests.Response()
        response.status_code = 200
        response._content = f"<html><body>{body}</body></html>".encode()
        response.encoding = "utf-8"
        return response

    async def head(self, url, **kwargs):
        return httpx.Response(200, headers={"Content-Type": "text/html"})

    async def identity(text):
        return text

    async def title(text):
        return "Без названия"

    async def heartbeat():
        try:
            assert await asyncio.to_thread(started.wait, 5)
            observations.append(not finished.is_set())
        finally:
            release.set()

    monkeypatch.setattr(nsu.requests, "get", fetch)
    monkeypatch.setattr(httpx.AsyncClient, "head", head)
    monkeypatch.setattr(rag, "clean_and_structure_text", identity)
    monkeypatch.setattr(rag, "generate_title_from_text", title)
    beat = asyncio.create_task(heartbeat())
    if target == "scores":
        assert await scores.parse_scores("https://example.test/") == []
    else:
        result = await rag.parse_page_for_rag(HttpUrl("https://example.test/"))
        assert body.strip() in result.text
    await beat
    if check == "responsive":
        assert observations == [True], "fetch blocked the event loop"
    else:
        assert len(calls) == 1, "HTML fallback fetched the same source again"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [requests.ConnectTimeout, requests.ReadTimeout, requests.HTTPError]
)
async def test_rag_fetch_failure_preserves_api_error(monkeypatch, error):
    async def head(self, url, **kwargs):
        return httpx.Response(200, headers={"Content-Type": "text/html"})

    def fetch(url, **kwargs):
        raise error("source unavailable")

    monkeypatch.setattr(httpx.AsyncClient, "head", head)
    monkeypatch.setattr(nsu.requests, "get", fetch)
    with pytest.raises(HTTPException) as caught:
        await rag.parse_page_for_rag(HttpUrl("https://example.test/"))
    assert caught.value.status_code == 400
    assert caught.value.detail == "Failed to parse page (possible block or invalid URL)"
