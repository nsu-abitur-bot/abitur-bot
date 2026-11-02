"""Controlled HTTP boundaries: real safe-fetch policy, no network or LLM calls."""

import asyncio
import socket
import threading

import httpx
import pytest
from fastapi import FastAPI

from api.routes import admission_score, rag
from parser import scores


@pytest.fixture
def public_dns(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **kw: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )


@pytest.mark.parametrize("target", ["rag", "scores"])
@pytest.mark.parametrize("error", [httpx.ConnectTimeout, httpx.ReadTimeout])
async def test_timeout_is_finite_and_reported(
    monkeypatch, public_dns, target, error, capsys, caplog
):
    from parser import nsu

    def fetch(self, request):
        assert request.extensions["timeout"] == {
            "connect": 5.0,
            "read": 30.0,
            "write": 30.0,
            "pool": 30.0,
        }
        raise error("delayed source")

    async def fetch_async(self, request):
        return fetch(self, request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fetch)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fetch_async)
    if target == "scores":
        assert await scores.parse_scores("https://example.test/") == []
        assert "Ошибка загрузки страницы" in caplog.text
    else:
        assert nsu.parse_page("https://example.test/", [], {}) == {}
        assert "Ошибка при запросе" in capsys.readouterr().out


def make_app(monkeypatch):
    async def identity(text):
        return text

    async def title(text):
        return "Без названия"

    monkeypatch.setattr(rag, "clean_and_structure_text", identity)
    monkeypatch.setattr(rag, "generate_title_from_text", title)
    app = FastAPI()
    app.include_router(rag.router)
    app.include_router(admission_score.router)
    app.dependency_overrides[admission_score.get_admission_score_service] = lambda: None

    @app.get("/probe")
    async def probe():
        return {"ok": True}

    return app


@pytest.mark.parametrize("target", ["rag", "scores", "scores_parse"])
async def test_routes_keep_loop_responsive(monkeypatch, public_dns, target):
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    expired = threading.Event()
    calls = []
    body = "Fallback source content. " * 10

    def hold():
        started.set()
        release.wait()
        finished.set()

    def fetch(self, request):
        if request.method == "GET":
            calls.append(request)
            if target == "rag":
                hold()
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html"},
            content=f"<html><body>{body}</body></html>".encode(),
        )

    async def fetch_async(self, request):
        if target == "scores" and request.method == "GET":
            await asyncio.to_thread(hold)
        return fetch(self, request)

    def parse(html):
        assert body in html
        hold()
        return []

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fetch)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fetch_async)
    if target == "scores_parse":
        monkeypatch.setattr(scores, "parse_scores_html", parse)

    def watchdog():
        expired.set()
        release.set()

    timer = threading.Timer(5, watchdog)
    timer.start()
    app = make_app(monkeypatch)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        if target == "rag":
            task = asyncio.create_task(
                client.post("/rag/parse", params={"url": "https://example.test/"})
            )
        else:
            task = asyncio.create_task(
                client.post(
                    "/admission-scores/preview", json={"url": "https://example.test/"}
                )
            )
        try:
            assert await asyncio.to_thread(started.wait, 3)
            ping = await client.get("/probe")
            assert ping.status_code == 200
            assert not finished.is_set() and not expired.is_set()
            release.set()
            response = await task
            assert response.status_code == 200, response.text
            if target == "rag":
                assert body.strip() in response.json()["text"]
            assert len(calls) == 1
        finally:
            release.set()
            timer.cancel()
            await task
            timer.join()


@pytest.mark.parametrize("target", ["rag", "scores"])
@pytest.mark.parametrize("redirect", [False, True])
async def test_routes_reject_unsafe_destination(
    monkeypatch, public_dns, target, redirect
):
    calls = []

    def fetch(self, request):
        calls.append(request)
        assert request.url.host == "93.184.216.34"
        return httpx.Response(302, headers={"location": "http://127.0.0.1/"})

    async def fetch_async(self, request):
        return fetch(self, request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fetch)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fetch_async)
    app = make_app(monkeypatch)
    url = "https://example.test/" if redirect else "http://127.0.0.1/"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        if target == "rag":
            response = await client.post("/rag/parse", params={"url": url})
            assert response.status_code == 400
            assert "public" in response.json()["detail"]
        else:
            response = await client.post("/admission-scores/preview", json={"url": url})
            assert response.status_code == 200
            assert response.json()["summary"]["total"] == 0
    assert len(calls) == int(redirect)


@pytest.mark.parametrize(
    "error", [httpx.ConnectTimeout, httpx.ReadTimeout, httpx.HTTPError]
)
async def test_rag_fetch_failure_preserves_api_error(monkeypatch, public_dns, error):
    def fetch(self, request):
        raise error("source unavailable")

    async def head(self, request):
        return httpx.Response(200, headers={"Content-Type": "text/html"})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fetch)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", head)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=make_app(monkeypatch)), base_url="http://test"
    ) as client:
        response = await client.post(
            "/rag/parse", params={"url": "https://example.test/"}
        )
    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "Failed to parse page (possible block or invalid URL)"
    )
