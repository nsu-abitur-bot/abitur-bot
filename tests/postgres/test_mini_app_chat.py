import asyncio
import hmac
import time
from datetime import timedelta
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from api.main import app
from db.postgres.db import get_async_session
from db.postgres.models import FeedbackReport, MiniAppTurn, User, timestamp
from db.postgres.services.mini_app import LEASE_DURATION
from llm.base import LLMResult
from pipeline import chat_access, llm_client
from rag.trace import RetrievalResult

SECRET = "local-test-key-with-at-least-32-characters"
CHATS = "/api/v1/mini-app/chats"


def auth(platform="telegram", external_id=42, instance="nsu-test", *, ttl=45):
    expires = int(time.time()) + ttl
    identity = f"{platform}:{external_id}:{expires}"
    claim = f"infovuz:v1:{instance}:{identity}"
    signature = hmac.digest(SECRET.encode(), claim.encode(), "sha256").hex()
    return {"Authorization": f"MiniApp {identity}:{signature}"}


@pytest.fixture
async def mini_api(test_engine, monkeypatch):
    monkeypatch.setenv("MINI_APP_INSTANCE_ID", "nsu-test")
    monkeypatch.setenv("MINI_APP_SHARED_SECRET", SECRET)
    maker = async_sessionmaker(test_engine, expire_on_commit=False)

    async def session():
        async with maker() as connection:
            yield connection

    monkeypatch.setitem(app.dependency_overrides, get_async_session, session)
    monkeypatch.setattr(chat_access, "AsyncSessionLocal", maker)
    monkeypatch.setattr(llm_client, "AsyncSessionLocal", maker)
    async with AsyncClient(
        transport=ASGITransport(app), base_url="http://test"
    ) as client:
        yield client, maker
    await asyncio.gather(*llm_client._background_tasks)


async def test_history_retry_isolation_and_ratings(mini_api, monkeypatch):
    client, maker = mini_api
    generation_started, allow_generation = asyncio.Event(), asyncio.Event()
    contexts = []
    fail_provider = True

    class NoFaq:
        async def match_async(self, question):
            return None

    class Provider:
        async def generate_with_tools(
            self, messages, tools, tool_executor, profile=None, on_delta=None
        ):
            nonlocal fail_provider
            if "Повтори" in messages[-1].content and fail_provider:
                fail_provider = False
                raise RuntimeError("injected provider outage")
            if "Ожидание" in messages[-1].content:
                generation_started.set()
                await allow_generation.wait()
            return LLMResult(text="Ответ из тестового каталога")

        async def generate_with_usage(self, messages, profile):
            return LLMResult(text='{"topic_id": null}')

    async def retrieval(query, conversation_history=None, search_query=None):
        contexts.append(conversation_history)
        return RetrievalResult(
            context="Материалы о приёме",
            sources=[
                {"url": "https://example.org/catalog", "title": "Каталог"},
                {"url": "javascript:unexpected", "title": "Недопустимый протокол"},
            ],
        )

    async def unexpected_redis():
        pytest.fail("Mini App uses persisted conversation history")

    monkeypatch.setattr(llm_client, "get_redis_client", unexpected_redis)
    monkeypatch.setattr(llm_client, "get_faq_matcher", NoFaq)
    monkeypatch.setattr(llm_client, "query_graph_with_sources", retrieval)
    monkeypatch.setattr(llm_client, "query_graph_with_crag", retrieval)
    monkeypatch.setattr(llm_client, "get_llm_provider", Provider)

    first, second, max_chat = (str(uuid4()) for _ in range(3))
    for chat_id, headers in [(first, auth()), (second, auth()), (max_chat, auth("max"))]:
        created = await client.post(CHATS, json={"id": chat_id}, headers=headers)
        assert created.status_code == 201, created.text
    assert (
        await client.post(CHATS, json={"id": first}, headers=auth())
    ).status_code == 201
    assert {chat["id"] for chat in (await client.get(CHATS, headers=auth())).json()} == {
        first,
        second,
    }
    assert [
        chat["id"] for chat in (await client.get(CHATS, headers=auth("max"))).json()
    ] == [max_chat]

    async def ask(chat_id, text, turn_id=None, headers=None):
        return await client.post(
            f"{CHATS}/{chat_id}/turns",
            json={"id": turn_id or str(uuid4()), "text": text},
            headers=headers or auth(),
        )

    for headers in [auth("max"), auth(external_id=43)]:
        assert (await client.get(f"{CHATS}/{first}", headers=headers)).status_code == 404
        assert (await ask(first, "Чужой вопрос", headers=headers)).status_code == 404
        assert (
            await client.post(CHATS, json={"id": first}, headers=headers)
        ).status_code == 404
    assert (await client.get(CHATS)).status_code == 401
    assert (await client.get(CHATS, headers=auth(ttl=-1))).status_code == 401
    assert (await client.get(CHATS, headers=auth(instance="other"))).status_code == 401

    answer = await ask(first, "Первый вопрос об общежитии")
    assert answer.status_code == 200, answer.text
    answer = answer.json()
    assert answer["sources"] == [
        {"url": "https://example.org/catalog", "title": "Каталог"}
    ]
    assert (await ask(second, "Вопрос о лабораториях")).status_code == 200
    assert "общежитии" not in contexts[-1]
    assert (await ask(first, "А когда подавать заявление?")).status_code == 200
    assert "Первый вопрос об общежитии" in contexts[-1]
    assert "Вопрос о лабораториях" not in contexts[-1]
    assert (await ask(max_chat, "Чат в MAX", headers=auth("max"))).status_code == 200
    assert "общежитии" not in contexts[-1]

    rate_url = f"{CHATS}/{first}/turns/{answer['id']}/rating"
    assert (
        await client.put(rate_url, json={"rating": "positive"}, headers=auth("max"))
    ).status_code == 404
    assert (
        await client.put(rate_url, json={"rating": "positive"}, headers=auth())
    ).json() == {"rating": "positive"}
    assert (
        await client.put(rate_url, json={"rating": "negative"}, headers=auth())
    ).json() == {"rating": "positive"}

    retry_id = str(uuid4())
    assert (await ask(first, "Повтори ещё раз", retry_id)).status_code == 503
    loaded = await client.get(f"{CHATS}/{first}", headers=auth())
    assert loaded.json()["turns"][-1]["status"] == "failed"
    recovered = await ask(first, "Повтори ещё раз", retry_id)
    assert recovered.status_code == 200, recovered.text
    assert (await ask(first, "Повтори ещё раз", retry_id)).json() == recovered.json()
    assert (await ask(first, "Другой текст", retry_id)).status_code == 409
    loaded = await client.get(f"{CHATS}/{first}", headers=auth())
    assert [turn["status"] for turn in loaded.json()["turns"]] == ["complete"] * 3
    assert loaded.json()["turns"][0]["rating"] == "positive"
    assert contexts[-1].count("Повтори ещё раз") == 1

    slow = asyncio.create_task(ask(first, "Ожидание ответа"))
    try:
        await asyncio.wait_for(generation_started.wait(), timeout=5)
        assert (await ask(first, "Второй параллельный вопрос")).status_code == 409
        loaded = await client.get(f"{CHATS}/{first}", headers=auth())
        assert loaded.json()["turns"][-1]["status"] == "pending"
    finally:
        allow_generation.set()
        assert (await slow).status_code == 200

    async with maker() as session:
        assert await session.scalar(select(func.count()).select_from(User)) == 2
        assert await session.scalar(select(func.count()).select_from(FeedbackReport)) == 1


async def test_stale_attempt_recovery_and_instance_boundary(mini_api, monkeypatch):
    client, maker = mini_api
    chat_id, turn_id = str(uuid4()), str(uuid4())
    await client.post(CHATS, json={"id": chat_id}, headers=auth())

    class AnswerFaq:
        async def match_async(self, question):
            return "Готовый ответ FAQ"

    monkeypatch.setattr(llm_client, "get_faq_matcher", AnswerFaq)
    async with maker.begin() as session:
        session.add(
            MiniAppTurn(
                id=turn_id,
                conversation_id=chat_id,
                question="Вопрос",
                attempt=1,
                started_at=timestamp() - LEASE_DURATION - timedelta(seconds=1),
                status="pending",
            )
        )
    loaded = await client.get(f"{CHATS}/{chat_id}", headers=auth())
    assert loaded.json()["turns"][0]["status"] == "failed"
    sent = await client.post(
        f"{CHATS}/{chat_id}/turns",
        json={"id": turn_id, "text": "Вопрос"},
        headers=auth(),
    )
    assert sent.status_code == 200, sent.text
    assert sent.json()["answer"] == "Готовый ответ FAQ"
    monkeypatch.setenv("MINI_APP_INSTANCE_ID", "other")
    assert (await client.get(CHATS, headers=auth(instance="other"))).json() == []
    assert (
        await client.get(f"{CHATS}/{chat_id}", headers=auth(instance="other"))
    ).status_code == 404
    assert (await client.get(CHATS, headers=auth())).status_code == 401
