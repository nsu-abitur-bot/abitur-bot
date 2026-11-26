import asyncio
import hmac
import time
from datetime import timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import asyncpg
import fakeredis.aioredis
import pytest
from httpx import ASGITransport, AsyncClient, Response
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker

from api.main import app
from bot.core import BotCore
from db.postgres.db import get_async_session
from db.postgres.models import FeedbackReport, MessageLog, MiniAppTurn, User, timestamp
from db.postgres.services.message_log import MessageLogService
from db.postgres.services.mini_app import LEASE_DURATION
from db.postgres.services.settings import SettingsService
from db.redis.client import RedisClient
from llm.base import LLMResult
from pipeline import llm_client
from pipeline.chat_access import RateLimitExceeded
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
    monkeypatch.setattr(llm_client, "AsyncSessionLocal", maker)
    async with AsyncClient(
        transport=ASGITransport(app), base_url="http://test"
    ) as client:
        yield client, maker
    await asyncio.gather(*llm_client._background_tasks)


@pytest.fixture
async def bot_chat(mini_api, monkeypatch):
    _, maker = mini_api
    async with fakeredis.aioredis.FakeRedis(decode_responses=True) as fake_redis:
        redis = RedisClient.__new__(RedisClient)
        redis.client = fake_redis
        monkeypatch.setattr("bot.core.AsyncSessionLocal", maker)
        monkeypatch.setattr("bot.core.get_redis_client", AsyncMock(return_value=redis))
        yield BotCore()


async def test_history_retry_isolation_and_ratings(mini_api, monkeypatch):
    client, maker = mini_api
    generation_started, allow_generation = asyncio.Event(), asyncio.Event()
    contexts = []
    queries = []
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
        queries.append(search_query or query)
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
    assert contexts[-1] is None
    assert (await ask(first, "А сколько оно стоит?")).status_code == 200
    followup_history = contexts[-1]
    assert followup_history is not None
    assert "Первый вопрос об общежитии" in followup_history
    assert "Вопрос о лабораториях" not in followup_history
    assert (await ask(max_chat, "Чат в MAX", headers=auth("max"))).status_code == 200
    assert contexts[-1] is None

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
    assert "Повтори ещё раз" not in (contexts[-1] or "")
    assert queries[-1].count("Повтори ещё раз") == 1

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
        await SettingsService(session).update_rate_limit_settings(100, 1)
    limited = await ask(first, "Вопрос сверх дневного лимита")
    assert limited.status_code == 429
    assert "дневной лимит" in limited.json()["detail"]


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


@pytest.mark.parametrize(
    ("external_ids", "system_limit", "user_limit", "limit_message"),
    [
        ((42, 42), 100, 1, "Вы исчерпали дневной лимит"),
        ((42, 43), 2, 100, "Сегодняшний общий лимит"),
    ],
    ids=["same-user", "last-system-slot"],
)
async def test_concurrent_dialogs_reserve_only_one_slot(
    mini_api,
    test_engine,
    monkeypatch,
    external_ids,
    system_limit,
    user_limit,
    limit_message,
):
    client, maker = mini_api
    requests = []
    for external_id in external_ids:
        chat_id, turn_id = str(uuid4()), str(uuid4())
        headers = auth(external_id=external_id)
        created = await client.post(CHATS, json={"id": chat_id}, headers=headers)
        assert created.status_code == 201
        requests.append(
            (f"{CHATS}/{chat_id}/turns", {"id": turn_id, "text": "Вопрос"}, headers)
        )

    async with maker() as session:
        await SettingsService(session).update_rate_limit_settings(
            system_limit, user_limit
        )
        if system_limit == 2:
            await MessageLogService(session).create_log(
                user_id=99,
                session_id="telegram:earlier",
                message_type="user_input",
                content="Предыдущий запрос",
                message_metadata={"source": "telegram"},
            )

    generation_started, finish_generation = asyncio.Event(), asyncio.Event()
    generated_ids = []

    async def generate(*args, log_entry_id, **kwargs):
        generated_ids.append(log_entry_id)
        if len(generated_ids) == 1:
            generation_started.set()
            await finish_generation.wait()
        return llm_client.LlmAnswer(text="Ответ")

    monkeypatch.setattr("api.routes.mini_app.ask_local_llm", generate)
    dsn = test_engine.url.set(drivername="postgresql").render_as_string(
        hide_password=False
    )
    blocker = await asyncpg.connect(dsn)
    tasks: list[asyncio.Task[Response]] = []
    try:
        async with blocker.transaction():
            # Allow reads but hold writes until both requests are waiting in PostgreSQL.
            await blocker.execute("LOCK TABLE message_logs IN SHARE MODE")
            tasks = [
                asyncio.create_task(client.post(url, json=body, headers=headers))
                for url, body, headers in requests
            ]
            async with asyncio.timeout(5):
                while (
                    await blocker.fetchval("""
                    SELECT count(DISTINCT pid) FROM pg_locks
                    WHERE NOT granted
                      AND database = (SELECT oid FROM pg_database
                                      WHERE datname = current_database())
                      AND (relation = 'message_logs'::regclass OR locktype = 'advisory')
                """)
                    < 2
                ):
                    await asyncio.sleep(0.005)
        await asyncio.wait_for(generation_started.wait(), timeout=5)
        done, pending = await asyncio.wait(
            tasks, timeout=5, return_when=asyncio.FIRST_COMPLETED
        )
        assert len(done) == len(pending) == 1
        rejected = done.pop().result()
        assert rejected.status_code == 429, rejected.text
        assert limit_message in rejected.json()["detail"]
        assert len(generated_ids) == 1
        async with maker() as session:
            reserved_log = await session.get(MessageLog, generated_ids[0])
            assert reserved_log is not None
            assert reserved_log.message_metadata == {"source": "mini-app"}
            assert await session.scalar(select(func.count()).select_from(MessageLog)) == (
                2 if system_limit == 2 else 1
            )
    finally:
        finish_generation.set()
        await blocker.close()
        async with asyncio.timeout(5):
            await asyncio.gather(*tasks, return_exceptions=True)
    assert sorted(task.result().status_code for task in tasks) == [200, 429]


async def test_retries_only_count_when_admitted_to_pipeline(mini_api, monkeypatch):
    client, maker = mini_api
    chat_id = str(uuid4())
    headers = auth()
    created = await client.post(CHATS, json={"id": chat_id}, headers=headers)
    assert created.status_code == 201
    url = f"{CHATS}/{chat_id}/turns"
    first = {"id": str(uuid4()), "text": "Первый вопрос"}
    second = {"id": str(uuid4()), "text": "Второй вопрос"}
    generated = []

    async def generate(message, *, log_entry_id, **kwargs):
        generated.append((message, log_entry_id))
        return llm_client.LlmAnswer(text="Ответ", failed=len(generated) == 1)

    monkeypatch.setattr("api.routes.mini_app.ask_local_llm", generate)
    async with maker() as session:
        await SettingsService(session).update_rate_limit_settings(100, 1)
    assert (await client.post(url, json=first, headers=headers)).status_code == 503
    assert (await client.post(url, json=first, headers=headers)).status_code == 429
    assert len(generated) == 1

    async with maker() as session:
        await SettingsService(session).update_rate_limit_settings(100, 2)
    recovered = await client.post(url, json=first, headers=headers)
    assert recovered.status_code == 200, recovered.text
    replay = await client.post(url, json=first, headers=headers)
    assert replay.status_code == 200
    assert replay.json() == recovered.json()
    assert (await client.post(url, json=second, headers=headers)).status_code == 429
    assert len(generated) == 2

    async with maker() as session:
        await SettingsService(session).update_rate_limit_settings(100, 3)
    assert (await client.post(url, json=second, headers=headers)).status_code == 200
    async with maker() as session:
        logs = list(await session.execute(select(MessageLog.content, MessageLog.id)))
    assert sorted(logs) == sorted(generated)
    assert [question for question, _ in generated] == [
        first["text"],
        first["text"],
        second["text"],
    ]


@pytest.mark.parametrize("channel", ["telegram", "max"])
async def test_messenger_and_mini_app_share_daily_limits(
    mini_api, bot_chat, monkeypatch, channel
):
    client, maker = mini_api
    generate = AsyncMock(return_value=llm_client.LlmAnswer(text="Ответ"))
    monkeypatch.setattr("bot.core.ask_local_llm", generate)
    monkeypatch.setattr("api.routes.mini_app.ask_local_llm", generate)
    async with maker() as session:
        await SettingsService(session).update_rate_limit_settings(2, 1)

    async def ask_bot(external_id):
        return await bot_chat.handle_message(
            channel, str(external_id), f"{channel}:{external_id}", "user", "Вопрос"
        )

    assert "Ответ" in (await ask_bot(42)).text
    for external_id, expected_status in [(42, 429), (43, 200)]:
        chat_id = str(uuid4())
        headers = auth(channel, external_id)
        created = await client.post(CHATS, json={"id": chat_id}, headers=headers)
        assert created.status_code == 201
        reply = await client.post(
            f"{CHATS}/{chat_id}/turns",
            json={"id": str(uuid4()), "text": "Вопрос"},
            headers=headers,
        )
        assert reply.status_code == expected_status, reply.text
        if expected_status == 429:
            assert reply.json()["detail"] == RateLimitExceeded.USER.value
    assert (await ask_bot(44)).text == RateLimitExceeded.SYSTEM.value
    assert generate.await_count == 2
    async with maker() as session:
        logs = await session.scalars(select(MessageLog).order_by(MessageLog.id))
        assert [log.message_metadata for log in logs] == [
            {"source": channel},
            {"source": "mini-app"},
        ]


async def test_pipeline_is_skipped_when_reservation_write_fails(
    mini_api, bot_chat, monkeypatch
):
    client, maker = mini_api
    generate = AsyncMock(return_value=llm_client.LlmAnswer(text="Ответ"))
    monkeypatch.setattr("bot.core.ask_local_llm", generate)
    monkeypatch.setattr("api.routes.mini_app.ask_local_llm", generate)
    async with maker.begin() as session:
        await session.execute(
            text("ALTER TABLE message_logs ADD CHECK (message_type <> 'user_input')")
        )

    chat_id = str(uuid4())
    created = await client.post(CHATS, json={"id": chat_id}, headers=auth())
    assert created.status_code == 201
    reply = await client.post(
        f"{CHATS}/{chat_id}/turns",
        json={"id": str(uuid4()), "text": "Вопрос"},
        headers=auth(),
    )
    assert reply.status_code == 503
    with pytest.raises(IntegrityError):
        await bot_chat.handle_message("telegram", "42", "telegram:42", "user", "Вопрос")
    generate.assert_not_awaited()
    async with maker() as session:
        assert await session.scalar(select(func.count()).select_from(MessageLog)) == 0
