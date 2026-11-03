"""Real SQL/API/cache transitions with synthetic embeddings, no messenger network."""

import asyncio
import hashlib
from time import perf_counter
from types import SimpleNamespace

import fakeredis.aioredis
import httpx
from fastapi import FastAPI
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

import bot.core as core_module
import faq.matcher as matcher_module
import pipeline.llm_client as pipeline_module
from api.routes.faq import router
from bot.formatting import format_for_channel
from db.postgres.db import get_async_session
from db.postgres.models import FeedbackReport, MessageLog, User
from db.redis.client import RedisClient
from faq.matcher import FAQMatcher
from faq.updates import watch_faq_updates


class ExactEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(b) for b in hashlib.sha256(t.casefold().encode()).digest()]
            for t in texts
        ]


async def test_http_edit_reaches_two_caches_and_channel_formatters(
    test_engine, tmp_path, monkeypatch, record_property
):
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    monkeypatch.setattr(matcher_module, "AsyncSessionLocal", factory)
    monkeypatch.setattr(core_module, "AsyncSessionLocal", factory)
    monkeypatch.setattr(pipeline_module, "AsyncSessionLocal", factory)
    redis = RedisClient.__new__(RedisClient)
    redis.client = fakeredis.aioredis.FakeRedis(decode_responses=True)

    async def get_redis():
        return redis

    async def no_classification(*args, **kwargs):
        return None

    def forbid_provider(*args, **kwargs):
        raise AssertionError("FAQ fixture must not call a provider")

    monkeypatch.setattr(core_module, "get_redis_client", get_redis)
    monkeypatch.setattr(pipeline_module, "get_redis_client", get_redis)
    monkeypatch.setattr(pipeline_module, "_classify_intent_bg", no_classification)
    monkeypatch.setattr(pipeline_module, "get_llm_provider", forbid_provider)
    monkeypatch.setattr(
        pipeline_module,
        "get_abbrev_expander",
        lambda: SimpleNamespace(expand=lambda value: value),
    )
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    async def session():
        async with factory() as value:
            yield value

    app.dependency_overrides[get_async_session] = session
    matchers = [
        FAQMatcher(
            faq_path=tmp_path / f"missing-{i}.yaml",
            threshold=0.99,
            embedder=ExactEmbeddings(),
        )
        for i in range(2)
    ]
    dsn = test_engine.url.set(drivername="postgresql").render_as_string(
        hide_password=False
    )
    watchers = [asyncio.create_task(watch_faq_updates(m, dsn)) for m in matchers]
    question = "Кто такой Иртегов?"

    async def wait_for_both(answer: str) -> None:
        async def ready():
            while True:
                values = await asyncio.gather(
                    *(m.match_async(question) for m in matchers)
                )
                if values == [answer, answer]:
                    return
                await asyncio.sleep(0.005)

        await asyncio.wait_for(ready(), timeout=10)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://fixture"
        ) as client:
            created = await client.post(
                "/api/v1/faq", json=dict(question=question, answer="fixture-v1")
            )
            assert created.status_code == 201
            await wait_for_both("fixture-v1")
            started = perf_counter()
            edited = await client.put(
                f"/api/v1/faq/{created.json()['id']}",
                json=dict(question=question, answer="fixture-v2"),
            )
            assert edited.status_code == 200
            await wait_for_both("fixture-v2")
            record_property(
                "http_edit_to_both_caches_ms", (perf_counter() - started) * 1000
            )
            for channel, matcher in zip(("telegram", "max"), matchers):
                answer = await matcher.match_async(question)
                assert isinstance(answer, str)
                assert format_for_channel(answer, [], channel) == "fixture-v2"
                monkeypatch.setattr(pipeline_module, "get_faq_matcher", lambda: matcher)
                core = core_module.BotCore()
                session_id = f"release390:{channel}:synthetic"
                reply = await core.handle_message(
                    channel=channel,
                    external_user_id="3900001",
                    session_id=session_id,
                    user_name="fixture",
                    user_text=question,
                )
                assert "fixture-v2" in reply.text
                while pipeline_module._background_tasks:
                    await asyncio.gather(*list(pipeline_module._background_tasks))
                await core.cmd_feedback(session_id)
                await core.handle_message(
                    channel=channel,
                    external_user_id="3900001",
                    session_id=session_id,
                    user_name="fixture",
                    user_text="synthetic feedback",
                )
            async with factory() as db:
                assert await db.scalar(select(func.count()).select_from(User)) == 2
                assert (
                    await db.scalar(
                        select(func.count())
                        .select_from(MessageLog)
                        .where(MessageLog.message_type == "user_input")
                    )
                    == 2
                )
                assert (
                    await db.scalar(select(func.count()).select_from(FeedbackReport)) == 2
                )
            record_property("synthetic_unique_channel_users", 2)
            record_property("synthetic_questions", 2)
            record_property("synthetic_feedback_reports", 2)
    finally:
        for watcher in watchers:
            watcher.cancel()
        await asyncio.gather(*watchers, return_exceptions=True)
        await redis.client.aclose()
