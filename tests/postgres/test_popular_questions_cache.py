import asyncio

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from api.services import popular_questions as service
from db.postgres.services.message_log import MessageLogService
from tests.postgres.test_popular_questions import FakeEmbeddings, _log_questions


@pytest.mark.asyncio
async def test_cached_result_skips_grouping_and_reuses_different_display_limits(
    session: AsyncSession, monkeypatch
):
    cache = service.PopularQuestionsCache(ttl=300)
    logs = MessageLogService(session)
    await _log_questions(logs, ["Как поступить?", "Сколько стоит общежитие?"])
    embedder = FakeEmbeddings()
    calls = 0
    cluster = service._cluster

    def count_clusters(*args):
        nonlocal calls
        calls += 1
        return cluster(*args)

    monkeypatch.setattr(service, "_cluster", count_clusters)
    first = await cache.get(logs, limit=1, raw_limit=10, embedder=embedder)
    second = await cache.get(logs, limit=2, raw_limit=10, embedder=embedder)
    assert len(first) == 1
    assert len(second) == 2
    assert calls == 1
    first[0]["variants"].append("changed by caller")
    assert "changed by caller" not in second[0]["variants"]


@pytest.mark.asyncio
async def test_concurrent_cache_misses_compute_once(session: AsyncSession):
    cache = service.PopularQuestionsCache(ttl=300)
    logs = MessageLogService(session)
    await _log_questions(logs, ["Как поступить?"])
    embedder = FakeEmbeddings()
    results = await asyncio.gather(
        *(cache.get(logs, raw_limit=10, embedder=embedder) for _ in range(4))
    )
    assert results == [results[0]] * 4
    assert embedder.calls == 1


@pytest.mark.asyncio
async def test_expiration_reloads_new_questions(session: AsyncSession):
    now = [0.0]
    cache = service.PopularQuestionsCache(ttl=300, clock=lambda: now[0])
    logs = MessageLogService(session)
    await _log_questions(logs, ["Как поступить?"])
    embedder = FakeEmbeddings()
    first = await cache.get(logs, raw_limit=10, embedder=embedder)
    await _log_questions(logs, ["Сколько стоит общежитие?"])
    assert await cache.get(logs, raw_limit=10, embedder=embedder) == first
    now[0] = 301
    assert len(await cache.get(logs, raw_limit=10, embedder=embedder)) == 2


@pytest.mark.asyncio
async def test_setting_revision_invalidates_cache(session: AsyncSession, monkeypatch):
    from config import instance

    cache = service.PopularQuestionsCache(ttl=300)
    logs = MessageLogService(session)
    await _log_questions(logs, ["Как поступить?"])
    embedder = FakeEmbeddings()
    await cache.get(logs, raw_limit=10, embedder=embedder)
    await _log_questions(logs, ["Сколько стоит общежитие?"])
    monkeypatch.setattr(instance, "_revision", 1)
    assert len(await cache.get(logs, raw_limit=10, embedder=embedder)) == 2


@pytest.mark.asyncio
async def test_failed_compute_is_not_cached(session: AsyncSession):
    class FailsOnce(FakeEmbeddings):
        def embed_documents(self, texts):
            if self.calls == 0:
                self.calls += 1
                raise RuntimeError("provider unavailable")
            return super().embed_documents(texts)

    cache = service.PopularQuestionsCache(ttl=300)
    logs = MessageLogService(session)
    await _log_questions(logs, ["Как поступить?"])
    embedder = FailsOnce()
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await cache.get(logs, raw_limit=10, embedder=embedder)
    assert len(await cache.get(logs, raw_limit=10, embedder=embedder)) == 1


@pytest.mark.asyncio
async def test_lru_bound_and_threshold_have_independent_results(session: AsyncSession):
    cache = service.PopularQuestionsCache(ttl=300, max_entries=1)
    logs = MessageLogService(session)
    await _log_questions(logs, ["Как поступить?"])
    embedder = FakeEmbeddings()
    await cache.get(logs, raw_limit=10, similarity_threshold=0.9, embedder=embedder)
    await _log_questions(logs, ["Сколько стоит общежитие?"])
    assert (
        len(
            await cache.get(
                logs, raw_limit=10, similarity_threshold=0.95, embedder=embedder
            )
        )
        == 2
    )
    # The first key was evicted and must see the new database contents too.
    assert (
        len(
            await cache.get(
                logs, raw_limit=10, similarity_threshold=0.9, embedder=embedder
            )
        )
        == 2
    )
    assert len(cache._entries) == 1
