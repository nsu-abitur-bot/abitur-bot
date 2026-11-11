import asyncio

import fakeredis.aioredis
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from api.services.popular_statistics import PopularStatistics
from db.postgres.services.message_log import MessageLogService
from tests.postgres.test_popular_questions import FakeEmbeddings, _log_questions


@pytest.fixture
async def redis():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


@pytest.mark.asyncio
async def test_page_read_never_runs_database_or_provider_work(redis):
    def forbidden_session():
        raise AssertionError("GET must not prepare statistics")

    stats = PopularStatistics(redis, forbidden_session)
    assert await stats.read(limit=5) == {
        "questions": [],
        "status": "pending",
        "updated_at": None,
    }


@pytest.mark.asyncio
async def test_background_prepares_and_persists_snapshot_without_reading_page(
    redis, session: AsyncSession, test_engine
):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    await _log_questions(MessageLogService(session), ["Как поступить?"])
    embedder = FakeEmbeddings()
    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    stats = PopularStatistics(redis, factory, embedder=embedder)
    await stats.refresh()
    first = await stats.read(limit=1)
    assert first["status"] == "ready"
    assert first["questions"][0]["count"] == 1
    assert first["updated_at"] is not None
    # Restarting the API reuses the stored snapshot, not a five-minute local TTL.
    restarted = PopularStatistics(redis, factory, embedder=embedder)
    assert await restarted.read(limit=1) == first
    await restarted.refresh()
    assert embedder.calls == 1


@pytest.mark.asyncio
async def test_new_question_is_processed_in_background(redis, session, test_engine):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    logs = MessageLogService(session)
    await _log_questions(logs, ["Как поступить?"])
    embedder = FakeEmbeddings()
    stats = PopularStatistics(redis, factory, embedder=embedder)
    await stats.refresh()
    await _log_questions(logs, ["Как поступить?", "Сколько стоит общежитие?"])
    # Visiting the page only reads the old prepared result.
    assert (await stats.read())["questions"][0]["count"] == 1
    assert embedder.calls == 1
    await stats.refresh()
    result = await stats.read()
    assert result["questions"][0]["count"] == 2
    assert len(result["questions"]) == 2
    assert embedder.calls == 2


@pytest.mark.asyncio
async def test_unchanged_source_skips_reclustering(
    redis, session, test_engine, monkeypatch
):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from api.services import popular_questions

    await _log_questions(MessageLogService(session), ["Как поступить?"])
    stats = PopularStatistics(
        redis,
        async_sessionmaker(test_engine, class_=AsyncSession),
        embedder=FakeEmbeddings(),
    )
    await stats.refresh()

    def forbidden_cluster(*args):
        raise AssertionError("No changes must mean no CPU recomputation")

    monkeypatch.setattr(popular_questions, "_cluster", forbidden_cluster)
    await stats.refresh()
    assert (await stats.read())["status"] == "ready"


@pytest.mark.asyncio
async def test_independent_workers_do_not_duplicate_refresh(redis, session, test_engine):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    await _log_questions(MessageLogService(session), ["Как поступить?"])
    factory = async_sessionmaker(test_engine, class_=AsyncSession)
    embedder = FakeEmbeddings()
    first = PopularStatistics(redis, factory, embedder=embedder)
    second = PopularStatistics(redis, factory, embedder=embedder)
    await asyncio.gather(first.refresh(), second.refresh())
    assert embedder.calls == 1
    assert (await second.read())["status"] == "ready"


@pytest.mark.asyncio
async def test_empty_prepared_history_is_ready_not_pending(redis, test_engine):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    stats = PopularStatistics(
        redis,
        async_sessionmaker(test_engine, class_=AsyncSession),
        embedder=FakeEmbeddings(),
    )
    await stats.refresh()
    result = await stats.read()
    assert result["status"] == "ready"
    assert result["questions"] == []


@pytest.mark.asyncio
async def test_worker_prepares_without_any_page_requests(redis, session, test_engine):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    await _log_questions(MessageLogService(session), ["Как поступить?"])
    embedder = FakeEmbeddings()
    stats = PopularStatistics(
        redis, async_sessionmaker(test_engine, class_=AsyncSession), embedder=embedder
    )
    worker = asyncio.create_task(stats.run())
    try:
        async with asyncio.timeout(3):
            while not await redis.exists(stats._key(500, 0.86)):
                await asyncio.sleep(0.01)
        assert embedder.calls == 1
        assert (await stats.read())["status"] == "ready"
    finally:
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker


@pytest.mark.asyncio
async def test_model_revision_get_is_pending_until_background_refresh(
    redis, session, test_engine, monkeypatch
):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from config import instance

    await _log_questions(MessageLogService(session), ["Как поступить?"])
    stats = PopularStatistics(
        redis,
        async_sessionmaker(test_engine, class_=AsyncSession),
        embedder=FakeEmbeddings(),
    )
    await stats.refresh()
    monkeypatch.setattr(instance, "_revision", 2)
    assert (await stats.read())["status"] == "pending"
    await stats.refresh()
    assert (await stats.read())["status"] == "ready"


@pytest.mark.asyncio
async def test_failed_background_refresh_keeps_previous_ready_snapshot(
    redis, session, test_engine
):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    class Failing(FakeEmbeddings):
        def embed_documents(self, texts):
            raise RuntimeError("Provider unavailable")

    factory = async_sessionmaker(test_engine, class_=AsyncSession)
    logs = MessageLogService(session)
    await _log_questions(logs, ["Как поступить?"])
    stats = PopularStatistics(redis, factory, embedder=FakeEmbeddings())
    await stats.refresh()
    first = await stats.read()
    await _log_questions(logs, ["Сколько стоит общежитие?"])
    stats.embedder = Failing()
    with pytest.raises(RuntimeError, match="Provider unavailable"):
        await stats.refresh()
    assert await stats.read() == first
    assert await redis.get(stats._key(500, 0.86) + ":lock") is None


@pytest.mark.asyncio
async def test_initial_background_failure_is_error_not_empty(redis, session, test_engine):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    class Failing(FakeEmbeddings):
        def embed_documents(self, texts):
            raise RuntimeError("Provider unavailable")

    await _log_questions(MessageLogService(session), ["Как поступить?"])
    stats = PopularStatistics(
        redis, async_sessionmaker(test_engine, class_=AsyncSession), embedder=Failing()
    )
    with pytest.raises(RuntimeError):
        await stats.refresh()
    assert (await stats.read())["status"] == "error"
    stats.embedder = FakeEmbeddings()
    await stats.refresh()
    assert (await stats.read())["status"] == "ready"


@pytest.mark.asyncio
async def test_unchanged_snapshot_expiration_is_extended(redis, session, test_engine):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    await _log_questions(MessageLogService(session), ["Как поступить?"])
    stats = PopularStatistics(
        redis, async_sessionmaker[AsyncSession](test_engine), embedder=FakeEmbeddings()
    )
    await stats.refresh()
    key = stats._key(500, 0.86)
    await redis.expire(key, 2)
    await stats.refresh()
    assert await redis.ttl(key) > 1000


@pytest.mark.asyncio
async def test_shared_redis_has_separate_instance_namespaces(redis, session, test_engine):
    from sqlalchemy.ext.asyncio import async_sessionmaker

    await _log_questions(MessageLogService(session), ["Как поступить?"])
    factory = async_sessionmaker[AsyncSession](test_engine)
    first = PopularStatistics(
        redis, factory, embedder=FakeEmbeddings(), namespace="first"
    )
    second = PopularStatistics(
        redis, factory, embedder=FakeEmbeddings(), namespace="second"
    )
    await first.refresh()
    assert (await first.read())["status"] == "ready"
    assert (await second.read())["status"] == "pending"


@pytest.mark.asyncio
async def test_timeout_and_shutdown_keep_lease_until_embedding_thread_finishes(
    redis, session, test_engine, monkeypatch
):
    import threading

    from sqlalchemy.ext.asyncio import async_sessionmaker

    from api.services import popular_statistics

    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    class Slow(FakeEmbeddings):
        def embed_documents(self, texts):
            self.calls += 1
            loop.call_soon_threadsafe(started.set)
            release.wait(timeout=3)
            return [[1.0, 0.0] for _ in texts]

    await _log_questions(MessageLogService(session), ["Как поступить?"])
    monkeypatch.setattr(popular_statistics, "REFRESH_TIMEOUT", 0.02)
    factory = async_sessionmaker[AsyncSession](test_engine)
    embedder = Slow()
    first = PopularStatistics(redis, factory, embedder=embedder)
    second = PopularStatistics(redis, factory, embedder=embedder)
    task = asyncio.create_task(first.refresh())
    try:
        await asyncio.wait_for(started.wait(), 2)
        async with asyncio.timeout(2):
            while not task.cancelling():
                await asyncio.sleep(0.001)
        assert not task.done()
        assert await redis.get(first._key(500, 0.86) + ":lock") is not None
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert await redis.get(first._key(500, 0.86) + ":lock") is not None
        await second.refresh()
        assert embedder.calls == 1
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert await redis.get(first._key(500, 0.86) + ":lock") is None
