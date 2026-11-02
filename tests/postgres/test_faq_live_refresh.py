"""Changes committed by the API reach the running bot through PostgreSQL."""

import asyncio
import hashlib
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

import faq.matcher as matcher_module
from db.postgres.services.faq import FaqDbService
from faq.matcher import FAQMatcher
from faq.updates import watch_faq_updates


class ExactEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(byte) for byte in hashlib.sha256(text.casefold().encode()).digest()]
            for text in texts
        ]


async def wait_for_answer(matcher: FAQMatcher, question: str, answer: str | None) -> None:
    async def check() -> None:
        while await matcher.match_async(question) != answer:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(check(), timeout=10)


@pytest.mark.asyncio
async def test_running_matcher_receives_committed_faq_changes(
    test_engine: AsyncEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_factory = async_sessionmaker(test_engine, expire_on_commit=False)
    monkeypatch.setattr(matcher_module, "AsyncSessionLocal", session_factory)
    matcher = FAQMatcher(
        faq_path=tmp_path / "missing.yaml",
        threshold=0.99,
        embedder=ExactEmbeddings(),
    )
    async with session_factory() as session:
        first = await FaqDbService(session).create(
            "Кто такой Иртегов?", [], "Преподаватель ФИТ НГУ."
        )

    dsn = test_engine.url.set(drivername="postgresql").render_as_string(
        hide_password=False
    )
    watcher = asyncio.create_task(watch_faq_updates(matcher, dsn))
    try:
        await wait_for_answer(matcher, "Кто такой Иртегов?", "Преподаватель ФИТ НГУ.")

        async with session_factory() as session:
            second = await FaqDbService(session).create(
                "Когда подать документы?", [], "До 25 июля."
            )
        await wait_for_answer(matcher, "Когда подать документы?", "До 25 июля.")

        async with session_factory() as session:
            await FaqDbService(session).update(
                first.id, "Кто такой Иртегов?", [], "Преподаватель НГУ."
            )
        await wait_for_answer(matcher, "Кто такой Иртегов?", "Преподаватель НГУ.")

        async with session_factory() as session:
            await FaqDbService(session).update(
                first.id, "Кто такой Иртегов?", ["Кто ведёт сети?"], "Преподаватель НГУ."
            )
        await wait_for_answer(matcher, "Кто ведёт сети?", "Преподаватель НГУ.")
        async with session_factory() as session:
            await FaqDbService(session).update(
                first.id, "Кто такой Иртегов?", [], "Преподаватель НГУ."
            )
        await wait_for_answer(matcher, "Кто ведёт сети?", None)

        async with session_factory() as session:
            await FaqDbService(session).delete(second.id)
        await wait_for_answer(matcher, "Когда подать документы?", None)
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_refresh_cannot_publish_over_newer_snapshot(
    test_engine: AsyncEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class BlockingEmbeddings(ExactEmbeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            if "old question" in texts:
                started.set()
                assert release.wait(10)
            return super().embed_documents(texts)

    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    monkeypatch.setattr(matcher_module, "AsyncSessionLocal", factory)
    matcher = FAQMatcher(
        faq_path=tmp_path / "missing.yaml", embedder=BlockingEmbeddings()
    )
    builder_name = (
        "_build_index" if hasattr(matcher, "_build_index") else "_process_items"
    )
    build = getattr(matcher, builder_name)

    def tracked_build(items: list[dict]) -> object:
        try:
            return build(items)
        finally:
            if items[0]["question"] == "old question":
                finished.set()

    monkeypatch.setattr(matcher, builder_name, tracked_build)
    async with factory() as session:
        entry = await FaqDbService(session).create("old question", [], "STALE")
    old = asyncio.create_task(matcher.refresh_from_db())
    try:
        assert await asyncio.to_thread(started.wait, 10)
        old.cancel()
        with pytest.raises(asyncio.CancelledError):
            await old
        async with factory() as session:
            await FaqDbService(session).update(entry.id, "new question", [], "CURRENT")
        await matcher.refresh_from_db()
        assert await matcher.match_async("new question") == "CURRENT"
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 10)
    assert await matcher.match_async("new question") == "CURRENT"
    assert matcher._items_signature == (("new question", (), "CURRENT"),)


@pytest.mark.asyncio
async def test_refresh_reuses_unchanged_and_preserves_last_good_on_failure(
    test_engine: AsyncEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlalchemy import text
    from sqlalchemy.exc import ProgrammingError

    class FailingEmbeddings(ExactEmbeddings):
        fail = False
        calls = 0

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            if self.fail:
                raise TimeoutError("synthetic document transport failure")
            return super().embed_documents(texts)

    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    monkeypatch.setattr(matcher_module, "AsyncSessionLocal", factory)
    embeddings = FailingEmbeddings()
    matcher = FAQMatcher(faq_path=tmp_path / "missing.yaml", embedder=embeddings)
    async with factory() as session:
        entry = await FaqDbService(session).create("known question", [], "last good")
    await matcher.refresh_from_db()
    index = matcher._index
    signature = matcher._items_signature
    calls = embeddings.calls
    await asyncio.gather(*(matcher.refresh_from_db() for _ in range(3)))
    assert matcher._index is index
    assert embeddings.calls == calls

    async with factory() as session:
        await FaqDbService(session).update(entry.id, "new question", [], "current")
    embeddings.fail = True
    with pytest.raises(TimeoutError):
        await matcher.refresh_from_db()
    assert matcher._index is index
    assert matcher._items_signature == signature
    embeddings.fail = False
    assert await matcher.match_async("known question") == "last good"

    async with test_engine.begin() as connection:
        await connection.execute(text("ALTER TABLE faq_entry RENAME TO faq_unavailable"))
    try:
        with pytest.raises(ProgrammingError):
            await matcher.refresh_from_db()
        assert matcher._index is index
        assert matcher._items_signature == signature
        assert await matcher.match_async("known question") == "last good"
    finally:
        async with test_engine.begin() as connection:
            await connection.execute(
                text("ALTER TABLE faq_unavailable RENAME TO faq_entry")
            )
    await matcher.refresh_from_db()
    assert await matcher.match_async("new question") == "current"


@pytest.mark.asyncio
async def test_watcher_failure_does_not_delay_cached_answers_and_retries(
    test_engine: AsyncEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    started = threading.Event()
    release = threading.Event()

    class BlockingFailure(ExactEmbeddings):
        fail = True

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            if "new question" in texts and self.fail:
                started.set()
                assert release.wait(10)
                raise TimeoutError("synthetic document transport failure")
            return super().embed_documents(texts)

    factory = async_sessionmaker(test_engine, expire_on_commit=False)
    monkeypatch.setattr(matcher_module, "AsyncSessionLocal", factory)
    embeddings = BlockingFailure()
    matcher = FAQMatcher(faq_path=tmp_path / "missing.yaml", embedder=embeddings)
    async with factory() as session:
        await FaqDbService(session).create("known question", [], "last good")
    dsn = test_engine.url.set(drivername="postgresql").render_as_string(
        hide_password=False
    )
    watcher = asyncio.create_task(watch_faq_updates(matcher, dsn))
    try:
        await wait_for_answer(matcher, "known question", "last good")
        async with factory() as session:
            await FaqDbService(session).create("new question", [], "current")
        assert await asyncio.to_thread(started.wait, 10)
        pending_answers = asyncio.gather(
            *(matcher.match_async("known question") for _ in range(3))
        )
        answers = await asyncio.wait_for(pending_answers, timeout=1)
        assert answers == ["last good"] * 3
        release.set()
        embeddings.fail = False
        await wait_for_answer(matcher, "new question", "current")
    finally:
        release.set()
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
