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

    await asyncio.wait_for(check(), timeout=3)


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

    dsn = str(test_engine.url).replace("postgresql+asyncpg://", "postgresql://", 1)
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
            await FaqDbService(session).delete(second.id)
        await wait_for_answer(matcher, "Когда подать документы?", None)
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
