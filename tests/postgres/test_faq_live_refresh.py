"""FAQ added in the admin process must reach a running bot matcher."""

import hashlib
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

import faq.matcher as matcher_module
from db.postgres.services.faq import FaqDbService
from faq.matcher import FAQMatcher


class ExactEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(byte) for byte in hashlib.sha256(text.casefold().encode()).digest()]
            for text in texts
        ]


@pytest.mark.asyncio
async def test_running_matcher_sees_faq_create_update_and_delete(
    test_engine: AsyncEngine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_factory = async_sessionmaker(test_engine, expire_on_commit=False)
    monkeypatch.setattr(matcher_module, "AsyncSessionLocal", session_factory)
    matcher = FAQMatcher(
        faq_path=tmp_path / "missing.yaml",
        threshold=0.99,
        embedder=ExactEmbeddings(),
        sync_from_db=True,
    )

    assert await matcher.match_async("Кто такой Иртегов?") is None

    async with session_factory() as session:
        service = FaqDbService(session)
        entry = await service.create(
            "Кто такой Иртегов?",
            ["Кто такой Дмитрий Валентинович Иртегов?"],
            "Преподаватель ФИТ НГУ.",
        )

    assert await matcher.match_async("Кто такой Иртегов?") == "Преподаватель ФИТ НГУ."
    assert (
        await matcher.match_async("Кто такой Дмитрий Валентинович Иртегов?")
        == "Преподаватель ФИТ НГУ."
    )

    async with session_factory() as session:
        service = FaqDbService(session)
        await service.update(entry.id, "Кто такой Иртегов?", [], "Преподаватель НГУ.")

    assert await matcher.match_async("Кто такой Иртегов?") == "Преподаватель НГУ."

    async with session_factory() as session:
        await FaqDbService(session).delete(entry.id)

    assert await matcher.match_async("Кто такой Иртегов?") is None
