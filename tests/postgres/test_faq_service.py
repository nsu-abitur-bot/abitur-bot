"""
Тесты для FaqDbService.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.services.faq import FaqDbService


@pytest.mark.asyncio
async def test_create_faq_entry(session: AsyncSession):
    service = FaqDbService(session)

    entry = await service.create(
        question="Как поступить?",
        aliases=["поступление", "поступить"],
        answer="Подайте документы через ЛК.",
    )

    assert entry.id is not None
    assert entry.question == "Как поступить?"
    assert entry.aliases == ["поступление", "поступить"]
    assert entry.answer == "Подайте документы через ЛК."


@pytest.mark.asyncio
async def test_get_faq_entry(session: AsyncSession):
    service = FaqDbService(session)
    created = await service.create(
        question="Где общежитие?",
        aliases=[],
        answer="На кампусе НГУ.",
    )

    found = await service.get(created.id)

    assert found is not None
    assert found.question == "Где общежитие?"


@pytest.mark.asyncio
async def test_update_faq_entry(session: AsyncSession):
    service = FaqDbService(session)
    created = await service.create(
        question="Старый вопрос",
        aliases=["старый"],
        answer="Старый ответ",
    )

    updated = await service.update(
        item_id=created.id,
        question="Новый вопрос",
        aliases=["новый"],
        answer="Новый ответ",
    )

    assert updated is not None
    assert updated.question == "Новый вопрос"
    assert updated.aliases == ["новый"]
    assert updated.answer == "Новый ответ"


@pytest.mark.asyncio
async def test_delete_faq_entry(session: AsyncSession):
    service = FaqDbService(session)
    created = await service.create(
        question="Удалить вопрос",
        aliases=[],
        answer="Удалить ответ",
    )

    deleted = await service.delete(created.id)
    found = await service.get(created.id)

    assert deleted is True
    assert found is None


@pytest.mark.asyncio
async def test_create_many_faq_entries(session: AsyncSession):
    service = FaqDbService(session)

    entries = await service.create_many(
        [
            {"question": "Q1", "aliases": ["a1"], "answer": "A1"},
            {"question": "Q2", "aliases": ["a2"], "answer": "A2"},
        ]
    )

    all_entries = await service.get_all()
    assert len(entries) == 2
    assert len(all_entries) == 2


@pytest.mark.asyncio
async def test_create_rolls_back_on_commit_error(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    service = FaqDbService(session)
    rollback_called = False
    original_rollback = session.rollback

    async def commit_fail() -> None:
        raise RuntimeError("commit failed")

    async def rollback_spy() -> None:
        nonlocal rollback_called
        rollback_called = True
        await original_rollback()

    monkeypatch.setattr(session, "commit", commit_fail)
    monkeypatch.setattr(session, "rollback", rollback_spy)

    with pytest.raises(RuntimeError, match="commit failed"):
        await service.create(
            question="Rollback check",
            aliases=[],
            answer="Rollback answer",
        )

    assert rollback_called is True
