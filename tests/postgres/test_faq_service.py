"""
Тесты для FaqDbService.
"""

from datetime import datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.faq import FaqItem
from api.services.faq import FaqService
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
async def test_upsert_many_faq_entries(session: AsyncSession):
    service = FaqDbService(session)

    entries, created, updated = await service.upsert_many(
        [
            {"question": "Q1", "aliases": ["a1"], "answer": "A1"},
            {"question": "Q2", "aliases": ["a2"], "answer": "A2"},
        ]
    )

    all_entries = await service.get_all()
    assert len(entries) == 2
    assert (created, updated) == (2, 0)
    assert len(all_entries) == 2


@pytest.mark.asyncio
async def test_upsert_many_reupload_updates_instead_of_duplicates(session: AsyncSession):
    service = FaqDbService(session)
    items = [
        {"question": "Вопрос 1", "aliases": [], "answer": "Ответ 1"},
        {"question": "Вопрос 2", "aliases": [], "answer": "Ответ 2"},
    ]

    entries, created, updated = await service.upsert_many(items)
    assert (created, updated) == (2, 0)

    reupload = [
        {"question": "Вопрос 1", "aliases": [], "answer": "Ответ 1 (обновлён)"},
        {"question": "Вопрос 2", "aliases": [], "answer": "Ответ 2"},
    ]
    entries, created, updated = await service.upsert_many(reupload)

    assert (created, updated) == (0, 2)
    all_entries = await service.get_all()
    assert len(all_entries) == 2
    answers = {entry.question: entry.answer for entry in all_entries}
    assert answers["Вопрос 1"] == "Ответ 1 (обновлён)"
    assert answers["Вопрос 2"] == "Ответ 2"


@pytest.mark.asyncio
async def test_upsert_many_case_and_spacing_variants_are_same_question(
    session: AsyncSession,
):
    service = FaqDbService(session)

    entries, created, updated = await service.upsert_many(
        [{"question": "Как поступить в НГУ?", "aliases": [], "answer": "Через конкурс."}]
    )
    assert (created, updated) == (1, 0)

    entries, created, updated = await service.upsert_many(
        [
            {
                "question": "  как поступить в нгу? ",
                "aliases": [],
                "answer": "Обновлённый ответ.",
            }
        ]
    )

    assert (created, updated) == (0, 1)
    all_entries = await service.get_all()
    assert len(all_entries) == 1
    assert all_entries[0].normalized_question == "как поступить в нгу?"
    assert all_entries[0].answer == "Обновлённый ответ."


@pytest.mark.asyncio
async def test_upsert_many_preserves_verification_metadata(session: AsyncSession):
    service = FaqDbService(session)
    confirmed_at = datetime(2026, 11, 5, 10, 0)
    created = await service.create(
        question="Как подать документы?",
        aliases=[],
        answer="Через личный кабинет.",
    )
    await service.update(
        item_id=created.id,
        question="Как подать документы?",
        aliases=[],
        answer="Через личный кабинет.",
        metadata={
            "verification_status": "confirmed",
            "confirmed_by": "Иванова",
            "confirmed_at": confirmed_at,
        },
    )

    entries, created_count, updated_count = await service.upsert_many(
        [{"question": "Как подать документы?", "aliases": [], "answer": "Другой ответ."}]
    )

    assert (created_count, updated_count) == (0, 1)
    assert len(entries) == 1
    assert entries[0].answer == "Другой ответ."
    assert entries[0].verification_status == "confirmed"
    assert entries[0].confirmed_by == "Иванова"
    assert entries[0].confirmed_at == confirmed_at


@pytest.mark.asyncio
async def test_upsert_many_last_wins_within_batch(session: AsyncSession):
    service = FaqDbService(session)

    entries, created, updated = await service.upsert_many(
        [
            {"question": "Один вопрос", "aliases": [], "answer": "Первый ответ"},
            {"question": "Один вопрос", "aliases": [], "answer": "Второй ответ"},
        ]
    )

    assert (created, updated) == (1, 0)
    assert len(entries) == 1
    assert entries[0].answer == "Второй ответ"


@pytest.mark.asyncio
async def test_create_duplicate_question_raises_value_error(session: AsyncSession):
    service = FaqDbService(session)
    await service.create(question="Дубликат", aliases=[], answer="Первый ответ")

    with pytest.raises(ValueError):
        await service.create(question="Дубликат", aliases=[], answer="Второй ответ")


@pytest.mark.asyncio
async def test_faq_verification_metadata_survives_legacy_update(session: AsyncSession):
    service = FaqService(session)
    confirmed_at = datetime(2026, 11, 4, 12, 0)
    created = await service.create(
        FaqItem(
            question="Когда подать документы?",
            answer="До 20 июля.",
            verification_status="confirmed",
            source_document="Правила приёма 2026",
            source_location="пункт 12",
            confirmed_by="Иванова",
            confirmed_at=confirmed_at,
        )
    )

    updated = await service.update(
        created.id or "", FaqItem(question="Когда подать?", answer="До 20 июля.")
    )

    assert updated.verification_status == "confirmed"
    assert updated.source_document == "Правила приёма 2026"
    assert updated.source_location == "пункт 12"
    assert updated.confirmed_by == "Иванова"
    assert updated.confirmed_at == confirmed_at
    assert (await service.get_all())[0] == updated


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
