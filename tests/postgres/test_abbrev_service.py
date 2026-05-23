"""
Тесты для AbbrevDbService.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.services.abbrev import AbbrevDbService


@pytest.mark.asyncio
async def test_create_abbreviation(session: AsyncSession):
    service = AbbrevDbService(session)

    entry = await service.create(
        short="ИТ",
        full="Информационные технологии",
    )

    assert entry.id is not None
    assert entry.short == "ИТ"
    assert entry.full == "Информационные технологии"


@pytest.mark.asyncio
async def test_create_duplicate_short_raises_value_error(session: AsyncSession):
    service = AbbrevDbService(session)
    await service.create(short="ФИТ", full="Факультет информационных технологий")

    with pytest.raises(ValueError, match="уже существует"):
        await service.create(short="ФИТ", full="Другое значение")


@pytest.mark.asyncio
async def test_get_abbreviation(session: AsyncSession):
    service = AbbrevDbService(session)
    created = await service.create(short="ММФ", full="Механико-математический факультет")

    found = await service.get(created.id)

    assert found is not None
    assert found.short == "ММФ"
    assert found.full == "Механико-математический факультет"


@pytest.mark.asyncio
async def test_update_abbreviation(session: AsyncSession):
    service = AbbrevDbService(session)
    created = await service.create(short="ФЕН", full="Старое значение")

    updated = await service.update(
        item_id=created.id,
        short="ФЕН",
        full="Факультет естественных наук",
    )

    assert updated is not None
    assert updated.short == "ФЕН"
    assert updated.full == "Факультет естественных наук"


@pytest.mark.asyncio
async def test_delete_abbreviation(session: AsyncSession):
    service = AbbrevDbService(session)
    created = await service.create(short="ГГФ", full="Геолого-геофизический факультет")

    deleted = await service.delete(created.id)
    found = await service.get(created.id)

    assert deleted is True
    assert found is None


@pytest.mark.asyncio
async def test_create_rolls_back_on_unexpected_commit_error(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    service = AbbrevDbService(session)
    rollback_called = False
    original_rollback = session.rollback

    async def commit_fail() -> None:
        raise RuntimeError("unexpected failure")

    async def rollback_spy() -> None:
        nonlocal rollback_called
        rollback_called = True
        await original_rollback()

    monkeypatch.setattr(session, "commit", commit_fail)
    monkeypatch.setattr(session, "rollback", rollback_spy)

    with pytest.raises(RuntimeError, match="unexpected failure"):
        await service.create(short="ФЛФ", full="Филологический факультет")

    assert rollback_called is True


@pytest.mark.asyncio
async def test_update_rolls_back_on_unexpected_commit_error(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    service = AbbrevDbService(session)
    entry = await service.create(short="ЮФ", full="Юридический факультет")
    rollback_called = False
    original_rollback = session.rollback

    async def commit_fail() -> None:
        raise RuntimeError("unexpected failure")

    async def rollback_spy() -> None:
        nonlocal rollback_called
        rollback_called = True
        await original_rollback()

    monkeypatch.setattr(session, "commit", commit_fail)
    monkeypatch.setattr(session, "rollback", rollback_spy)

    with pytest.raises(RuntimeError, match="unexpected failure"):
        await service.update(
            item_id=entry.id,
            short="ЮФ",
            full="Новое значение",
        )

    assert rollback_called is True
