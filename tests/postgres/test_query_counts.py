"""Количество SQL-запросов в пакетных операциях (регрессия N+1, #300).

Предпросмотр и импорт проходных баллов, список факультетов и сохранение
результатов проверки документов ходили в БД в цикле — по 1–3 SELECT на
строку/факультет. Здесь фиксируется верхняя граница числа SELECT-ов для
этих операций. Тесты гоняются в реальной тестовой PostgreSQL
(см. tests/postgres/conftest.py), счётчик слушает before_cursor_execute.
"""

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.services.admission_score import AdmissionScoreService
from db.postgres.services.document import (
    DOCUMENT_STATUS_CHANGED,
    DOCUMENT_STATUS_UNCHANGED,
    DocumentService,
)
from db.postgres.services.faculty import FacultyService


class _SelectCounter:
    """Считает SELECT-запросы, ушедшие через движок сессии."""

    def __init__(self, session: AsyncSession) -> None:
        self.count = 0
        # Движок под async-сессией: событие before_cursor_execute — синхронное.
        self._engine = session.sync_session.bind

    def _on_cursor_execute(self, conn, cursor, statement, params, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            self.count += 1

    def __enter__(self) -> "_SelectCounter":
        event.listen(self._engine, "before_cursor_execute", self._on_cursor_execute)
        return self

    def __exit__(self, *exc_info) -> None:
        event.remove(self._engine, "before_cursor_execute", self._on_cursor_execute)


async def _seed_directory(session: AsyncSession, faculties: int, programs: int) -> None:
    """`faculties` факультетов по `programs` направлений, у каждого по алиасу."""
    service = FacultyService(session)
    await service.upsert_from_seed(
        [
            {
                "name": f"Факультет номер {index}",
                "aliases": [f"Ф{index}"],
                "programs": [
                    {
                        "name": f"Направление {program} факультета {index}",
                        "level": "bachelor",
                    }
                    for program in range(programs)
                ],
            }
            for index in range(faculties)
        ]
    )


async def _seed_documents(session: AsyncSession, count: int) -> list[str]:
    service = DocumentService(session)
    ids = []
    for index in range(count):
        document = await service.create_or_update_for_source(
            graph_id="test_graph",
            source_url=f"https://example.test/{index}",
            title=f"Документ {index}",
            content_hash=f"hash-{index}",
            content_length=100,
        )
        ids.append(document.id)
    return ids


@pytest.mark.asyncio
async def test_resolve_program_id_bounded_queries(session: AsyncSession):
    """Резолв строк предпросмотра: справочник грузится один раз, не по строке.

    Раньше каждая строка стоила два SELECT (все факультеты + направления
    факультета) — на сотнях строк импорт держал event loop минутами (#300).
    """
    await _seed_directory(session, faculties=3, programs=3)
    service = AdmissionScoreService(session)

    rows = [
        (f"Ф{faculty}", f"Направление {program} факультета {faculty}", "bachelor")
        for faculty in range(3)
        for program in range(3)
    ]

    with _SelectCounter(session) as counter:
        matched = [
            await service.resolve_program_id(faculty, program, level)
            for faculty, program, level in rows
        ]

    assert sum(pid is not None for pid in matched) == 9
    # Один справочник факультетов + один selectinload направлений на сервис.
    assert counter.count <= 3, f"N+1 вернулся: {counter.count} SELECT на 9 строк"


@pytest.mark.asyncio
async def test_faculty_directory_two_queries(session: AsyncSession):
    """Справочник факультетов с направлениями — два запроса вместо 1+N."""
    await _seed_directory(session, faculties=5, programs=4)
    service = FacultyService(session)

    with _SelectCounter(session) as counter:
        directory = await service.load_directory(
            faculties_only_active=False, programs_only_active=False
        )

    assert len(directory.faculties) == 5
    assert all(
        len(directory.programs_by_faculty[faculty.id]) == 4
        for faculty in directory.faculties
    )
    assert counter.count <= 2, f"N+1 вернулся: {counter.count} SELECT на 5 факультетов"


@pytest.mark.asyncio
async def test_save_check_results_single_batch(session: AsyncSession):
    """Результаты проверки пишутся одним IN-запросом, а не get_by_id в цикле."""
    document_ids = await _seed_documents(session, count=5)
    service = DocumentService(session)
    results = [
        {
            "id": document_id,
            "status": DOCUMENT_STATUS_UNCHANGED,
            "content_hash": f"hash-{index}",
            "message": None,
        }
        for index, document_id in enumerate(document_ids)
    ]

    with _SelectCounter(session) as counter:
        await service.save_check_results(results)

    assert counter.count <= 1, f"N+1 вернулся: {counter.count} SELECT на 5 документов"
    refreshed = {
        document.id: document for document in await service.get_by_ids(document_ids)
    }
    assert all(
        refreshed[document_id].last_checked_hash == f"hash-{index}"
        for index, document_id in enumerate(document_ids)
    )


@pytest.mark.asyncio
async def test_update_changed_documents_batches_lookups(session: AsyncSession):
    """Доработка изменённых документов берёт их одним IN-запросом."""
    from api.services.document_update import DocumentUpdateService

    document_ids = await _seed_documents(session, count=4)
    checks = [
        {
            "id": document_id,
            "title": f"Документ {index}",
            "source_url": f"https://example.test/{index}",
            "status": DOCUMENT_STATUS_CHANGED,
            "content_hash": f"new-hash-{index}",
            "previous_hash": f"hash-{index}",
            "message": None,
        }
        for index, document_id in enumerate(document_ids)
    ]

    update_service = DocumentUpdateService(session)

    async def fake_check_documents(document_ids=None):
        return checks

    async def fake_update_document(document, content_hash):
        return {"id": document.id, "status": "обновлён"}

    update_service.check_documents = fake_check_documents
    update_service._update_document = fake_update_document

    with _SelectCounter(session) as counter:
        results = await update_service.update_changed_documents()

    assert len(results) == 4
    assert counter.count <= 2, f"N+1 вернулся: {counter.count} SELECT на 4 документов"
