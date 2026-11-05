"""Регрессии для удаления факультета и конфликтов имён на реальной БД."""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.dependencies import require_admin
from api.main import app
from api.routes.admission_score import get_admission_score_service
from api.routes.faculty import get_faculty_service
from db.postgres.models import AdmissionScore, Program
from db.postgres.services.admission_score import AdmissionScoreService
from db.postgres.services.faculty import FacultyService


@pytest.mark.asyncio
async def test_delete_faculty_preserves_program_and_scores(session: AsyncSession):
    service = FacultyService(session)
    faculty = await service.create_faculty("Факультет для удаления")
    program = await service.create_program(faculty.id, "Программная инженерия")
    score = AdmissionScore(
        program_id=program.id, year=2025, form="budget", passing_score=250
    )
    session.add(score)
    await session.commit()

    assert await service.delete_faculty(faculty.id) is True

    retained_faculty = await service.get_faculty_by_id(faculty.id)
    assert retained_faculty is not None
    assert retained_faculty.is_active is False
    assert await service.find_faculty_by_alias("Факультет для удаления") is None
    assert (
        await session.scalar(select(Program).where(Program.id == program.id))
    ) is not None
    retained_score = await session.scalar(
        select(AdmissionScore).where(AdmissionScore.id == score.id)
    )
    assert retained_score is not None
    assert retained_score.passing_score == 250

    await service.update_faculty(faculty.id, is_active=True)
    assert await service.find_faculty_by_alias("Факультет для удаления") is not None
    assert await service.delete_faculty("missing") is False


@pytest.mark.asyncio
async def test_duplicate_faculty_name_returns_conflict(session: AsyncSession):
    app.dependency_overrides[require_admin] = lambda: object()
    app.dependency_overrides[get_faculty_service] = lambda: FacultyService(session)
    try:
        async with AsyncClient(
            transport=ASGITransport(app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            first = await client.post(
                "/api/v1/faculties", json={"name": "Факультет-дубликат"}
            )
            assert first.status_code == 201
            second = await client.post(
                "/api/v1/faculties", json={"name": "Другой факультет"}
            )
            assert second.status_code == 201

            duplicate = await client.post(
                "/api/v1/faculties", json={"name": "Факультет-дубликат"}
            )
            assert duplicate.status_code == 409
            rename = await client.put(
                f"/api/v1/faculties/{second.json()['id']}",
                json={"name": "Факультет-дубликат"},
            )
            assert rename.status_code == 409
            assert (await client.get("/api/v1/faculties")).status_code == 200
    finally:
        app.dependency_overrides.pop(get_faculty_service, None)
        app.dependency_overrides.pop(require_admin, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("lookup", ["all", "faculty", "owner"])
async def test_active_program_search_tracks_faculty_status(session: AsyncSession, lookup):
    service = FacultyService(session)
    faculty = await service.create_faculty("Тестовый факультет", aliases=["ТФ"])
    active = await service.create_program(faculty.id, "Активная программа")
    inactive = await service.create_program(
        faculty.id, "Неактивная программа", is_active=False
    )

    async def search():
        if lookup == "owner":
            return await service.find_faculty_for_program(active.name)
        if lookup == "faculty":
            return list(await service.get_programs_by_faculty(faculty.id))
        return list(await service.get_all_programs())

    expected = faculty if lookup == "owner" else [active]
    assert await search() == expected
    await service.delete_faculty(faculty.id)
    assert await search() == (None if lookup == "owner" else [])
    assert {p.id for p in await service.get_all_programs(only_active=False)} == {
        active.id,
        inactive.id,
    }
    assert {
        p.id for p in await service.get_programs_by_faculty(faculty.id, only_active=False)
    } == {active.id, inactive.id}
    retained_active = await service.get_program_by_id(active.id)
    retained_inactive = await service.get_program_by_id(inactive.id)
    assert retained_active is not None and retained_active.is_active is True
    assert retained_inactive is not None and retained_inactive.is_active is False
    await service.update_faculty(faculty.id, is_active=True)
    assert await search() == expected
    assert await service.find_faculty_for_program(inactive.name) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("filters", [{}, {"faculty": "ТФ"}, {"program": "Информатика"}])
async def test_score_search_tracks_faculty_status(session: AsyncSession, filters):
    faculties = FacultyService(session)
    faculty = await faculties.create_faculty("Тестовый факультет", aliases=["ТФ"])
    program = await faculties.create_program(faculty.id, "Информатика")
    session.add(
        AdmissionScore(program_id=program.id, year=2025, form="budget", passing_score=250)
    )
    await session.commit()
    service = AdmissionScoreService(session)
    before = await service.query_scores(**filters)
    assert len(before) == 1
    await faculties.delete_faculty(faculty.id)
    assert await service.query_scores(**filters) == []
    assert await service.query_scores(**filters, only_active=False) == before
    app.dependency_overrides[require_admin] = lambda: object()
    app.dependency_overrides[get_admission_score_service] = lambda: service
    try:
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://test"
        ) as client:
            history = await client.get("/api/v1/admission-scores", params=filters)
            assert history.status_code == 200
            assert history.json() == before
            active_only = await client.get(
                "/api/v1/admission-scores", params={**filters, "only_active": "true"}
            )
            assert active_only.status_code == 200
            assert active_only.json() == []
    finally:
        app.dependency_overrides.pop(get_admission_score_service, None)
        app.dependency_overrides.pop(require_admin, None)
    await faculties.update_faculty(faculty.id, is_active=True)
    assert await service.query_scores(**filters) == before
    await faculties.update_program(program.id, is_active=False)
    assert await service.query_scores(**filters) == []
    assert await service.query_scores(**filters, only_active=False) == before
