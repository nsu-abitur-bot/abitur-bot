"""Регрессии для удаления факультета и конфликтов имён на реальной БД."""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.dependencies import require_admin
from api.main import app
from api.routes.faculty import get_faculty_service
from db.postgres.models import AdmissionScore, Program
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
