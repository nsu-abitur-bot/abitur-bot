"""Проверка сохранения уровня через API на реальной тестовой PostgreSQL."""

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.dependencies import require_admin
from api.main import app
from api.routes.admission_score import get_admission_score_service
from api.routes.faculty import get_faculty_service
from db.postgres.services.admission_score import AdmissionScoreService, ScoreRow
from db.postgres.services.faculty import FacultyService


@pytest.mark.asyncio
async def test_scores_api_preserves_program_level(session: AsyncSession, monkeypatch):
    faculties = FacultyService(session)
    faculty = await faculties.create_faculty("Колледж и институт")
    scores = AdmissionScoreService(session)
    for level in ("spo", "bachelor"):
        await faculties.create_program(faculty.id, "Информатика", level=level)
    await scores.upsert_from_rows(
        [
            ScoreRow(
                faculty_name=faculty.name,
                program_name="Информатика",
                code=None,
                level=level,
                year=2025,
                form="budget",
                passing_score=score,
                average_score=None,
            )
            for level, score in (("spo", 5), ("bachelor", 250))
        ]
    )
    monkeypatch.setitem(app.dependency_overrides, require_admin, lambda: None)
    monkeypatch.setitem(
        app.dependency_overrides, get_admission_score_service, lambda: scores
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/admission-scores")
    assert response.status_code == 200
    assert {(row["level"], row["passing_score"]) for row in response.json()} == {
        ("spo", 5),
        ("bachelor", 250),
    }


@pytest.mark.asyncio
async def test_spo_api_roundtrip(session: AsyncSession, monkeypatch):
    service = FacultyService(session)
    monkeypatch.setitem(app.dependency_overrides, require_admin, lambda: None)
    monkeypatch.setitem(app.dependency_overrides, get_faculty_service, lambda: service)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        faculty = await client.post("/api/v1/faculties", json={"name": "Колледж"})
        assert faculty.status_code == 201
        faculty_id = faculty.json()["id"]
        created = await client.post(
            f"/api/v1/faculties/{faculty_id}/programs",
            json={"name": "Информационные системы", "level": "spo"},
        )
        assert created.status_code == 201
        program_id = created.json()["id"]
        assert created.json()["level"] == "spo"
        assert created.json()["code"] is None

        response = await client.put(
            f"/api/v1/faculties/programs/{program_id}",
            json={"code": "09.02.07", "level": "spo"},
        )
        assert response.status_code == 200
        assert response.json()["level"] == "spo"

        session.expunge_all()
        reopened = await client.get("/api/v1/faculties")
        assert reopened.status_code == 200
        program = reopened.json()[0]["programs"][0]
        assert program["id"] == program_id
        assert program["level"] == "spo"
        assert program["code"] == "09.02.07"
