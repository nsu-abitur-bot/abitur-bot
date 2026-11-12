from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.security import create_access_token
from api.main import app
from db.postgres.db import get_async_session
from db.postgres.models import AdminRole, AdmissionScore
from db.postgres.services.admin import AdminService
from db.postgres.services.faculty import FacultyService
from db.postgres.services.feedback_report import FeedbackReportService

AdminClient = tuple[AsyncClient, dict[AdminRole, dict[str, str]]]


@pytest.fixture
async def admin_client(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[AdminClient]:
    headers = {}
    service = AdminService(session)
    for role in AdminRole:
        admin = await service.create(role.value, "unused-password-hash", role)
        headers[role] = {"Authorization": f"Bearer {create_access_token(admin.id)}"}
    await session.commit()
    monkeypatch.setitem(app.dependency_overrides, get_async_session, lambda: session)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client, headers


@pytest.mark.asyncio
async def test_feedback_viewer_can_read_but_cannot_change_status(
    session: AsyncSession, admin_client: AdminClient
) -> None:
    report = await FeedbackReportService(session).create_report(
        user_id=200002, session_id="review", channel="telegram", comment="Report"
    )
    path = f"/api/v1/feedback/{report.id}"
    client, headers = admin_client
    viewer = headers[AdminRole.viewer]
    payload = {"status": "reviewed"}

    assert (await client.get(path, headers=viewer)).status_code == 200
    assert (await client.patch(path, json=payload)).status_code == 401
    assert (await client.patch(path, json=payload, headers=viewer)).status_code == 403
    assert (await client.get(path, headers=viewer)).json()["status"] == "open"
    response = await client.patch(path, json=payload, headers=headers[AdminRole.admin])
    assert response.status_code == 200
    assert response.json()["status"] == "reviewed"


@pytest.mark.asyncio
@pytest.mark.parametrize("delete_program", [False, True], ids=["faculty", "program"])
async def test_directory_delete_requires_superadmin(
    session: AsyncSession, admin_client: AdminClient, delete_program: bool
) -> None:
    service = FacultyService(session)
    faculty = await service.create_faculty("Факультет")
    program = await service.create_program(faculty.id, "Направление")
    score = AdmissionScore(
        program_id=program.id, year=2025, form="budget", passing_score=250
    )
    session.add(score)
    await session.commit()
    score_id = score.id
    path = "/api/v1/faculties" + (
        f"/programs/{program.id}" if delete_program else f"/{faculty.id}"
    )
    client, headers = admin_client

    assert (await client.delete(path)).status_code == 401
    for role in (AdminRole.viewer, AdminRole.admin):
        assert (await client.delete(path, headers=headers[role])).status_code == 403

    await session.refresh(faculty)
    await session.refresh(score)
    assert faculty.is_active
    assert score.passing_score == 250

    response = await client.delete(path, headers=headers[AdminRole.superadmin])
    assert response.status_code == 204
    session.expunge_all()
    retained_score = await session.get(AdmissionScore, score_id)
    if delete_program:
        assert retained_score is None
    else:
        assert retained_score is not None
        retained_faculty = await service.get_faculty_by_id(faculty.id)
        assert retained_faculty is not None and not retained_faculty.is_active
