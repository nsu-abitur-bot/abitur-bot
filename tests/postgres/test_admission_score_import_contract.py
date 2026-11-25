"""Контракт импорта с настоящим обработчиком, сервисом и PostgreSQL."""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.dependencies import require_admin
from api.main import app
from api.routes.admission_score import import_scores
from api.schemas.admission_score import ImportRequest, ScoreRowSchema
from db.postgres.db import get_async_session
from db.postgres.models import AdmissionScore
from db.postgres.services.admission_score import AdmissionScoreService
from db.postgres.services.faculty import FacultyService


async def _seed_program(session: AsyncSession) -> None:
    await FacultyService(session).upsert_from_seed(
        [
            {
                "name": "Faculty",
                "aliases": ["F"],
                "programs": [{"name": "Program", "level": "bachelor"}],
            }
        ]
    )


def _row(level: str = "bachelor", year: int = 2024) -> ScoreRowSchema:
    return ScoreRowSchema(
        faculty_name=" f ",
        program_name=" PROGRAM ",
        year=year,
        admission_basis=" Budget ",
        passing_score=250,
        level=level,
    )


@pytest.mark.parametrize("level", ["", "unknown", " bachelor ", "бакалавриат"])
async def test_import_preserves_level_normalization(
    session: AsyncSession, level: str
) -> None:
    """Неизвестный уровень исторически снимает фильтр, а не теряет строку."""
    await _seed_program(session)

    response = await import_scores(
        ImportRequest(rows=[_row(level)]), AdmissionScoreService(session)
    )

    assert response.model_dump() == {"created": 1, "updated": 0, "skipped": 0}
    score = (await session.execute(select(AdmissionScore))).scalar_one()
    assert score.passing_score == 250
    assert score.admission_basis == "budget"


async def test_import_normalized_duplicates_update_one_record(
    session: AsyncSession,
) -> None:
    await _seed_program(session)
    second = _row().model_copy(
        update={
            "faculty_name": "Faculty",
            "program_name": "Program",
            "admission_basis": "budget",
        }
    )
    second.passing_score = None

    response = await import_scores(
        ImportRequest(rows=[_row(), second]), AdmissionScoreService(session)
    )

    assert response.model_dump() == {"created": 1, "updated": 1, "skipped": 0}
    score = (await session.execute(select(AdmissionScore))).scalar_one()
    assert score.passing_score == 250


async def test_import_database_error_rolls_back_autoflushed_rows(
    session: AsyncSession,
) -> None:
    await _seed_program(session)

    with pytest.raises(DBAPIError):
        await import_scores(
            ImportRequest(rows=[_row(year=2024), _row(year=2**40)]),
            AdmissionScoreService(session),
        )
    await session.rollback()

    count = await session.scalar(select(func.count()).select_from(AdmissionScore))
    assert count == 0


async def test_v1_import_filter_and_confirm_study_form(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _seed_program(session)
    monkeypatch.setitem(app.dependency_overrides, require_admin, lambda: object())
    monkeypatch.setitem(app.dependency_overrides, get_async_session, lambda: session)
    path = "/api/v1/admission-scores"
    common = {"faculty_name": "Faculty", "program_name": "Program", "year": 2024}
    combinations = {
        ("full_time", "budget"): 251,
        ("full_time", "paid"): 202,
        ("correspondence", "budget"): 223,
        ("correspondence", "paid"): 184,
    }
    rows = [
        {
            **common,
            "study_form": study_form,
            "admission_basis": basis,
            "passing_score": score,
        }
        for (study_form, basis), score in combinations.items()
    ]
    legacy = {**common, "form": " Budget ", "passing_score": 211.5}

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"{path}/import", json={"rows": [*rows, legacy, {**legacy, "year": 2023}]}
        )
        assert response.status_code == 200, response.text
        assert response.json() == {"created": 6, "updated": 0, "skipped": 0}

        result = await client.post(
            f"{path}/import", json={"rows": [{**legacy, "passing_score": None}]}
        )
        assert result.json() == {"created": 0, "updated": 1, "skipped": 0}
        result = await client.get(path, params={"year": 2024})
        scores = result.json()
        assert len(scores) == 5
        assert {
            (row["study_form"], row["admission_basis"]): row["passing_score"]
            for row in scores
        } == {**combinations, (None, "budget"): 211.5}
        assert all(row["form"] == row["admission_basis"] for row in scores)

        filtered = await client.get(
            path, params={"year": 2024, "study_form": "correspondence", "form": "paid"}
        )
        assert [row["passing_score"] for row in filtered.json()] == [184]
        filtered = await client.get(
            path, params={"study_form": "full_time", "admission_basis": "budget"}
        )
        assert [row["passing_score"] for row in filtered.json()] == [251]

        unknown = next(row for row in scores if row["study_form"] is None)
        conflict = await client.patch(
            f"{path}/{unknown['id']}", json={"study_form": "full_time"}
        )
        assert conflict.status_code == 409

        old = (await client.get(path, params={"year": 2023})).json()[0]
        confirmed = await client.patch(
            f"{path}/{old['id']}", json={"study_form": "correspondence"}
        )
        assert confirmed.status_code == 200
        assert confirmed.json() == {**old, "study_form": "correspondence"}
        assert (await client.get(path, params={"year": 2023})).json() == [
            confirmed.json()
        ]

        updated = await client.patch(
            f"{path}/{old['id']}", json={"admission_basis": "paid"}
        )
        assert updated.status_code == 200
        assert updated.json() == {
            **confirmed.json(),
            "admission_basis": "paid",
            "form": "paid",
        }
        reset = await client.patch(f"{path}/{old['id']}", json={"study_form": None})
        assert reset.status_code == 200
        assert reset.json() == {**updated.json(), "study_form": None}

        conflict = await client.get(
            path, params={"form": "paid", "admission_basis": "budget"}
        )
        assert conflict.status_code == 422
        conflict = await client.post(
            f"{path}/import", json={"rows": [{**legacy, "admission_basis": "paid"}]}
        )
        assert conflict.status_code == 422


async def test_v1_seat_count_import_edit_and_legacy_rows(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _seed_program(session)
    monkeypatch.setitem(app.dependency_overrides, require_admin, lambda: object())
    monkeypatch.setitem(app.dependency_overrides, get_async_session, lambda: session)
    path = "/api/v1/admission-scores"
    common = {"faculty_name": "Faculty", "program_name": "Program", "year": 2025}
    budget = {**common, "admission_basis": "budget", "study_form": "full_time"}
    paid = {**budget, "admission_basis": "paid"}
    source_url = "https://example.test/admission/2025"
    rows = [
        {**budget, "seat_count": 0, "source_url": source_url},
        {**paid, "seat_count": 12},
        {**budget, "study_form": "correspondence", "seat_count": 5},
        {**budget, "year": 2024, "seat_count": 7},
        {**common, "form": "budget", "passing_score": 241.5},
    ]

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(f"{path}/import", json={"rows": rows})
        assert response.json() == {"created": 5, "updated": 0, "skipped": 0}
        response = await client.post(
            f"{path}/import", json={"rows": [{**budget}, {**paid, "seat_count": None}]}
        )
        assert response.json() == {"created": 0, "updated": 2, "skipped": 0}

        items = (await client.get(path)).json()
        assert {
            (item["year"], item["study_form"], item["admission_basis"]): item[
                "seat_count"
            ]
            for item in items
        } == {
            (2024, "full_time", "budget"): 7,
            (2025, "full_time", "budget"): 0,
            (2025, "full_time", "paid"): 12,
            (2025, "correspondence", "budget"): 5,
            (2025, None, "budget"): None,
        }
        zero = next(item for item in items if item["seat_count"] == 0)
        assert zero["source_url"] == source_url
        assert zero["passing_score"] is None and zero["average_score"] is None
        legacy = next(item for item in items if item["study_form"] is None)
        assert legacy["passing_score"] == 241.5

        patch_path = f"{path}/{zero['id']}"
        for invalid in (-1, 1.5, True, "12", 2**31):
            payload = {"seat_count": invalid}
            assert (await client.patch(patch_path, json=payload)).status_code == 422
            result = await client.post(
                f"{path}/import", json={"rows": [{**budget, **payload}]}
            )
            assert result.status_code == 422

        updated_source = "https://example.test/updated"
        result = await client.patch(
            patch_path, json={"seat_count": 17, "source_url": updated_source}
        )
        assert result.json() == {**zero, "seat_count": 17, "source_url": updated_source}
        result = await client.patch(patch_path, json={"seat_count": None})
        assert result.json() == {**zero, "seat_count": None, "source_url": updated_source}
