"""Контракт импорта с настоящим обработчиком, сервисом и PostgreSQL."""

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from api.routes.admission_score import import_scores
from api.schemas.admission_score import ImportRequest, ScoreRowSchema
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
        form=" Budget ",
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
    assert score.form == "budget"


async def test_import_normalized_duplicates_update_one_record(
    session: AsyncSession,
) -> None:
    await _seed_program(session)
    second = _row().model_copy(
        update={"faculty_name": "Faculty", "program_name": "Program", "form": "budget"}
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
