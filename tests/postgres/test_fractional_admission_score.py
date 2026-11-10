import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.admission_score import ScoreItem, ScoreRowSchema
from db.postgres.models import Faculty, Program
from db.postgres.services.admission_score import AdmissionScoreService, ScoreRow


@pytest.mark.asyncio
async def test_college_score_round_trip(session: AsyncSession):
    faculty = Faculty(name="Университетский колледж")
    session.add(faculty)
    await session.flush()
    program = Program(
        faculty_id=faculty.id,
        name="Сетевое и системное администрирование",
        code="09.02.06",
        level="spo",
    )
    session.add(program)
    await session.commit()
    payload = ScoreRowSchema(
        faculty_name=faculty.name,
        program_name=program.name,
        code=program.code,
        level="spo",
        year=2024,
        form="budget",
        passing_score=4.43,
        average_score=4.42,
    )
    service = AdmissionScoreService(session)
    row = ScoreRow(**payload.model_dump())
    assert await service.upsert_from_rows([row]) == {
        "created": 1,
        "updated": 0,
        "skipped": 0,
    }
    session.expire_all()
    scores = await service.query_scores(year=2024, level="spo")
    result = ScoreItem(**scores[0])
    assert result.passing_score == 4.43
    assert result.average_score == 4.42
    assert await service.upsert_from_rows([row]) == {
        "created": 0,
        "updated": 1,
        "skipped": 0,
    }
