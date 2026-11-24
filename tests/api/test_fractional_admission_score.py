import pytest

from api.schemas.admission_score import ScoreItem, ScoreRowSchema


@pytest.mark.parametrize("schema", [ScoreRowSchema, ScoreItem])
def test_college_passing_score_keeps_fraction(schema):
    row = schema(
        **({"id": "some-score"} if schema is ScoreItem else {}),
        faculty_name="Университетский колледж",
        program_name="Сетевое и системное администрирование",
        code="09.02.06",
        level="spo",
        year=2024,
        admission_basis="budget",
        passing_score=4.43,
        average_score=4.42,
    )
    assert row.model_dump()["passing_score"] == 4.43
