"""Регрессии поиска направлений с опечатками в вопросе и справочнике."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.models import Program
from db.postgres.services.admission_score import (
    AdmissionScoreService,
    ScoreRow,
    match_programs,
)
from db.postgres.services.faculty import FacultyService


@pytest.mark.parametrize(
    "program_name, query",
    [
        ("Программная инженерия", "програмная инженерия"),
        ("Програмная инженирия", "программная инженерия"),
        ("Програмная инженирия", "программную инженерию"),
        ("Компьютерные науки", "компютерные науки"),
        (
            "Информатика и вычислительная техника. Программная инженерия",
            "направление програмная инженерия, НГУ",
        ),
    ],
)
def test_match_programs_tolerates_typos(program_name: str, query: str) -> None:
    program = Program(id="program", name=program_name, code="09.03.01")

    assert match_programs([program], query) == [program]


@pytest.mark.parametrize(
    "query",
    ["програмная ветеринария", "ветеринария", "", "направление и НГУ"],
)
def test_match_programs_requires_every_significant_word(query: str) -> None:
    program = Program(id="program", name="Программная инженерия", code="09.03.01")

    assert match_programs([program], query) == []


@pytest.mark.parametrize(
    "program_name, query",
    [
        ("Программная инженерия", "программная инженерия"),
        ("Информатика. Программная инженерия", "программная инженерия"),
        ("Программная инженерия и компьютерные науки", "программная инженерия"),
        ("Программная инженерия и компьютерные науки", "инженерия программная"),
        ("Программная инженерия", "программной инженерии"),
    ],
)
def test_match_programs_stricter_tiers_win_over_fuzzy(
    program_name: str, query: str
) -> None:
    exact = Program(id="exact", name=program_name, code="09.03.01")
    fuzzy = Program(id="fuzzy", name="Програмная инженирия", code="09.03.01")

    assert match_programs([fuzzy, exact], query) == [exact]


def test_match_programs_fuzzy_respects_code_candidates() -> None:
    selected = Program(id="selected", name="Програмная инженирия", code="09.03.01")
    other_code = Program(id="other", name="Програмная инженирия", code="09.03.04")

    assert match_programs([other_code, selected], "программная инженерия 09.03.01") == [
        selected
    ]


@pytest.mark.asyncio
async def test_query_scores_finds_sibgiu_program_with_database_typos(
    session: AsyncSession,
) -> None:
    faculty_service = FacultyService(session)
    faculty = await faculty_service.create_faculty("Институт цифровых технологий")
    program_name = "Програмная инженирия"
    await faculty_service.create_program(
        faculty.id, program_name, code="09.03.01", level="bachelor"
    )
    service = AdmissionScoreService(session)
    stats = await service.upsert_from_rows(
        [
            ScoreRow(
                faculty_name=faculty.name,
                program_name=program_name,
                code="09.03.01",
                year=2025,
                form="budget",
                passing_score=141,
                average_score=None,
            )
        ]
    )
    assert stats == {"created": 1, "updated": 0, "skipped": 0}

    by_code = await service.query_scores(program="09.03.01", year=2025, form="budget")
    assert len(by_code) == 1
    assert by_code[0]["passing_score"] == 141

    for query in ("программная инженерия", "программную инженерию"):
        rows = await service.query_scores(program=query, year=2025, form="budget")
        assert rows == by_code
        assert rows[0]["program_name"] == program_name
