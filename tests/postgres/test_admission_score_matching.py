"""Поиск проходных баллов по тому, как направление называют в вопросе (#235).

Справочник и баллы — настоящие: сид `db/seed/faculties.json` и сохранённая
страница итогов приёма НГУ, разобранная тем же парсером, что и в проде. Так
тест видит ровно те названия, что лежат в базе: «Информатика и вычислительная
техника. Программная инженерия и компьютерные науки», а не «ПИиКН».
"""

from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.models import Program
from db.postgres.services.admission_score import (
    AdmissionScoreService,
    match_programs,
    query_variants,
)
from db.postgres.services.faculty import FacultyService
from db.seed.load_faculties import DEFAULT_SEED_PATH, load_seed_file
from parser.scores import parse_scores_html

_FIXTURE = (
    Path(__file__).parent.parent / "parser" / "fixtures" / "html" / "nsu_scores.html"
)

PIKN = "Информатика и вычислительная техника. Программная инженерия и компьютерные науки"
KNIS = "Информатика и вычислительная техника. Компьютерные науки и системотехника"
PMI = "Прикладная математика и информатика"


def _seed_programs() -> list[Program]:
    """Направления из сида как объекты модели — без базы."""
    return [
        Program(id=f"{f['name']}/{p['name']}", name=p["name"], code=p.get("code"))
        for f in load_seed_file(DEFAULT_SEED_PATH)
        for p in f.get("programs", [])
    ]


@pytest.mark.parametrize(
    "query, expected",
    [
        ("ПИиКН", {PIKN}),
        ("ПИКН", {PIKN}),
        ("КНиС", {KNIS}),
        ("ПМИ", {PMI}),
        ("прикладная математика", {PMI}),
        ("Программная инженерия", {PIKN}),
        ("программная инженерия, НГУ", {PIKN}),
        ("Лечебное дело", {"Лечебное дело"}),
        ("09.03.01", {PIKN, KNIS}),
        ("Ветеринария", set()),
    ],
)
def test_match_programs_on_seed(query: str, expected: set[str]):
    names = {p.name for p in match_programs(_seed_programs(), query)}
    assert names == expected


def test_match_programs_folds_yo():
    programs = [Program(id="1", name="Химия. Зелёная химия", code=None)]
    assert match_programs(programs, "зеленая химия") == programs


def test_query_variants_splits_expanded_abbreviation():
    assert query_variants("ПИиКН (Программная инженерия и компьютерные науки)") == [
        "ПИиКН (Программная инженерия и компьютерные науки)",
        "Программная инженерия и компьютерные науки",
        "ПИиКН ",
    ]
    assert query_variants("ФИТ (Факультет ИТ)", inner=False) == [
        "ФИТ (Факультет ИТ)",
        "ФИТ ",
    ]


@pytest_asyncio.fixture
async def service(session: AsyncSession) -> AdmissionScoreService:
    await FacultyService(session).upsert_from_seed(load_seed_file(DEFAULT_SEED_PATH))
    svc = AdmissionScoreService(session)
    stats = await svc.upsert_from_rows(
        parse_scores_html(_FIXTURE.read_text(encoding="utf-8"))
    )
    assert stats["created"] > 400
    return svc


def _programs(rows: list[dict]) -> set[str]:
    return {r["program_name"] for r in rows}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "program",
    [
        "ПИиКН",
        "пиикн",
        "ПИКН",
        "ПИиКН (Программная инженерия и компьютерные науки)",
        "Программная инженерия и компьютерные науки",
        "программная инженерия и компьютерные науки",
        "«Программная инженерия и компьютерные науки» (09.03.01)",
        "Программная инженерия",
        "направление программная инженерия",
    ],
)
async def test_pikn_phrasings_find_scores(service: AdmissionScoreService, program: str):
    rows = await service.query_scores(program=program)
    assert _programs(rows) == {PIKN}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "faculty", ["ФИТ", "фит", "ФИТ НГУ", "Факультет информационных технологий"]
)
async def test_pikn_with_faculty(service: AdmissionScoreService, faculty: str):
    rows = await service.query_scores(faculty=faculty, program="ПИиКН", year=2024)
    assert _programs(rows) == {PIKN}
    assert all(r["passing_score"] for r in rows if r["form"] == "budget")


@pytest.mark.asyncio
async def test_wrong_faculty_still_finds_program(service: AdmissionScoreService):
    """Модель угадала факультет неверно — направление всё равно находится."""
    rows = await service.query_scores(faculty="ММФ", program="ПИиКН")
    assert _programs(rows) == {PIKN}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "program",
    ["ПМИ", "ПМиИ", "прикладная математика", "Прикладная математика и информатика"],
)
async def test_pmi_phrasings(service: AdmissionScoreService, program: str):
    rows = await service.query_scores(program=program)
    assert _programs(rows) == {PMI}


@pytest.mark.asyncio
async def test_knis_abbreviation(service: AdmissionScoreService):
    rows = await service.query_scores(program="КНиС")
    assert _programs(rows) == {KNIS}


@pytest.mark.asyncio
async def test_code_matches_all_programs_with_it(service: AdmissionScoreService):
    rows = await service.query_scores(program="09.03.01")
    assert _programs(rows) == {PIKN, KNIS}


@pytest.mark.asyncio
async def test_faculty_name_in_program_slot(service: AdmissionScoreService):
    """«ФИТ» в поле направления — это факультет, а не «Физика и ИТ» физфака."""
    rows = await service.query_scores(program="ФИТ")
    assert _programs(rows) == {PIKN, KNIS}


@pytest.mark.asyncio
async def test_exact_name_still_wins_over_partial(service: AdmissionScoreService):
    rows = await service.query_scores(program="Лечебное дело")
    assert _programs(rows) == {"Лечебное дело"}


@pytest.mark.asyncio
async def test_unknown_program_returns_empty(service: AdmissionScoreService):
    assert await service.query_scores(program="Ветеринария") == []
    assert await service.query_scores(faculty="Несуществующий") == []
