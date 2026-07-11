"""Тесты для AdmissionScoreService (реальная тестовая БД, без моков)."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.services.admission_score import (
    AdmissionScoreService,
    ScoreRow,
)
from db.postgres.services.faculty import FacultyService


async def _seed_fit(session: AsyncSession) -> FacultyService:
    """ФИТ и два его направления бакалавриата."""
    service = FacultyService(session)
    await service.upsert_from_seed(
        [
            {
                "name": "Факультет информационных технологий",
                "aliases": ["ФИТ"],
                "programs": [
                    {
                        "name": "Программная инженерия и компьютерные науки",
                        "code": "09.03.04",
                        "level": "bachelor",
                    },
                    {
                        "name": "Компьютерные науки и системотехника",
                        "code": "09.03.01",
                        "level": "bachelor",
                    },
                ],
            }
        ]
    )
    return service


def _fit_rows() -> list[ScoreRow]:
    return [
        ScoreRow(
            faculty_name="ФИТ",
            program_name="Программная инженерия и компьютерные науки",
            code="09.03.04",
            year=2024,
            form="budget",
            passing_score=246,
            average_score=84.2,
        ),
        ScoreRow(
            faculty_name="Факультет информационных технологий",
            program_name="Компьютерные науки и системотехника",
            code="09.03.01",
            year=2024,
            form="budget",
            passing_score=260,
            average_score=88.1,
        ),
    ]


@pytest.mark.asyncio
async def test_upsert_creates_then_updates_idempotently(session: AsyncSession):
    await _seed_fit(session)
    service = AdmissionScoreService(session)

    stats = await service.upsert_from_rows(_fit_rows())
    assert stats == {"created": 2, "updated": 0, "skipped": 0}

    # Повторный upsert тех же строк — обновление, без дубликатов.
    stats_again = await service.upsert_from_rows(_fit_rows())
    assert stats_again == {"created": 0, "updated": 2, "skipped": 0}

    # Изменение значения обновляет существующую строку.
    changed = _fit_rows()
    changed[0].passing_score = 250
    stats_changed = await service.upsert_from_rows(changed)
    assert stats_changed == {"created": 0, "updated": 2, "skipped": 0}

    scores = await service.query_scores(faculty="ФИТ")
    updated = next(
        s
        for s in scores
        if s["program_name"] == "Программная инженерия и компьютерные науки"
    )
    assert updated["passing_score"] == 250


@pytest.mark.asyncio
async def test_upsert_skips_unknown_faculty_and_program(session: AsyncSession):
    await _seed_fit(session)
    service = AdmissionScoreService(session)

    rows = [
        ScoreRow(
            faculty_name="Несуществующий факультет",
            program_name="Что-то",
            code=None,
            year=2024,
            form="budget",
            passing_score=200,
            average_score=None,
        ),
        ScoreRow(
            faculty_name="ФИТ",
            program_name="Направление, которого нет на ФИТ",
            code=None,
            year=2024,
            form="budget",
            passing_score=210,
            average_score=None,
        ),
        # Недопустимая форма обучения тоже пропускается.
        ScoreRow(
            faculty_name="ФИТ",
            program_name="Компьютерные науки и системотехника",
            code="09.03.01",
            year=2024,
            form="целевое",
            passing_score=220,
            average_score=None,
        ),
    ]
    stats = await service.upsert_from_rows(rows)
    assert stats == {"created": 0, "updated": 0, "skipped": 3}

    assert await service.query_scores() == []


@pytest.mark.asyncio
async def test_query_scores_distinguishes_programs(session: AsyncSession):
    """Ключевая проверка: 2024 бюджет — 246 vs 260 по разным направлениям."""
    await _seed_fit(session)
    service = AdmissionScoreService(session)
    await service.upsert_from_rows(_fit_rows())

    by_faculty = await service.query_scores(faculty="ФИТ", year=2024, form="budget")
    assert len(by_faculty) == 2
    scores_by_program = {s["program_name"]: s["passing_score"] for s in by_faculty}
    assert scores_by_program["Программная инженерия и компьютерные науки"] == 246
    assert scores_by_program["Компьютерные науки и системотехника"] == 260

    # Проверяем плоскую форму словаря.
    sample = by_faculty[0]
    assert set(sample.keys()) == {
        "faculty_name",
        "program_name",
        "code",
        "year",
        "form",
        "passing_score",
        "average_score",
    }
    assert sample["faculty_name"] == "Факультет информационных технологий"


@pytest.mark.asyncio
async def test_query_scores_by_program_filter(session: AsyncSession):
    await _seed_fit(session)
    service = AdmissionScoreService(session)
    await service.upsert_from_rows(_fit_rows())

    only_cs = await service.query_scores(
        faculty="ФИТ", program="Компьютерные науки и системотехника"
    )
    assert len(only_cs) == 1
    assert only_cs[0]["passing_score"] == 260
    assert only_cs[0]["code"] == "09.03.01"

    # Тот же фильтр по направлению без факультета.
    only_cs_no_fac = await service.query_scores(
        program="Компьютерные науки и системотехника"
    )
    assert len(only_cs_no_fac) == 1
    assert only_cs_no_fac[0]["passing_score"] == 260


@pytest.mark.asyncio
async def test_query_scores_unknown_returns_empty(session: AsyncSession):
    await _seed_fit(session)
    service = AdmissionScoreService(session)
    await service.upsert_from_rows(_fit_rows())

    assert await service.query_scores(faculty="Несуществующий") == []
    assert await service.query_scores(year=1999) == []


@pytest.mark.asyncio
async def test_resolve_program_id(session: AsyncSession):
    """Публичный резолвер program_id для предпросмотра импорта."""
    await _seed_fit(session)
    service = AdmissionScoreService(session)

    # По алиасу факультета и точному имени направления.
    resolved = await service.resolve_program_id(
        "ФИТ", "Компьютерные науки и системотехника"
    )
    assert resolved is not None

    # Неизвестный факультет / направление → None (не пишет в БД).
    assert await service.resolve_program_id("Несуществующий факультет", "Что-то") is None
    assert await service.resolve_program_id("ФИТ", "Направление, которого нет") is None


async def _seed_granularity(session: AsyncSession) -> FacultyService:
    """Справочник в гранулярности «Поле. Профиль» + один специалитет.

    Воспроизводит реальное расхождение: справочник хранит склеенные названия,
    а страница итогов приёма отдаёт «Поле» и «Профиль» отдельными строками.
    """
    service = FacultyService(session)
    await service.upsert_from_seed(
        [
            {
                "name": "Факультет информационных технологий",
                "aliases": ["ФИТ"],
                "programs": [
                    {
                        "name": "Информатика и вычислительная техника. "
                        "Компьютерные науки и системотехника",
                        "code": "09.03.01",
                        "level": "bachelor",
                    },
                    {
                        "name": "Информатика и вычислительная техника. "
                        "Программная инженерия и компьютерные науки",
                        "code": "09.03.01",
                        "level": "bachelor",
                    },
                ],
            },
            {
                "name": "Физический факультет",
                "aliases": ["ФФ"],
                "programs": [
                    {
                        "name": "Физика. Фундаментальная и экспериментальная физика",
                        "code": "03.03.02",
                        "level": "bachelor",
                    }
                ],
            },
            {
                "name": "Институт медицины",
                "aliases": ["мед"],
                "programs": [
                    {
                        "name": "Лечебное дело",
                        "code": "31.05.01",
                        "level": "specialist",
                    }
                ],
            },
        ]
    )
    return service


@pytest.mark.asyncio
async def test_strict_matching_profile_field_and_aggregate(session: AsyncSession):
    await _seed_granularity(session)
    svc = AdmissionScoreService(session)
    rows = [
        # Голые профили (как на странице итогов) → матч на «Поле. Профиль».
        ScoreRow(
            "ФИТ",
            "Компьютерные науки и системотехника",
            "09.03.01",
            2024,
            "budget",
            260,
            None,
        ),
        ScoreRow(
            "ФИТ",
            "Программная инженерия и компьютерные науки",
            "09.03.01",
            2024,
            "budget",
            246,
            None,
        ),
        # Агрегат поля над двумя профилями → ambiguous_field → пропуск.
        ScoreRow(
            "ФИТ",
            "Информатика и вычислительная техника",
            "09.03.01",
            2024,
            "budget",
            999,
            None,
        ),
        # Поле над ЕДИНСТВЕННЫМ профилем → field_prefix → матч.
        ScoreRow("ФФ", "Физика", "03.03.02", 2024, "budget", 190, None),
    ]
    stats = await svc.upsert_from_rows(rows)
    assert stats == {"created": 3, "updated": 0, "skipped": 1}

    fit = await svc.query_scores(faculty="ФИТ", year=2024, form="budget")
    by = {s["program_name"]: s["passing_score"] for s in fit}
    assert (
        by["Информатика и вычислительная техника. Компьютерные науки и системотехника"]
        == 260
    )
    assert (
        by[
            "Информатика и вычислительная техника. Программная инженерия и компьютерные науки"
        ]
        == 246
    )
    # Агрегат 999 никуда не записался (не перезаписал профиль).
    assert 999 not in by.values()

    ff = await svc.query_scores(faculty="ФФ", year=2024, form="budget")
    assert len(ff) == 1 and ff[0]["passing_score"] == 190


@pytest.mark.asyncio
async def test_strict_matching_no_substring_collision(session: AsyncSession):
    """Профиль и агрегат поля не схлопываются в один program_id."""
    await _seed_granularity(session)
    svc = AdmissionScoreService(session)
    stats = await svc.upsert_from_rows(
        [
            ScoreRow(
                "ФИТ",
                "Компьютерные науки и системотехника",
                "09.03.01",
                2024,
                "budget",
                260,
                None,
            ),
            ScoreRow(
                "ФИТ",
                "Информатика и вычислительная техника",
                "09.03.01",
                2024,
                "budget",
                111,
                None,
            ),  # агрегат → пропуск
        ]
    )
    assert stats["created"] == 1 and stats["skipped"] == 1
    cs = await svc.query_scores(program="Компьютерные науки и системотехника", year=2024)
    assert len(cs) == 1 and cs[0]["passing_score"] == 260


@pytest.mark.asyncio
async def test_query_finds_specialist_by_default_level(session: AsyncSession):
    """level=None по умолчанию находит специалитет (медицина)."""
    await _seed_granularity(session)
    svc = AdmissionScoreService(session)
    await svc.upsert_from_rows(
        [
            ScoreRow(
                "мед",
                "Лечебное дело",
                "31.05.01",
                2024,
                "budget",
                258,
                None,
                level="specialist",
            ),
        ]
    )
    res = await svc.query_scores(program="Лечебное дело", year=2024)
    assert len(res) == 1 and res[0]["passing_score"] == 258


@pytest.mark.asyncio
async def test_get_available_years_descending(session: AsyncSession):
    await _seed_fit(session)
    service = AdmissionScoreService(session)

    rows = _fit_rows()
    rows.append(
        ScoreRow(
            faculty_name="ФИТ",
            program_name="Компьютерные науки и системотехника",
            code="09.03.01",
            year=2023,
            form="budget",
            passing_score=255,
            average_score=None,
        )
    )
    await service.upsert_from_rows(rows)

    years = await service.get_available_years()
    assert years == [2024, 2023]
