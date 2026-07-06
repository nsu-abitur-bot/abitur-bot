"""Тесты для FacultyService (реальная тестовая БД, без моков)."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from db.postgres.services.faculty import (
    FacultyService,
    normalize_level,
    normalize_name,
)


async def _seed_fit_and_econ(session: AsyncSession) -> FacultyService:
    """ФИТ → его направления; экономфак → бизнес-информатика (bachelor)."""
    service = FacultyService(session)
    await service.upsert_from_seed(
        [
            {
                "name": "Факультет информационных технологий",
                "aliases": ["ФИТ"],
                "programs": [
                    {
                        "name": "Программная инженерия",
                        "code": "09.03.04",
                        "level": "bachelor",
                    },
                    {
                        "name": "Программная инженерия",
                        "code": "09.04.04",
                        "level": "master",
                    },
                ],
            },
            {
                "name": "Экономический факультет",
                "aliases": ["ЭФ"],
                "programs": [
                    {
                        "name": "Бизнес-информатика",
                        "code": "38.03.05",
                        "level": "bachelor",
                    }
                ],
            },
        ]
    )
    return service


@pytest.mark.asyncio
async def test_normalize_helpers():
    assert normalize_name("  ФИТ  ") == "фит"
    assert normalize_name("«Программная инженерия».") == "программная инженерия"
    assert normalize_level("Bachelor") == "bachelor"
    assert normalize_level("аспирантура") is None
    assert normalize_level(None) is None


@pytest.mark.asyncio
async def test_upsert_is_idempotent(session: AsyncSession):
    service = await _seed_fit_and_econ(session)
    faculties = await service.get_all_faculties()
    assert len(faculties) == 2

    # Повторная загрузка не создаёт дубликатов.
    await _seed_fit_and_econ(session)
    faculties_again = await service.get_all_faculties()
    assert len(faculties_again) == 2

    all_programs = await service.get_all_programs()
    # 2 у ФИТ (bachelor + master) + 1 у экономфака.
    assert len(all_programs) == 3


@pytest.mark.asyncio
async def test_find_faculty_by_alias_and_name(session: AsyncSession):
    service = await _seed_fit_and_econ(session)

    by_alias = await service.find_faculty_by_alias("ФИТ")
    assert by_alias is not None
    assert by_alias.name == "Факультет информационных технологий"

    # Регистр и пробелы не важны.
    by_alias_lower = await service.find_faculty_by_alias("  фит ")
    assert by_alias_lower is not None
    assert by_alias_lower.id == by_alias.id

    by_name = await service.find_faculty_by_alias("Экономический факультет")
    assert by_name is not None
    assert by_name.name == "Экономический факультет"

    assert await service.find_faculty_by_alias("Несуществующий") is None


@pytest.mark.asyncio
async def test_get_programs_by_faculty_and_level(session: AsyncSession):
    service = await _seed_fit_and_econ(session)
    fit = await service.find_faculty_by_alias("ФИТ")
    assert fit is not None

    all_fit = await service.get_programs_by_faculty(fit.id)
    assert {p.level for p in all_fit} == {"bachelor", "master"}

    bachelor_only = await service.get_programs_by_faculty(fit.id, level="bachelor")
    assert len(bachelor_only) == 1
    assert bachelor_only[0].level == "bachelor"

    master_only = await service.get_programs_by_faculty(fit.id, level="master")
    assert len(master_only) == 1
    assert master_only[0].level == "master"


@pytest.mark.asyncio
async def test_business_informatics_belongs_to_econ_not_fit(session: AsyncSession):
    """Ключевая проверка авторитетной таблицы: бизнес-информатика на экономфаке."""
    service = await _seed_fit_and_econ(session)

    owner = await service.find_faculty_for_program("Бизнес-информатика", level="bachelor")
    assert owner is not None
    assert owner.name == "Экономический факультет"

    fit = await service.find_faculty_by_alias("ФИТ")
    assert fit is not None
    fit_programs = await service.get_programs_by_faculty(fit.id, level="bachelor")
    assert all(
        normalize_name(p.name) != normalize_name("Бизнес-информатика")
        for p in fit_programs
    )


@pytest.mark.asyncio
async def test_create_program_rejects_bad_level(session: AsyncSession):
    service = FacultyService(session)
    faculty = await service.create_faculty("Тестовый факультет", aliases=["ТФ"])
    with pytest.raises(ValueError):
        await service.create_program(
            faculty.id, "Какое-то направление", level="аспирантура"
        )
