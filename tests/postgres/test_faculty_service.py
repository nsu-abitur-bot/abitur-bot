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


@pytest.mark.asyncio
async def test_soft_seed_adds_missing_without_overwriting(session: AsyncSession):
    """Мягкий сид доливает недостающее и НЕ затирает правки из админки."""
    service = FacultyService(session)
    await service.upsert_from_seed(
        [
            {
                "name": "Гуманитарный институт",
                "aliases": ["ГИ"],
                "programs": [
                    {"name": "История", "code": "46.03.01", "level": "bachelor"}
                ],
            }
        ]
    )
    faculty = await service.get_faculty_by_name("Гуманитарный институт")
    assert faculty is not None

    # Правки «из админки»: свой алиас и выключенное направление.
    await service.update_faculty(faculty.id, aliases=["ГИ", "админский-алиас"])
    programs = await service.get_programs_by_faculty(faculty.id, only_active=False)
    history = next(p for p in programs if p.name == "История")
    await service.update_program(history.id, is_active=False)

    stats = await service.upsert_from_seed(
        [
            {
                "name": "Гуманитарный институт",
                "aliases": ["ГИ", "Гуманитарный институт НГУ"],
                "programs": [
                    # Другой код у существующего — НЕ должен перезаписаться.
                    {"name": "История", "code": "99.99.99", "level": "bachelor"},
                    {"name": "Журналистика", "code": "42.03.02", "level": "bachelor"},
                ],
            }
        ],
        soft=True,
    )

    refreshed = await service.get_faculty_by_name("Гуманитарный институт")
    assert refreshed is not None
    assert "админский-алиас" in refreshed.aliases  # правка админа цела
    assert "Гуманитарный институт НГУ" in refreshed.aliases  # новый долит

    progs = {
        p.name: p
        for p in await service.get_programs_by_faculty(refreshed.id, only_active=False)
    }
    assert progs["История"].is_active is False  # не включили обратно
    assert progs["История"].code == "46.03.01"  # код не перезаписан
    assert "Журналистика" in progs  # новое направление добавлено
    assert stats["programs_created"] == 1


@pytest.mark.asyncio
async def test_soft_seed_fills_only_empty_code(session: AsyncSession):
    service = FacultyService(session)
    await service.upsert_from_seed(
        [
            {
                "name": "Физический факультет",
                "aliases": ["ФФ"],
                "programs": [{"name": "Физическая информатика", "level": "bachelor"}],
            }
        ]
    )
    faculty = await service.get_faculty_by_name("Физический факультет")
    assert faculty is not None
    progs = await service.get_programs_by_faculty(faculty.id, only_active=False)
    assert progs[0].code is None

    await service.upsert_from_seed(
        [
            {
                "name": "Физический факультет",
                "aliases": ["ФФ"],
                "programs": [
                    {
                        "name": "Физическая информатика",
                        "code": "03.03.02",
                        "level": "bachelor",
                    }
                ],
            }
        ],
        soft=True,
    )
    progs = await service.get_programs_by_faculty(faculty.id, only_active=False)
    assert progs[0].code == "03.03.02"  # пустой код заполнен


@pytest.mark.asyncio
async def test_hard_seed_still_overwrites(session: AsyncSession):
    """Жёсткий режим (soft=False) сохраняет прежнее поведение перезаписи."""
    service = FacultyService(session)
    await service.upsert_from_seed(
        [
            {
                "name": "Гуманитарный институт",
                "aliases": ["ГИ"],
                "programs": [
                    {"name": "История", "code": "46.03.01", "level": "bachelor"}
                ],
            }
        ]
    )
    faculty = await service.get_faculty_by_name("Гуманитарный институт")
    assert faculty is not None
    await service.update_faculty(faculty.id, aliases=["ГИ", "админский-алиас"])
    programs = await service.get_programs_by_faculty(faculty.id, only_active=False)
    await service.update_program(programs[0].id, is_active=False)

    await service.upsert_from_seed(
        [
            {
                "name": "Гуманитарный институт",
                "aliases": ["ГИ"],
                "programs": [
                    {"name": "История", "code": "46.03.01", "level": "bachelor"}
                ],
            }
        ],
        soft=False,
    )

    refreshed = await service.get_faculty_by_name("Гуманитарный институт")
    assert refreshed is not None
    assert refreshed.aliases == ["ГИ"]  # алиасы заменены целиком
    progs = await service.get_programs_by_faculty(refreshed.id, only_active=False)
    assert progs[0].is_active is True  # принудительно включено обратно
