"""Поиск образовательных программ в справочнике подразделений."""

import logging

from db.postgres.db import AsyncSessionLocal
from db.postgres.models import EDUCATION_LEVELS
from db.postgres.services.faculty import FacultyService, normalize_level
from llm.base import ToolSpec

logger = logging.getLogger(__name__)

LEVEL_LABELS = {
    "bachelor": "бакалавриат",
    "specialist": "специалитет",
    "master": "магистратура",
    "spo": "СПО",
}

EDUCATION_PROGRAMS_TOOL = ToolSpec(
    name="get_education_programs",
    description=(
        "Образовательные программы из актуального справочника подразделений, "
        "включая программы колледжа (СПО). Возвращает название, код, подразделение "
        "и уровень каждой программы."
    ),
    parameters={
        "type": "object",
        "properties": {
            "faculty": {
                "type": "string",
                "description": "Название или алиас подразделения. Необязательно.",
            },
            "level": {
                "type": "string",
                "enum": sorted(EDUCATION_LEVELS),
                "description": (
                    "Уровень: spo — СПО, bachelor — бакалавриат, specialist — "
                    "специалитет, master — магистратура. Укажи уровень из вопроса."
                ),
            },
        },
        "additionalProperties": False,
    },
)


async def execute_education_programs(arguments: dict) -> str:
    """Возвращает активные программы с сохранением фильтра по уровню."""
    raw_level = arguments.get("level")
    level = normalize_level(str(raw_level)) if raw_level is not None else None
    if raw_level is not None and level is None:
        return "Неизвестный уровень образования."
    try:
        async with AsyncSessionLocal() as session:
            service = FacultyService(session)
            faculties = await service.get_all_faculties(only_active=True)
            faculty_names = {f.id: f.name for f in faculties}
            faculty_query = arguments.get("faculty")
            if faculty_query:
                faculty = await service.find_faculty_by_alias(str(faculty_query))
                if faculty is None:
                    return "Подразделение не найдено в справочнике."
                programs = await service.get_programs_by_faculty(faculty.id, level=level)
            else:
                programs = await service.get_all_programs(level=level)
            if not programs:
                return "По этому запросу программ в справочнике нет."
            lines = ["Образовательные программы из справочника:"]
            for program in programs:
                code = f" ({program.code})" if program.code else ""
                label = LEVEL_LABELS.get(program.level, program.level)
                lines.append(
                    f"- {program.name}{code} — {label}; "
                    f"{faculty_names[program.faculty_id]}"
                )
            return "\n".join(lines)
    except Exception:
        logger.exception("Ошибка выполнения инструмента get_education_programs")
        return "Не удалось получить программы из справочника."
