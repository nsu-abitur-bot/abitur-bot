"""Инструмент get_admission_scores: проходные/средние баллы прошлых лет.

Позволяет модели брать реальные числа из структурированного хранилища
(:class:`AdmissionScoreService`), а не выдумывать их. Возвращает компактный
человекочитаемый текст, который затем используется как авторитетный контекст.
"""

import logging
from typing import Optional

from db.postgres.services.admission_score import AdmissionScoreService

from db.postgres.db import AsyncSessionLocal
from llm.base import ToolSpec

logger = logging.getLogger(__name__)


ADMISSION_SCORES_TOOL = ToolSpec(
    name="get_admission_scores",
    description=(
        "Проходные и средние баллы прошлых лет по направлениям НГУ (бакалавриат). "
        "Используй для любых вопросов про проходной/средний балл, чтобы не "
        "выдумывать числа."
    ),
    parameters={
        "type": "object",
        "properties": {
            "faculty": {
                "type": "string",
                "description": (
                    "Название или аббревиатура факультета НГУ "
                    "(например, «ФИТ», «ММФ»). Необязательно."
                ),
            },
            "program": {
                "type": "string",
                "description": (
                    "Название направления подготовки "
                    "(например, «Компьютерные науки и системотехника»). "
                    "Необязательно."
                ),
            },
            "year": {
                "type": "integer",
                "description": "Год приёма (например, 2024). Необязательно.",
            },
            "form": {
                "type": "string",
                "enum": ["budget", "paid"],
                "description": (
                    "Форма обучения: budget — бюджет, paid — платное (по договору). "
                    "Необязательно."
                ),
            },
            "metric": {
                "type": "string",
                "enum": ["passing", "average", "both"],
                "description": (
                    "Какой балл нужен: passing — проходной (по умолчанию), "
                    "average — средний, both — оба. Указывай average или both "
                    "ТОЛЬКО если пользователь явно спросил про средний балл."
                ),
            },
        },
        "additionalProperties": False,
    },
)


_FORM_LABELS = {"budget": "бюджет", "paid": "платно"}


def _coerce_year(value: object) -> Optional[int]:
    """Приводит year к int (модель иногда присылает строку). None при ошибке."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _format_score_value(row: dict, metric: str) -> str:
    """Форматирует значение балла для одной строки под выбранную метрику."""
    passing = row.get("passing_score")
    average = row.get("average_score")

    parts: list[str] = []
    if metric in ("passing", "both"):
        parts.append("проходной " + (str(passing) if passing is not None else "—"))
    if metric in ("average", "both"):
        avg_text = f"{average:g}" if isinstance(average, (int, float)) else "—"
        parts.append("средний " + avg_text)
    return ", ".join(parts)


def _format_scores(rows: list[dict], metric: str) -> str:
    """Компактный RU-текст, сгруппированный по направлению, затем по году/форме."""
    header = "Баллы прошлых лет (бакалавриат НГУ):"
    lines: list[str] = [header]

    current_key: Optional[tuple[str, Optional[str]]] = None
    for row in rows:
        program = row.get("program_name") or "—"
        code = row.get("code")
        faculty = row.get("faculty_name") or ""
        key = (program, code)
        if key != current_key:
            current_key = key
            code_suffix = f" ({code})" if code else ""
            faculty_suffix = f" — {faculty}" if faculty else ""
            lines.append("")
            lines.append(f"{program}{code_suffix}{faculty_suffix}:")

        year = row.get("year")
        form_label = _FORM_LABELS.get(row.get("form", ""), row.get("form", ""))
        value = _format_score_value(row, metric)
        lines.append(f"- {year}, {form_label}: {value}")

    return "\n".join(lines).strip()


async def execute_admission_scores(arguments: dict) -> str:
    """Выполняет запрос проходных/средних баллов и форматирует результат.

    Пустые/отсутствующие фильтры не передаются в сервис. Если данных нет —
    возвращает единый маркер, по которому модель понимает, что цифр нет.
    """
    metric = str(arguments.get("metric") or "passing").lower()
    if metric not in ("passing", "average", "both"):
        metric = "passing"

    faculty = arguments.get("faculty")
    program = arguments.get("program")
    form = arguments.get("form")
    year = _coerce_year(arguments.get("year"))

    try:
        async with AsyncSessionLocal() as session:
            rows = await AdmissionScoreService(session).query_scores(
                faculty=str(faculty) if faculty else None,
                program=str(program) if program else None,
                year=year,
                form=str(form) if form else None,
            )
    except Exception:
        logger.exception("Ошибка выполнения инструмента get_admission_scores")
        return "Не удалось получить данные о проходных баллах."

    if not rows:
        return "По этому запросу данных о проходных баллах нет."

    return _format_scores(rows, metric)
