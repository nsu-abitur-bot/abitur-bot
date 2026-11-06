"""Инструмент get_admission_scores: проходные/средние баллы прошлых лет.

Позволяет модели брать реальные числа из структурированного хранилища
(:class:`AdmissionScoreService`), а не выдумывать их. Возвращает компактный
человекочитаемый текст, который затем используется как авторитетный контекст.
"""

import logging
from typing import Literal, Optional

from abbrev.expander import get_abbrev_expander
from config.university import is_nsu
from db.postgres.db import AsyncSessionLocal
from db.postgres.models import EDUCATION_LEVELS
from db.postgres.services.admission_score import AdmissionScoreService
from db.postgres.services.faculty import normalize_level
from llm.base import ToolSpec
from pipeline.tools.education_programs import LEVEL_LABELS

logger = logging.getLogger(__name__)


class AdmissionScoresResult(str):
    """Текст для провайдера и проверяемый статус для пайплайна."""

    status: Literal["ok", "empty", "error"]
    scope: dict

    def __new__(
        cls,
        text: str,
        *,
        status: Literal["ok", "empty", "error"],
        scope: dict | None = None,
    ) -> "AdmissionScoresResult":
        instance = super().__new__(cls, text)
        instance.status = status
        instance.scope = dict(scope or {})
        return instance


ADMISSION_SCORES_TOOL = ToolSpec(
    name="get_admission_scores",
    description=(
        "Проходные и средние баллы прошлых лет по направлениям НГУ "
        "с фильтрацией по уровню образования. "
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
                    "Направление подготовки так, как его назвал пользователь: "
                    "полное название, его часть или сокращение (например, "
                    "«Компьютерные науки и системотехника», «ПИиКН», "
                    "«прикладная математика»). Необязательно."
                ),
            },
            "level": {
                "type": "string",
                "enum": sorted(EDUCATION_LEVELS),
                "description": (
                    "Уровень из вопроса: spo — СПО, bachelor — бакалавриат, "
                    "specialist — специалитет, master — магистратура."
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
    header = (
        "Проходные баллы прошлых лет (НГУ):"
        if is_nsu()
        else "Проходные баллы прошлых лет:"
    )
    lines: list[str] = [header]

    current_key: Optional[tuple] = None
    for row in rows:
        program = row.get("program_name") or "—"
        code = row.get("code")
        faculty = row.get("faculty_name") or ""
        level = row.get("level")
        key = (program, code, faculty, level)
        if key != current_key:
            current_key = key
            code_suffix = f" ({code})" if code else ""
            faculty_suffix = f" — {faculty}" if faculty else ""
            lines.append("")
            level_suffix = f" [{LEVEL_LABELS.get(level, level)}]" if level else ""
            lines.append(f"{program}{code_suffix}{faculty_suffix}{level_suffix}:")

        year = row.get("year")
        form_label = _FORM_LABELS.get(row.get("form", ""), row.get("form", ""))
        value = _format_score_value(row, metric)
        lines.append(f"- {year}, {form_label}: {value}")

    return "\n".join(lines).strip()


def _str_or_none(value: object) -> Optional[str]:
    return str(value) if value else None


def _expand(value: object) -> Optional[str]:
    """Расшифровывает аббревиатуры из справочника админки.

    «ПИиКН» превращается в «ПИиКН (Программная инженерия и компьютерные
    науки)» — сервис пробует обе части. Так сокращения, которые завели в
    админке, работают и в поиске баллов, а не только в тексте вопроса.
    """
    text = _str_or_none(value)
    if text is None:
        return None
    try:
        return get_abbrev_expander().expand(text)
    except Exception:
        logger.warning("Не удалось расшифровать аббревиатуры в %r", text)
        return text


async def execute_admission_scores(arguments: dict) -> str:
    """Выполняет запрос проходных/средних баллов и форматирует результат.

    Пустые/отсутствующие фильтры не передаются в сервис. Если данных нет —
    возвращает единый маркер, по которому модель понимает, что цифр нет.
    """
    raw_level = arguments.get("level")
    level = normalize_level(str(raw_level)) if raw_level is not None else None
    if raw_level is not None and level is None:
        return AdmissionScoresResult("Неизвестный уровень образования.", status="error")
    metric = str(arguments.get("metric") or "passing").lower()
    if metric not in ("passing", "average", "both"):
        metric = "passing"

    faculty = _expand(arguments.get("faculty"))
    program = _expand(arguments.get("program"))
    form = arguments.get("form")
    year = _coerce_year(arguments.get("year"))

    try:
        async with AsyncSessionLocal() as session:
            service = AdmissionScoreService(session)
            rows = await service.query_scores(
                faculty=faculty,
                program=program,
                year=year,
                form=_str_or_none(form),
                level=level,
            )
            note = ""
            if not rows and year is not None:
                # Модель подставляет текущий год, а итоги приёма за него ещё не
                # опубликованы: лучше показать прошлые годы, чем «данных нет».
                rows = await service.query_scores(
                    faculty=faculty, program=program, form=_str_or_none(form), level=level
                )
                if rows:
                    note = f"За {year} год данных нет, показаны доступные годы."
    except Exception:
        logger.exception("Ошибка выполнения инструмента get_admission_scores")
        return AdmissionScoresResult(
            "Не удалось получить данные о проходных баллах.", status="error"
        )

    if not rows:
        return AdmissionScoresResult(
            "По этому запросу данных о проходных баллах нет.", status="empty"
        )

    scope = {
        key: arguments[key]
        for key in ("program", "faculty", "level", "year", "form")
        if arguments.get(key) is not None
    }
    scope["metric"] = metric
    # Один фактически найденный вариант сохраняем для следующего уточнения.
    # Несколько лет/форм не превращаем в случайно выбранный последний вариант.
    for key, column in (
        ("program", "code"),
        ("faculty", "faculty_name"),
        ("level", "level"),
        ("year", "year"),
        ("form", "form"),
    ):
        values = {row.get(column) for row in rows}
        if len(values) == 1 and None not in values:
            scope[key] = next(iter(values))
        elif key in ("year", "form"):
            scope.pop(key, None)
    text = _format_scores(rows, metric)
    return AdmissionScoresResult(
        f"{note}\n\n{text}" if note else text, status="ok", scope=scope
    )
