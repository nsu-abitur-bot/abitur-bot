"""Инструмент get_admission_scores: баллы и объявленные места приёма.

Позволяет модели брать реальные числа из структурированного хранилища
(:class:`AdmissionScoreService`), а не выдумывать их. Возвращает компактный
человекочитаемый текст, который затем используется как авторитетный контекст.
"""

import logging
from html import escape
from typing import Literal, Optional

from abbrev.expander import get_abbrev_expander
from config.university import is_nsu
from db.postgres.db import AsyncSessionLocal
from db.postgres.models import (
    ADMISSION_BASES,
    EDUCATION_LEVELS,
    STUDY_FORM_LABELS,
    StudyForm,
)
from db.postgres.services.admission_score import AdmissionScoreService
from db.postgres.services.faculty import normalize_level
from llm.base import ToolSpec
from pipeline.tools.education_programs import LEVEL_LABELS

logger = logging.getLogger(__name__)


class AdmissionScoresResult(str):
    """Текст для провайдера и проверяемый статус для пайплайна."""

    status: Literal["ok", "empty", "error"]
    scope: dict
    use_as_answer: bool

    @property
    def program_ids(self) -> frozenset[str]:
        return frozenset(self.scope.get("_program_ids", []))

    def __new__(
        cls,
        text: str,
        *,
        status: Literal["ok", "empty", "error"],
        scope: dict | None = None,
        use_as_answer: bool = False,
    ) -> "AdmissionScoresResult":
        instance = super().__new__(cls, text)
        instance.status = status
        instance.scope = dict(scope or {})
        instance.use_as_answer = use_as_answer
        return instance


ADMISSION_SCORES_TOOL = ToolSpec(
    name="get_admission_scores",
    description=(
        "Проходные и средние баллы прошлых лет и объявленные места по программам НГУ. "
        "Для баллов в каждой записи возвращает оба доступных показателя с программой, "
        "годом, формой обучения и основанием приёма. "
        "Фильтры: год, уровень, форма обучения и основание приёма. Для запроса мест "
        "укажи metric=seats, а для мест вместе с баллами добавь include_seat_count. "
        "Число объявленных мест не равно числу зачисленных или свободных мест "
        "и не гарантирует поступление."
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
            "admission_basis": {
                "type": "string",
                "enum": sorted(ADMISSION_BASES),
                "description": (
                    "Основание приёма: budget — бюджет, paid — платное. "
                    "Без фильтра вернутся все основания."
                ),
            },
            "study_form": {
                "type": "string",
                "enum": [form.value for form in STUDY_FORM_LABELS],
                "description": (
                    "Форма обучения: full_time — очная, correspondence — заочная. "
                    "Указывай только форму, прямо запрошенную пользователем. "
                    "Для сравнения форм или без уточнения оставь поле пустым."
                ),
            },
            "metric": {
                "type": "string",
                "enum": ["passing", "average", "both", "seats"],
                "description": (
                    "Что интересует пользователя: passing — проходной (по умолчанию), "
                    "average — средний, both — оба, seats — только объявленные места. "
                    "Для passing, average и both результат содержит оба доступных "
                    "показателя."
                ),
            },
            "include_seat_count": {
                "type": "boolean",
                "description": "Покажи объявленные места вместе с баллами.",
            },
        },
        "additionalProperties": False,
    },
)


_BASIS_LABELS = {"budget": "бюджет", "paid": "платно"}


def normalize_score_arguments(arguments: dict) -> dict:
    """Поддерживает прежнее имя основания приёма во внутренних вызовах."""
    if "form" in arguments:
        arguments = dict(arguments)
        legacy = arguments.pop("form")
        if arguments.setdefault("admission_basis", legacy) != legacy:
            raise ValueError("form и admission_basis должны совпадать")
    return arguments


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


def _format_score_values(row: dict) -> str:
    passing = row.get("passing_score")
    average = row.get("average_score")
    if passing is None and average is None:
        return "проходной и средний баллы не указаны"
    passing_text = str(passing).removesuffix(".0") if passing is not None else "не указан"
    average_text = str(average).removesuffix(".0") if average is not None else "не указан"
    return f"проходной {passing_text}, средний {average_text}"


def _format_scores(
    rows: list[dict], metric: str, *, include_seat_count: bool = False
) -> str:
    """Компактный текст с классификацией каждой строки данных приёма."""
    show_seats = metric == "seats" or include_seat_count
    if metric == "seats":
        header = "Объявленные места"
    elif show_seats:
        header = "Баллы и объявленные места"
    else:
        header = "Проходные и средние баллы прошлых лет"
    university = " (НГУ)" if is_nsu() else ""
    lines = [f"{header}{university}:"]

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
        basis = row["admission_basis"]
        basis_label = _BASIS_LABELS.get(basis, basis)
        study_form = row.get("study_form")
        study_label = (
            STUDY_FORM_LABELS.get(study_form, study_form)
            if study_form is not None
            else "форма обучения не указана"
        )
        values = [_format_score_values(row)] if metric != "seats" else []
        seat_count = row.get("seat_count")
        if show_seats:
            values.append(
                f"объявлено мест: {seat_count}"
                if seat_count is not None
                else "нет данных о количестве мест"
            )
        source_url = row.get("source_url")
        if source_url and (metric != "seats" or seat_count is not None):
            values.append(f"источник: {escape(source_url, quote=False)}")
        lines.append(f"- {year}, {study_label}, {basis_label}: {', '.join(values)}")

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

    Отличает пустую выборку и сбой запроса от записей с незаполненными баллами.
    """
    try:
        arguments = normalize_score_arguments(arguments)
    except ValueError as exc:
        return AdmissionScoresResult(str(exc), status="error")
    program_ids = arguments.get("program_ids")
    if program_ids is not None and (
        not isinstance(program_ids, list)
        or not program_ids
        or any(not isinstance(value, str) or not value for value in program_ids)
    ):
        return AdmissionScoresResult("Некорректные ID программ.", status="error")
    raw_level = arguments.get("level")
    level = normalize_level(str(raw_level)) if raw_level is not None else None
    if raw_level is not None and level is None:
        return AdmissionScoresResult("Неизвестный уровень образования.", status="error")
    metric = str(arguments.get("metric") or "passing").lower()
    if metric not in ("passing", "average", "both", "seats"):
        metric = "passing"
    include_seat_count = arguments.get("include_seat_count") is True
    show_seats = metric == "seats" or include_seat_count

    faculty = _expand(arguments.get("faculty"))
    program = _expand(arguments.get("program"))
    basis = _str_or_none(arguments.get("admission_basis"))
    if basis is not None:
        basis = basis.strip().lower()
        if basis not in ADMISSION_BASES:
            return AdmissionScoresResult("Неизвестное основание приёма.", status="error")
    raw_study_form = _str_or_none(arguments.get("study_form"))
    try:
        study_form = StudyForm(raw_study_form) if raw_study_form is not None else None
    except ValueError:
        return AdmissionScoresResult("Неизвестная форма обучения.", status="error")
    year = _coerce_year(arguments.get("year"))

    try:
        async with AsyncSessionLocal() as session:
            service = AdmissionScoreService(session)
            filters = dict(
                faculty=faculty,
                program=program,
                admission_basis=basis,
                study_form=study_form,
                level=level,
                program_ids=program_ids,
                include_identity=True,
            )
            rows = await service.query_scores(year=year, **filters)
            note = ""
            if (
                year is not None
                and not show_seats
                and not any(
                    row["passing_score"] is not None or row["average_score"] is not None
                    for row in rows
                )
            ):
                other_years = await service.query_scores(only_with_scores=True, **filters)
                if other_years:
                    # Объявленные места могут появиться до публикации итоговых баллов.
                    note = (
                        f"В записях за {year} год баллы не указаны. "
                        "Показаны также доступные баллы за другие годы."
                        if rows
                        else f"За {year} год данных нет, показаны доступные годы."
                    )
                    rows = sorted(
                        [*rows, *other_years],
                        key=lambda row: (
                            row["program_name"],
                            row["program_id"],
                            row["year"],
                        ),
                    )
    except Exception:
        logger.exception("Ошибка выполнения инструмента get_admission_scores")
        return AdmissionScoresResult(
            "Не удалось получить данные о приёме.", status="error"
        )

    scope = {
        key: arguments[key]
        for key in (
            "program",
            "faculty",
            "level",
            "year",
            "admission_basis",
            "study_form",
        )
        if arguments.get(key) is not None
    }
    scope["metric"] = metric
    if include_seat_count:
        scope["include_seat_count"] = True
    scope["_program_ids"] = sorted(
        set(program_ids)
        if program_ids is not None
        else {row["program_id"] for row in rows}
    )
    scope["_program_names"] = {row["program_id"]: [row["program_name"]] for row in rows}
    if not rows:
        if show_seats:
            subject = "о количестве мест" if metric == "seats" else "о баллах и местах"
            period = f" за {year} год" if year is not None else ""
            text = f"По этому запросу нет данных {subject}{period}."
        else:
            text = "По этому запросу нет данных о баллах."
        return AdmissionScoresResult(
            text,
            status="empty",
            scope=scope,
            use_as_answer=True,
        )

    # Один фактически найденный вариант сохраняем для следующего уточнения.
    # Форму обучения сохраняем только при явном выборе пользователя.
    for key, column in (
        ("program", "code"),
        ("faculty", "faculty_name"),
        ("level", "level"),
        ("year", "year"),
        ("admission_basis", "admission_basis"),
    ):
        values = {row.get(column) for row in rows}
        if len(values) == 1 and None not in values:
            scope[key] = next(iter(values))
        elif key in ("year", "admission_basis"):
            scope.pop(key, None)
    text = _format_scores(rows, metric, include_seat_count=include_seat_count)
    return AdmissionScoresResult(
        f"{note}\n\n{text}" if note else text,
        status="ok",
        scope=scope,
        use_as_answer=True,
    )
