"""Сервис для проходных баллов направлений подготовки.

Структурированное хранилище проходных баллов прошлых лет (одна строка на
тройку program/year/form). Позволяет отвечать на числовые вопросы из SQL,
а не из RAG. Разрешение program_id идёт через авторитетный справочник
факультетов (:class:`FacultyService`).
"""

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import ADMISSION_FORMS, AdmissionScore, Faculty, Program
from .faculty import FacultyService, normalize_level, normalize_name

logger = logging.getLogger(__name__)


@dataclass
class ScoreRow:
    """Сырая строка проходного балла (до разрешения в program_id)."""

    faculty_name: str
    program_name: str
    code: Optional[str]
    year: int
    form: str
    passing_score: Optional[int]
    average_score: Optional[float]
    level: str = "bachelor"


class AdmissionScoreService:
    """Идемпотентный upsert и выборки проходных баллов."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.faculty_service = FacultyService(session)

    async def _resolve_program(
        self, faculty_name: str, program_name: str, level: str
    ) -> Optional[Program]:
        """Разрешает (факультет, направление) в объект Program.

        Сначала ищем факультет по названию/алиасу, затем среди его направлений
        нужного уровня. Совпадение по нормализованному имени — сначала точное,
        затем (как запасной вариант) самое длинное вхождение подстроки.
        Возвращает None, если ничего не разрешилось.
        """
        faculty = await self.faculty_service.find_faculty_by_alias(faculty_name)
        if faculty is None:
            return None

        canonical_level = normalize_level(level) or level
        programs = await self.faculty_service.get_programs_by_faculty(
            faculty.id, level=canonical_level
        )

        target = normalize_name(program_name)
        if not target:
            return None

        # Точное совпадение по нормализованному имени.
        for program in programs:
            if normalize_name(program.name) == target:
                return program

        # Запасной вариант: самое длинное вхождение подстроки (детерминировано).
        best: Optional[Program] = None
        best_len = 0
        for program in programs:
            norm = normalize_name(program.name)
            if norm and (norm in target or target in norm):
                overlap = min(len(norm), len(target))
                if overlap > best_len:
                    best = program
                    best_len = overlap
        return best

    async def resolve_program_id(
        self, faculty_name: str, program_name: str, level: str = "bachelor"
    ) -> Optional[str]:
        """Разрешает (факультет, направление) в program_id без записи в БД.

        Публичная обёртка над :meth:`_resolve_program` для предпросмотра импорта
        (подсчёт сопоставленных/несопоставленных строк). Возвращает None, если
        факультет или направление не найдены в справочнике.
        """
        program = await self._resolve_program(faculty_name, program_name, level)
        return program.id if program is not None else None

    async def upsert_from_rows(self, rows: Iterable[ScoreRow]) -> dict[str, int]:
        """Идемпотентный upsert по ключу (program_id, year, form).

        Разрешает program_id через факультет (по алиасу) и направление (по
        нормализованному имени). Если факультет или направление не находятся —
        строка молча пропускается (logger.info). Возвращает счётчики
        ``{"created": int, "updated": int, "skipped": int}``.
        """
        stats = {"created": 0, "updated": 0, "skipped": 0}

        for row in rows:
            form = (row.form or "").strip().lower()
            if form not in ADMISSION_FORMS:
                logger.info(
                    "Пропущена строка проходного балла: недопустимая форма %r "
                    "(факультет=%r, направление=%r, год=%s)",
                    row.form,
                    row.faculty_name,
                    row.program_name,
                    row.year,
                )
                stats["skipped"] += 1
                continue

            program = await self._resolve_program(
                row.faculty_name, row.program_name, row.level
            )
            if program is None:
                logger.info(
                    "Пропущена строка проходного балла: не разрешён факультет/"
                    "направление (факультет=%r, направление=%r, уровень=%r)",
                    row.faculty_name,
                    row.program_name,
                    row.level,
                )
                stats["skipped"] += 1
                continue

            stmt = select(AdmissionScore).where(
                AdmissionScore.program_id == program.id,
                AdmissionScore.year == row.year,
                AdmissionScore.form == form,
            )
            result = await self.session.execute(stmt)
            existing = result.scalar_one_or_none()

            if existing is None:
                self.session.add(
                    AdmissionScore(
                        program_id=program.id,
                        year=row.year,
                        form=form,
                        passing_score=row.passing_score,
                        average_score=row.average_score,
                        source_url=None,
                    )
                )
                stats["created"] += 1
            else:
                existing.passing_score = row.passing_score
                existing.average_score = row.average_score
                stats["updated"] += 1

        await self.session.commit()
        return stats

    async def query_scores(
        self,
        *,
        faculty: Optional[str] = None,
        program: Optional[str] = None,
        year: Optional[int] = None,
        form: Optional[str] = None,
        level: str = "bachelor",
    ) -> list[dict]:
        """Выборка проходных баллов с разрешением строковых фильтров.

        Факультет разрешается по алиасу, направление — по нормализованному
        имени среди направлений факультета. Возвращает список плоских словарей,
        отсортированный по (program_name, year, form). Пустой список, если
        ничего не найдено.
        """
        stmt = (
            select(AdmissionScore, Program, Faculty)
            .join(Program, AdmissionScore.program_id == Program.id)
            .join(Faculty, Program.faculty_id == Faculty.id)
        )

        canonical_level = normalize_level(level)
        if canonical_level is not None:
            stmt = stmt.where(Program.level == canonical_level)

        if faculty is not None:
            resolved_faculty = await self.faculty_service.find_faculty_by_alias(faculty)
            if resolved_faculty is None:
                return []
            stmt = stmt.where(Program.faculty_id == resolved_faculty.id)

        if program is not None:
            target_program: Optional[Program] = None
            if faculty is not None:
                # Факультет уже разрешён — ищем направление среди его программ.
                target_program = await self._resolve_program(faculty, program, level)
                if target_program is None:
                    return []
                stmt = stmt.where(AdmissionScore.program_id == target_program.id)
            else:
                normalized_program = normalize_name(program)
                stmt = stmt.where(func.lower(Program.name) == normalized_program)

        if year is not None:
            stmt = stmt.where(AdmissionScore.year == year)

        if form is not None:
            normalized_form = form.strip().lower()
            stmt = stmt.where(AdmissionScore.form == normalized_form)

        result = await self.session.execute(stmt)
        rows: list[dict] = []
        for score, prog, fac in result.all():
            rows.append(
                {
                    "faculty_name": fac.name,
                    "program_name": prog.name,
                    "code": prog.code,
                    "year": score.year,
                    "form": score.form,
                    "passing_score": score.passing_score,
                    "average_score": score.average_score,
                }
            )

        rows.sort(key=lambda r: (r["program_name"], r["year"], r["form"]))
        return rows

    async def get_available_years(self) -> list[int]:
        """Список различных годов, по убыванию."""
        stmt = select(AdmissionScore.year).distinct().order_by(AdmissionScore.year.desc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
