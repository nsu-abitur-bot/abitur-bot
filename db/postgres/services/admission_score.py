"""Сервис для проходных баллов направлений подготовки.

Структурированное хранилище проходных баллов прошлых лет (одна строка на
тройку program/year/form). Позволяет отвечать на числовые вопросы из SQL,
а не из RAG. Разрешение program_id идёт через авторитетный справочник
факультетов (:class:`FacultyService`).
"""

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import ADMISSION_FORMS, AdmissionScore, Faculty, Program
from .faculty import FacultyService, normalize_level, normalize_name

logger = logging.getLogger(__name__)

# Разделитель «Поле. Профиль» в названиях направлений справочника.
_FIELD_PROFILE_SEP = ". "


def _has_field_profile(name: str) -> bool:
    return _FIELD_PROFILE_SEP in name


def _field_part(name: str) -> str:
    """Часть названия ДО первого «. » (поле/УГСН), нормализованная."""
    return normalize_name(name.split(_FIELD_PROFILE_SEP, 1)[0])


def _profile_part(name: str) -> str:
    """Часть названия ПОСЛЕ последнего «. » (профиль), нормализованная."""
    return normalize_name(name.rsplit(_FIELD_PROFILE_SEP, 1)[-1])


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

    async def _resolve_program_detailed(
        self, faculty_name: str, program_name: str, level: Optional[str]
    ) -> tuple[Optional[Program], str]:
        """Строго разрешает (факультет, направление) в Program + причину.

        Две страницы НГУ используют разную гранулярность: страница итогов приёма
        даёт «Поле» и «Профиль» отдельными строками, а справочник — склеенное
        «Поле. Профиль». Поэтому матчим по приоритету:
        1) точное нормализованное имя;
        2) уникальный ПРОФИЛЬ (суффикс после «. »);
        3) уникальное ПОЛЕ (префикс до «. »).
        Substring НЕ используем: для авторитетного хранилища числовых фактов
        неоднозначное лучше пропустить, чем поставить неверный балл. Строки-
        агрегаты поля/УГСН (несколько профилей под одним полем) осознанно
        отсекаются (`ambiguous_field`). Возвращает (program|None, reason).
        """
        faculty = await self.faculty_service.find_faculty_by_alias(faculty_name)
        if faculty is None:
            return None, "faculty_unknown"

        canonical_level = normalize_level(level) or level
        programs = await self.faculty_service.get_programs_by_faculty(
            faculty.id, level=canonical_level
        )

        target = normalize_name(program_name)
        if not target:
            return None, "empty"

        # 1. Точное совпадение полного имени.
        exact = [p for p in programs if normalize_name(p.name) == target]
        if len(exact) == 1:
            return exact[0], "exact"
        if len(exact) > 1:
            return None, "ambiguous_exact"

        # 2. Уникальный профиль (суффикс «Поле. ПРОФИЛЬ»).
        by_profile = [
            p
            for p in programs
            if _has_field_profile(p.name) and _profile_part(p.name) == target
        ]
        if len(by_profile) == 1:
            return by_profile[0], "profile_suffix"
        if len(by_profile) > 1:
            return None, "ambiguous_profile"

        # 3. Уникальное поле (префикс «ПОЛЕ. Профиль») — только если ровно одно.
        by_field = [
            p
            for p in programs
            if _has_field_profile(p.name) and _field_part(p.name) == target
        ]
        if len(by_field) == 1:
            return by_field[0], "field_prefix"
        if len(by_field) > 1:
            return None, "ambiguous_field"

        return None, "unknown"

    async def _resolve_program(
        self, faculty_name: str, program_name: str, level: Optional[str]
    ) -> Optional[Program]:
        """Разрешает (факультет, направление) в объект Program или None."""
        program, _reason = await self._resolve_program_detailed(
            faculty_name, program_name, level
        )
        return program

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

    async def _match_program_ids_global(
        self, program_name: str, level: Optional[str]
    ) -> list[str]:
        """program_id всех направлений, подходящих под имя, БЕЗ факультета.

        Та же строгая логика (точное имя → профиль-суффикс → филд-префикс), но
        по всему справочнику. Возвращает ВСЕ id первого сработавшего уровня.
        """
        target = normalize_name(program_name)
        if not target:
            return []
        programs = await self.faculty_service.get_all_programs(
            level=normalize_level(level)
        )
        exact = [p.id for p in programs if normalize_name(p.name) == target]
        if exact:
            return exact
        by_profile = [
            p.id
            for p in programs
            if _has_field_profile(p.name) and _profile_part(p.name) == target
        ]
        if by_profile:
            return by_profile
        return [
            p.id
            for p in programs
            if _has_field_profile(p.name) and _field_part(p.name) == target
        ]

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

            program, reason = await self._resolve_program_detailed(
                row.faculty_name, row.program_name, row.level
            )
            if program is None:
                logger.info(
                    "Пропущена строка проходного балла [%s]: направление не "
                    "разрешено (факультет=%r, направление=%r, уровень=%r)",
                    reason,
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
        level: Optional[str] = None,
    ) -> list[dict]:
        """Выборка проходных баллов с разрешением строковых фильтров.

        Факультет разрешается по алиасу, направление — по строгому матчу имени
        (точное → профиль-суффикс → филд-префикс). ``level=None`` (по умолчанию)
        означает любой уровень — так находятся и специалитетные программы
        (медицина), а не только бакалавриат. Возвращает список плоских словарей,
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
            if faculty is not None:
                # Факультет уже разрешён — ищем направление среди его программ.
                target_program = await self._resolve_program(faculty, program, level)
                if target_program is None:
                    return []
                stmt = stmt.where(AdmissionScore.program_id == target_program.id)
            else:
                # Без факультета — матчим по справочнику глобально (точное имя →
                # профиль-суффикс → филд-префикс), затем фильтруем по id.
                ids = await self._match_program_ids_global(program, level)
                if not ids:
                    return []
                stmt = stmt.where(AdmissionScore.program_id.in_(ids))

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
