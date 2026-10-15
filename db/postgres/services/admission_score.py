"""Сервис для проходных баллов направлений подготовки.

Структурированное хранилище проходных баллов прошлых лет (одна строка на
тройку program/year/form). Позволяет отвечать на числовые вопросы из SQL,
а не из RAG. Разрешение program_id идёт через авторитетный справочник
факультетов (:class:`FacultyService`).
"""

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

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


# --- Поиск направления по формулировке из вопроса ---------------------------
#
# Импорт баллов матчит строго (см. _resolve_program_detailed): там ошибка
# пишет чужой балл в базу. Поиск по вопросу — другое дело: модель передаёт
# направление так, как его назвал абитуриент («ПИиКН», «программная
# инженерия», «ПИиКН (Программная инженерия и компьютерные науки)» после
# расширения аббревиатур), а в справочнике лежит «Информатика и
# вычислительная техника. Программная инженерия и компьютерные науки».
# Строгое сравнение такие запросы не находит, и бот отвечает, что баллов нет.
# Поэтому при поиске после строгих уровней идут мягкие: аббревиатура, код
# ФГОС, совпадение слов. Ответ всё равно подписан полным названием и
# факультетом, так что лишняя строка в выдаче видна модели, а не подменяет
# собой нужную.

_CODE_RE = re.compile(r"\d{2}\.\d{2}\.\d{2}")
_WORD_RE = re.compile(r"[a-zа-я0-9]+")
# Слова, которые описывают запрос, а не называют направление.
_NOISE_WORDS = frozenset(
    {
        "направление",
        "направления",
        "направлении",
        "программа",
        "программы",
        "профиль",
        "профиля",
        "специальность",
        "специальности",
        "бакалавриат",
        "бакалавриата",
        "специалитет",
        "специалитета",
        "нгу",
        "на",
        "по",
        "в",
    }
)
_CONJUNCTION = "и"


def _fold(value: str) -> str:
    """normalize_name + «ё» как «е»: в вопросах «ё» пишут через раз."""
    return normalize_name(value).replace("ё", "е")


def _words(value: str) -> list[str]:
    return _WORD_RE.findall(_fold(_CODE_RE.sub(" ", value)))


def _stem(word: str) -> str:
    """Грубая основа: отрезаем окончание, чтобы «прикладная» ~ «прикладной»."""
    return word[: max(4, len(word) - 2)]


def _acronyms(name: str) -> set[str]:
    """Аббревиатуры названия: «ПИиКН» и «ПИКН» для «Программная инженерия и
    компьютерные науки». Считаются для полного названия и для каждой его
    части «Поле. Профиль» — абитуриенты сокращают именно профиль."""
    result: set[str] = set()
    for part in {name, *name.split(_FIELD_PROFILE_SEP)}:
        words = _WORD_RE.findall(_fold(part))
        if len(words) < 2:
            continue
        with_conj = "".join(w if w == _CONJUNCTION else w[0] for w in words)
        without_conj = "".join(w[0] for w in words if w != _CONJUNCTION)
        result.update(a for a in (with_conj, without_conj) if len(a) >= 2)
    return result


def query_variants(value: str, inner: bool = True) -> list[str]:
    """Варианты строки запроса, от самого точного к более общим.

    Расширитель аббревиатур превращает «ПИиКН» в «ПИиКН (Программная инженерия
    и компьютерные науки)», и модель часто передаёт в инструмент именно эту
    форму. Пробуем её целиком, затем часть в скобках и часть до скобок.
    ``inner=False`` — без части в скобках.
    """
    variants = [value]
    match = re.match(r"^(.*?)\(([^()]*)\)\s*$", value.strip())
    if match:
        if inner:
            variants.append(match.group(2))
        variants.append(match.group(1))
    seen: set[str] = set()
    result: list[str] = []
    for variant in variants:
        if _fold(variant) and _fold(variant) not in seen:
            seen.add(_fold(variant))
            result.append(variant)
    return result


def match_programs(programs: Sequence[Program], query: str) -> list[Program]:
    """Направления, подходящие под формулировку из вопроса.

    Уровни по порядку, срабатывает первый непустой:
    1) точное имя, профиль, поле — как при импорте;
    2) аббревиатура названия или профиля («ПИиКН», «ПМИ», «КНиС»);
    3) код ФГОС («09.03.01»);
    4) значимые слова запроса есть в названии — подряд, вразброс, затем по
       основам («прикладная математика», «программная инженерия»).
    """
    target = _fold(query)
    if not target:
        return []

    tiers = (
        lambda p: _fold(p.name) == target,
        lambda p: _has_field_profile(p.name) and _fold(_profile_part(p.name)) == target,
        lambda p: _has_field_profile(p.name) and _fold(_field_part(p.name)) == target,
    )
    for tier in tiers:
        found = [p for p in programs if tier(p)]
        if found:
            return found

    compact = re.sub(r"[^a-zа-я0-9]", "", target)
    if " " not in target and len(compact) >= 3:
        found = [p for p in programs if compact in _acronyms(p.name)]
        if found:
            return found

    codes = set(_CODE_RE.findall(query))
    words = [w for w in _words(query) if w not in _NOISE_WORDS and w != _CONJUNCTION]
    if codes and not words:
        return [p for p in programs if p.code in codes]
    if not words:
        return []

    candidates = [p for p in programs if not codes or p.code in codes] or list(programs)
    phrase = f" {' '.join(words)} "
    for matches in (
        # Слова идут подряд: «прикладная математика» — это ПМИ, а не «Прикладные
        # математика и физика. Прикладная физика…», где те же слова вразброс.
        lambda p: phrase in f" {' '.join(_words(p.name))} ",
        lambda p: set(words) <= set(_words(p.name)),
    ):
        found = [p for p in candidates if matches(p)]
        if found:
            return found
    stems = {_stem(w) for w in words}
    return [p for p in candidates if stems <= {_stem(w) for w in _words(p.name)}]


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

    async def _find_faculty_loose(
        self, query: str, inner: bool = True
    ) -> Optional[Faculty]:
        """Факультет по формулировке из вопроса: «ФИТ», «ФИТ НГУ»,
        «ФИТ (Факультет информационных технологий)»."""
        for variant in query_variants(query, inner=inner):
            for candidate in (variant, re.sub(r"\bнгу\b", " ", _fold(variant))):
                faculty = await self.faculty_service.find_faculty_by_alias(candidate)
                if faculty is not None:
                    return faculty
        return None

    async def _match_program_ids_loose(
        self, query: str, faculty: Optional[Faculty], level: Optional[str]
    ) -> list[str]:
        """program_id направлений под формулировку из вопроса.

        Сначала ищем среди направлений указанного факультета, затем по всему
        справочнику: модель нередко угадывает факультет неверно (ПИиКН на
        ММФ), и тогда честнее показать баллы с подписью настоящего факультета,
        чем ответить, что данных нет. Название факультета в поле направления
        («ФИТ») означает все его направления — иначе «ФИТ» совпал бы с
        аббревиатурой «Физика и информационные технологии» физфака.
        """
        # Только направления, по которым есть баллы: иначе «Прикладные
        # математика и физика» могли бы совпасть с профилем без баллов вместо
        # строки поля, куда их записал импорт, и ответ снова был бы пустым.
        scored = set(
            (await self.session.execute(select(AdmissionScore.program_id).distinct()))
            .scalars()
            .all()
        )
        programs = [
            p
            for p in await self.faculty_service.get_all_programs(
                level=normalize_level(level)
            )
            if p.id in scored
        ]
        # Скобки не смотрим: в «Прикладная математика ФИТ (Факультет
        # информационных технологий)» факультет — уточнение, а не запрос.
        as_faculty = await self._find_faculty_loose(query, inner=False)
        if as_faculty is not None:
            return [p.id for p in programs if p.faculty_id == as_faculty.id]

        variants = query_variants(query)
        scopes: list[Sequence[Program]] = []
        if faculty is not None:
            scopes.append([p for p in programs if p.faculty_id == faculty.id])
        scopes.append(programs)

        for scope in scopes:
            for variant in variants:
                found = match_programs(scope, variant)
                if found:
                    return [p.id for p in found]

        return []

    async def upsert_from_rows(self, rows: Iterable[ScoreRow]) -> dict[str, int]:
        """Идемпотентный upsert по ключу (program_id, year, form).

        Разрешает program_id через факультет (по алиасу) и направление (по
        нормализованному имени). Если факультет или направление не находятся —
        строка молча пропускается (logger.info). Возвращает счётчики
        ``{"created": int, "updated": int, "skipped": int}``.

        Пустой балл в строке не затирает уже сохранённый: если НГУ поменяет
        вёрстку страницы итогов, парсер начнёт отдавать строки без чисел, и
        без этой проверки импорт молча обнулил бы баллы по всему университету.
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
                if row.passing_score is not None:
                    existing.passing_score = row.passing_score
                if row.average_score is not None:
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

        Факультет разрешается по алиасу, направление — по формулировке из
        вопроса (:func:`match_programs`: точное имя → профиль → поле →
        аббревиатура → код → слова). ``level=None`` (по умолчанию)
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

        resolved_faculty = None
        if faculty is not None:
            resolved_faculty = await self._find_faculty_loose(faculty)
            # Без направления фильтровать не по чему; с направлением —
            # нераспознанный факультет не повод терять ответ.
            if resolved_faculty is None and program is None:
                return []

        if program is not None:
            ids = await self._match_program_ids_loose(program, resolved_faculty, level)
            if not ids:
                return []
            stmt = stmt.where(AdmissionScore.program_id.in_(ids))
        elif resolved_faculty is not None:
            stmt = stmt.where(Program.faculty_id == resolved_faculty.id)

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
