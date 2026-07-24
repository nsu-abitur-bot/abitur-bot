"""Сервис для справочника факультетов и направлений подготовки.

Этот справочник используется как авторитетный источник для CRAG-фильтрации:
он позволяет узнать, какие направления принадлежат конкретному факультету и
уровню образования, а также найти факультет по названию/алиасу/аббревиатуре.
"""

import re
from typing import Iterable, Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import EDUCATION_LEVELS, EducationLevel, Faculty, Program


def normalize_name(value: str) -> str:
    """Нормализует название/алиас для нечувствительного к регистру сравнения."""
    value = (value or "").strip().lower()
    # Схлопываем пробелы и убираем кавычки/лишнюю пунктуацию по краям.
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \t\n«»\"'.,")


def normalize_level(level: Optional[str]) -> Optional[str]:
    """Приводит уровень образования к каноническому значению или None."""
    if not level:
        return None
    normalized = level.strip().lower()
    if normalized in EDUCATION_LEVELS:
        return normalized
    return None


class FacultyService:
    """CRUD и выборки по факультетам и направлениям подготовки."""

    def __init__(self, session: AsyncSession):
        self.session = session

    # --- Факультеты ---------------------------------------------------------

    async def create_faculty(
        self,
        name: str,
        aliases: Optional[Iterable[str]] = None,
        is_active: bool = True,
    ) -> Faculty:
        faculty = Faculty(
            name=name.strip(),
            aliases=[a.strip() for a in (aliases or []) if a and a.strip()],
            is_active=is_active,
        )
        self.session.add(faculty)
        await self.session.commit()
        await self.session.refresh(faculty)
        return faculty

    async def get_faculty_by_id(self, faculty_id: str) -> Optional[Faculty]:
        stmt = select(Faculty).where(Faculty.id == faculty_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_faculty_by_name(self, name: str) -> Optional[Faculty]:
        """Точный (нечувствительный к регистру) поиск факультета по названию."""
        normalized = normalize_name(name)
        stmt = select(Faculty).where(func.lower(Faculty.name) == normalized)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all_faculties(self, only_active: bool = True) -> Sequence[Faculty]:
        stmt = select(Faculty)
        if only_active:
            stmt = stmt.where(Faculty.is_active)
        stmt = stmt.order_by(Faculty.name)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def find_faculty_by_alias(
        self, query: str, only_active: bool = True
    ) -> Optional[Faculty]:
        """Находит факультет по названию, аббревиатуре или алиасу.

        Сравнение нечувствительно к регистру и устойчиво к лишним пробелам и
        кавычкам. Сначала пробуем точное совпадение по имени, затем по алиасам.
        """
        normalized = normalize_name(query)
        if not normalized:
            return None

        faculties = await self.get_all_faculties(only_active=only_active)

        for faculty in faculties:
            if normalize_name(faculty.name) == normalized:
                return faculty

        for faculty in faculties:
            for alias in faculty.aliases or []:
                if normalize_name(alias) == normalized:
                    return faculty
        return None

    async def update_faculty(
        self,
        faculty_id: str,
        name: Optional[str] = None,
        aliases: Optional[Iterable[str]] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[Faculty]:
        faculty = await self.get_faculty_by_id(faculty_id)
        if faculty is None:
            return None
        if name is not None:
            faculty.name = name.strip()
        if aliases is not None:
            faculty.aliases = [a.strip() for a in aliases if a and a.strip()]
        if is_active is not None:
            faculty.is_active = is_active
        await self.session.commit()
        await self.session.refresh(faculty)
        return faculty

    async def delete_faculty(self, faculty_id: str) -> bool:
        faculty = await self.get_faculty_by_id(faculty_id)
        if faculty is None:
            return False
        await self.session.delete(faculty)
        await self.session.commit()
        return True

    # --- Направления --------------------------------------------------------

    async def create_program(
        self,
        faculty_id: str,
        name: str,
        level: str = EducationLevel.bachelor.value,
        code: Optional[str] = None,
        is_active: bool = True,
    ) -> Program:
        canonical_level = normalize_level(level)
        if canonical_level is None:
            raise ValueError(
                f"Недопустимый уровень образования: '{level}'. "
                f"Допустимые значения: {sorted(EDUCATION_LEVELS)}"
            )
        program = Program(
            faculty_id=faculty_id,
            name=name.strip(),
            level=canonical_level,
            code=code.strip() if code else None,
            is_active=is_active,
        )
        self.session.add(program)
        await self.session.commit()
        await self.session.refresh(program)
        return program

    async def get_program_by_id(self, program_id: str) -> Optional[Program]:
        stmt = select(Program).where(Program.id == program_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_programs_by_faculty(
        self,
        faculty_id: str,
        level: Optional[str] = None,
        only_active: bool = True,
    ) -> Sequence[Program]:
        """Возвращает направления факультета, опционально фильтруя по уровню."""
        stmt = select(Program).where(Program.faculty_id == faculty_id)
        if only_active:
            stmt = stmt.where(Program.is_active)
        canonical_level = normalize_level(level)
        if canonical_level is not None:
            stmt = stmt.where(Program.level == canonical_level)
        stmt = stmt.order_by(Program.name)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_all_programs(
        self, level: Optional[str] = None, only_active: bool = True
    ) -> Sequence[Program]:
        stmt = select(Program)
        if only_active:
            stmt = stmt.where(Program.is_active)
        canonical_level = normalize_level(level)
        if canonical_level is not None:
            stmt = stmt.where(Program.level == canonical_level)
        stmt = stmt.order_by(Program.name)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def find_faculty_for_program(
        self, program_name: str, level: Optional[str] = None
    ) -> Optional[Faculty]:
        """Находит факультет, которому принадлежит направление с данным именем.

        Используется CRAG-фильтром, чтобы определить, на каком факультете
        реально находится упомянутое в чанке направление.
        """
        normalized = normalize_name(program_name)
        if not normalized:
            return None
        stmt = (
            select(Program)
            .where(func.lower(Program.name) == normalized)
            .where(Program.is_active)
        )
        canonical_level = normalize_level(level)
        if canonical_level is not None:
            stmt = stmt.where(Program.level == canonical_level)
        result = await self.session.execute(stmt)
        program = result.scalars().first()
        if program is None:
            return None
        return await self.get_faculty_by_id(program.faculty_id)

    async def update_program(
        self,
        program_id: str,
        name: Optional[str] = None,
        level: Optional[str] = None,
        code: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[Program]:
        program = await self.get_program_by_id(program_id)
        if program is None:
            return None
        if name is not None:
            program.name = name.strip()
        if level is not None:
            canonical_level = normalize_level(level)
            if canonical_level is None:
                raise ValueError(f"Недопустимый уровень образования: '{level}'")
            program.level = canonical_level
        if code is not None:
            program.code = code.strip() or None
        if is_active is not None:
            program.is_active = is_active
        await self.session.commit()
        await self.session.refresh(program)
        return program

    async def delete_program(self, program_id: str) -> bool:
        program = await self.get_program_by_id(program_id)
        if program is None:
            return False
        await self.session.delete(program)
        await self.session.commit()
        return True

    # --- Наполнение из сида -------------------------------------------------

    async def upsert_from_seed(
        self, faculties: Sequence[dict], soft: bool = False
    ) -> dict[str, int]:
        """Идемпотентно загружает справочник из сид-структуры.

        ``soft=True`` — «мягкий» режим: добавляется только то, чего ещё нет
        (новые факультеты, новые алиасы, новые направления, код у направления
        без кода). Существующие данные НЕ затираются: алиасы доливаются
        объединением, ``is_active`` и уже заполненные коды не трогаются. Это
        безопасно гонять при каждом старте — правки из админки переживают
        деплой, а новые записи из сида подхватываются.

        ``soft=False`` (по умолчанию) — жёсткая перезапись: алиасы и коды
        заменяются значениями из сида, ``is_active`` принудительно True.

        Формат каждого элемента ``faculties``::

            {
                "name": "Факультет информационных технологий",
                "aliases": ["ФИТ"],
                "programs": [
                    {"name": "Программная инженерия",
                     "code": "09.03.04", "level": "bachelor"},
                    ...
                ],
            }

        Возвращает счётчики созданных/обновлённых факультетов и направлений.
        """
        stats = {
            "faculties_created": 0,
            "faculties_updated": 0,
            "programs_created": 0,
            "programs_updated": 0,
        }

        for raw_faculty in faculties:
            name = str(raw_faculty.get("name", "")).strip()
            if not name:
                continue
            aliases = [
                str(a).strip() for a in raw_faculty.get("aliases", []) if str(a).strip()
            ]

            faculty = await self.get_faculty_by_name(name)
            if faculty is None:
                faculty = Faculty(name=name, aliases=aliases, is_active=True)
                self.session.add(faculty)
                await self.session.flush()
                stats["faculties_created"] += 1
            elif soft:
                # Доливаем только недостающие алиасы; is_active не трогаем.
                existing_aliases = list(faculty.aliases or [])
                known = {normalize_name(a) for a in existing_aliases}
                added: list[str] = []
                for alias in aliases:
                    alias_key = normalize_name(alias)
                    if alias_key and alias_key not in known:
                        known.add(alias_key)
                        added.append(alias)
                if added:
                    faculty.aliases = existing_aliases + added
                    stats["faculties_updated"] += 1
            else:
                faculty.aliases = aliases
                faculty.is_active = True
                stats["faculties_updated"] += 1

            existing = await self.get_programs_by_faculty(faculty.id, only_active=False)
            existing_by_key = {(normalize_name(p.name), p.level): p for p in existing}

            for raw_program in raw_faculty.get("programs", []):
                prog_name = str(raw_program.get("name", "")).strip()
                if not prog_name:
                    continue
                level = normalize_level(raw_program.get("level"))
                if level is None:
                    raise ValueError(
                        f"Недопустимый уровень у направления '{prog_name}': "
                        f"{raw_program.get('level')!r}"
                    )
                code = raw_program.get("code")
                code = str(code).strip() if code else None

                key = (normalize_name(prog_name), level)
                program = existing_by_key.get(key)
                if program is None:
                    self.session.add(
                        Program(
                            faculty_id=faculty.id,
                            name=prog_name,
                            level=level,
                            code=code,
                            is_active=True,
                        )
                    )
                    stats["programs_created"] += 1
                elif soft:
                    # Заполняем только пустой код; is_active не трогаем.
                    if code and not program.code:
                        program.code = code
                        stats["programs_updated"] += 1
                else:
                    program.code = code
                    program.is_active = True
                    stats["programs_updated"] += 1

        await self.session.commit()
        return stats
