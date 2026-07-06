"""CRAG (Corrective RAG) слой между ретривалом и генерацией.

Идея: после того как LightRAG вернул чанки-кандидаты, мы НЕ генерируем ответ
вслепую, а сначала «думаем»:

1. Понимаем, упомянут ли в вопросе конкретный факультет и/или уровень
   образования (через авторитетный справочник Faculty/Program).
2. LLM-грейдинг релевантности каждого чанка вопросу.
3. Авторитетная фильтрация по факультету: если в вопросе явно назван факультет,
   то чанки, говорящие про направления ЧУЖИХ факультетов, отсекаются. Именно это
   убирает кейс «проходные на ФИТ» → «бизнес-информатика» (она на экономфаке).
4. Если после фильтра кандидатов мало — одна попытка переформулировать запрос и
   до-достать чанки (без бесконечных циклов).

Слой включается/выключается и настраивается через переменные окружения, чтобы
его можно было быстро откатить (CRAG_ENABLED=false).
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from db.postgres.db import AsyncSessionLocal
from db.postgres.models import Faculty, Program
from db.postgres.services.faculty import FacultyService, normalize_name

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "да"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class CragConfig:
    """Конфигурация CRAG, читается из окружения."""

    enabled: bool = field(default_factory=lambda: _env_bool("CRAG_ENABLED", True))
    # Порог LLM-оценки релевантности чанка (0..1), ниже — отсекаем.
    relevance_threshold: float = field(
        default_factory=lambda: _env_float("CRAG_RELEVANCE_THRESHOLD", 0.5)
    )
    # Минимальное число чанков после фильтра, ниже которого пробуем доретрив.
    min_chunks: int = field(default_factory=lambda: _env_int("CRAG_MIN_CHUNKS", 2))
    # Разрешать ли одну попытку переформулировать запрос и до-достать чанки.
    allow_refine: bool = field(
        default_factory=lambda: _env_bool("CRAG_ALLOW_REFINE", True)
    )
    # Включать ли авторитетную фильтрацию по таблице факультетов.
    use_faculty_table: bool = field(
        default_factory=lambda: _env_bool("CRAG_USE_FACULTY_TABLE", True)
    )
    # Максимум чанков, отправляемых на LLM-грейдинг (контроль латентности).
    max_graded_chunks: int = field(
        default_factory=lambda: _env_int("CRAG_MAX_GRADED_CHUNKS", 12)
    )


def get_crag_config() -> CragConfig:
    """Синхронный конфиг из окружения (fallback для тестов/без БД)."""
    return CragConfig()


def _parse_bool(raw: Optional[str], default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "да"}


def _parse_float(raw: Optional[str], default: float) -> float:
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _parse_int(raw: Optional[str], default: int) -> int:
    try:
        return int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


async def load_crag_config() -> CragConfig:
    """Эффективный конфиг CRAG: значения из БД (settings) поверх env-дефолтов.

    Настройки управляются через веб-админку (таблица settings). Если ключа в БД
    нет — берётся значение из окружения/дефолт. При недоступности БД возвращаем
    env-конфиг (fail-safe), чтобы бот продолжал работать.
    """
    cfg = CragConfig()
    try:
        from db.postgres.services.settings import SettingsService

        async with AsyncSessionLocal() as session:
            raw = await SettingsService(session).get_crag_raw()
    except Exception as exc:
        logger.warning(
            "CRAG: не удалось загрузить настройки из БД, используем env/дефолты: %s",
            exc,
        )
        return cfg

    if not raw:
        return cfg

    return CragConfig(
        enabled=_parse_bool(raw.get("crag_enabled"), cfg.enabled),
        relevance_threshold=_parse_float(
            raw.get("crag_relevance_threshold"), cfg.relevance_threshold
        ),
        min_chunks=_parse_int(raw.get("crag_min_chunks"), cfg.min_chunks),
        allow_refine=_parse_bool(raw.get("crag_allow_refine"), cfg.allow_refine),
        use_faculty_table=_parse_bool(
            raw.get("crag_use_faculty_table"), cfg.use_faculty_table
        ),
        max_graded_chunks=_parse_int(
            raw.get("crag_max_graded_chunks"), cfg.max_graded_chunks
        ),
    )


@dataclass
class FacultyHint:
    """Результат разбора вопроса: какой факультет/уровень в нём упомянут."""

    faculty: Optional[Faculty] = None
    level: Optional[str] = None


# Маркеры уровня образования в тексте вопроса.
_LEVEL_MARKERS: dict[str, tuple[str, ...]] = {
    "master": ("магистратур", "магистр", "магистерск"),
    "specialist": ("специалитет", "специалист"),
    "bachelor": ("бакалавриат", "бакалавр", "бакалаврск"),
}


def detect_level(question: str) -> Optional[str]:
    """Определяет уровень образования по тексту вопроса (или None)."""
    lowered = question.lower()
    for level, markers in _LEVEL_MARKERS.items():
        if any(marker in lowered for marker in markers):
            return level
    return None


async def detect_faculty_hint(question: str) -> FacultyHint:
    """Ищет в вопросе упоминание факультета (по названию/алиасу) и уровня.

    Сопоставление идёт по справочнику Faculty: проверяем, встречается ли имя
    факультета или любой из его алиасов как отдельное слово/подстрока вопроса.
    """
    level = detect_level(question)
    normalized_q = normalize_name(question)
    if not normalized_q:
        return FacultyHint(faculty=None, level=level)

    # Готовим набор «слов» вопроса для матчинга коротких аббревиатур (ФИТ, ЭФ),
    # чтобы случайно не словить аббревиатуру внутри другого слова.
    question_tokens = set(re.findall(r"[\wа-яё]+", normalized_q))

    try:
        async with AsyncSessionLocal() as session:
            faculties = await FacultyService(session).get_all_faculties(only_active=True)
            best: Optional[Faculty] = None
            best_len = 0
            for faculty in faculties:
                candidates = [faculty.name, *(faculty.aliases or [])]
                for candidate in candidates:
                    norm = normalize_name(candidate)
                    if not norm:
                        continue
                    matched = False
                    if " " in norm:
                        # Многословное название — ищем как подстроку.
                        matched = norm in normalized_q
                    else:
                        # Одно слово/аббревиатура — точное совпадение токена.
                        matched = norm in question_tokens
                    # Берём самое длинное совпадение (наиболее специфичное).
                    if matched and len(norm) > best_len:
                        best = faculty
                        best_len = len(norm)
            return FacultyHint(faculty=best, level=level)
    except Exception as exc:
        logger.warning("CRAG: не удалось определить факультет из вопроса: %s", exc)
        return FacultyHint(faculty=None, level=level)


@dataclass
class CragChunk:
    """Нормализованный чанк-кандидат для CRAG."""

    index: int
    content: str
    source_url: Optional[str]
    file_path: Optional[str]


async def _faculty_conflict(chunk: CragChunk, hint: FacultyHint) -> bool:
    """True, если чанк явно говорит про направление ЧУЖОГО факультета.

    Логика: если в вопросе назван факультет X, а в тексте чанка встречается
    название направления, которое по справочнику принадлежит факультету Y != X,
    считаем чанк конфликтующим и отсекаем. Если направление принадлежит X или не
    найдено в справочнике — конфликта нет (не отсекаем по этому правилу).
    """
    if hint.faculty is None:
        return False

    content_norm = normalize_name(chunk.content)
    if not content_norm:
        return False

    try:
        async with AsyncSessionLocal() as session:
            service = FacultyService(session)
            # Все направления чужих факультетов на нужном уровне.
            programs = await service.get_all_programs(level=hint.level, only_active=True)
            for program in programs:
                if program.faculty_id == hint.faculty.id:
                    continue
                prog_norm = normalize_name(program.name)
                if not prog_norm or len(prog_norm) < 4:
                    continue
                if prog_norm in content_norm:
                    # Убедимся, что направление НЕ принадлежит и нужному факультету
                    # (одноимённые направления на разных факультетах).
                    own = await _program_belongs_to_faculty(service, program, hint)
                    if not own:
                        logger.info(
                            "CRAG: чанк #%d отсечён — упоминает '%s' "
                            "(факультет %s), а спрашивали про %s",
                            chunk.index,
                            program.name,
                            program.faculty_id,
                            hint.faculty.name,
                        )
                        return True
    except Exception as exc:
        logger.warning("CRAG: ошибка авторитетной фильтрации: %s", exc)
        return False
    return False


async def _program_belongs_to_faculty(
    service: FacultyService, program: Program, hint: FacultyHint
) -> bool:
    """Есть ли одноимённое направление и у факультета из вопроса."""
    if hint.faculty is None:
        return False
    own_programs = await service.get_programs_by_faculty(
        hint.faculty.id, level=hint.level, only_active=True
    )
    target = normalize_name(program.name)
    return any(normalize_name(p.name) == target for p in own_programs)


def _build_grading_prompt(
    question: str, hint: FacultyHint, chunk: CragChunk
) -> tuple[str, str]:
    faculty_clause = ""
    if hint.faculty is not None:
        faculty_clause = (
            f"\nВ вопросе явно назван факультет: «{hint.faculty.name}». "
            "Фрагмент релевантен ТОЛЬКО если относится именно к этому факультету. "
            "Если фрагмент про другой факультет/направление другого факультета — "
            "он НЕ релевантен."
        )
    level_clause = ""
    if hint.level is not None:
        level_map = {
            "bachelor": "бакалавриат",
            "specialist": "специалитет",
            "master": "магистратура",
        }
        level_clause = (
            f"\nУровень образования в вопросе: {level_map.get(hint.level, hint.level)}. "
            "Фрагмент про другой уровень считается нерелевантным."
        )

    system_prompt = (
        "Ты — строгий оценщик релевантности (grader) для системы Corrective RAG. "
        "Оцени, помогает ли фрагмент ответить на вопрос пользователя. "
        "Верни СТРОГО JSON без обрамления: "
        '{"score": <число от 0 до 1>, "relevant": <true|false>}. '
        "score=1 — фрагмент прямо отвечает на вопрос; "
        "score=0 — фрагмент не относится к вопросу."
        f"{faculty_clause}{level_clause}"
    )
    user_prompt = f"Вопрос: {question}\n\nФрагмент:\n{chunk.content}"
    return system_prompt, user_prompt


def _parse_grade(raw: str) -> tuple[float, Optional[bool]]:
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return 0.0, None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return 0.0, None
    score = data.get("score")
    relevant = data.get("relevant")
    try:
        score_val = float(score)
    except (TypeError, ValueError):
        score_val = 0.0
    score_val = max(0.0, min(1.0, score_val))
    relevant_val = relevant if isinstance(relevant, bool) else None
    return score_val, relevant_val


async def grade_chunk(
    question: str, hint: FacultyHint, chunk: CragChunk, config: CragConfig
) -> bool:
    """LLM-грейдинг одного чанка. True — оставить, False — отсеять."""
    from llm.factory import get_llm_provider
    from llm.profiles import LLMProfiles

    system_prompt, user_prompt = _build_grading_prompt(question, hint, chunk)
    provider = get_llm_provider()
    try:
        raw = await provider.generate(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ],
            profile=LLMProfiles.INTENT,
        )
    except Exception as exc:
        logger.warning("CRAG: ошибка грейдинга чанка #%d: %s", chunk.index, exc)
        # Fail-open: при ошибке LLM не теряем потенциально полезный чанк.
        return True

    score, relevant = _parse_grade(raw)
    keep = score >= config.relevance_threshold
    if relevant is False and score < config.relevance_threshold:
        keep = False
    logger.info(
        "CRAG grade: chunk #%d score=%.2f relevant=%s keep=%s",
        chunk.index,
        score,
        relevant,
        keep,
    )
    return keep


async def filter_chunks(
    question: str,
    chunks: list[CragChunk],
    config: Optional[CragConfig] = None,
) -> tuple[list[CragChunk], FacultyHint]:
    """Применяет CRAG-фильтр к списку чанков.

    Порядок: сначала дешёвая авторитетная фильтрация по таблице факультетов
    (без LLM), затем LLM-грейдинг релевантности по оставшимся (с ограничением
    max_graded_chunks для контроля латентности).
    """
    config = config or get_crag_config()
    hint = await detect_faculty_hint(question)

    if not chunks:
        return [], hint

    # 1. Авторитетная фильтрация по факультету (без LLM).
    after_authority: list[CragChunk] = []
    if config.use_faculty_table and hint.faculty is not None:
        for chunk in chunks:
            if await _faculty_conflict(chunk, hint):
                continue
            after_authority.append(chunk)
    else:
        after_authority = list(chunks)

    # 2. LLM-грейдинг релевантности.
    to_grade = after_authority[: config.max_graded_chunks]
    rest = after_authority[config.max_graded_chunks :]
    kept: list[CragChunk] = []
    for chunk in to_grade:
        if await grade_chunk(question, hint, chunk, config):
            kept.append(chunk)
    # Чанки за пределами лимита грейдинга оставляем как есть (уже прошли авторитет).
    kept.extend(rest)

    logger.info(
        "CRAG filter: input=%d after_authority=%d kept=%d faculty=%s level=%s",
        len(chunks),
        len(after_authority),
        len(kept),
        hint.faculty.name if hint.faculty else None,
        hint.level,
    )
    return kept, hint


async def refine_query(question: str, hint: FacultyHint) -> str:
    """Одна попытка переформулировать запрос для повторного ретрива."""
    from llm.factory import get_llm_provider
    from llm.profiles import LLMProfiles

    faculty_clause = (
        f" Сохрани упоминание факультета «{hint.faculty.name}»."
        if hint.faculty is not None
        else ""
    )
    system_prompt = (
        "Ты помогаешь улучшить поисковый запрос к базе знаний. "
        "Переформулируй вопрос так, чтобы он точнее находил релевантные документы: "
        "добавь ключевые синонимы, раскрой аббревиатуры." + faculty_clause + " "
        "Верни ТОЛЬКО переформулированный запрос одной строкой, без пояснений."
    )
    provider = get_llm_provider()
    try:
        raw = await provider.generate(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question),
            ],
            profile=LLMProfiles.INTENT,
        )
    except Exception as exc:
        logger.warning("CRAG: ошибка переформулировки запроса: %s", exc)
        return question
    refined = (raw or "").strip().splitlines()
    result = refined[0].strip() if refined else ""
    return result or question


CragRetrieveFn = Callable[[str], Awaitable[list[CragChunk]]]


async def run_crag(
    question: str,
    chunks: list[CragChunk],
    retrieve_fn: Optional[CragRetrieveFn] = None,
    config: Optional[CragConfig] = None,
) -> tuple[list[CragChunk], FacultyHint]:
    """Полный CRAG-цикл: фильтр → (при нехватке) одна доретрив-итерация.

    ``retrieve_fn`` — функция, выполняющая повторный ретрив по новому запросу.
    Если она не передана или доретрив выключен, ограничиваемся одним проходом.
    """
    config = config or get_crag_config()
    kept, hint = await filter_chunks(question, chunks, config)

    if len(kept) >= config.min_chunks or not config.allow_refine or retrieve_fn is None:
        return kept, hint

    logger.info(
        "CRAG: после фильтра осталось %d (<%d) — пробуем доретрив",
        len(kept),
        config.min_chunks,
    )
    refined = await refine_query(question, hint)
    if normalize_name(refined) == normalize_name(question):
        return kept, hint

    try:
        new_chunks = await retrieve_fn(refined)
    except Exception as exc:
        logger.warning("CRAG: ошибка доретрива: %s", exc)
        return kept, hint

    # Объединяем уникальные по content чанки и фильтруем повторно.
    seen = {c.content for c in kept}
    merged = list(kept)
    for c in new_chunks:
        if c.content not in seen:
            merged.append(c)
            seen.add(c.content)

    refined_kept, _ = await filter_chunks(question, merged, config)
    logger.info(
        "CRAG: после доретрива kept=%d (было %d)",
        len(refined_kept),
        len(kept),
    )
    # Возвращаем лучший из двух наборов.
    return (refined_kept if len(refined_kept) >= len(kept) else kept), hint
