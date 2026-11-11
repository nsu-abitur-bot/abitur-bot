"""Поиск требований ЕГЭ по метаданным и чанкам одной карточки."""

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from lightrag import LightRAG

from db.postgres.services.faculty import normalize_name
from rag.crag import detect_level
from rag.trace import LightragTrace, RetrievalResult

logger = logging.getLogger(__name__)

_CODE = re.compile(r"(?<![\w.])\d{2}\.\d{2}\.\d{2}(?!\w|\.\d)")
_EGE = re.compile(r"\bегэ\b|\bедин\w+\s+государственн\w+\s+экзамен\w*", re.I)
_EXAMS = re.compile(r"\bэкзамен\w*|\bвступительн\w+\s+испытан\w*", re.I)
_OTHER_EXAMS = re.compile(r"внутренн|расписани|\bкогда\b|после\s+спо", re.I)
_ADMISSION_SCORES = re.compile(r"\b(?:проходн\w*|средн\w*)\s+балл\w*\b", re.I)
_HISTORY_ROLE = re.compile(r"(?m)^(user|assistant|system): ")
_CLARIFY_PROFILE = "Уточните профиль, чтобы я назвал требования к ЕГЭ."
_SUBJECT = re.compile(
    r"математик|русск\w*\s+язык|физик|информатик|географ|хими|биологи|"
    r"истори|обществознани|литератур|иностранн\w*\s+язык|английск\w*\s+язык",
    re.I,
)
_EXAM_GROUP = re.compile(
    r"^(?:обязательн|по выбору|(?:один|одна)\s+на выбор)|\b(?:по|на)\s+выбор\w*",
    re.I,
)
_SECTION_END = re.compile(
    r"^(?:для поступающих после|после\s+(?:спо|колледж)|внутренние|"
    r"срок(?:и)?\s+(?:при[её]ма|обучения)|форма обучения|бюджетные места|"
    r"платные места|стоимость|контакты|что нужно знать)",
    re.I,
)
_INLINE_HEADING = re.compile(
    r"Единый\s+государственный\s+экзамен\s*\(ЕГЭ\)|"
    r"Для\s+поступающих\s+после\s+среднего\s+профессионального\s+образования|"
    r"Что\s+нужно\s+знать\s+для\s+поступления"
)
_PROFILE = re.compile(r"^(?:профиль(?: подготовки)?|специализация)\s*[:—–-]\s*(.+)", re.I)
_PROFILE_REQUEST = re.compile(
    r"\b(?:профиль|профиля|профиле|профилю|профилем|специализаци\w*)\b"
    r"(?:\s+подготовки)?(?:\s*[:«\"—–-]|\s+(?!на\b|для\b|в\b)[а-яёa-z])",
    re.I,
)
_CODE_HEADING = re.compile(
    r"^(?:(?:код(?: (?:специальности|направления(?: подготовки)?))?|"
    r"специальность|направление(?: подготовки)?)\s*[:—–-]?\s*)?"
    r"(\d{2}\.\d{2}\.\d{2})(?!\w|\.\d)",
    re.I,
)
_GENERIC_TITLE = re.compile(
    r"^(?:правила|список|перечень|условия|порядок|проходные|минимальные|"
    r"документы|сроки|экзамены|егэ|каталог)\b",
    re.I,
)
_CODE_LEVEL = {"02": "spo", "03": "bachelor", "04": "master", "05": "specialist"}
_LEVEL_LABEL = {
    "spo": "СПО",
    "bachelor": "бакалавриат",
    "master": "магистратура",
    "specialist": "специалитет",
}


@dataclass(frozen=True)
class ProgramCard:
    doc_id: str
    url: str | None
    title: str
    code: str
    name: str
    profile: str
    level: str | None

    @property
    def label(self) -> str:
        label = f"{self.code} «{self.name}»" if self.name else self.code
        if self.profile:
            return f"{label} — {self.profile}"
        return label or self.title


def is_program_exam_question(question: str) -> bool:
    """Распознаёт запрос требований к экзаменам до поиска по карточкам."""
    return (
        not _OTHER_EXAMS.search(question)
        and not _ADMISSION_SCORES.search(question)
        and bool(_EGE.search(question) or _EXAMS.search(question))
    )


def _pending_exam_question(history: str | None) -> str | None:
    if history:
        parts = _HISTORY_ROLE.split(history)
        for index in range(len(parts) - 2, 2, -2):
            if parts[index] == "assistant":
                if parts[index + 1].startswith(_CLARIFY_PROFILE):
                    if parts[index - 2] == "user":
                        return parts[index - 1].strip()
                break
    return None


def can_query_program_exams(question: str, history: str | None) -> bool:
    """Проверяет, нужен ли резолвер карточек, включая уточнение профиля."""
    return not _ADMISSION_SCORES.search(question) and (
        is_program_exam_question(question) or _pending_exam_question(history) is not None
    )


def _heading(line: str) -> str:
    return line.strip().strip("#* ").strip()


def _name_key(text: str) -> str:
    return " ".join(re.findall(r"\w+", normalize_name(text).replace("ё", "е")))


def _mentioned(name: str, question_key: str) -> bool:
    key = _name_key(name)
    return bool(key) and f" {key} " in question_key


def read_program_card(
    doc_id: str, url: str | None, title: str, summary: str
) -> ProgramCard | None:
    """Восстанавливает метаданные старых карточек из заголовка и начала текста."""
    lines = [_heading(line) for line in summary.splitlines() if line.strip()]
    first_line = lines[0] if lines else ""
    if _CODE_HEADING.match(first_line):
        heading = first_line
    else:
        early_lines = [line for line in lines[1:4] if _CODE_HEADING.match(line)]
        heading = early_lines[0] if len(early_lines) == 1 else title
    codes = set(_CODE.findall(heading))
    # Заголовок списка программ не идентифицирует отдельную карточку.
    if len(codes) != 1:
        return None
    code = next(iter(codes))
    heading_match = _CODE_HEADING.match(heading)
    raw_name = (
        heading[heading_match.end() :]
        if heading_match
        else _CODE.sub("", heading, count=1)
    ).strip(" \t«»()—–-:.,")
    if not raw_name:
        raw_name = title if title and not _CODE.search(title) else ""
        if not raw_name and not _CODE.search(first_line):
            raw_name = first_line
        if not raw_name and len(lines) > 1 and detect_level(lines[1]) is None:
            raw_name = lines[1]
    name, _, profile = re.sub(r"\s*\|.*$", "", raw_name).partition(" — ")
    if not profile:
        name, _, profile = re.sub(r"\s[-–]\s", " — ", name, count=1).partition(" — ")
    profile = next(
        (match[1] for line in lines[1:] if (match := _PROFILE.match(line))),
        profile,
    )
    name = name.strip(" \t«»()")
    if _GENERIC_TITLE.match(name) or (not name and not heading_match):
        return None
    level = _CODE_LEVEL.get(code.split(".")[1])
    for line in lines[:3]:
        declared = detect_level(line)
        if declared is not None and (_CODE.search(line) or len(line.split()) <= 3):
            level = declared
            break
    return ProgramCard(doc_id, url, title or name or code, code, name, profile, level)


def _select_cards(
    question: str, cards: list[ProgramCard], trace: LightragTrace
) -> tuple[list[ProgramCard], ProgramCard | None]:
    codes = set(_CODE.findall(question))
    requested_level = detect_level(question)
    explicit_profile = bool(_PROFILE_REQUEST.search(question))
    question_key = f" {_name_key(question)} "
    candidates: list[tuple[int, ProgramCard]] = []
    recognized = None
    rejected = []
    for card in cards:
        if codes and card.code not in codes:
            rejected.append((card.doc_id, "code_mismatch"))
            continue
        profile_match = _mentioned(card.profile, question_key)
        name_match = _mentioned(card.name, question_key)
        if not codes and not (profile_match or name_match):
            rejected.append((card.doc_id, "name_mismatch"))
            continue
        recognized = card
        if explicit_profile and not profile_match:
            rejected.append((card.doc_id, "profile_mismatch"))
            continue
        if requested_level and card.level and card.level != requested_level:
            rejected.append((card.doc_id, "level_mismatch"))
            continue
        score = len(_name_key(card.name)) if name_match and not codes else 0
        if profile_match:
            score += len(_name_key(card.profile))
        candidates.append((score, card))

    best = max((score for score, _ in candidates), default=0)
    selected = []
    for score, card in candidates:
        if score < best:
            rejected.append((card.doc_id, "less_specific"))
            continue
        trace.note(
            f"program_exam doc_id={card.doc_id} selected code={card.code}"
            f" level={card.level}"
        )
        selected.append(card)
    for doc_id, reason in rejected[:24]:
        trace.note(f"program_exam doc_id={doc_id} rejected={reason}")
    if len(rejected) > 24:
        trace.note(f"program_exam other_rejections={len(rejected) - 24}")
    return selected, recognized


def _section_lines(text: str) -> Iterator[str]:
    # Старый HTML-импорт соединял заголовки и списки внутри main без пробелов.
    for line in text.splitlines(keepends=True):
        start = 0
        for marker in _INLINE_HEADING.finditer(line):
            if line[start : marker.start()].strip("#* \t"):
                yield line[start : marker.start()]
                start = marker.start()
        yield line[start:]


def _exam_spans(text: str, code: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    offset = 0
    current_code = code
    for line in _section_lines(text):
        heading = _heading(line)
        code_heading = _CODE_HEADING.match(heading)
        if code_heading:
            current_code = code_heading[1]
        is_ege = (
            current_code == code
            and bool(_EGE.search(heading))
            and not _SECTION_END.match(heading)
        )
        if (
            start is not None
            and not is_ege
            and (
                code_heading
                or _SECTION_END.match(heading)
                or (line.lstrip().startswith("#") and not _EXAM_GROUP.search(heading))
            )
        ):
            spans.append((start, offset))
            start = None
        if (
            is_ege
            and start is None
            and (
                line.lstrip().startswith("#")
                or len(heading.split()) <= 8
                or _SUBJECT.search(heading)
            )
        ):
            start = offset
        offset += len(line)
    if start is not None:
        spans.append((start, len(text)))
    return [(begin, end) for begin, end in spans if _SUBJECT.search(text[begin:end])]


@dataclass(frozen=True)
class ExamChunk:
    chunk_id: str
    section: int
    start: int
    end: int


@dataclass
class IndexedExams:
    card: ProgramCard
    text: str
    chunks: list[ExamChunk]

    def passages(self) -> dict[int, list[ExamChunk]]:
        sections: dict[int, list[ExamChunk]] = {}
        for chunk in self.chunks:
            sections.setdefault(chunk.section, []).append(chunk)
        return sections

    def render(self, chunks: list[ExamChunk]) -> str:
        return self.text[chunks[0].start : max(chunk.end for chunk in chunks)].strip()


async def _get_indexed_exams(
    rag: LightRAG, card: ProgramCard, trace: LightragTrace
) -> IndexedExams:
    status = await rag.doc_status.get_by_id(card.doc_id)
    full_doc = await rag.full_docs.get_by_id(card.doc_id)
    text = str((full_doc or {}).get("content") or "")
    spans = _exam_spans(text, card.code)
    ids = (status or {}).get("chunks_list") or []
    chunks: list[ExamChunk] = []
    previous_start = -1
    for chunk_id, stored in zip(ids, await rag.text_chunks.get_by_ids(ids)):
        content = str((stored or {}).get("content") or "").strip(" \r\n\t\ufffd")
        start = text.find(content, previous_start + 1) if content else -1
        if start < 0:
            trace.note(
                f"program_exam doc_id={card.doc_id} chunk_id={chunk_id}"
                " rejected=chunk_unavailable"
            )
            continue
        previous_start = start
        end = start + len(content)
        found = False
        for section, (begin, finish) in enumerate(spans):
            if start < finish and begin < end:
                chunks.append(
                    ExamChunk(chunk_id, section, max(start, begin), min(end, finish))
                )
                found = True
        if not found:
            trace.note(
                f"program_exam doc_id={card.doc_id} chunk_id={chunk_id}"
                " rejected=outside_exam_section"
            )
    indexed = IndexedExams(card, text, chunks)
    complete: list[ExamChunk] = []
    for section, pieces in indexed.passages().items():
        begin, end = spans[section]
        covered = begin
        for piece in pieces:
            if piece.start > covered and text[covered : piece.start].strip():
                break
            covered = max(covered, piece.end)
        if not text[covered:end].strip():
            complete.extend(pieces)
        else:
            for piece in pieces:
                trace.note(
                    f"program_exam doc_id={card.doc_id} chunk_id={piece.chunk_id}"
                    " rejected=incomplete_exam_section"
                )
    return IndexedExams(card, text, complete)


def _requirements(exams: IndexedExams) -> tuple[str, ...]:
    return tuple(
        sorted(_name_key(exams.render(chunks)) for chunks in exams.passages().values())
    )


async def _rank_sections(
    rag: LightRAG, question: str, exams: IndexedExams, trace: LightragTrace
) -> list[str]:
    sections = exams.passages()
    scores = {}
    embed = rag.embedding_func
    if len(exams.chunks) > 1 and embed is not None:
        ids = list(dict.fromkeys(chunk.chunk_id for chunk in exams.chunks))
        try:
            vectors = await rag.chunks_vdb.get_vectors_by_ids(ids)
            embedding = await embed([question])
            query = np.asarray(embedding[0], dtype=float)
            query_norm = np.linalg.norm(query)
            for chunk_id, values in vectors.items():
                vector = np.asarray(values, dtype=float)
                norm = query_norm * np.linalg.norm(vector)
                if norm:
                    scores[chunk_id] = float(query @ vector / norm)
        except Exception as exc:
            logger.warning("Unable to rank indexed exam chunks: %s", exc)
            trace.note("program_exam rank_fallback=embedding_unavailable")
            scores = {}

    def rank(section: list[ExamChunk]) -> float:
        return max(scores.get(chunk.chunk_id, 0.0) for chunk in section)

    result = []
    for chunks in sorted(sections.values(), key=rank, reverse=True):
        for chunk in chunks:
            trace.note(
                f"program_exam doc_id={exams.card.doc_id} chunk_id={chunk.chunk_id}"
                f" kept score={scores.get(chunk.chunk_id, 0.0):.3f}"
            )
        result.append(exams.render(chunks))
    return result


async def retrieve_program_exams(
    rag: LightRAG,
    question: str,
    cards: list[ProgramCard],
    trace: LightragTrace,
    max_sources: int,
    conversation_history: str | None = None,
) -> RetrievalResult | None:
    """Возвращает контекст одной карточки или готовое уточнение/отсутствие данных."""
    if _ADMISSION_SCORES.search(question):
        return None
    follow_up = None
    if is_program_exam_question(question):
        selected, recognized = _select_cards(question, cards, trace)
    else:
        previous = _pending_exam_question(conversation_history)
        if previous is None:
            return None
        pending, _ = _select_cards(previous, cards, trace)
        reply = _PROFILE.sub(r"\1", question.strip())
        selected = [
            card
            for card in pending
            if _name_key(reply) == _name_key(card.profile or card.name)
        ]
        if not selected:
            return None
        recognized = selected[0]
        follow_up = f"{previous}\n{question}"

    codes = sorted(set(_CODE.findall(question)))
    if not selected:
        label = ", ".join(codes) if codes else recognized.label if recognized else ""
        if not label:
            return None
        level = _LEVEL_LABEL.get(detect_level(question) or "")
        qualifier = f" (уровень образования — {level})" if level else ""
        return RetrievalResult(
            context="",
            trace=trace.get_lines(),
            answer=f"В базе знаний нет данных о ЕГЭ для программы {label}{qualifier}.",
        )

    indexed = [await _get_indexed_exams(rag, card, trace) for card in selected]
    if len(indexed) > 1 and (
        len({item.card.code for item in indexed}) > 1
        or len({_requirements(item) for item in indexed}) > 1
    ):
        labels = "; ".join(dict.fromkeys(item.card.label for item in indexed))
        return RetrievalResult(
            context="",
            trace=trace.get_lines(),
            answer=f"{_CLARIFY_PROFILE} Варианты: {labels}.",
        )
    exams = indexed[0]
    if not exams.chunks:
        return RetrievalResult(
            context="",
            trace=trace.get_lines(),
            answer=f"В карточке {exams.card.label} нет данных о ЕГЭ.",
        )

    generation_question = follow_up or question
    parts = await _rank_sections(rag, generation_question, exams, trace)
    header = f"Программа {exams.card.label}"
    level = _LEVEL_LABEL.get(exams.card.level or "")
    if level:
        level_context = f"Уровень образования — {level}"
        header += f"\n{level_context}"
        if detect_level(generation_question) is None:
            generation_question += f"\n{level_context}."
    context = "\n\n---\n\n".join([header, *parts])
    sources = (
        [{"url": exams.card.url, "title": exams.card.title}]
        if exams.card.url and max_sources > 0
        else []
    )
    return RetrievalResult(
        context=context,
        sources=sources,
        trace=trace.get_lines(),
        generation_question=generation_question,
    )
