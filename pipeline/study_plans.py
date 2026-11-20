import asyncio
import json
import logging
import re
from dataclasses import dataclass
from html import escape
from time import monotonic
from urllib.parse import unquote, urldefrag, urlsplit

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import select

from config.university import is_nsu
from db.postgres.db import AsyncSessionLocal
from db.postgres.models import Document
from db.postgres.services.document import ACTIVE_DOCUMENT_STATUSES
from llm.factory import get_llm_provider
from llm.profiles import LLMProfiles
from parser.pdf import PdfPage
from parser.study_plan_table import PdfDiscipline
from parser.url import fetch_pdf_pages
from pipeline.prompts import STUDY_PLAN_EVIDENCE_PROMPT
from rag.retriever import DEFAULT_GRAPH_ID

logger = logging.getLogger(__name__)

_NSU_PLAN_URL = (
    "https://www.nsu.ru/n/information-technologies-department/education_fit/"
    "programs/OOP/09-03-01/piikn/UP_09.03.01_{year}_PIiKN.pdf"
)
_PLAN = re.compile(r"учебн\w+\s+план", re.IGNORECASE)
_STUDY = re.compile(
    r"\b(?:семестр\w*|дисциплин\w*|факультатив\w*|"
    r"обязател\w*\s+ли|предмет\w*\s+по\s+выбору)\b",
    re.IGNORECASE,
)
_ADMISSIONS = re.compile(r"\b(?:егэ|вступительн\w+|для поступления)\b", re.IGNORECASE)
_NON_DISCIPLINE = re.compile(
    r"\b(?:оплат\w*|стоимость|общежити\w*|стипенди\w*|сроки|начинается|"
    r"каникул\w*|отчислен\w*)\b",
    re.IGNORECASE,
)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_COHORT = re.compile(
    r"(?:г\s*од\s+(?:начала\s+подготовки|набора|при[её]ма)|"
    r"набора|при[её]ма)\s*[:—-]?\s*((?:19|20)\d{2})(?!\d)",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://[^\s<>\"«»]+", re.IGNORECASE)
_TOKENS = re.compile(r"[a-zа-я0-9]+")
_PIIKN = re.compile(
    r"\bпиикн\b|\bпрограммн\w+\s+инженери\w+\s+и\s+компьютерн\w+"
    r"\s+наук\w*\b",
    re.IGNORECASE,
)
_NO_EVIDENCE = "Не удалось подтвердить сведения о дисциплине по странице учебного плана."
_NEED_PLAN = (
    "Укажите программу и год набора или пришлите публичную ссылку на учебный план в PDF."
)
_OBLIGATION = {
    "обязательная часть": r"\bобязательн\w+ (?:часть|дисциплин\w*|предмет\w*)",
    "дисциплина по выбору": r"\b(?:дисциплин\w+|предмет\w+) по выбору",
    "факультатив": r"\bфакультатив\w*",
}


@dataclass(frozen=True)
class _PlanSource:
    url: str
    title: str
    year_hint: int | None = None


@dataclass(frozen=True)
class _Evidence:
    source: _PlanSource
    year: int | None
    page: int
    discipline: PdfDiscipline


@dataclass(frozen=True)
class StudyPlanAnswer:
    text: str
    sources: list[dict]


_cached_plans: dict[str, tuple[float, list[PdfPage]]] = {}


def _tokens(text: str) -> list[str]:
    return _TOKENS.findall(text.casefold().replace("ё", "е"))


def _stem(word: str) -> str:
    if len(word) < 5 or not re.fullmatch("[а-я]+", word):
        return word
    return re.sub(
        r"(?:ого|ему|ому|ыми|ими|ая|ую|ые|ий|ый|ой|ое|ом|ах|ям|[аяуеиыюо])$", "", word
    )


def _mentions(text: str, phrase: str) -> bool:
    words = [_stem(word) for word in _tokens(text)]
    pattern = [_stem(word) for word in _tokens(phrase)]
    return bool(pattern) and any(
        words[offset : offset + len(pattern)] == pattern
        for offset in range(len(words) - len(pattern) + 1)
    )


def _pdf_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and parsed.username is None
            and unquote(parsed.path).casefold().endswith(".pdf")
        )
    except ValueError:
        return False


def _year_hint(text: str) -> int | None:
    years = set(_YEAR.findall(text))
    return int(years.pop()) if len(years) == 1 else None


async def _registered_sources(question: str) -> list[_PlanSource]:
    candidates = []
    if is_nsu() and _PIIKN.search(question):
        candidates.extend(
            _PlanSource(
                _NSU_PLAN_URL.format(year=year), f"Учебный план ПИиКН {year}", year
            )
            for year in (2026, 2025, 2024)
        )

    try:
        async with AsyncSessionLocal() as session:
            rows = await session.execute(
                select(Document.title, Document.source_url)
                .where(
                    Document.graph_id == DEFAULT_GRAPH_ID,
                    Document.status.in_(ACTIVE_DOCUMENT_STATUSES),
                    Document.source_url.is_not(None),
                    Document.title.ilike("%учебн%план%"),
                )
                .order_by(Document.updated_at.desc())
            )
            documents = rows.all()
    except Exception:
        logger.warning("Unable to load registered study plans", exc_info=True)
        return candidates
    for title, url in documents:
        if not url or not _pdf_url(url):
            continue
        program = re.sub(r"[«»\"()]", "", _YEAR.sub("", _PLAN.sub("", title))).strip()
        program = re.sub(r"\b(?:года?|набора|для)\b", "", program).strip(" -—,.:")
        if _mentions(question, program) or (
            _PIIKN.search(question) and _PIIKN.search(title)
        ):
            candidates.append(_PlanSource(url, title, _year_hint(title)))
    return candidates


async def _read_pages(url: str) -> list[PdfPage]:
    cached = _cached_plans.get(url)
    if cached and monotonic() < cached[0]:
        return cached[1]
    pages = await fetch_pdf_pages(url)
    if pages and all(page.text for page in pages):
        if len(_cached_plans) >= 8:
            _cached_plans.pop(next(iter(_cached_plans)))
        _cached_plans[url] = (monotonic() + 900, pages)
    return pages


def _cohort(pages: list[PdfPage]) -> int | None:
    years = {int(year) for page in pages for year in _COHORT.findall(page.text)}
    return years.pop() if len(years) == 1 else None


def _semesters_from_quote(quote: str) -> tuple[int, ...]:
    ordinal = r"[1-9]\d?(?:\s*[-–]\s*(?:й|м|ом))?"
    numbers = rf"({ordinal}(?:\s*[,и]\s*{ordinal})*)"
    matches = re.findall(
        rf"{numbers}\s+семестр\w*|семестр\w*\s*[:№-]?\s*{numbers}", quote
    )
    if len(matches) != 1:
        return ()
    return tuple(
        sorted({int(value) for value in re.findall(r"\d+", "".join(matches[0]))})
    )


def _read_model_evidence(
    response: str, pages: dict[int, str], question: str
) -> list[tuple[int, PdfDiscipline]]:
    try:
        response = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", response)
        items = json.loads(response)
    except (ValueError, RecursionError):
        return []
    if not isinstance(items, list):
        return []
    facts = []
    for item in items[:5]:
        if not isinstance(item, dict):
            continue
        page, name, quote = item.get("page"), item.get("name"), item.get("quote")
        if type(page) is not int or page not in pages:
            continue
        if not isinstance(name, str) or not isinstance(quote, str):
            continue
        name, quote = " ".join(name.split()), " ".join(quote.split())
        if not name or not quote or len(quote) > 360:
            continue
        if not _mentions(question, name) or not _mentions(quote, name):
            continue
        if not any(
            quote.casefold() in " ".join(line.split()).casefold()
            for line in pages[page].splitlines()
        ):
            continue
        status = [
            label
            for label, pattern in _OBLIGATION.items()
            if re.search(pattern, quote.lower())
        ]
        if re.search(r"\bне\b|\bнеобязател", quote.lower()):
            status = [label for label in status if label != "обязательная часть"]
        facts.append(
            (
                page,
                PdfDiscipline(
                    name,
                    _semesters_from_quote(quote.lower()),
                    status[0] if len(status) == 1 else None,
                    quote,
                ),
            )
        )
    return facts


async def _model_evidence(
    question: str, pages: list[PdfPage]
) -> list[tuple[int, PdfDiscipline]]:
    relevant: dict[int, str] = {}
    budget = 35_000
    question_words = {
        _stem(word)
        for word in _tokens(question)
        if len(word) >= 5 and not _STUDY.fullmatch(word)
    }
    for page in pages:
        if not page.text or page.disciplines:
            continue
        lines = [" ".join(line.split()) for line in page.text.splitlines()]
        hits = [
            offset
            for offset, line in enumerate(lines)
            if question_words.intersection(map(_stem, _tokens(line)))
        ]
        for offset in hits:
            excerpt = " ".join(
                line for line in lines[max(0, offset - 1) : offset + 2] if line
            )
            if excerpt in relevant.get(page.number, ""):
                continue
            if len(excerpt) > budget:
                continue
            relevant[page.number] = f"{relevant.get(page.number, '')}\n{excerpt}".strip()
            budget -= len(excerpt)

    if not relevant:
        return []
    response = await get_llm_provider().generate(
        [
            SystemMessage(content=STUDY_PLAN_EVIDENCE_PROMPT),
            HumanMessage(
                content=json.dumps(
                    {"question": question, "pages": relevant}, ensure_ascii=False
                )
            ),
        ],
        profile=LLMProfiles.PARSER,
    )
    verified_pages = {page.number: page.text for page in pages if page.number in relevant}
    return _read_model_evidence(response, verified_pages, question)


def _native_evidence(
    question: str, pages: list[PdfPage]
) -> list[tuple[int, PdfDiscipline]]:
    matches = [
        (page.number, discipline)
        for page in pages
        for discipline in page.disciplines
        if _mentions(question, discipline.name)
    ]
    return [
        (page, discipline)
        for page, discipline in matches
        if not any(
            other.name != discipline.name
            and len(_tokens(other.name)) > len(_tokens(discipline.name))
            and _mentions(other.name, discipline.name)
            for _, other in matches
        )
    ]


async def _find_evidence(
    question: str, sources: list[_PlanSource], requested_year: int | None
) -> list[_Evidence]:
    fallback: list[_Evidence] = []
    deadline = monotonic() + 25
    for index, source in enumerate(sources):
        try:
            allowance = (deadline - monotonic()) / (len(sources) - index)
            if allowance <= 0:
                break
            async with asyncio.timeout(allowance):
                pages = await _read_pages(source.url)
                year = _cohort(pages)
                facts = _native_evidence(question, pages) or await _model_evidence(
                    question, pages
                )
        except Exception as exc:
            logger.warning("Study plan could not be read (%s)", type(exc).__name__)
            continue
        if not facts:
            continue
        evidence = [_Evidence(source, year, page, fact) for page, fact in facts]
        if requested_year is None or year == requested_year:
            return evidence
        if not fallback or (fallback[0].year is None and year is not None):
            fallback = evidence
    return fallback


def _render(evidence: list[_Evidence], requested_year: int | None) -> StudyPlanAnswer:
    statements = []
    sources = []
    for item in evidence[:3]:
        discipline = item.discipline
        facts = []
        if discipline.semesters:
            semesters = ", ".join(f"{number}-й" for number in discipline.semesters)
            label = "семестр" if len(discipline.semesters) == 1 else "семестры"
            facts.append(f"{semesters} {label}")
        if discipline.obligation:
            facts.append(discipline.obligation)
        if item.year is None:
            scope = "Год набора в документе не подтверждён."
        elif requested_year is not None and item.year != requested_year:
            scope = (
                f"Это план другого набора ({item.year}). "
                f"Для набора {requested_year} сведения не подтверждены."
            )
        else:
            scope = f"Учебный план для набора {item.year} года."
        detail = f" — {', '.join(facts)}" if facts else ""
        statements.append(
            escape(
                f"{discipline.name}{detail}. {scope}\n"
                f"Подтверждение (стр. {item.page}): «{discipline.evidence}»."
            )
        )
        reference = {
            "url": f"{urldefrag(item.source.url).url}#page={item.page}",
            "title": f"{item.source.title}, стр. {item.page}",
        }
        if reference not in sources:
            sources.append(reference)
    return StudyPlanAnswer("\n\n".join(statements), sources)


async def answer_study_plan(question: str) -> StudyPlanAnswer | None:
    """Answer curriculum questions with evidence tied to the original PDF page."""
    scope = _URL.sub("", question)
    url_match = _URL.search(question)
    url = url_match.group(0).rstrip(".,;:!?)]") if url_match else ""
    direct = bool(url and _pdf_url(url))
    has_subject_question = bool(
        _PLAN.search(scope)
        or _STUDY.search(scope)
        or (
            (direct or _PIIKN.search(scope))
            and re.search(r"\b(?:есть\s+ли|когда\s+(?:проходят|изучают))\b", scope, re.I)
        )
    )
    if (
        not has_subject_question
        or _ADMISSIONS.search(scope)
        or _NON_DISCIPLINE.search(scope)
    ):
        return None
    year = _year_hint(scope)
    try:
        if direct:
            candidates = [_PlanSource(urldefrag(url).url, "Учебный план")]
        else:
            async with asyncio.timeout(2):
                candidates = await _registered_sources(scope)
        if not candidates:
            return StudyPlanAnswer(_NEED_PLAN, []) if _PLAN.search(scope) else None
        unique = {item.url: item for item in candidates}
        ordered = sorted(
            unique.values(),
            key=lambda item: (
                year is not None and item.year_hint != year,
                -(item.year_hint or 0),
            ),
        )
        evidence = await _find_evidence(scope, ordered[:3], year)
    except TimeoutError:
        return StudyPlanAnswer(_NO_EVIDENCE, [])
    return _render(evidence, year) if evidence else StudyPlanAnswer(_NO_EVIDENCE, [])
