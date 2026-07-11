"""Парсер страницы «Итоги приёма прошлых лет» сайта НГУ.

Страница организована как институт/факультет → направление → таблица.
Заголовок факультета/института — это ``<span class="bold">…факультет/институт…</span>``.
Перед каждой таблицей идёт ``<p>`` с названием направления, кодом ФГОС и уровнем,
например: «Программная инженерия и компьютерные науки (09.03.01, бакалавриат)».

В таблице **годы — это столбцы** (2015–2025), а **метрика — это строка**.
Нужные строки: «Проходной балл» и «Средний балл», встречающиеся один раз в
группе «Зачислены на бюджет» и один раз в группе «Зачислены платно».

Парсинг детерминированный (BeautifulSoup, без LLM). Возвращает список
:class:`ScoreRow` — по одной строке на тройку (направление, год, форма).
"""

import logging
import re

import requests
from bs4 import BeautifulSoup, Tag

from db.postgres.services.admission_score import ScoreRow

logger = logging.getLogger(__name__)

DEFAULT_SCORES_URL = "https://www.nsu.ru/n/education/apply-info/itogi-priema/"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

# Заголовок факультета/института — жирный span с этими словами в тексте.
_FACULTY_RE = re.compile(r"факультет|институт", re.IGNORECASE)

# Парентеза направления: содержит уровень образования (и обычно код ФГОС).
_PROGRAM_PAREN_RE = re.compile(
    r"\(([^()]*(?:бакалавриат|специалитет|специалист)[^()]*)\)\s*$",
    re.IGNORECASE,
)
_CODE_RE = re.compile(r"\d{2}\.\d{2}\.\d{2}")

_FORM_BUDGET_RE = re.compile(r"зачислены\s+на\s+бюджет", re.IGNORECASE)
_FORM_PAID_RE = re.compile(r"зачислены\s+платно", re.IGNORECASE)

_EMPTY_CELLS = {"", "-", "—", "–", "0"}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _is_faculty_heading(el: Tag) -> bool:
    """Жирный span (``class="bold"``) с названием факультета/института.

    Именно ``class="bold"`` отличает заголовки факультетов от обычных ссылок
    навигации (у тех такого класса нет), хотя и те и другие обёрнуты в ``<a>``.
    """
    if el.name != "span":
        return False
    classes = el.get("class") or []
    if "bold" not in classes:
        return False
    text = _clean(el.get_text(" ", strip=True))
    return bool(_FACULTY_RE.search(text)) and len(text) < 120


def _parse_program_heading(text: str) -> tuple[str, str | None, str] | None:
    """Разбирает заголовок направления на (название, код, уровень).

    Возвращает None, если это не заголовок направления (нет уровня в скобках).
    """
    text = _clean(text)
    match = _PROGRAM_PAREN_RE.search(text)
    if match is None:
        return None

    name = text[: match.start()].strip()
    if not name:
        return None

    inner = match.group(1)
    code_match = _CODE_RE.search(inner)
    code = code_match.group(0) if code_match else None

    lower = inner.lower()
    level = (
        "specialist" if ("специалитет" in lower or "специалист" in lower) else "bachelor"
    )
    return name, code, level


def _parse_number(text: str) -> float | None:
    """«-», «0», пусто → None; иначе число (запятая как десятичный разделитель).

    Проходной/средний балл никогда не равны 0, поэтому «0» трактуем как «нет
    данных» (на странице так помечены незаполненные годы и отсутствие набора)."""
    text = _clean(text)
    if text in _EMPTY_CELLS:
        return None
    normalized = text.replace(",", ".").replace(" ", "")
    try:
        return float(normalized)
    except ValueError:
        return None


def _parse_table(
    table: Tag,
    faculty_name: str,
    program_name: str,
    code: str | None,
    level: str,
) -> list[ScoreRow]:
    """Парсит одну таблицу направления в список ScoreRow."""
    rows = table.find_all("tr")
    if not rows:
        return []

    # Первая строка — годы по столбцам. Столбец 0 — метка метрики.
    header_cells = rows[0].find_all(["td", "th"])
    year_by_col: dict[int, int] = {}
    for idx, cell in enumerate(header_cells):
        raw = _clean(cell.get_text(" ", strip=True))
        if re.fullmatch(r"(19|20)\d{2}", raw):
            year_by_col[idx] = int(raw)
    if not year_by_col:
        return []

    # form -> metric -> {year: value}
    collected: dict[str, dict[str, dict[int, float]]] = {
        "budget": {"passing": {}, "average": {}},
        "paid": {"passing": {}, "average": {}},
    }

    current_form: str | None = None
    for tr in rows[1:]:
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        label = _clean(cells[0].get_text(" ", strip=True))
        label_lower = label.lower()

        if _FORM_BUDGET_RE.search(label_lower):
            current_form = "budget"
            continue
        if _FORM_PAID_RE.search(label_lower):
            current_form = "paid"
            continue

        if current_form is None:
            continue

        if label_lower.startswith("проходной балл"):
            metric = "passing"
        elif label_lower.startswith("средний балл"):
            metric = "average"
        else:
            continue

        for col_idx, year in year_by_col.items():
            if col_idx >= len(cells):
                continue
            value = _parse_number(cells[col_idx].get_text(" ", strip=True))
            if value is not None:
                collected[current_form][metric][year] = value

    result: list[ScoreRow] = []
    for form in ("budget", "paid"):
        passing = collected[form]["passing"]
        average = collected[form]["average"]
        years = sorted(set(passing) | set(average))
        for year in years:
            p_val = passing.get(year)
            a_val = average.get(year)
            if p_val is None and a_val is None:
                continue
            result.append(
                ScoreRow(
                    faculty_name=faculty_name,
                    program_name=program_name,
                    code=code,
                    year=year,
                    form=form,
                    passing_score=int(round(p_val)) if p_val is not None else None,
                    average_score=a_val,
                    level=level,
                )
            )
    return result


def parse_scores_html(html: str) -> list[ScoreRow]:
    """Детерминированно парсит HTML страницы итогов приёма в список ScoreRow."""
    soup = BeautifulSoup(html, "html.parser")

    result: list[ScoreRow] = []
    current_faculty: str | None = None
    pending: tuple[str, str | None, str] | None = None  # (program, code, level)

    for el in soup.find_all(["span", "p", "table"]):
        if el.name == "span":
            if _is_faculty_heading(el):
                current_faculty = _clean(el.get_text(" ", strip=True))
                pending = None
            continue

        if el.name == "p":
            parsed = _parse_program_heading(el.get_text(" ", strip=True))
            if parsed is not None:
                pending = parsed
            continue

        # el.name == "table"
        if pending is None or current_faculty is None:
            continue
        program_name, code, level = pending
        result.extend(_parse_table(el, current_faculty, program_name, code, level))
        pending = None

    return result


async def parse_scores(url: str = DEFAULT_SCORES_URL) -> list[ScoreRow]:
    """Загружает и парсит страницу «Итоги приёма прошлых лет» НГУ.

    Возвращает список :class:`ScoreRow` — по одной строке на тройку
    (направление, год, форма). Проходной/средний баллы за годы без данных
    пропускаются.
    """
    try:
        response = requests.get(url, headers=_HEADERS, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.error("Ошибка загрузки страницы итогов приёма %s: %s", url, exc)
        return []

    return parse_scores_html(response.text)
