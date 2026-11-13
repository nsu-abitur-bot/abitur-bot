import re
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PdfDiscipline:
    name: str
    semesters: tuple[int, ...]
    obligation: str | None
    evidence: str


_Word = tuple[float, float, float, float, str, int, int, int]
_CODE = re.compile(r"[БB]1\.[А-ЯA-Z0-9.]+|ФТД\.[А-ЯA-Z0-9.]*", re.IGNORECASE)
_SEMESTERS = re.compile(r",?[1-9][0-9]?(?:,[1-9][0-9]?)*,?")


def _row_words(words: Sequence[_Word], anchor: _Word) -> list[_Word]:
    # Words in tiny table cells can be slightly misaligned with the row number.
    tolerance = (anchor[3] - anchor[1]) * 0.45
    return sorted(
        (word for word in words if abs(word[3] - anchor[3]) <= tolerance),
        key=lambda word: word[0],
    )


def read_study_plan_table(words: Sequence[_Word]) -> tuple[PdfDiscipline, ...]:
    """Read NSU-style study-plan rows by the positions of their column headers."""
    headings = next(
        (_row_words(words, word) for word in words if word[4].casefold() == "семестрам"),
        [],
    )
    if not headings:
        return ()
    by_semester = next((word for word in headings if word[4].casefold() == "по"), None)
    totals = next((word for word in headings if word[4].casefold() == "количество"), None)
    discipline = next(
        (
            word
            for word in words
            if word[4].casefold() == "дисциплины"
            and any(part[4].casefold() == "название" for part in _row_words(words, word))
        ),
        None,
    )
    if by_semester is None or totals is None or discipline is None:
        return ()
    assessments = sorted(
        word[0]
        for word in words
        if word[4].casefold() in {"зачет", "зачёт", "экзамен", "коллоквиум"}
        and by_semester[1] < word[1] < discipline[1] + 45
        and word[0] < totals[0]
    )
    if len(assessments) < 2:
        return ()
    # Headers are vertical and cells center-aligned. Use midpoints at the two
    # outside edges of the assessment group to exclude workload columns.
    first_column = assessments[0] - (assessments[1] - assessments[0]) / 2
    last_column = assessments[-1] + (totals[0] - assessments[-1]) / 2
    if not discipline[0] < first_column < by_semester[0] < last_column:
        return ()
    numbered_rows = sorted(
        (
            word
            for word in words
            if word[0] < discipline[0]
            and word[1] > discipline[1]
            and _CODE.fullmatch(word[4])
        ),
        key=lambda word: word[1],
    )
    sections: dict[str, str] = {}
    results = []
    for index, anchor in enumerate(numbered_rows):
        code = anchor[4].rstrip(".").upper().replace("B", "Б")
        tolerance = (anchor[3] - anchor[1]) * 0.45
        start = anchor[3] - tolerance
        end = (
            numbered_rows[index + 1][3] - tolerance
            if index + 1 < len(numbered_rows)
            else anchor[3] + tolerance
        )
        row = [word for word in words if start <= word[3] < end]
        name_words = [word for word in row if anchor[2] < word[0] < first_column]
        name_words.sort(
            key=lambda word: (round((word[3] - anchor[3]) / (tolerance * 2)), word[0])
        )
        name = " ".join(word[4] for word in name_words)
        if not name:
            continue
        normalized = name.casefold()
        if "обязательная часть" in normalized or "обязательные дисциплины" in normalized:
            sections[code] = "обязательная часть"
            continue
        if "дисциплины по выбору" in normalized or "факультативы" in normalized:
            sections[code] = (
                "дисциплина по выбору" if "по выбору" in normalized else "факультатив"
            )
            continue
        if (
            not code.rsplit(".", 1)[-1].isdigit()
            or "формируемая участниками" in normalized
            or "дисциплины (модули)" in normalized
        ):
            continue
        obligation = next(
            (
                sections[key]
                for key in sorted(sections, key=len, reverse=True)
                if code.startswith(f"{key}.")
            ),
            None,
        )
        if "элективная дисциплина" in normalized:
            obligation = "элективная дисциплина"
        semesters: set[int] = set()
        for word in row:
            if first_column <= (word[0] + word[2]) / 2 < last_column:
                if _SEMESTERS.fullmatch(word[4]):
                    semesters.update(map(int, word[4].strip(",").split(",")))
        evidence = f"{anchor[4]} {name}"
        results.append(
            PdfDiscipline(name, tuple(sorted(semesters)), obligation, evidence)
        )
    return tuple(results)
