"""User context for questions that need the immediately preceding topic."""

import re
from collections.abc import Sequence

_ROLE = re.compile(r"(?m)^(user|assistant|system): ")
_SPEAKER = re.compile(r"^\[from [^\n\]]+\]\s*", re.I)
_BASIS = (
    r"(?:на\s+)?(?:платн\w*|платк\w*|бюджет\w*)"
    r"(?:\s+(?:основ\w*|отделен\w*|мест\w*|обучени\w*))?"
)
_YEAR = r"(?:(?:в|за|на)\s+)?20\d{2}(?:\s+год\w*)?"
_DOCUMENTS = (
    r"(?:какие\s+(?:(?:там|ещ[её])\s+)?документы(?:\s+(?:там\s+)?нужны)?"
    r"|какие\s+документы\s+(?:надо|нужно)\s+подать)"
    r"(?:\s+(?:там|туда))?"
)
_FOLLOW_UP = re.compile(
    rf"(?:(?:а|и|тогда)\s+)?(?:{_BASIS}|{_YEAR}"
    rf"|(?:в|за)\s+(?:другом|следующем|прошлом|этом)\s+году"
    rf"|(?:можно\s+)?подробнее|сколько\s+(?:это|он[аои]?|там)\s+стоит)"
    rf"|(?:а|и|там|туда|тогда)\s+{_DOCUMENTS}",
    re.I,
)
_OVERRIDES = tuple(
    re.compile(rf"(?<!\w){pattern}(?!\w)", re.I) for pattern in (_BASIS, _YEAR)
)


def _follow_up(question: str) -> bool:
    text = _SPEAKER.sub("", question.strip()).strip(" ?!.,")
    return _FOLLOW_UP.fullmatch(text) is not None


def prior_user_questions(question: str, previous: Sequence[str]) -> list[str]:
    """Select the latest topic and its follow-ups for an elliptical question."""
    if not _follow_up(question):
        return []

    updates = {pattern for pattern in _OVERRIDES if pattern.search(question)}
    context: list[str] = []
    for turn in reversed(previous):
        if not turn.strip():
            continue
        text = turn
        for pattern in _OVERRIDES:
            if pattern in updates:
                text = pattern.sub("", text)
        if text.strip(" ?!.,").casefold() not in {"", "а", "и", "тогда"}:
            context.append(" ".join(text.split()))
        if not _follow_up(turn):
            break
        updates.update(pattern for pattern in _OVERRIDES if pattern.search(turn))
    return list(reversed(context))


def prior_search_history(question: str, history: str | None) -> str | None:
    """Read the user turns from the retriever's role-tagged history format."""
    parts = _ROLE.split(history or "")
    previous = [
        text.strip() for role, text in zip(parts[1::2], parts[2::2]) if role == "user"
    ]
    prior = prior_user_questions(question, previous)
    return "\n".join(f"user: {turn}" for turn in prior) or None
