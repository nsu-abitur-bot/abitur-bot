"""Проверка черновика ответа по источникам и переадресация к сотруднику."""

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from config.instance import get_setting
from config.latency import stage
from config.university import identity, is_nsu
from llm.base import BaseLLMProvider
from llm.profiles import LLMProfile
from pipeline.prompts import ANSWER_REVIEW_PROMPT

logger = logging.getLogger(__name__)

REVIEW_PROFILE = LLMProfile(max_tokens=900, timeout=15, temperature=0.0)
NO_GROUNDS = "Я не нашёл информации в базе знаний, чтобы ответить на этот вопрос."
REVIEW_UNAVAILABLE = "Сейчас я не могу подтвердить ответ на этот вопрос."


@dataclass(frozen=True)
class ReviewDecision:
    verdict: Literal["supported", "neutral", "clarify", "refer", "unavailable"]
    question: str = ""


@dataclass(frozen=True)
class HandoffContact:
    name: str
    role: str
    channel: str
    source_url: str = ""
    source_title: str = ""


_NSU_ADMISSIONS = HandoffContact(
    name="Марина Шашкова",
    role="ответственный секретарь приёмной комиссии НГУ",
    channel="почта приёмной комиссии priem@nsu.ru, телефон +7 (383) 363-40-37",
    source_url="https://events.nsu.ru/abiturient",
    source_title="Контакты приёмной комиссии НГУ",
)
_NSU_FIT_PHD = HandoffContact(
    name="Арис Саввич Терсенов",
    role="заместитель декана ФИТ по аспирантуре",
    channel="a.tersenov@nsu.ru",
    source_url="https://education.nsu.ru/apply_fit/",
    source_title="Контакты ФИТ НГУ",
)


def handoff(question: str, *, unavailable: bool = False) -> tuple[str, list[dict]]:
    """Формирует ответ из настроек вуза или официального справочника НГУ."""
    prefix = REVIEW_UNAVAILABLE if unavailable else NO_GROUNDS
    if is_nsu():
        lowered = question.casefold()
        contact = (
            _NSU_FIT_PHD
            if "аспирантур" in lowered
            and re.search(r"\bфит\b|факультет\w* информационных технологий", lowered)
            else _NSU_ADMISSIONS
        )
        return (
            f"{prefix} Уточнить порядок действий поможет {contact.name}, "
            f"{contact.role}. Для связи — {contact.channel}.",
            [{"title": contact.source_title, "url": contact.source_url}],
        )

    name = get_setting("ADMISSIONS_HANDOFF_NAME", "").strip()
    university = identity()
    channels = ", ".join(
        channel for channel in (university["email"], university["phone"]) if channel
    )
    if name and channels:
        return (
            f"{prefix} Уточнить вопрос поможет сотрудник приёмной комиссии "
            f"{name}. Для связи — {channels}.",
            [],
        )
    if channels:
        return f"{prefix} Контакты приёмной комиссии — {channels}.", []
    return prefix, []


async def review_answer(
    provider: BaseLLMProvider,
    question: str,
    history: list[dict[str, str]],
    draft: str,
    evidence: list[str],
) -> ReviewDecision:
    """Возвращает решение о публикации. Сбой проверки блокирует черновик."""
    if not draft.strip():
        return ReviewDecision("refer")

    payload = json.dumps(
        {
            "question": question,
            "history": history,
            "draft": draft,
            "sources": evidence,
        },
        ensure_ascii=False,
    )
    try:
        with stage("answer_review"):
            result = await provider.generate(
                [
                    SystemMessage(content=ANSWER_REVIEW_PROMPT),
                    HumanMessage(content=payload),
                ],
                profile=REVIEW_PROFILE,
            )
            result = result.strip()
            fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", result, flags=re.DOTALL)
            decision = json.loads(fenced[1] if fenced else result)
        if isinstance(decision, dict):
            verdict = decision.get("verdict")
            if verdict in ("neutral", "refer"):
                return ReviewDecision(verdict)
            if verdict == "clarify":
                followup = decision.get("question")
                if isinstance(followup, str):
                    followup = followup.strip()
                    if re.fullmatch(r"[^<>\r\n?]{1,499}\?", followup):
                        return ReviewDecision("clarify", question=followup)
                return ReviewDecision("unavailable")
            quotes = decision.get("quotes")
            if (
                verdict == "supported"
                and isinstance(quotes, list)
                and quotes
                and all(
                    isinstance(quote, str)
                    and quote.strip()
                    and any(quote in source for source in evidence)
                    for quote in quotes
                )
            ):
                return ReviewDecision("supported")
            if verdict == "supported":
                return ReviewDecision("refer")
    except Exception as exc:
        logger.warning("Answer review failed: %s", type(exc).__name__)
    return ReviewDecision("unavailable")
