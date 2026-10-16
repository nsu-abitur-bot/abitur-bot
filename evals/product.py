"""The live adapter uses the same entry point and reference data as the bot."""

import asyncio

from abbrev.expander import get_abbrev_expander
from db.postgres.db import AsyncSessionLocal
from db.postgres.services.abbrev import AbbrevDbService
from db.postgres.services.faq import FaqDbService
from evals.checks import Observation
from evals.schema import Scenario
from faq.matcher import get_faq_matcher
from pipeline.llm_client import ask_local_llm


async def prepare_product() -> None:
    # Fail explicitly on an unavailable reference DB, rather than measuring empty FAQ.
    async with AsyncSessionLocal() as session:
        faq = await FaqDbService(session).get_all()
        abbrev = await AbbrevDbService(session).get_all()
    matcher = get_faq_matcher()
    await asyncio.to_thread(
        matcher.load_items,
        [{"question": e.question, "aliases": e.aliases, "answer": e.answer} for e in faq],
    )
    if faq and matcher.size == 0:
        raise RuntimeError("FAQ initialization failed")
    get_abbrev_expander().load_items([{"short": e.short, "full": e.full} for e in abbrev])


async def run_product(scenario: Scenario, session_id: str) -> Observation:
    answer = await ask_local_llm(scenario.question, session_id=session_id, user_id=0)
    return Observation(
        text=answer.text,
        tools_called=answer.trace.tools_called,
        faq_matched=answer.trace.faq_matched,
    )
