"""The live adapter uses the same entry point and reference data as the bot."""

import asyncio

from abbrev.expander import get_abbrev_expander
from db.postgres.db import AsyncSessionLocal
from db.postgres.services.abbrev import AbbrevDbService
from db.postgres.services.faq import FaqDbService
from evals.checks import Observation
from evals.schema import Scenario
from faq.matcher import get_faq_matcher
from pipeline.llm_client import LlmAnswer, ask_local_llm


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
    if scenario.setup_questions:
        history = [*scenario.history]
        for question in scenario.setup_questions:
            prior = await ask_local_llm(question, session_id, prior_history=history)
            history.append({"role": "user", "content": question})
            if prior.history_entry:
                history.append(prior.history_entry)
        answer = await ask_local_llm(scenario.question, session_id, prior_history=history)
    else:
        answer = await _run_single_question(scenario, session_id)

    sources = [
        s.get("url") if isinstance(s, dict) else str(s) for s in (answer.sources or [])
    ]
    return Observation(
        text=answer.text,
        tools_called=answer.trace.tools_called,
        faq_matched=answer.trace.faq_matched,
        sources=[s for s in sources if s],
    )


async def _run_single_question(scenario: Scenario, session_id: str) -> LlmAnswer:
    if scenario.history:
        from pipeline.llm_client import get_redis_client

        redis_client = await get_redis_client()
        for entry in scenario.history:
            await redis_client.add_message(session_id, entry)

    return await ask_local_llm(scenario.question, session_id=session_id, user_id=0)
