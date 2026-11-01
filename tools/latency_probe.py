"""Bounded synthetic Gemini baseline: six sequential generations, two embeddings.

Run with `uv run python -m tools.latency_probe` in the intended environment.
Reads runtime settings; never starts a bot, loads RAG or writes application data.
Output deliberately excludes responses, exception messages and connection settings.
"""

import asyncio
import json
import logging
from time import perf_counter

from langchain_core.messages import HumanMessage

from bootstrap.runtime_settings import refresh_settings
from config.instance import get_setting
from db.postgres.db import engine
from llm.factory import get_llm_provider
from llm.profiles import LLMProfiles

QUESTIONS = (
    "Какие документы нужны для поступления в НГУ?",
    "Есть ли общежитие для первокурсников НГУ?",
    "Какие проходные баллы были на ФИТ НГУ?",
)


async def main() -> None:
    """Use the installed model/profile unchanged; SDK retry policy is unchanged."""
    logging.disable(logging.CRITICAL)
    try:
        await refresh_settings()
        if get_setting("LLM_PROVIDER", "gemini") != "gemini":
            raise RuntimeError("Gemini baseline requires the current Gemini provider")
        provider = get_llm_provider()
        print(
            json.dumps(
                {
                    "provider": type(provider).__name__,
                    "model": get_setting("GEMINI_MODEL"),
                    "embedding_model": get_setting("GEMINI_EMBEDDING_MODEL"),
                    "scope": "synthetic provider-only; SDK retries unchanged",
                }
            ),
            flush=True,
        )
        for pass_number in range(2):
            for index, question in enumerate(QUESTIONS):
                started = perf_counter()
                row: dict = {
                    "phase": "first-pass" if pass_number == 0 else "warm-pass",
                    "question_id": index,
                }
                try:
                    result = await provider.generate_with_usage(
                        [HumanMessage(content=question)], profile=LLMProfiles.CHAT
                    )
                    row.update(status="ok", output_chars=len(result.text))
                except Exception as exc:
                    row.update(status=type(exc).__name__)
                row["duration_ms"] = int((perf_counter() - started) * 1000)
                print(json.dumps(row), flush=True)
            started = perf_counter()
            row = {"phase": "embedding", "pass": pass_number}
            try:
                embedder = provider.get_embeddings_model()
                result = await asyncio.to_thread(embedder.embed_documents, [QUESTIONS[0]])
                row.update(status="ok", dimensions=len(result[0]))
            except Exception as exc:
                row.update(status=type(exc).__name__)
            row["duration_ms"] = int((perf_counter() - started) * 1000)
            print(json.dumps(row), flush=True)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(json.dumps({"setup_error": type(exc).__name__}), flush=True)
        raise SystemExit(1) from None
