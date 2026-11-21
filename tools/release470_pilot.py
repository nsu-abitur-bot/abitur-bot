"""One approved raw pipeline question and real Telegram delivery, under runner meter.

Synthetic worker ingress is explicit: this does not claim Telegram polling or
BotCore's username-prefix path. Never install another ledger.capture() here.
"""

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.release470_store import BudgetStore

REPORTS = Path("/reports")
PLAN = REPORTS / "pilot-plan-approved-2026-11-21.json"
BUDGET = REPORTS / "release470-budget.sqlite3"


async def preflight() -> dict:
    from bootstrap.runtime_settings import runtime_settings
    from config.instance import get_setting
    from db.postgres.db import engine

    if (engine.url.host, engine.url.database) != ("postgres", "eval470"):
        raise RuntimeError("Unexpected isolated target")
    async with runtime_settings():
        values = {
            key: get_setting(key)
            for key in (
                "DEEPSEEK_MODEL",
                "FAQ_EMBEDDING_PROVIDER",
                "LIGHTRAG_EMBEDDING_PROVIDER",
            )
        }
        return {
            "cgroup_memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
            "cgroup_memory_current": Path("/sys/fs/cgroup/memory.current")
            .read_text()
            .strip(),
            "providers": values,
            "telegram_proxy_configured": bool(get_setting("TELEGRAM_SOCKS5_PROXY")),
            "recipient_mount_present": Path("/run/eval470/recipient.json").is_file(),
            "budget": BudgetStore(BUDGET).snapshot(),
            "provider_calls": 0,
            "messages_sent": 0,
        }


async def run_case(case_id: str, result: dict, *, refresh_faq: bool = True) -> None:
    from aiogram import Bot
    from aiogram.client.session.aiohttp import AiohttpSession

    from bootstrap.runtime_settings import runtime_settings
    from bot.formatting import format_for_channel
    from bot.streaming import TelegramStreamer
    from config.instance import get_setting
    from db.postgres.db import AsyncSessionLocal, engine
    from db.postgres.services.user import UserService
    from faq.matcher import get_faq_matcher
    from pipeline import llm_client

    if (engine.url.host, engine.url.database) != ("postgres", "eval470"):
        raise RuntimeError("Unexpected isolated target")
    plan = json.loads(PLAN.read_text())
    question = plan["questions"][case_id]
    if hashlib.sha256(question.encode()).hexdigest() != plan["case_plan"][case_id]:
        raise RuntimeError("Question binding differs")
    store = BudgetStore(BUDGET)
    before = store.snapshot()
    if before["halted"] or before["pending_or_uncertain"]:
        raise RuntimeError("Budget is not ready")
    result["budget_before"] = before
    async with runtime_settings():
        if (
            any(
                get_setting(key) != "cloudru"
                for key in ("FAQ_EMBEDDING_PROVIDER", "LIGHTRAG_EMBEDDING_PROVIDER")
            )
            or get_setting("DEEPSEEK_MODEL") != "deepseek-flash"
        ):
            raise RuntimeError("Unexpected provider configuration")
        # Recipient file is used only for authorized routing, never emitted.
        recipient = json.loads(Path("/run/eval470/recipient.json").read_text())
        chat_id = recipient["chat_id"]
        proxy = get_setting("TELEGRAM_SOCKS5_PROXY")
        if not proxy:
            raise RuntimeError("The approved Telegram proxy is required")
        bot = Bot(
            token=get_setting("BOT_TOKEN") or "", session=AiohttpSession(proxy=proxy)
        )
        async with AsyncSessionLocal() as db:
            user = await UserService(db).ensure_user_by_telegram_id(-47020261121)
            user_id = user.user_id
        if refresh_faq:
            await get_faq_matcher().refresh_from_db()
        result["setup_finished_at_utc"] = datetime.now(UTC).isoformat()
        result["rag_stage_entered"] = False
        original_retrieve = llm_client._retrieve_context

        async def observed_retrieve(*args: Any, **kwargs: Any):
            result["rag_stage_entered"] = True
            return await original_retrieve(*args, **kwargs)

        llm_client._retrieve_context = observed_retrieve

        class Delivery:
            def __getattr__(self, name: str):
                return getattr(bot, name)

            async def send_message(self, *args: Any, **kwargs: Any):
                response = await bot.send_message(*args, **kwargs)
                result["delivered_parts"] += 1
                result["last_bot_api_accept_at_utc"] = datetime.now(UTC).isoformat()
                return response

        delivery: Any = Delivery()
        streamer = TelegramStreamer(delivery, str(chat_id))
        started = time.monotonic()
        result["question_started_at_utc"] = datetime.now(UTC).isoformat()
        result["delivered_parts"] = 0
        try:
            answer = await llm_client.ask_local_llm(
                question,
                session_id=f"EVAL470:{plan['run_id']}:{case_id}",
                user_id=user_id,
                stream_callback=streamer.update,
                status_callback=streamer.set_status,
            )
            tasks = list(llm_client._background_tasks)
            if tasks:
                await asyncio.gather(*tasks)
            state = store.snapshot()
            if state["halted"] or state["pending_or_uncertain"]:
                raise RuntimeError("Meter stopped the pilot")
            if not answer.text or answer.text.strip() in {
                "Что-то пошло не так",
                "Ответ не найден",
            }:
                raise RuntimeError("Pipeline did not produce a complete answer")
            result["faq_matched"] = answer.trace.faq_matched
            result["tools_called"] = answer.trace.tools_called
            result["source_count"] = len(answer.sources)
            result["answer_sha256"] = hashlib.sha256(answer.text.encode()).hexdigest()
            result["answer_characters"] = len(answer.text)
            await streamer.finalize(
                format_for_channel(answer.text, answer.sources, "telegram"),
                parse_mode="HTML",
                fallback_plain=True,
            )
            result["full_delivery_seconds"] = round(time.monotonic() - started, 4)
            result["completed"] = result["delivered_parts"] > 0
        finally:
            llm_client._retrieve_context = original_retrieve
            await streamer.close()
            await bot.session.close()
            await llm_client.cleanup_redis()
        result["budget_after"] = store.snapshot()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        required=False,
        choices=("rag-1", "rag-2", "faq-1", "faq-2", "tools-1", "tools-2"),
    )
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    logging.disable(logging.CRITICAL)
    if args.preflight:
        print(json.dumps(asyncio.run(preflight()), indent=2))
        return
    if args.case is None:
        parser.error("--case is required for a paid question")
    output = REPORTS / f"pilot-{args.case}-20261121.json"
    result = {
        "case_id": args.case,
        "completed": False,
        "ingress": "synthetic worker; real pipeline/formatting/Telegram delivery",
        "telegram_polling_verified": False,
        "botcore_prefix_path_verified": False,
    }
    # Claim once before any setup or request. Failed cases are not auto-retried.
    with output.open("x") as stream:
        json.dump(result, stream)
    try:
        with (
            open(os.devnull, "w") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            asyncio.run(asyncio.wait_for(run_case(args.case, result), 480))
    except BaseException as error:
        result["error_type"] = type(error).__name__
        try:
            result["budget_after"] = BudgetStore(BUDGET).snapshot()
        except Exception:
            result["budget_unreadable"] = True
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["completed"] or "error_type" in result:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
