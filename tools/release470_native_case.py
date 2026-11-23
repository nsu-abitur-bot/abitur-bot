"""One not-yet-started case through native BotCore in a single metered worker.

Explicitly excludes real polling and simultaneous independent API/bot processes.
Progress is persisted before delivery so an abrupt exit cannot erase its boundary.
"""

import argparse
import asyncio
import contextlib
import hashlib
import importlib.util
import json
import logging
import os
import time
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any, cast


def load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(filename)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Missing case module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def run(case, plan, ledger, row, save):
    from aiogram import Bot
    from aiogram.client.session.aiohttp import AiohttpSession

    from bootstrap.runtime_settings import runtime_settings
    from bot import core
    from bot.streaming import TelegramStreamer
    from config.instance import get_setting
    from db.postgres.db import engine
    from faq.matcher import get_faq_matcher
    from faq.updates import watch_faq_updates
    from pipeline import llm_client

    if (engine.url.host, engine.url.database) != ("postgres", "eval470"):
        raise RuntimeError("Unexpected isolated database")
    if any(ledger.store.snapshot()["cases"].get(case, {}).values()):
        raise RuntimeError("Case already has accounted requests; no automatic repeat")
    continuation = load("eval470_continuation", "release470_continue.py")
    helper = load("eval470_faq", "release470_faq_api.py")
    manifest = Path("/reports/faq-single-records-20261123.json")
    application = continuation.faq_application() if case == "faq-2" else None
    created = False
    watcher = None
    bot = None
    try:
        if case == "faq-2":
            row["api_create"] = await helper.operate(
                "create", manifest_path=manifest, application=application
            )
            created = True
            save()
        async with runtime_settings():
            if (
                any(
                    get_setting(key) != "cloudru"
                    for key in ("FAQ_EMBEDDING_PROVIDER", "LIGHTRAG_EMBEDDING_PROVIDER")
                )
                or get_setting("DEEPSEEK_MODEL") != "deepseek-flash"
            ):
                raise RuntimeError("Unexpected providers")
            matcher = get_faq_matcher()
            await matcher.refresh_from_db()
            if case == "faq-2":
                watcher = asyncio.create_task(
                    watch_faq_updates(
                        matcher,
                        engine.url.render_as_string(hide_password=False).replace(
                            "postgresql+asyncpg://", "postgresql://", 1
                        ),
                    )
                )
                # Allow native listener registration before the committed edit.
                await asyncio.sleep(1)
                row["api_update"] = await helper.operate(
                    "update",
                    manifest_path=manifest,
                    refresh_before=False,
                    application=application,
                )
                started = time.monotonic()
                for _ in range(600):
                    questions, answers, _ = matcher._index
                    position = questions.index(plan["questions"]["faq-1"])
                    if answers[position] == "EVAL470 synthetic answer v2":
                        break
                    await asyncio.sleep(0.1)
                else:
                    raise RuntimeError("FAQ notification did not refresh the cache")
                row["notification_refresh_seconds"] = time.monotonic() - started
                save()
            chat = json.loads(Path("/run/eval470/recipient.json").read_text())["chat_id"]
            proxy = get_setting("TELEGRAM_SOCKS5_PROXY")
            if not proxy:
                raise RuntimeError("Approved proxy required")
            bot = Bot(
                token=get_setting("BOT_TOKEN") or "", session=AiohttpSession(proxy=proxy)
            )
            delivery_bot = bot

            class Delivery:
                def __getattr__(self, name):
                    return getattr(delivery_bot, name)

                async def send_message(self, *args, **kwargs):
                    row["delivery_attempted_at_utc"] = datetime.now(UTC).isoformat()
                    save()
                    response = await delivery_bot.send_message(*args, **kwargs)
                    row["parts"] += 1
                    row["last_bot_api_accept_at_utc"] = datetime.now(UTC).isoformat()
                    save()
                    return response

            original = core.ask_local_llm

            @wraps(original)
            async def observed(*args, **kwargs):
                answer = await original(*args, **kwargs)
                row["faq_matched"] = answer.trace.faq_matched
                row["tools_called"] = answer.trace.tools_called
                row["sources_count"] = len(answer.sources)
                row["answer_sha256"] = hashlib.sha256(answer.text.encode()).hexdigest()
                row["pipeline_returned_at_utc"] = datetime.now(UTC).isoformat()
                if answer.text.strip() in {"Что-то пошло не так", "Ответ не найден", ""}:
                    raise RuntimeError("Pipeline returned a failure answer")
                save()
                return answer

            core.ask_local_llm = observed
            for name in ("query_graph_with_sources", "query_graph_with_crag"):
                function = getattr(llm_client, name)

                def observe_graph(function, name):
                    @wraps(function)
                    async def query(*args, **kwargs):
                        value = await function(*args, **kwargs)
                        row.setdefault("graph_returns", []).append(
                            {
                                "function": name,
                                "context_characters": len(value.context or ""),
                                "sources_count": len(value.sources),
                            }
                        )
                        save()
                        return value

                    return query

                setattr(llm_client, name, observe_graph(function, name))
            streamer = TelegramStreamer(cast(Bot, Delivery()), str(chat))
            started = time.monotonic()
            row["question_started_at_utc"] = datetime.now(UTC).isoformat()
            save()
            try:
                reply = await core.BotCore().handle_message(
                    channel="telegram",
                    external_user_id=str(chat),
                    session_id=str(chat),
                    user_name="EVAL470",
                    user_text=plan["questions"][case],
                    stream_callback=streamer.update,
                    status_callback=streamer.set_status,
                )
                tasks = list(llm_client._background_tasks)
                if tasks:
                    await asyncio.gather(*tasks)
                if "pipeline_returned_at_utc" not in row:
                    raise RuntimeError("BotCore did not reach the pipeline")
                if case == "faq-2":
                    row["expected_faq_answer_present"] = (
                        "EVAL470 synthetic answer v2" in reply.text
                    )
                await streamer.finalize(
                    reply.text,
                    parse_mode=reply.parse_mode,
                    fallback_plain=reply.fallback_plain_on_format_error,
                )
                row["seconds"] = round(time.monotonic() - started, 4)
                row["completed"] = row["parts"] > 0
            finally:
                await streamer.close()
                await llm_client.cleanup_redis()
    finally:
        if watcher is not None:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher
        if bot is not None:
            await bot.session.close()
        if created:
            row["api_cleanup"] = await helper.operate(
                "cleanup", manifest_path=manifest, application=application
            )
            save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("faq-2", "tools-1", "tools-2"), required=True)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    plan = json.loads(Path("/reports/pilot-plan-approved-2026-11-21.json").read_text())
    continuation = load("eval470_continuation", "release470_continue.py")
    ledger = continuation.connect_fix(plan)
    receipt = Path(f"/reports/native-{args.case}-20261123.json")
    with receipt.open("x"):
        pass
    row: dict[str, Any] = {
        "case_id": args.case,
        "parts": 0,
        "ingress": "synthetic native BotCore",
        "telegram_polling_tested": False,
        "api_bot_separate_processes": False,
        "budget_before": ledger.store.snapshot(),
    }

    def save():
        row["budget_after"] = ledger.store.snapshot()
        row["memory_current"] = int(Path("/sys/fs/cgroup/memory.current").read_text())
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(row, indent=2))
        temporary.replace(receipt)

    try:
        save()
        asyncio.run(run(args.case, plan, ledger, row, save))
    except BaseException as error:
        row["failed"] = type(error).__name__
    finally:
        save()
        print(json.dumps(row, indent=2))
    if "failed" in row:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
