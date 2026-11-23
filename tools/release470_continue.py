"""Explicit remaining four cases, native BotCore, independent hot API/bot caches.

The initial input is synthetic, not Telegram polling. Actual Bot API delivery is
used. Run only through the installed pilot runner; never create another capture.
"""

import argparse
import asyncio
import contextlib
import hashlib
import importlib.util
import json
import logging
import multiprocessing
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast


def load_file(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(filename)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Missing continuation module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def connect_fix(plan, *, probe=False):
    from pipeline import llm_client

    guard = load_file("eval470_owner", "release470_owner_guard.py")
    fixed = load_file("eval470_fixed", "release470_runtime.py")
    old_question = llm_client.ask_local_llm
    ledger = guard.installed_ledger(old_question, plan["case_plan"])
    guard.attach_owner_guard(ledger)
    if not probe:
        # Replace exactly the recognized question-label wrapper with its tested
        # successor. SDK hooks, root meter, transport gate and store stay intact.
        llm_client.ask_local_llm = fixed.instrument_question(
            getattr(old_question, "__wrapped__"), ledger
        )
        from bot.core import BotCore

        recipient = json.loads(Path("/run/eval470/recipient.json").read_text())
        chat = recipient.get("chat_id")
        if type(chat) is not int or chat <= 0:
            raise RuntimeError("Unverified recipient")
        BotCore.handle_message = fixed.instrument_bot_core(
            BotCore.handle_message, ledger, recipient=str(chat)
        )
    return ledger


def api_child(pipe):
    async def serve():
        helper = load_file("eval470_faq", "release470_faq_api.py")
        from faq.matcher import get_faq_matcher

        manifest = Path("/reports/faq-continue-records-20261123.json")
        application = faq_application()
        created = False
        try:
            result = await helper.operate(
                "create", manifest_path=manifest, application=application
            )
            created = True
            await get_faq_matcher().refresh_from_db()
            pipe.send({"created": result, "cache_hot": True})
            command = await asyncio.to_thread(pipe.recv)
            if command == "update":
                result = await helper.operate(
                    "update",
                    manifest_path=manifest,
                    refresh_before=False,
                    application=application,
                )
                pipe.send({"updated": result})
                await asyncio.to_thread(pipe.recv)
        finally:
            if created:
                pipe.send(
                    {
                        "cleanup": await helper.operate(
                            "cleanup", manifest_path=manifest, application=application
                        )
                    }
                )

    try:
        asyncio.run(serve())
    except BaseException as error:
        pipe.send({"failed": type(error).__name__})
    finally:
        pipe.close()


def faq_application():
    """Native routes and auth dependency, without unrelated import workers."""
    from fastapi import Depends, FastAPI

    from api.auth.dependencies import require_admin
    from api.auth.router import router as auth_router
    from api.routes.faq import router as faq_router

    app = FastAPI()
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(
        faq_router, prefix="/api/v1", dependencies=[Depends(require_admin)]
    )
    return app


async def receive(pipe):
    if not await asyncio.to_thread(pipe.poll, 180):
        raise RuntimeError("API child did not finish within bounded wait")
    message = pipe.recv()
    if "failed" in message:
        raise RuntimeError("API child failed: " + message["failed"])
    return message


async def remaining(plan, ledger, pipe, result):
    from aiogram import Bot
    from aiogram.client.session.aiohttp import AiohttpSession

    from bootstrap.runtime_settings import runtime_settings
    from bot.core import BotCore
    from bot.streaming import TelegramStreamer
    from config.instance import get_setting
    from db.postgres.db import engine
    from faq.matcher import get_faq_matcher
    from faq.updates import watch_faq_updates
    from pipeline import llm_client

    if (engine.url.host, engine.url.database) != ("postgres", "eval470"):
        raise RuntimeError("Unexpected test database")
    result["api"] = await receive(pipe)
    async with runtime_settings():
        if (
            any(
                get_setting(key) != "cloudru"
                for key in ("FAQ_EMBEDDING_PROVIDER", "LIGHTRAG_EMBEDDING_PROVIDER")
            )
            or get_setting("DEEPSEEK_MODEL") != "deepseek-flash"
        ):
            raise RuntimeError("Unexpected provider selection")
        chat = json.loads(Path("/run/eval470/recipient.json").read_text())["chat_id"]
        proxy = get_setting("TELEGRAM_SOCKS5_PROXY")
        if not proxy:
            raise RuntimeError("Approved Telegram proxy required")
        bot = Bot(
            token=get_setting("BOT_TOKEN") or "", session=AiohttpSession(proxy=proxy)
        )
        matcher = get_faq_matcher()
        watcher = asyncio.create_task(
            watch_faq_updates(
                matcher,
                engine.url.render_as_string(hide_password=False).replace(
                    "postgresql+asyncpg://", "postgresql://", 1
                ),
            )
        )

        async def wait_cache(answer):
            for _ in range(600):
                questions, answers, _vectors = matcher._index
                if plan["questions"]["faq-1"] in questions:
                    if answers[questions.index(plan["questions"]["faq-1"])] == answer:
                        return
                await asyncio.sleep(0.1)
            raise RuntimeError("Native FAQ watcher did not update its hot cache")

        try:
            await wait_cache("EVAL470 synthetic answer v1")
            result["bot_cache_hot_before_edit"] = True
            for case in ("faq-1", "faq-2", "tools-1", "tools-2"):
                if case == "faq-2":
                    pipe.send("update")
                    result["api_update"] = await receive(pipe)
                    started = time.monotonic()
                    await wait_cache("EVAL470 synthetic answer v2")
                    result["bot_notification_refresh_seconds"] = (
                        time.monotonic() - started
                    )
                row: dict[str, Any] = {
                    "case_id": case,
                    "ingress": "synthetic native BotCore",
                    "telegram_polling_tested": False,
                    "parts": 0,
                }
                result["cases"].append(row)

                class Delivery:
                    def __getattr__(self, name):
                        return getattr(bot, name)

                    async def send_message(self, *args, **kwargs):
                        response = await bot.send_message(*args, **kwargs)
                        row["parts"] += 1
                        row["last_bot_api_accept_at_utc"] = datetime.now(UTC).isoformat()
                        return response

                # Observe actual graph entrypoint results, not just an outer RAG stage.
                original_graph = llm_client.query_graph_with_sources
                original_crag = llm_client.query_graph_with_crag

                def observe(function, name):
                    async def query(*args, **kwargs):
                        value = await function(*args, **kwargs)
                        row.setdefault("graph_returns", []).append(
                            {
                                "function": name,
                                "result_type": type(value).__name__,
                                "context_characters": len(value.context or ""),
                                "sources_count": len(value.sources),
                            }
                        )
                        return value

                    return query

                llm_client.query_graph_with_sources = observe(original_graph, "sources")
                llm_client.query_graph_with_crag = observe(original_crag, "crag")
                streamer = TelegramStreamer(cast(Bot, Delivery()), str(chat))
                started = time.monotonic()
                row["question_started_at_utc"] = datetime.now(UTC).isoformat()
                try:
                    reply = await BotCore().handle_message(
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
                    if not reply.text or reply.text == "Ответ не найден":
                        raise RuntimeError("BotCore did not produce an answer")
                    if case.startswith("faq-"):
                        expected = "EVAL470 synthetic answer v" + (
                            "1" if case == "faq-1" else "2"
                        )
                        row["expected_faq_answer_present"] = expected in reply.text
                    await streamer.finalize(
                        reply.text,
                        parse_mode=reply.parse_mode,
                        fallback_plain=reply.fallback_plain_on_format_error,
                    )
                    row["seconds"] = round(time.monotonic() - started, 4)
                    row["answer_sha256"] = hashlib.sha256(reply.text.encode()).hexdigest()
                    row["completed"] = row["parts"] > 0
                    state = ledger.store.snapshot()
                    if state["halted"] or state["pending_or_uncertain"]:
                        raise RuntimeError("Meter halted")
                finally:
                    llm_client.query_graph_with_sources = original_graph
                    llm_client.query_graph_with_crag = original_crag
                    await streamer.close()
        finally:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher
            await bot.session.close()
            await llm_client.cleanup_redis()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    plan = json.loads(Path("/reports/pilot-plan-approved-2026-11-21.json").read_text())
    ledger = connect_fix(plan, probe=args.probe)
    if args.probe:
        app = faq_application()
        print(
            json.dumps(
                {
                    "compatible": True,
                    "api_routes": len(app.routes),
                    "memory_current": int(
                        Path("/sys/fs/cgroup/memory.current").read_text()
                    ),
                    "budget": ledger.store.snapshot(),
                    "provider_calls": 0,
                }
            )
        )
        return
    for case in ("faq-1", "faq-2", "tools-1", "tools-2"):
        if any(ledger.store.snapshot()["cases"].get(case, {}).values()):
            raise RuntimeError("A remaining case already has accounted requests")
    # Preload safe code only so fork shares these pages, without API lifespan,
    # import queue, listeners or database connections.
    app = faq_application()
    if not app.routes:
        raise RuntimeError("Native API routes missing")
    receipt = Path("/reports/pilot-continue-20261123.json")
    with receipt.open("x"):
        pass
    result = {
        "cases": [],
        "budget_before": ledger.store.snapshot(),
        "api_bot_separate_processes": True,
        "provider_guard": "operator capture + active ledger policy guard",
    }
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe()
    # Fork before any app DB connection, event loop, provider request or thread.
    process = context.Process(target=api_child, args=(child,))
    process.start()
    child.close()
    try:
        asyncio.run(remaining(plan, ledger, parent, result))
    except BaseException as error:
        result["failed"] = type(error).__name__
    finally:
        if process.is_alive():
            parent.send("cleanup")
        try:
            if parent.poll(180):
                result["api_cleanup"] = parent.recv()
        except (EOFError, OSError):
            result["api_cleanup_unknown"] = True
        process.join(30)
        if process.is_alive():
            # Do not orphan a live process; runner close remains the outer safety net.
            process.terminate()
            process.join(5)
            result["api_cleanup_unknown"] = True
        result["budget_after"] = ledger.store.snapshot()
        result["finished_at_utc"] = datetime.now(UTC).isoformat()
        receipt.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    if "failed" in result or result.get("api_cleanup_unknown"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
