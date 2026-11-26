"""Opt-in full application profiling, with synthetic Telegram transport only.

Run inside the configured deployment environment, never via pytest. Reads settings
and allowlisted reference rows, creates a uniquely named test database and KB copy,
and never polls Telegram or sends a message. No settings/secrets are serialized.
A persistent attempt ledger caps paid SDK attempts across process restarts.
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import threading
import traceback
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

REFERENCE_TABLES = (
    "settings",
    "faq_entry",
    "abbreviation",
    "faculty",
    "program",
    "admission_score",
    "document",
    "topic",
)
QUESTIONS = (
    ("fit_programs", "Какие программы бакалавриата есть на ФИТ НГУ?"),
    (
        "distance_contract",
        "Можно ли заключить договор платного обучения на ФИТ НГУ дистанционно?",
    ),
    ("scores_tool", "Какие проходные баллы были на ФИТ НГУ в 2025 году?"),
    ("missing_2027_price", "Сколько стоит обучение на ФИТ НГУ в 2027 году?"),
)


class NumericTimings(logging.Handler):
    """Only explicitly allowlisted numeric/static timing records leave the process."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if not message.startswith(("latency ", "latency_request ")):
            return
        data = json.loads(message.split(" ", 1)[1])
        data["event"] = message.split(" ", 1)[0]
        data["finished_ms"] = round(perf_counter() * 1000, 3)
        if "duration_ms" in data:
            data["started_ms"] = round(data["finished_ms"] - data["duration_ms"], 3)
        self.rows.append(data)
        print(json.dumps(data), flush=True)


class SyntheticTelegram:
    """Zero-network transport: its times MUST NOT be reported as Telegram RTT."""

    def __init__(self) -> None:
        self.final_parts = 0

    async def send_chat_action(self, *args: Any, **kwargs: Any) -> None:
        await asyncio.sleep(0)

    async def send_message_draft(self, *args: Any, **kwargs: Any) -> None:
        await asyncio.sleep(0)

    async def send_message(self, *args: Any, **kwargs: Any) -> None:
        self.final_parts += 1
        await asyncio.sleep(0)


def instrument_sdk(ledger: Path, limit: int) -> None:
    """Count actual Gemini HTTP attempts, including retries, without request bodies."""
    from google.genai._api_client import BaseApiClient

    from config.latency import _request_id, stage

    call_state: ContextVar[dict[str, Any] | None] = ContextVar("sdk_call", default=None)
    lock = threading.Lock()

    def reserve() -> None:
        with lock:
            used = json.loads(ledger.read_text())["attempts"] if ledger.exists() else 0
            if used >= limit:
                raise RuntimeError("Profiling request budget exhausted")
            ledger.write_text(json.dumps({"attempts": used + 1, "limit": limit}))
            call = call_state.get()
            if call is not None:
                call["attempts"] += 1

    async_request = BaseApiClient._async_request
    sync_request = BaseApiClient._request

    def finish(call: dict[str, Any]) -> None:
        print(
            json.dumps(
                {
                    "event": "sdk_call",
                    "request_id": _request_id.get(),
                    "call_id": call["id"],
                    "attempts": call["attempts"],
                    "retries": max(0, call["attempts"] - 1),
                }
            ),
            flush=True,
        )

    async def async_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        call = {"id": uuid4().hex, "attempts": 0}
        token = call_state.set(call)
        try:
            with stage("provider_sdk_call_async"):
                return await async_request(self, *args, **kwargs)
        finally:
            finish(call)
            call_state.reset(token)

    def sync_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        call = {"id": uuid4().hex, "attempts": 0}
        token = call_state.set(call)
        try:
            with stage("provider_sdk_call_sync"):
                return sync_request(self, *args, **kwargs)
        finally:
            finish(call)
            call_state.reset(token)

    BaseApiClient._async_request = async_call
    BaseApiClient._request = sync_call
    async_original = BaseApiClient._async_request_once
    sync_original = BaseApiClient._request_once

    async def async_attempt(self: Any, *args: Any, **kwargs: Any) -> Any:
        reserve()
        with stage("provider_http_attempt_async"):
            return await async_original(self, *args, **kwargs)

    def sync_attempt(self: Any, *args: Any, **kwargs: Any) -> Any:
        reserve()
        with stage("provider_http_attempt_sync"):
            return sync_original(self, *args, **kwargs)

    BaseApiClient._async_request_once = async_attempt
    BaseApiClient._request_once = sync_attempt


async def load_reference_matchers() -> None:
    """Match bot startup without importing main and installing application log sinks."""
    from abbrev.expander import get_abbrev_expander
    from db.postgres.db import AsyncSessionLocal
    from db.postgres.services.abbrev import AbbrevDbService
    from db.postgres.services.faq import FaqDbService
    from faq.matcher import get_faq_matcher

    async with AsyncSessionLocal() as session:
        entries = await FaqDbService(session).get_all()
        get_faq_matcher().load_items(
            [
                {"question": e.question, "aliases": e.aliases, "answer": e.answer}
                for e in entries
            ]
        )
        abbrevs = await AbbrevDbService(session).get_all()
        get_abbrev_expander().load_items(
            [{"short": e.short, "full": e.full} for e in abbrevs]
        )


def configure_numeric_logging() -> NumericTimings:
    """Suppress ordinary logs, whose arguments can include dialogs or secrets."""
    for item in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(item, logging.Logger):
            item.handlers.clear()
            item.propagate = True
            item.setLevel(logging.NOTSET)
    logging.getLogger().handlers = [logging.NullHandler()]
    logging.getLogger().setLevel(logging.CRITICAL)
    latency = logging.getLogger("abitur.latency")
    timings = NumericTimings()
    latency.handlers = [timings]
    latency.propagate = False
    latency.setLevel(logging.INFO)
    return timings


async def run(args: argparse.Namespace, snapshot_override: Any = None) -> None:
    timings = configure_numeric_logging()

    from sqlalchemy import select, text
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from config.instance import get_setting, install
    from db.postgres import db
    from db.postgres.models import Base
    from db.postgres.services.instance_settings import InstanceSettingsService

    source_engine = db.engine
    if snapshot_override is None:
        async with db.AsyncSessionLocal() as session:
            snapshot = await InstanceSettingsService(session).read()
    else:
        snapshot = snapshot_override
    install(snapshot.values, snapshot.revision)
    if get_setting("LLM_PROVIDER", "gemini") != "gemini":
        raise RuntimeError("Harness requires Gemini attempt instrumentation")
    if os.getenv("LIGHTRAG_STORAGE", "json") != "json":
        raise RuntimeError("Postgres LightRAG snapshot is not implemented")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Fresh snapshot output directory required")
    args.output.mkdir(parents=True, exist_ok=True)
    ledger_path = args.ledger or (args.output / "budget.json")
    instrument_sdk(ledger_path, args.attempt_limit)

    url = source_engine.url.set(database=args.database)
    admin = create_async_engine(
        source_engine.url.set(database="postgres"), isolation_level="AUTOCOMMIT"
    )
    async with admin.connect() as connection:
        exists = await connection.scalar(
            text("SELECT 1 FROM pg_database WHERE datname=:n"), {"n": args.database}
        )
        if exists:
            raise RuntimeError("Fresh isolated database name required")
        await connection.execute(text(f'CREATE DATABASE "{args.database}"'))
    await admin.dispose()
    target = create_async_engine(url)
    db.engine = target
    db.AsyncSessionLocal = async_sessionmaker(target, expire_on_commit=False)
    async with target.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    counts = {}
    async with source_engine.connect() as source, target.begin() as destination:
        for name in REFERENCE_TABLES:
            table = Base.metadata.tables[name]
            query = select(table)
            if name == "settings":
                query = query.where(table.c.key != "instance_settings_v1")
            rows = [dict(row) for row in (await source.execute(query)).mappings()]
            counts[name] = len(rows)
            if rows:
                await destination.execute(table.insert(), rows)
    await source_engine.dispose()

    original_kb = Path(os.getenv("LIGHTRAG_WORKSPACE_BASE", "./data/lightrag"))
    isolated_kb = args.output / "kb"
    if not isolated_kb.exists():
        shutil.copytree(
            original_kb,
            isolated_kb,
            ignore=shutil.ignore_patterns("*llm*cache*", "*response_cache*", "*.pkl"),
        )
    values = dict(snapshot.values)
    values["LIGHTRAG_WORKSPACE_BASE"] = str(isolated_kb)
    os.environ["LIGHTRAG_WORKSPACE_BASE"] = str(isolated_kb)
    install(values, snapshot.revision)
    os.environ["REDIS_URL"] = args.redis_url
    os.environ["DATABASE_URL"] = url.render_as_string(hide_password=False)
    os.environ["DB_NAME"] = args.database

    from aiogram.types import Chat, Message, User

    from bot import core as core_module
    from bot.core import BotCore
    from bot.telegram_bot import answer_message
    from config.latency import stage
    from llm.profiles import LLMProfiles
    from pipeline import llm_client

    timings = configure_numeric_logging()
    original_ask = core_module.ask_local_llm
    trace_flags: dict[str, Any] = {}

    async def capture_answer(*arguments: Any, **keywords: Any) -> Any:
        answer = await original_ask(*arguments, **keywords)
        trace_flags.update(
            faq_matched=answer.trace.faq_matched,
            tools_called=list(answer.trace.tools_called),
            sources_count=len(answer.sources),
        )
        return answer

    core_module.ask_local_llm = capture_answer
    print(
        json.dumps(
            {
                "event": "manifest",
                "scope": "real application; synthetic Telegram",
                "source_sha": args.source_sha,
                "reference_counts": counts,
                "model": get_setting("GEMINI_MODEL"),
                "chat_max_tokens": LLMProfiles.CHAT.max_tokens,
                "kb_files": sum(1 for p in isolated_kb.rglob("*") if p.is_file()),
                "attempt_limit": args.attempt_limit,
            }
        ),
        flush=True,
    )
    with stage("startup_faq_abbrev"):
        await load_reference_matchers()
    case_number = 0
    for repeat in range(args.repeats):
        for index, (name, question) in enumerate(QUESTIONS):
            ledger = (
                json.loads(ledger_path.read_text())
                if ledger_path.exists()
                else {"attempts": 0}
            )
            if ledger["attempts"] >= args.attempt_limit:
                print(json.dumps({"event": "budget_stop", **ledger}), flush=True)
                return
            if name not in args.cases:
                continue
            telegram = SyntheticTelegram()
            # Unique session per question/repeat prevents dialog history confounding.
            synthetic_id = 376000000 + args.run_id * 100 + repeat * 10 + index
            message = Message(
                message_id=1,
                date=datetime.now(UTC),
                chat=Chat(id=synthetic_id, type="private"),
                from_user=User(id=synthetic_id, is_bot=False, first_name="Synthetic"),
                text=question,
            )
            print(
                json.dumps(
                    {
                        "event": "case",
                        "case": name,
                        "repeat": repeat,
                        "process_first": case_number == 0,
                    }
                ),
                flush=True,
            )
            case_number += 1
            trace_flags.clear()
            before_rows = len(timings.rows)
            started = perf_counter()
            await answer_message(telegram, BotCore(), message)  # type: ignore[arg-type]
            wall = (perf_counter() - started) * 1000
            # Background intent time is visible, but is not part of response wall time.
            while llm_client._background_tasks:
                await asyncio.gather(
                    *list(llm_client._background_tasks), return_exceptions=True
                )
            print(
                json.dumps(
                    {
                        "event": "case_result",
                        "case": name,
                        "repeat": repeat,
                        "wall_ms": round(wall, 3),
                        "final_parts": telegram.final_parts,
                        **trace_flags,
                        "generation_completed": any(
                            row.get("stage") == "generation_tools_total"
                            and row.get("status") == "ok"
                            for row in timings.rows[before_rows:]
                        ),
                        "errors": sum(
                            row.get("status") == "error"
                            for row in timings.rows[before_rows:]
                        ),
                    }
                ),
                flush=True,
            )
    await llm_client.cleanup_redis()
    from db.redis.client import get_redis_client

    redis = await get_redis_client()
    await redis.close()
    await target.dispose()


def main() -> None:
    """Validate isolation and bound this explicitly invoked paid experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--redis-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--repeats", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--attempt-limit", type=int, default=14, choices=range(1, 53))
    parser.add_argument("--ledger", type=Path)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[name for name, _ in QUESTIONS],
        default=[name for name, _ in QUESTIONS],
    )
    args = parser.parse_args()
    if (
        not args.database.startswith("latency376_")
        or not args.database.replace("_", "").isalnum()
    ):
        parser.error("database must be an isolated latency376_ name")
    try:
        asyncio.run(run(args))
    except Exception as exc:
        # Never emit traceback/exception text: SDK exceptions may contain secrets.
        print(
            json.dumps(
                {
                    "event": "harness_error",
                    "error_type": type(exc).__name__,
                    "frames": [
                        {
                            "file": Path(f.filename).name,
                            "line": f.lineno,
                            "function": f.name,
                        }
                        for f in traceback.extract_tb(exc.__traceback__)
                    ],
                }
            ),
            flush=True,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
