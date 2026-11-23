"""Explicit isolated test entrypoints; never used by normal API/bot deployment."""

import argparse
import hashlib
import inspect
import json
import os
import runpy
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

from tools.admission_usage import IndexingLimit
from tools.release470_budget import Release470Ledger

FIXED_BUDGET = Path("/reports/release470-budget.sqlite3")
_bot_question: ContextVar[tuple[str, str, str] | None] = ContextVar(
    "release470_verified_bot_question", default=None
)


def instrument_bot_core(function, ledger: Release470Ledger, *, recipient: str):
    """Bind approved user_text before BotCore adds its username prefix.

    This is restricted to the already authorized private Telegram recipient.
    It never strips a prefix supplied directly to the pipeline.
    """
    signature = inspect.signature(function)
    plan = ledger.store.policy["case_plan"]

    @wraps(function)
    async def wrapped(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs).arguments
        text, name = arguments.get("user_text"), arguments.get("user_name")
        if (
            arguments.get("channel") != "telegram"
            or arguments.get("external_user_id") != recipient
            or arguments.get("session_id") != recipient
            or not isinstance(text, str)
            or not isinstance(name, str)
            or not 1 <= len(name) <= 64
            or any(ord(char) < 32 or char in "[]" for char in name)
        ):
            raise IndexingLimit("Unapproved isolated Telegram input")
        raw_digest = hashlib.sha256(text.encode()).hexdigest()
        cases = [case for case, expected in plan.items() if raw_digest == expected]
        if len(cases) != 1:
            raise IndexingLimit("Question is outside the immutable pilot plan")
        formatted_digest = hashlib.sha256(f"[from {name}] {text}".encode()).hexdigest()
        token = _bot_question.set((cases[0], raw_digest, formatted_digest))
        try:
            with ledger.question(cases[0]):
                return await function(*args, **kwargs)
        finally:
            _bot_question.reset(token)

    return wrapped


def instrument_question(function, ledger: Release470Ledger):
    """Label each planned question before intent/FAQ/RAG/background work starts."""
    signature = inspect.signature(function)
    plan = ledger.store.policy["case_plan"]

    @wraps(function)
    async def wrapped(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs).arguments
        message = arguments.get("message")
        if not isinstance(message, str):
            raise IndexingLimit("Unknown pilot message contract")
        digest = hashlib.sha256(message.encode()).hexdigest()
        cases = [case for case, expected in plan.items() if digest == expected]
        bound = _bot_question.get()
        if bound is not None:
            case, raw_digest, formatted_digest = bound
            if plan.get(case) != raw_digest or digest != formatted_digest:
                raise IndexingLimit("BotCore changed the verified question binding")
            cases = [case]
        if len(cases) != 1:
            raise IndexingLimit("Question is outside the immutable pilot plan")
        with ledger.question(cases[0]):
            return await function(*args, **kwargs)

    return wrapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("api", "bot"), required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Separate owner authorization and --execute are required")
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    ledger = Release470Ledger(Path("/reports"), budget_path=FIXED_BUDGET)
    if ledger.store.policy["offline_only"]:
        raise IndexingLimit("An offline fixture budget cannot launch live processes")
    # Install hooks before provider clients, routes or handlers can retain aliases.
    with ledger.capture():
        from db.postgres.db import engine

        if (engine.url.host, engine.url.database) != ("postgres", "eval470"):
            raise IndexingLimit("Only the fixed isolated database is allowed")
        from pipeline import llm_client

        llm_client.ask_local_llm = instrument_question(llm_client.ask_local_llm, ledger)
        ledger.context("setup", "setup")
        if args.component == "api":
            import uvicorn

            uvicorn.run("api.main:app", host="0.0.0.0", port=8000, workers=1)
        else:
            from bot.core import BotCore

            # Routing metadata remains private and is never added to reports.
            recipient = json.loads(Path("/run/eval470/recipient.json").read_text())
            recipient_id = recipient.get("chat_id")
            if type(recipient_id) is not int or recipient_id <= 0:
                raise IndexingLimit("A verified private Telegram recipient is required")
            BotCore.handle_message = instrument_bot_core(
                BotCore.handle_message, ledger, recipient=str(recipient_id)
            )
            runpy.run_module("main", run_name="__main__")


if __name__ == "__main__":
    main()
