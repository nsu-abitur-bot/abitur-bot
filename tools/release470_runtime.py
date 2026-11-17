"""Explicit isolated test entrypoints; never used by normal API/bot deployment."""

import argparse
import hashlib
import inspect
import os
import runpy
from functools import wraps
from pathlib import Path

from tools.admission_usage import IndexingLimit
from tools.release470_budget import Release470Ledger

FIXED_BUDGET = Path("/reports/release470-budget.sqlite3")


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
            runpy.run_module("main", run_name="__main__")


if __name__ == "__main__":
    main()
