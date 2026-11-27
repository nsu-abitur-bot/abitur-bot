"""Opt-in CRAG comparison on public KB fragments; no chat delivery or index writes.

stdin is a private JSON snapshot: settings, faculty_seed, initial and extra chunks.
The target PostgreSQL database must be a dedicated crag219_* database.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import types
from contextvars import ContextVar
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import httpx
from langchain_core.messages import BaseMessage
from openai import AsyncOpenAI

from llm.profiles import LLMProfile


async def run(args: argparse.Namespace, snapshot: dict[str, Any]) -> dict[str, Any]:
    from sqlalchemy.engine import make_url

    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    url = make_url(os.environ["DATABASE_URL"])
    if not url.database or not url.database.startswith("crag219_"):
        raise ValueError("Dedicated crag219_* database required")
    from config.instance import install
    from db.postgres.db import AsyncSessionLocal, engine
    from db.postgres.models import Base
    from db.postgres.services.faculty import FacultyService
    from llm.factory import get_llm_provider

    install(snapshot["settings"], snapshot["revision"])
    async with engine.begin() as connection:
        # Never reset an existing database, even one with the expected prefix.
        existing = await connection.exec_driver_sql(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema='public'"
        )
        if existing.scalar_one():
            raise ValueError("Fresh database required")
        await connection.run_sync(Base.metadata.create_all)
    async with AsyncSessionLocal() as session:
        await FacultyService(session).upsert_from_seed(snapshot["faculty_seed"])
    initial, extra = snapshot["initial"], snapshot["extra"]
    if not initial or not extra:
        raise ValueError("A populated retrieval snapshot is required")
    from rag import crag as candidate

    baseline = types.ModuleType("crag219_baseline")
    sys.modules[baseline.__name__] = baseline
    source = args.baseline.read_text()
    exec(compile(source, str(args.baseline), "exec"), baseline.__dict__)
    provider = get_llm_provider()
    client = getattr(provider, "client", None)
    if not isinstance(client, AsyncOpenAI):
        raise ValueError("This harness requires the configured DeepSeek/OpenAI transport")
    active: ContextVar[dict | None] = ContextVar("crag219_call", default=None)
    attempts = 0
    exhausted = False
    results = []

    async def request_hook(request: httpx.Request) -> None:
        nonlocal attempts, exhausted
        if attempts >= args.attempt_limit:
            exhausted = True
            raise RuntimeError("Benchmark request limit reached")
        attempts += 1
        call = active.get()
        if call is not None:
            call["http_attempts"] += 1
        starts = {}

        async def trace(event: str, _info: dict[str, Any]) -> None:
            if call is None:
                return
            name, _, state = event.rpartition(".")
            if state == "started":
                starts[name] = perf_counter()
            elif state in {"complete", "failed"} and name in starts:
                elapsed = (perf_counter() - starts.pop(name)) * 1000
                call["transport_ms"][name] = call["transport_ms"].get(name, 0) + elapsed

        request.extensions["trace"] = trace

    client._client.event_hooks.setdefault("request", []).append(request_hook)
    try:
        for label in ("baseline", "candidate") * args.repeats:
            module = cast(Any, baseline if label == "baseline" else candidate)
            calls = []

            class Measured:
                async def generate(
                    self, messages: list[BaseMessage], profile: LLMProfile | None = None
                ) -> str:
                    if exhausted:
                        raise RuntimeError("Benchmark request limit reached")
                    call: dict[str, Any] = {
                        "prompt_sha256": hashlib.sha256(
                            json.dumps(
                                [m.content for m in messages], ensure_ascii=False
                            ).encode()
                        ).hexdigest(),
                        "kind": "grade"
                        if "Фрагмент:\n" in messages[-1].content
                        else "refine",
                        "http_attempts": 0,
                        "transport_ms": {},
                    }
                    calls.append(call)
                    token = active.set(call)
                    started = perf_counter()
                    try:
                        result = await provider.generate_with_usage(
                            messages, profile=profile
                        )
                        call["tokens"] = {
                            "input": result.usage.prompt_tokens,
                            "output": result.usage.completion_tokens,
                        }
                        return result.text
                    except Exception as exc:
                        call["error_type"] = type(exc).__name__
                        raise
                    finally:
                        call["duration_ms"] = (perf_counter() - started) * 1000
                        active.reset(token)

            measured = Measured()
            module.get_llm_provider = lambda: measured
            chunks = [
                module.CragChunk(
                    index=i,
                    content=x["content"],
                    source_url=x["source_url"],
                    file_path=x["source_url"],
                )
                for i, x in enumerate(initial)
            ]
            retrieved = 0

            async def retrieve(_question: str) -> list[Any]:
                nonlocal retrieved
                retrieved += 1
                return [
                    module.CragChunk(
                        index=i + 100,
                        content=x["content"],
                        source_url=x["source_url"],
                        file_path=x["source_url"],
                    )
                    for i, x in enumerate(initial + extra)
                ]

            started = perf_counter()
            kept, _ = await module.run_crag(
                args.question, chunks, retrieve_fn=retrieve, config=module.CragConfig()
            )
            row = {
                "arm": label,
                "duration_ms": (perf_counter() - started) * 1000,
                "logical_calls": len(calls),
                "http_attempts": sum(c["http_attempts"] for c in calls),
                "grades": sum(c["kind"] == "grade" for c in calls),
                "retrievals": retrieved,
                "kept_sha256": [
                    hashlib.sha256(c.content.encode()).hexdigest() for c in kept
                ],
                "calls": calls,
            }
            results.append(row)
            if exhausted or any("error_type" in c for c in calls):
                raise RuntimeError(
                    "Incomplete comparison; provider error or request limit"
                )
    finally:
        client._client.event_hooks["request"].remove(request_hook)
        await client.close()
        await engine.dispose()
    return {
        "scope": (
            "isolated CRAG component; real provider, fixed real KB candidates; "
            "no Telegram/RAG delivery"
        ),
        "baseline_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "candidate_source_sha256": hashlib.sha256(
            Path(candidate.__file__).read_bytes()
        ).hexdigest(),
        "question": args.question,
        "input_sha256": hashlib.sha256(
            json.dumps(initial + extra, sort_keys=True).encode()
        ).hexdigest(),
        "http_attempts": attempts,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--question", default="Какие программы бакалавриата есть на ФИТ НГУ?"
    )
    parser.add_argument("--attempt-limit", type=int, choices=range(1, 81), default=60)
    parser.add_argument("--repeats", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    try:
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as output:
            report = asyncio.run(run(args, json.load(sys.stdin)))
            json.dump(report, output, ensure_ascii=False, indent=2)
        print(
            json.dumps(
                {
                    "status": "completed",
                    "http_attempts": report["http_attempts"],
                    "arms": [
                        {k: v for k, v in row.items() if k != "calls"}
                        for row in report["results"]
                    ],
                }
            )
        )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
