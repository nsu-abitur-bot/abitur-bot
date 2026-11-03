"""Measure controlled product-code paths. Never a live latency/quality benchmark."""

import argparse
import asyncio
import json
import os
import platform
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from evals.checks import check_assertion
from evals.schema import load_scenarios
from tools.release_metrics import request_summary


async def probe(repeats: int, concurrency: int) -> dict:
    if repeats < 1 or concurrency < 1:
        raise ValueError("Positive repeats and concurrency required")
    if os.environ.get("PYTHON_DOTENV_DISABLED") != "1":
        raise ValueError("Disable dotenv before importing the application")
    from evals.offline import offline_pipeline

    scenarios = [s for s in load_scenarios("evals/scenarios.yaml") if not s.requires_llm]
    samples = []
    semaphore = asyncio.Semaphore(concurrency)
    start_utc = datetime.now(UTC).isoformat()
    with offline_pipeline() as runner:

        async def run(scenario, scheduled: float) -> None:
            async with semaphore:
                started = perf_counter()
                path = "unclassified"
                status = "error"
                try:
                    async with asyncio.timeout(10):
                        observation = await runner(
                            scenario, f"release-probe:{uuid4().hex}"
                        )
                    path = (
                        "faq"
                        if observation.faq_matched
                        else "tools"
                        if observation.tools_called
                        else "rag"
                    )
                    status = (
                        "ok"
                        if all(
                            check_assertion(a, observation) for a in scenario.assertions
                        )
                        else "error"
                    )
                except TimeoutError:
                    status = "timeout"
                except Exception:
                    status = "error"
                samples.append(
                    {
                        "case_id": scenario.id,
                        "path": path,
                        "status": status,
                        "duration_ms": (perf_counter() - scheduled) * 1000,
                        "service_ms": (perf_counter() - started) * 1000,
                    }
                )

        started = perf_counter()
        tasks = [
            asyncio.create_task(run(s, perf_counter()))
            for _ in range(repeats)
            for s in scenarios
        ]
        await asyncio.gather(*tasks)
        elapsed = perf_counter() - started
    groups = defaultdict(list)
    for row in samples:
        groups[row["path"]].append(row)
    return {
        "schema_version": 1,
        "scope": "controlled_offline; fake retrieval/providers/transport, memory history",
        "started_at": start_utc,
        "finished_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "concurrency": concurrency,
        "repeats": repeats,
        "elapsed_seconds": elapsed,
        "completed_per_second": len(samples) / elapsed,
        "summary": request_summary(samples),
        "by_path": {key: request_summary(rows) for key, rows in groups.items()},
        "provider_attempts": None,
        "tokens": None,
        "cost": None,
        "limitation": (
            "Not production throughput, delivery, model cost or admission quality"
        ),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(probe(args.repeats, args.concurrency))
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "by_path": report["by_path"]}))
    if report["summary"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
