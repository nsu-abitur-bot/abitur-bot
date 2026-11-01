"""Summarize content-free JSONL timings, never add overlapping stage durations."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep failures explicit and require completed generation for performance stats."""
    requests: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("request_id"):
            requests[row["request_id"]].append(row)
    output: list[dict[str, Any]] = []
    for request_id, events in requests.items():
        stages: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            if event.get("stage"):
                stages[event["stage"]].append(event)
        if "telegram_total" not in stages:
            continue
        generated = any(
            row.get("status") == "ok" for row in stages.get("generation_tools_total", [])
        )
        total = stages["telegram_total"][-1]
        first = next(iter(stages.get("generation_first_delta", [])), None)
        first_from_entry = None
        if first and "finished_ms" in first and "started_ms" in total:
            first_from_entry = round(first["finished_ms"] - total["started_ms"], 3)
        sdk_calls = [row for row in events if row.get("event") == "sdk_call"]
        attempt_spans = [
            row
            for row in events
            if row.get("stage", "").startswith("provider_http_attempt_")
        ]
        attempts = (
            sum(row["attempts"] for row in sdk_calls)
            if sdk_calls
            else len(attempt_spans)
            if attempt_spans
            else None
        )
        output.append(
            {
                "request_id": request_id,
                "generation_completed": generated,
                "wall_ms": total["duration_ms"],
                "first_text_from_handler_ms": first_from_entry,
                "http_attempts": attempts,
                "sdk_retries": sum(row["retries"] for row in sdk_calls)
                if sdk_calls
                else None,
                "errors": [
                    row["stage"] for row in events if row.get("status") == "error"
                ],
                "stages_ms": {
                    name: [row["duration_ms"] for row in value]
                    for name, value in stages.items()
                },
            }
        )
    valid = [row for row in output if row["generation_completed"]]
    wall = sorted(row["wall_ms"] for row in valid)
    return {
        "requests": output,
        "completed_generations": len(valid),
        "excluded_incomplete_generations": len(output) - len(valid),
        "wall_median_ms": statistics.median(wall) if wall else None,
        "wall_p95_nearest_rank_ms": wall[math.ceil(0.95 * len(wall)) - 1]
        if wall
        else None,
        "limitation": "Small-sample nearest-rank p95 is usually the maximum; "
        "synthetic transport is not Telegram RTT; FAQ-only requests "
        "need explicit trace classification and are excluded here.",
    }


def main() -> None:
    """Read only prefiltered JSONL artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.startswith("{")
    ]
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
