"""Recalculate #390 measurements from immutable #388 evidence, without services."""

import argparse
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

from evals.review_admission import calculate_review
from tools.release_metrics import request_summary

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs/reports/release-388-full-data"
RUN = "full-20261110-02"


def require(value: object, message: str) -> None:
    if not value:
        raise ValueError(message)


def summarize_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Use the recorded Nov 10 tariff, never an inferred current billing rate."""
    costs = Decimal(0)
    prompt = output = embedding = generations = oauth = 0
    for row in rows:
        require(row["usage_complete"] is True, "Incomplete usage")
        require(row["http_status"] == 200 and not row["error_type"], "Failed attempt")
        usage = row["usage"]
        if row["operation"] == "oauth":
            oauth += 1
            continue
        fields = ("prompt_tokens", "total_tokens")
        if row["provider"] == "deepseek":
            fields += (
                "completion_tokens",
                "prompt_cache_hit_tokens",
                "prompt_cache_miss_tokens",
            )
        require(
            all(type(usage.get(k)) is int and usage[k] >= 0 for k in fields),
            "Invalid token counts",
        )
        if row["provider"] == "deepseek" and row["operation"] == "generation":
            p, c, hit, miss = (usage[k] for k in fields if k != "total_tokens")
            require(p == hit + miss and usage["total_tokens"] == p + c, "Token mismatch")
            costs += Decimal(miss) + Decimal(hit) * Decimal("0.02") + Decimal(c) * 4
            prompt += p
            output += c
            generations += 1
        elif row["provider"] == "gigachat" and row["operation"] == "embedding":
            require(usage["total_tokens"] == usage["prompt_tokens"], "Token mismatch")
            embedding += usage["prompt_tokens"]
        else:
            raise ValueError("Unsupported provider operation")
    return {
        "http_attempts": len(rows),
        "deepseek_generations": generations,
        "deepseek_input_tokens": prompt,
        "deepseek_output_tokens": output,
        "deepseek_historical_cny": str(costs / 1_000_000),
        "gigachat_embedding_tokens": embedding,
        "gigachat_cost": None,
        "oauth_attempts": oauth,
    }


def calculate() -> dict[str, Any]:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    fingerprints = {}
    verified: dict[str, bytes] = {}
    for spec in manifest["files"]:
        name = spec["file"]
        require(Path(name).name == name and name not in fingerprints, "Invalid manifest")
        raw = (DATA / name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        require(digest == spec["sha256"], "Evidence SHA mismatch: " + name)
        fingerprints[name] = digest
        verified[name] = raw
    required = {
        f"{condition}-{RUN}-batch-{batch:02}-{kind}"
        for condition in ("with-kb", "empty-kb")
        for batch in range(5)
        for kind in ("report.json", "usage.jsonl", "intents.jsonl")
    } | {"with-kb-aggregate.json", "empty-kb-aggregate.json"}
    require(required <= verified.keys(), "Missing required evidence")
    conditions = {}
    all_rows = []
    for condition in ("with-kb", "empty-kb"):
        results = []
        rows = []
        for batch in range(5):
            prefix = f"{condition}-{RUN}-batch-{batch:02}"
            report = json.loads(verified[prefix + "-report.json"])
            require(
                report["status"] == "completed" and report["total"] == 10, "Bad batch"
            )
            usage = [
                json.loads(line)
                for line in verified[prefix + "-usage.jsonl"].splitlines()
            ]
            intents = [
                json.loads(line)
                for line in verified[prefix + "-intents.jsonl"].splitlines()
            ]
            require(len(usage) == len(intents), "Unsettled intents")
            require(
                all(
                    type(r["attempt"]) is int and r["attempt"] > 0
                    for r in usage + intents
                ),
                "Invalid attempt ID",
            )
            by_attempt = {r["attempt"]: r for r in usage}
            by_intent = {r["attempt"]: r for r in intents}
            require(len(by_attempt) == len(usage), "Duplicate attempt")
            require(len(by_intent) == len(intents), "Duplicate intent")
            require(by_attempt.keys() == by_intent.keys(), "Attempt ID sets differ")
            for intent in intents:
                require(intent["attempt"] in by_attempt, "Missing attempt")
                row = by_attempt[intent["attempt"]]
                require(
                    all(row.get(k) == v for k, v in intent.items()), "Intent mismatch"
                )
                require(
                    row["phase"] in {"answer", "judge", "calibration"}, "Unknown phase"
                )
                if row["phase"] != "calibration":
                    require(row["case_id"] in report["selected_ids"], "Unknown case")
            require(
                len(usage)
                == report["accounting"]["attempts_started"]
                == report["accounting"]["attempts_finished"],
                "Accounting mismatch",
            )
            results.extend(report["results"])
            rows.extend(usage)
        require(
            len(results) == len({r["id"] for r in results}) == 50, "Wrong case coverage"
        )
        observations = [
            {
                "duration_ms": r["elapsed_seconds"] * 1000,
                "status": "timeout"
                if r.get("error") == "TimeoutError"
                else "error"
                if r.get("error")
                else "ok",
            }
            for r in results
        ]
        phases = {
            phase: summarize_usage([r for r in rows if r["phase"] == phase])
            for phase in ("answer", "judge", "calibration")
        }
        counts = Counter(
            r["case_id"]
            for r in rows
            if r["phase"] == "answer" and r["operation"] == "generation"
        )
        conditions[condition] = {
            "answer_usage_per_case": {
                case["id"]: summarize_usage(
                    [
                        r
                        for r in rows
                        if r["phase"] == "answer" and r["case_id"] == case["id"]
                    ]
                )
                for case in results
            },
            "usage_by_phase": phases,
            "total_usage": summarize_usage(rows),
            "evaluation_latency_including_judge": request_summary(observations),
            "bot_only_latency": None,
            "faq_matched_cases": sum(r["faq_matched"] for r in results),
            "cases_calling_tools": sum(bool(r["tools_called"]) for r in results),
            "answer_generation_attempts_per_case": {
                "mean": sum(counts.values()) / 50,
                "max": max((counts[r["id"]] for r in results), default=0),
                "denominator": 50,
            },
        }
        all_rows.extend(rows)
    return {
        "schema_version": 1,
        "run_id": RUN,
        "scope": "Historical isolated bfe4f8a/custom, not current NSU channel acceptance",
        "pricing": {
            "snapshot_date": "2026-11-10",
            "deepseek_cny_per_million": {
                "cache_hit": "0.02",
                "cache_miss": "1",
                "output": "4",
            },
            "source": "388-full-evaluation-2026-11-10.md and original run calculation",
            "current_tariff_verified": False,
            "billing_debit_verified": False,
        },
        "verified_inputs": fingerprints,
        "conditions": conditions,
        "total_usage": summarize_usage(all_rows),
        "quality": calculate_review(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    text = json.dumps(calculate(), ensure_ascii=False, indent=2) + "\n"
    if args.report:
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
