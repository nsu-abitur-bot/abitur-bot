"""Read only non-content budget metadata; no app, provider or secret-file access."""

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from tools.release470_store import BudgetStore, canonical

FIELDS = ("input", "output", "embedding", "micro_cny", "micro_rub")
RECEIPT_FIELDS = (
    "provider",
    "currency",
    "status",
    "usage",
    "usage_complete",
    "vectors_valid",
    "response_model",
    "error",
    "elapsed_seconds",
)
USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
)


def accounting_events(events: list[dict], attempt: int) -> list[dict]:
    """Also retain batch resume events that clear pending IDs in their snapshot."""
    matched = []
    pending: set[int] = set()
    for event in events:
        current = set(event["snapshot"]["pending_or_uncertain"])
        if event.get("attempt") == attempt or (
            attempt in pending and attempt not in current
        ):
            matched.append(event)
        pending = current
    return matched


def confirmed_usage(final: dict, held: dict, operation: str, events: list[dict]) -> bool:
    """A settled status after resume cannot manufacture a provider receipt."""
    finals = [event for event in events if event.get("event") == "final"]
    status = final.get("status")
    usage = final.get("usage")
    if (
        not finals
        or any(event.get("uncertain") is not False for event in finals)
        or type(status) is not int
        or not 200 <= status < 300
        or final.get("usage_complete") is not True
        or final.get("error") is not False
        or not isinstance(usage, dict)
    ):
        return False
    required = ["prompt_tokens", "total_tokens"]
    if operation == "generation":
        required.append("completion_tokens")
    if any(type(usage.get(key)) is not int or usage[key] < 0 for key in required):
        return False
    if operation == "embedding" and final.get("provider") == "cloudru":
        return (
            final.get("vectors_valid") is True
            and usage["prompt_tokens"] == usage["total_tokens"] == held["embedding"]
        )
    if operation == "generation" and final.get("provider") == "deepseek":
        return (
            usage["prompt_tokens"] == held["input"]
            and usage["completion_tokens"] == held["output"]
            and usage["total_tokens"] == held["input"] + held["output"]
        )
    return False


def public_receipt(final: dict) -> dict:
    result = {key: final.get(key) for key in RECEIPT_FIELDS}
    usage = final.get("usage")
    result["usage"] = {
        key: usage[key]
        for key in USAGE_FIELDS
        if isinstance(usage, dict) and type(usage.get(key)) is int and usage[key] >= 0
    }
    return result


def receipt(path: Path) -> dict:
    store = BudgetStore(path)
    events = store.journal()
    settled = dict.fromkeys(FIELDS, 0)
    accepted = dict.fromkeys(FIELDS, 0)
    unresolved = dict.fromkeys(FIELDS, 0)
    uncertain = []
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        for row in db.execute(
            "SELECT id,case_id,operation,status,held,final FROM attempts"
        ):
            held = json.loads(row["held"])
            final = json.loads(row["final"]) if row["final"] else {}
            related = accounting_events(events, row["id"])
            if row["status"] == "settled" and confirmed_usage(
                final, held, row["operation"], related
            ):
                for field in FIELDS:
                    settled[field] += held[field]
            else:
                accepted_reserve = row["status"] == "settled"
                totals = accepted if accepted_reserve else unresolved
                for field in FIELDS:
                    totals[field] += held[field]
                uncertain.append(
                    {
                        "id": row["id"],
                        "case_id": row["case_id"],
                        "operation": row["operation"],
                        "accounting_status": "accepted_without_confirmed_usage"
                        if accepted_reserve
                        else "unresolved",
                        "held_reserve": held,
                        "receipt": public_receipt(final),
                        "accounting_events": [
                            {
                                "event": event.get("event"),
                                "at_utc": datetime.fromtimestamp(
                                    event["at"], UTC
                                ).isoformat(),
                            }
                            for event in related
                        ],
                    }
                )
        journal_digest = db.execute(
            "SELECT digest FROM journal ORDER BY seq DESC LIMIT 1"
        ).fetchone()[0]
    return {
        "policy_sha256": hashlib.sha256(canonical(store.policy).encode()).hexdigest(),
        "snapshot": store.snapshot(),
        "confirmed_settled_totals": settled,
        "accepted_conservative_reserve_totals": accepted,
        "unresolved_reserve_totals": unresolved,
        "uncertain_reserves_not_confirmed_spend": uncertain,
        "journal_digest": journal_digest,
    }


if __name__ == "__main__":
    print(json.dumps(receipt(Path("/reports/release470-budget.sqlite3")), indent=2))
