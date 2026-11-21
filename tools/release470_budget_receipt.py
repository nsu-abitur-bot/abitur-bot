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


def receipt(path: Path) -> dict:
    store = BudgetStore(path)
    events = store.journal()
    settled = dict.fromkeys(FIELDS, 0)
    uncertain = []
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        for row in db.execute(
            "SELECT id,case_id,operation,status,held,final FROM attempts"
        ):
            held = json.loads(row["held"])
            if row["status"] == "settled":
                for field in FIELDS:
                    settled[field] += held[field]
            else:
                final = json.loads(row["final"]) if row["final"] else {}
                uncertain.append(
                    {
                        "id": row["id"],
                        "case_id": row["case_id"],
                        "operation": row["operation"],
                        "held_reserve": held,
                        "receipt": {key: final.get(key) for key in RECEIPT_FIELDS},
                        "accounting_events": [
                            {
                                "event": event.get("event"),
                                "at_utc": datetime.fromtimestamp(
                                    event["at"], UTC
                                ).isoformat(),
                            }
                            for event in events
                            if event.get("attempt") == row["id"]
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
        "uncertain_reserves_not_confirmed_spend": uncertain,
        "journal_digest": journal_digest,
    }


if __name__ == "__main__":
    print(json.dumps(receipt(Path("/reports/release470-budget.sqlite3")), indent=2))
