"""One durable, transactional budget shared by all isolated pilot processes."""

import hashlib
import json
import re
import sqlite3
import time
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from tools.admission_usage import IndexingLimit

POLICY = {
    "schema": 1,
    "scope": "release470",
    "currency": "CNY",
    "caps": {
        "input": 300000,
        "output": 10000,
        "embedding": 10000,
        "micro_cny": 3000000,
        "attempts": 512,
    },
    "question": {"input": 32000, "output": 1024, "embedding": 512, "generations": 40},
    "rates_micro_cny": {"input": 2, "output": 8},
}
KINDS = ("input", "output", "embedding", "micro_cny")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class BudgetStore:
    """No implicit creation, reset, timeout refund or directory-based resume."""

    @staticmethod
    def create(
        path: Path, *, offline_only: bool = True, case_plan: dict | None = None
    ) -> None:
        """Explicit owner initialization, once, before attaching any process."""
        plan = case_plan or {}
        ids = {"faq-1", "faq-2", "rag-1", "rag-2", "tools-1", "tools-2"}
        if (
            type(offline_only) is not bool
            or not isinstance(plan, dict)
            or set(plan) - ids
            or len(set(plan.values())) != len(plan)
            or any(
                not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
                for v in plan.values()
            )
            or (not offline_only and not plan)
        ):
            raise IndexingLimit("A fixed unique question plan is required")
        with path.open("xb"):
            pass
        with closing(sqlite3.connect(path)) as db, db:
            db.executescript("""
                CREATE TABLE metadata (id INTEGER PRIMARY KEY CHECK(id=1), policy TEXT);
                CREATE TABLE attempts (
                    id INTEGER PRIMARY KEY, case_id TEXT NOT NULL,
                    operation TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL, held TEXT NOT NULL, final TEXT
                );
                CREATE TABLE journal (
                    seq INTEGER PRIMARY KEY, body TEXT NOT NULL, digest TEXT NOT NULL
                );
                CREATE TABLE state (id INTEGER PRIMARY KEY CHECK(id=1), halted INTEGER);
                INSERT INTO state VALUES (1,0);
            """)
            db.execute(
                "INSERT INTO metadata VALUES (1,?)",
                (canonical({**POLICY, "offline_only": offline_only, "case_plan": plan}),),
            )
            policy = canonical(
                {**POLICY, "offline_only": offline_only, "case_plan": plan}
            )
            body = canonical(
                {
                    "event": "initialize",
                    "at": time.time(),
                    "policy_sha256": hashlib.sha256(policy.encode()).hexdigest(),
                    "snapshot": {
                        "totals": dict.fromkeys(KINDS, 0),
                        "cases": {},
                        "attempts_started": 0,
                        "attempts_finished": 0,
                        "pending_or_uncertain": [],
                        "halted": False,
                    },
                }
            )
            db.execute(
                "INSERT INTO journal(body,digest) VALUES (?,?)",
                (body, hashlib.sha256(body.encode()).hexdigest()),
            )
        path.chmod(0o600)

    def __init__(self, path: Path) -> None:
        if not path.is_file() or path.is_symlink():
            raise IndexingLimit("A pre-existing fixed pilot budget is required")
        self.path = path.resolve()
        with self.transaction() as db:
            if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise IndexingLimit("Budget integrity check failed")
            self.policy = json.loads(
                db.execute("SELECT policy FROM metadata WHERE id=1").fetchone()[0]
            )
            if any(self.policy.get(k) != v for k, v in POLICY.items()):
                raise IndexingLimit("Pilot policy differs from the initialized budget")
            self._verify(db)

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _verify(self, db) -> None:
        previous = ""
        last = None
        sequence = 0
        for row in db.execute("SELECT * FROM journal ORDER BY seq"):
            sequence += 1
            if row["seq"] != sequence:
                raise IndexingLimit("Pilot journal sequence is incomplete")
            expected = hashlib.sha256((previous + row["body"]).encode()).hexdigest()
            if row["digest"] != expected:
                raise IndexingLimit("Pilot journal integrity check failed")
            previous = expected
            last = json.loads(row["body"])
        count = db.execute("SELECT count(*) FROM attempts").fetchone()[0]
        reserves = sum(
            json.loads(r[0])["event"] == "reserve"
            for r in db.execute("SELECT body FROM journal")
        )
        if count != reserves:
            raise IndexingLimit("Attempts and reservation journal disagree")
        first = db.execute("SELECT body FROM journal WHERE seq=1").fetchone()
        if (
            first is None
            or json.loads(first[0]).get("policy_sha256")
            != hashlib.sha256(canonical(self.policy).encode()).hexdigest()
        ):
            raise IndexingLimit("Immutable pilot policy differs from initialization")
        if last is not None and last["snapshot"] != self._snapshot(db):
            raise IndexingLimit("Budget state differs from its committed journal")

    def _snapshot(self, db) -> dict:
        totals = dict.fromkeys(KINDS, 0)
        cases: dict[str, dict] = {}
        pending = []
        for row in db.execute("SELECT * FROM attempts ORDER BY id"):
            values = json.loads(row["held"])
            case = cases.setdefault(
                row["case_id"], {**dict.fromkeys(KINDS, 0), "generations": 0}
            )
            for kind in KINDS:
                totals[kind] += values[kind]
                case[kind] += values[kind]
            case["generations"] += int(row["operation"] == "generation")
            if row["status"] != "settled":
                pending.append(row["id"])
        return {
            "totals": totals,
            "cases": cases,
            "attempts_started": db.execute("SELECT count(*) FROM attempts").fetchone()[0],
            "attempts_finished": db.execute(
                "SELECT count(*) FROM attempts WHERE final IS NOT NULL"
            ).fetchone()[0],
            "pending_or_uncertain": pending,
            "halted": bool(
                db.execute("SELECT halted FROM state WHERE id=1").fetchone()[0]
            ),
        }

    def _event(self, db, event: dict) -> None:
        row = db.execute(
            "SELECT digest FROM journal ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        previous = row[0] if row else ""
        body = canonical({**event, "at": time.time(), "snapshot": self._snapshot(db)})
        digest = hashlib.sha256((previous + body).encode()).hexdigest()
        db.execute("INSERT INTO journal(body,digest) VALUES (?,?)", (body, digest))

    def snapshot(self) -> dict:
        with self.transaction() as db:
            self._verify(db)
            return self._snapshot(db)

    def output_limit(self, case_id: str, requested: int) -> int:
        state = self.snapshot()
        if state["halted"] or state["pending_or_uncertain"]:
            raise IndexingLimit("Budget has an unresolved physical attempt")
        cap = (
            POLICY["caps"]["output"]
            if case_id == "setup"
            else POLICY["question"]["output"]
        )
        result = min(
            requested,
            1024,
            cap - state["cases"].get(case_id, {}).get("output", 0),
            POLICY["caps"]["output"] - state["totals"]["output"],
        )
        if type(result) is not int or result < 1:
            raise IndexingLimit("Release output allowance exhausted")
        return result

    def reserve(
        self, case_id: str, operation: str, held: dict, fingerprint: str, evidence: dict
    ) -> int:
        with self.transaction() as db:
            self._verify(db)
            state = self._snapshot(db)
            # The transport gate serializes active physical calls. Pending here
            # therefore means a crashed/interrupted previous owner, never a refund.
            if state["halted"] or state["pending_or_uncertain"]:
                raise IndexingLimit("Unresolved attempt; reconcile before any new call")
            if state["attempts_started"] >= POLICY["caps"]["attempts"]:
                raise IndexingLimit("Physical attempt cap reached")
            case = state["cases"].get(case_id, {})
            per = POLICY["caps"] if case_id == "setup" else POLICY["question"]
            for kind in KINDS:
                if type(held[kind]) is not int or held[kind] < 0:
                    raise IndexingLimit("Invalid reservation")
                if state["totals"][kind] + held[kind] > POLICY["caps"][kind]:
                    raise IndexingLimit("Global pilot budget exhausted")
                if kind in per and case.get(kind, 0) + held[kind] > per[kind]:
                    raise IndexingLimit("Question token budget exhausted")
            if (
                case_id != "setup"
                and operation == "generation"
                and case.get("generations", 0) >= 40
            ):
                raise IndexingLimit("Question generation cap reached")
            cursor = db.execute(
                "INSERT INTO attempts(case_id,operation,fingerprint,status,held) "
                "VALUES (?,?,?,'reserved',?)",
                (case_id, operation, fingerprint, canonical(held)),
            )
            attempt = cursor.lastrowid
            if attempt is None:
                raise IndexingLimit("Reservation identity was not created")
            self._event(
                db,
                {
                    "event": "reserve",
                    "attempt": attempt,
                    "case_id": case_id,
                    "operation": operation,
                    "held": held,
                    "evidence": evidence,
                },
            )
            return int(attempt)

    def settle(
        self,
        attempt: int,
        fingerprint: str,
        actual: dict | None,
        receipt: dict,
        uncertain: bool,
    ) -> None:
        with self.transaction() as db:
            self._verify(db)
            row = db.execute("SELECT * FROM attempts WHERE id=?", (attempt,)).fetchone()
            if row is None or row["fingerprint"] != fingerprint:
                raise IndexingLimit("Attempt identity mismatch")
            final = canonical(receipt)
            if row["final"] is not None:

                def stable(v):
                    return {
                        k: x for k, x in json.loads(v).items() if k != "elapsed_seconds"
                    }

                if stable(row["final"]) != stable(final):
                    raise IndexingLimit("Conflicting final receipt")
                return
            held = json.loads(row["held"])
            if actual is not None:
                for kind in KINDS:
                    if type(actual[kind]) is not int or actual[kind] < 0:
                        raise IndexingLimit("Invalid final usage")
                    uncertain |= actual[kind] > held[kind]
                # Unknown/error outcomes retain their whole reserve. If usage
                # exceeded a bound, preserve the larger observed amount honestly.
                held = {
                    k: max(held[k], actual[k]) if uncertain else actual[k] for k in KINDS
                }
            status = "uncertain" if uncertain else "settled"
            db.execute(
                "UPDATE attempts SET status=?,held=?,final=? WHERE id=?",
                (status, canonical(held), final, attempt),
            )
            if uncertain:
                db.execute("UPDATE state SET halted=1 WHERE id=1")
            self._event(
                db,
                {
                    "event": "final",
                    "attempt": attempt,
                    "receipt": receipt,
                    "uncertain": uncertain,
                },
            )

    def journal(self) -> list[dict]:
        with self.transaction() as db:
            self._verify(db)
            return [
                json.loads(r[0])
                for r in db.execute("SELECT body FROM journal ORDER BY seq")
            ]
