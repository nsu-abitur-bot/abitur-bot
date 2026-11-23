"""Uncertain reserves must never be presented as confirmed provider spending."""

import json
import tempfile
import unittest
from pathlib import Path

from tools.release470_budget_receipt import receipt
from tools.release470_store import BudgetStore


class OperatorStore(BudgetStore):
    """Offline operator contract: accept the reserve without inventing usage."""

    def accept_reserved_cost(self):
        with self.transaction() as db:
            self._verify(db)
            attempts = [
                row[0]
                for row in db.execute("SELECT id FROM attempts WHERE status != 'settled'")
            ]
            db.execute("UPDATE attempts SET status='settled' WHERE status != 'settled'")
            db.execute("UPDATE state SET halted=0 WHERE id=1")
            self._event(
                db, {"event": "operator_accept_reserved_cost", "attempts": attempts}
            )


class ReceiptTests(unittest.TestCase):
    def test_deepseek_confirmed_usage_is_exported_separately_from_reserves(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.sqlite3"
            BudgetStore.create(path)
            store = BudgetStore(path)
            actual = {
                "input": 10,
                "output": 2,
                "embedding": 0,
                "micro_cny": 36,
                "micro_rub": 0,
            }
            attempt = store.reserve("rag-1", "generation", actual, "a" * 64, {})
            store.settle(
                attempt,
                "a" * 64,
                actual,
                {
                    "provider": "deepseek",
                    "status": 200,
                    "error": False,
                    "usage_complete": True,
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                },
                False,
            )
            exported = receipt(path)
            self.assertEqual(exported["confirmed_settled_totals"], actual)
            self.assertEqual(exported["uncertain_reserves_not_confirmed_spend"], [])
            self.assertEqual(
                exported["accepted_conservative_reserve_totals"]["micro_cny"], 0
            )

    def test_export_after_operator_resume_keeps_original_unknown_reserve_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.sqlite3"
            BudgetStore.create(
                path,
                cloudru={
                    "approved": True,
                    "micro_rub_limit": 1000000,
                    "embedding_limit": 10000,
                    "rate_hundredths": 3222,
                    "tokenizer_sha256": "0" * 64,
                    "tokenizer_revision": "offline-fixture",
                },
            )
            store = OperatorStore(path)
            held = {
                "input": 0,
                "output": 0,
                "embedding": 94,
                "micro_cny": 0,
                "micro_rub": 3029,
            }
            attempt = store.reserve("setup", "embedding", held, "a" * 64, {})
            original_receipt = {
                "provider": "cloudru",
                "status": None,
                "usage": {},
                "usage_complete": False,
                "error": True,
            }
            store.settle(attempt, "a" * 64, None, original_receipt, True)
            store.accept_reserved_cost()
            state_before = store.snapshot()
            events_before = store.journal()
            exported = receipt(path)
            self.assertFalse(exported["snapshot"]["halted"])
            self.assertEqual(exported["snapshot"]["pending_or_uncertain"], [])
            self.assertEqual(exported["confirmed_settled_totals"]["micro_rub"], 0)
            self.assertEqual(exported["confirmed_settled_totals"]["embedding"], 0)
            reserve = exported["uncertain_reserves_not_confirmed_spend"][0]
            self.assertEqual(reserve["held_reserve"], held)
            self.assertIn(
                "operator_accept_reserved_cost",
                [event["event"] for event in reserve["accounting_events"]],
            )
            self.assertEqual(exported["snapshot"]["totals"], held)
            self.assertEqual(exported["accepted_conservative_reserve_totals"], held)
            self.assertEqual(exported["unresolved_reserve_totals"]["micro_rub"], 0)
            self.assertEqual(
                reserve["accounting_status"], "accepted_without_confirmed_usage"
            )
            self.assertEqual(store.snapshot(), state_before)
            self.assertEqual(store.journal(), events_before)
            known = {**held, "embedding": 10, "micro_rub": 323}
            next_attempt = store.reserve("setup", "embedding", known, "b" * 64, {})
            store.settle(
                next_attempt,
                "b" * 64,
                known,
                {
                    "provider": "cloudru",
                    "status": 200,
                    "usage_complete": True,
                    "vectors_valid": True,
                    "error": False,
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                },
                False,
            )
            continued = receipt(path)
            self.assertEqual(continued["confirmed_settled_totals"], known)
            self.assertEqual(continued["accepted_conservative_reserve_totals"], held)
            self.assertEqual(continued["snapshot"]["totals"]["micro_rub"], 3352)

    def test_original_uncertainty_is_retained_even_with_complete_usage_in_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.sqlite3"
            BudgetStore.create(path)
            store = OperatorStore(path)
            held = {
                "input": 10,
                "output": 2,
                "embedding": 0,
                "micro_cny": 36,
                "micro_rub": 0,
            }
            attempt = store.reserve("rag-1", "generation", held, "a" * 64, {})
            store.settle(
                attempt,
                "a" * 64,
                None,
                {
                    "provider": "deepseek",
                    "status": 200,
                    "error": False,
                    "usage_complete": True,
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                },
                True,
            )
            store.accept_reserved_cost()
            exported = receipt(path)
            self.assertEqual(exported["confirmed_settled_totals"]["micro_cny"], 0)
            self.assertEqual(exported["accepted_conservative_reserve_totals"], held)

    def test_settled_cost_and_unknown_reserve_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.sqlite3"
            BudgetStore.create(
                path,
                cloudru={
                    "approved": True,
                    "micro_rub_limit": 1000000,
                    "embedding_limit": 10000,
                    "rate_hundredths": 3222,
                    "tokenizer_sha256": "0" * 64,
                    "tokenizer_revision": "offline-fixture",
                },
            )
            store = BudgetStore(path)
            known = {
                "input": 0,
                "output": 0,
                "embedding": 10,
                "micro_cny": 0,
                "micro_rub": 323,
            }
            first = store.reserve("setup", "embedding", known, "a" * 64, {})
            store.settle(
                first,
                "a" * 64,
                known,
                {
                    "provider": "cloudru",
                    "status": 200,
                    "usage_complete": True,
                    "vectors_valid": True,
                    "error": False,
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                },
                False,
            )
            unknown = {**known, "embedding": 20, "micro_rub": 645}
            second = store.reserve("setup", "embedding", unknown, "b" * 64, {})
            store.settle(
                second,
                "b" * 64,
                None,
                {
                    "provider": "cloudru",
                    "error": True,
                    "raw_error": "do-not-export",
                    "usage": {"private_error": "do-not-export"},
                },
                True,
            )
            exported = receipt(path)
            self.assertEqual(exported["confirmed_settled_totals"]["micro_rub"], 323)
            self.assertEqual(exported["snapshot"]["totals"]["micro_rub"], 968)
            self.assertEqual(len(exported["uncertain_reserves_not_confirmed_spend"]), 1)
            self.assertEqual(exported["unresolved_reserve_totals"]["micro_rub"], 645)
            self.assertEqual(
                exported["accepted_conservative_reserve_totals"]["micro_rub"], 0
            )
            self.assertTrue(exported["snapshot"]["halted"])
            self.assertNotIn("do-not-export", json.dumps(exported))


if __name__ == "__main__":
    unittest.main()
