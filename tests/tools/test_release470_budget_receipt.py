"""Uncertain reserves must never be presented as confirmed provider spending."""

import json
import tempfile
import unittest
from pathlib import Path

from tools.release470_budget_receipt import receipt
from tools.release470_store import BudgetStore


class ReceiptTests(unittest.TestCase):
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
            store.settle(first, "a" * 64, known, {"provider": "cloudru"}, False)
            unknown = {**known, "embedding": 20, "micro_rub": 645}
            second = store.reserve("setup", "embedding", unknown, "b" * 64, {})
            store.settle(
                second,
                "b" * 64,
                None,
                {"provider": "cloudru", "error": True, "raw_error": "do-not-export"},
                True,
            )
            exported = receipt(path)
            self.assertEqual(exported["confirmed_settled_totals"]["micro_rub"], 323)
            self.assertEqual(exported["snapshot"]["totals"]["micro_rub"], 968)
            self.assertEqual(len(exported["uncertain_reserves_not_confirmed_spend"]), 1)
            self.assertTrue(exported["snapshot"]["halted"])
            self.assertNotIn("do-not-export", json.dumps(exported))


if __name__ == "__main__":
    unittest.main()
