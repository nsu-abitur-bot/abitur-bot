"""Owner caps remain strict even with a larger operator policy."""

import contextvars
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from tools.admission_usage import IndexingLimit
from tools.release470_owner_guard import CAPS, attach_owner_guard, check_reservation


def snapshot():
    return {
        "halted": False,
        "pending_or_uncertain": [],
        "attempts_started": 59,
        "totals": dict.fromkeys(CAPS, 0),
        "cases": {},
    }


class OwnerTests(unittest.TestCase):
    def test_every_global_cap_includes_current_reserve(self):
        for kind, cap in CAPS.items():
            state = snapshot()
            state["totals"][kind] = cap
            with self.subTest(kind=kind), self.assertRaises(IndexingLimit):
                check_reservation(state, "setup", "embedding", {kind: 1})

    def test_question_limits_pending_and_generation_cap(self):
        for fields in ({"input": 32001}, {"output": 1025}, {"embedding": 513}):
            with self.assertRaises(IndexingLimit):
                check_reservation(snapshot(), "faq-1", "generation", fields)
        state = snapshot()
        state["cases"]["faq-1"] = {"generations": 40}
        with self.assertRaises(IndexingLimit):
            check_reservation(state, "faq-1", "generation", {})
        for change in ({"halted": True}, {"pending_or_uncertain": [58]}):
            with self.assertRaises(IndexingLimit):
                check_reservation({**snapshot(), **change}, "setup", "embedding", {})

    def test_original_reserve_and_remaining_output_are_preserved(self):
        state = snapshot()
        state["cases"]["tools-1"] = {"output": 1000}
        original = Mock(return_value=60)
        policy = {"caps": {"micro_rub": 1000000000}}
        before = copy.deepcopy(policy)
        ledger = SimpleNamespace(
            store=SimpleNamespace(
                snapshot=lambda: state, reserve=original, policy=policy
            ),
            _case=contextvars.ContextVar("test_case", default="tools-1"),
            output_limit=lambda value: value,
        )
        attach_owner_guard(ledger)
        self.assertEqual(ledger.output_limit(1024), 24)
        self.assertEqual(
            ledger.store.reserve("tools-1", "generation", {"output": 24}, "hash", {}), 60
        )
        original.assert_called_once()
        self.assertEqual(policy, before)
        with self.assertRaises(IndexingLimit):
            ledger.store.reserve("tools-1", "generation", {"output": 25}, "hash", {})
        self.assertEqual(original.call_count, 1)
        with self.assertRaises(IndexingLimit):
            attach_owner_guard(ledger)
