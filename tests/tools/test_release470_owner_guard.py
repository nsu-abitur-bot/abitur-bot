"""Mirror current policy limits without changing the operator meter or policy."""

import contextvars
import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx

from tools.admission_usage import IndexingLimit
from tools.release470_budget import Release470Ledger
from tools.release470_owner_guard import FIELDS, attach_owner_guard, check_reservation
from tools.release470_store import POLICY, BudgetStore


def policy():
    result = copy.deepcopy(POLICY)
    result["caps"]["micro_rub"] = 1000000
    return result


def snapshot():
    return {
        "halted": False,
        "pending_or_uncertain": [],
        "attempts_started": 59,
        "totals": dict.fromkeys(FIELDS, 0),
        "cases": {},
    }


class OwnerTests(unittest.TestCase):
    def test_amended_policy_allows_a_reserve_past_the_old_one_ruble_cap(self):
        state = snapshot()
        state["totals"]["micro_rub"] = 1000000
        original = Mock(return_value=60)
        ledger = SimpleNamespace(
            store=SimpleNamespace(
                snapshot=lambda: state,
                reserve=original,
                policy={
                    "caps": {
                        **policy()["caps"],
                        "micro_rub": 1000000000,
                        "micro_cny": 100000000,
                        "attempts": 100000,
                    },
                    "question": {
                        "input": 256000,
                        "output": 32000,
                        "embedding": 100000,
                        "generations": 100,
                    },
                },
            ),
            _case=contextvars.ContextVar("amended_case", default="setup"),
            output_limit=lambda value: value,
        )
        before = copy.deepcopy(ledger.store.policy)
        attach_owner_guard(ledger)
        self.assertEqual(
            ledger.store.reserve("setup", "embedding", {"micro_rub": 1}, "hash", {}), 60
        )
        original.assert_called_once()
        self.assertEqual(ledger.store.policy, before)
        state["totals"]["micro_cny"] = 3000000
        self.assertEqual(
            ledger.store.reserve("setup", "generation", {"micro_cny": 1}, "hash", {}), 60
        )
        for kind, cap in (("micro_rub", 1000000000), ("micro_cny", 100000000)):
            state["totals"][kind] = cap
            with self.subTest(kind=kind), self.assertRaises(IndexingLimit):
                ledger.store.reserve("setup", "embedding", {kind: 1}, "hash", {})
            state["totals"][kind] = 0
        self.assertEqual(original.call_count, 2)

    def test_amended_question_and_attempt_limits_replace_legacy_constants(self):
        state = snapshot()
        state["attempts_started"] = 512
        state["cases"]["tools-1"] = {"output": 1024, "generations": 40}
        active = policy()
        active["caps"].update(output=2000000, attempts=100000)
        active["question"].update(output=32000, generations=100)
        ledger = SimpleNamespace(
            store=SimpleNamespace(
                policy=active, snapshot=lambda: state, reserve=Mock(return_value=513)
            ),
            _case=contextvars.ContextVar("current_case", default="tools-1"),
            output_limit=Mock(side_effect=lambda value: min(value, 1024)),
        )
        original_output = ledger.output_limit
        attach_owner_guard(ledger)
        # The upstream per-request bound is still retained.
        self.assertEqual(ledger.output_limit(32000), 1024)
        original_output.assert_called_once_with(30976)
        self.assertEqual(
            ledger.store.reserve("tools-1", "generation", {"output": 1}, "hash", {}), 513
        )
        state["cases"]["tools-1"]["generations"] = 100
        with self.assertRaises(IndexingLimit):
            ledger.store.reserve("tools-1", "generation", {}, "hash", {})
        state["attempts_started"] = 100000
        with self.assertRaises(IndexingLimit):
            ledger.store.reserve("setup", "embedding", {}, "hash", {})

    def test_incomplete_policy_cannot_attach_guard(self):
        ledger = SimpleNamespace(store=SimpleNamespace(policy={"caps": {}}))
        with self.assertRaises(IndexingLimit):
            attach_owner_guard(ledger)

    def test_owner_guard_retains_operator_http_meter_and_durable_reservation(self):
        class Counter:
            evidence = {"kind": "offline-synthetic"}

            def __call__(self, payload):
                return 100

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.sqlite3"
            BudgetStore.create(path)
            ledger = Release470Ledger(
                Path(directory), budget_path=path, counter=Counter()
            )
            ledger.context("pilot", "tools-1")
            attach_owner_guard(ledger)
            response = httpx.Response(
                200,
                json={
                    "model": "deepseek-flash",
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "total_tokens": 110,
                    },
                },
            )
            transport = Mock(return_value=response)
            with (
                patch.dict(os.environ, {"PYTHON_DOTENV_DISABLED": "1"}),
                ledger.capture(patch_provider=False),
                httpx.Client(transport=httpx.MockTransport(transport)) as client,
            ):
                client.post(
                    "https://api.deepseek.com/chat/completions",
                    json={
                        "model": "deepseek-flash",
                        "thinking": {"type": "disabled"},
                        "max_tokens": 24,
                        "messages": [{"role": "user", "content": "synthetic"}],
                    },
                )
            state = BudgetStore(path).snapshot()
            self.assertEqual(state["attempts_started"], 1)
            self.assertEqual(state["attempts_finished"], 1)
            self.assertEqual(state["totals"]["output"], 10)
            self.assertEqual(state["pending_or_uncertain"], [])
            transport.assert_called_once()

    def test_every_global_cap_includes_current_reserve(self):
        for kind in FIELDS:
            cap = policy()["caps"][kind]
            state = snapshot()
            state["totals"][kind] = cap
            with self.subTest(kind=kind), self.assertRaises(IndexingLimit):
                check_reservation(state, "setup", "embedding", {kind: 1}, policy())

    def test_question_limits_pending_and_generation_cap(self):
        for fields in ({"input": 32001}, {"output": 1025}, {"embedding": 513}):
            with self.assertRaises(IndexingLimit):
                check_reservation(snapshot(), "faq-1", "generation", fields, policy())
        state = snapshot()
        state["cases"]["faq-1"] = {"generations": 40}
        with self.assertRaises(IndexingLimit):
            check_reservation(state, "faq-1", "generation", {}, policy())
        for change in ({"halted": True}, {"pending_or_uncertain": [58]}):
            with self.assertRaises(IndexingLimit):
                check_reservation(
                    {**snapshot(), **change}, "setup", "embedding", {}, policy()
                )

    def test_original_reserve_and_remaining_output_are_preserved(self):
        state = snapshot()
        state["cases"]["tools-1"] = {"output": 1000}
        original = Mock(return_value=60)
        active_policy = policy()
        before = copy.deepcopy(active_policy)
        ledger = SimpleNamespace(
            store=SimpleNamespace(
                snapshot=lambda: state, reserve=original, policy=active_policy
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
        self.assertEqual(active_policy, before)
        with self.assertRaises(IndexingLimit):
            ledger.store.reserve("tools-1", "generation", {"output": 25}, "hash", {})
        self.assertEqual(original.call_count, 1)
        with self.assertRaises(IndexingLimit):
            attach_owner_guard(ledger)
