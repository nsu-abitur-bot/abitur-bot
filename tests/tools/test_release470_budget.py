"""Offline guards: no provider, application startup or database is involved."""

import tempfile
import unittest
from pathlib import Path

import httpx

from tools.admission_usage import IndexingLimit
from tools.release470_budget import Release470Ledger


class ReleaseBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ledger = Release470Ledger(Path(self.temp.name))
        self.ledger.context("pilot", "rag-1")

    def request(self, **changes) -> httpx.Request:
        payload = {
            "model": "deepseek-flash",
            "max_tokens": 1024,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "synthetic"}],
        }
        return httpx.Request(
            "POST",
            "https://api.deepseek.com/chat/completions",
            json={**payload, **changes},
        )

    def finish(self, row, **changes) -> None:
        self.ledger.finish(
            row,
            httpx.Response(
                200,
                json={
                    "model": "deepseek-flash",
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "total_tokens": 110,
                    },
                    **changes,
                },
            ),
            None,
        )

    def test_oauth_form_is_metered_without_json_decoding(self) -> None:
        calls = []

        def dispatch(request):
            calls.append(request)
            self.assertEqual(request.content, b"scope=GIGACHAT_API_PERS")
            return httpx.Response(200, json={"access_token": "synthetic"})

        request = httpx.Request(
            "POST",
            "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
            data={"scope": "GIGACHAT_API_PERS"},
        )
        self.ledger.context("setup", "setup")
        with (
            self.ledger.capture(),
            httpx.Client(transport=httpx.MockTransport(dispatch)) as client,
        ):
            self.assertEqual(client.send(request).status_code, 200)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.ledger.sequence, 1)
        self.assertEqual(self.ledger.completed, 1)
        self.assertFalse(self.ledger.halted)
        self.assertEqual(self.ledger.totals, {"input": 0, "output": 0, "embedding": 0})

    def test_provider_output_uses_remaining_question_allowance(self) -> None:
        from llm.providers.deepseek import DeepSeekProvider

        row = self.ledger.begin(self.request())
        self.finish(row)
        provider = object.__new__(DeepSeekProvider)
        provider.model_name = "deepseek-flash"
        with self.ledger.capture():
            self.assertEqual(provider._params(None)["max_tokens"], 1014)
        self.assertEqual(provider._params(None)["max_tokens"], 1024)

    def test_outstanding_output_is_reserved_before_next_attempt(self) -> None:
        row = self.ledger.begin(self.request())
        with self.assertRaises(IndexingLimit):
            self.ledger.begin(self.request())
        self.finish(row)
        self.assertEqual(self.ledger.accounting()["token_totals"]["input"], 100)

    def test_question_budget_persists_after_context_switch(self) -> None:
        row = self.ledger.begin(self.request())
        self.finish(row)
        self.ledger.context("pilot", "faq-1")
        self.ledger.context("pilot", "rag-1")
        self.assertEqual(self.ledger.cases["rag-1"]["input"], 100)
        self.ledger.cases["rag-1"]["input"] = 32000
        with self.assertRaises(IndexingLimit):
            self.ledger.begin(self.request())

    def test_physical_generation_limit_applies_before_attempt(self) -> None:
        self.ledger.cases["rag-1"] = {
            "input": 0,
            "output": 0,
            "embedding": 0,
            "generations": 40,
        }
        with self.assertRaises(IndexingLimit):
            self.ledger.begin(self.request())
        self.assertEqual(self.ledger.sequence, 0)

    def test_missing_usage_stops_further_spending(self) -> None:
        row = self.ledger.begin(self.request())
        self.finish(row, usage={})
        with self.assertRaises(IndexingLimit):
            self.ledger.begin(self.request())

    def test_unknown_model_response_halts(self) -> None:
        row = self.ledger.begin(self.request())
        self.finish(row, model="unapproved-model")
        self.assertTrue(self.ledger.halted)

    def test_unapproved_request_model_rejected(self) -> None:
        with self.assertRaises(IndexingLimit):
            self.ledger.begin(self.request(model="deepseek-chat"))
        self.assertEqual(self.ledger.sequence, 0)

    def test_output_and_thinking_must_be_bounded(self) -> None:
        for change in ({"max_tokens": 1025}, {"thinking": {"type": "enabled"}}):
            with self.assertRaises(IndexingLimit):
                self.ledger.begin(self.request(**change))
        self.assertEqual(self.ledger.sequence, 0)

    def test_prior_setup_is_counted_in_global_budget(self) -> None:
        self.ledger.context("setup", "setup")
        self.ledger.totals["input"] = 300000
        with self.assertRaises(IndexingLimit):
            self.ledger.begin(self.request())

    def test_capture_blocks_retry_before_second_transport_call(self) -> None:
        calls = []

        def dispatch(request):
            calls.append(request.url.host)
            return httpx.Response(503, json={})

        with (
            self.ledger.capture(),
            httpx.Client(transport=httpx.MockTransport(dispatch)) as client,
        ):
            self.assertEqual(client.send(self.request()).status_code, 503)
            with self.assertRaises(IndexingLimit):
                client.send(self.request())
        self.assertEqual(calls, ["api.deepseek.com"])
        self.assertEqual(self.ledger.sequence, 1)


if __name__ == "__main__":
    unittest.main()
