"""Synthetic Cloud.ru SDK/usage tests, with no provider requests or secrets."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import httpx
from openai import OpenAI

from tools.admission_usage import IndexingLimit
from tools.release470_budget import Release470Ledger
from tools.release470_cloudru import MODEL, CloudCounter, cost_micro_rub
from tools.release470_store import BudgetStore


class Counter:
    evidence = {"kind": "offline-synthetic"}

    def __call__(self, payload):
        return 100


def cloud_body(**changes):
    return {"model": MODEL, "input": ["synthetic"], "encoding_format": "float", **changes}


def reply(tokens=100, **changes):
    return httpx.Response(
        200,
        json={
            "model": MODEL,
            "data": [{"index": 0, "embedding": [1.0] * 2048}],
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            **changes,
        },
    )


class CloudTests(unittest.TestCase):
    def setUp(self):
        guard = patch.dict(os.environ, {"PYTHON_DOTENV_DISABLED": "1"})
        guard.start()
        self.addCleanup(guard.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "budget.sqlite3"
        self.settings = {
            "approved": True,
            "micro_rub_limit": 1000000,
            "embedding_limit": 10000,
            "rate_hundredths": 3222,
            "tokenizer_sha256": "0" * 64,
            "tokenizer_revision": "offline-fixture-only",
        }
        BudgetStore.create(self.path, cloudru=self.settings)
        self.ledger = Release470Ledger(
            self.root,
            budget_path=self.path,
            counter=Counter(),
            embedding_counter=Counter(),
        )
        self.ledger.context("pilot", "rag-1")

    def begin(self, **changes):
        return self.ledger.begin(
            httpx.Request(
                "POST",
                "https://foundation-models.api.cloud.ru/v1/embeddings",
                json=cloud_body(**changes),
            )
        )

    def test_rub_cost_rounding(self):
        self.assertEqual(cost_micro_rub(10000, 3222), 322200)
        self.assertEqual(cost_micro_rub(1, 3222), 33)
        with self.assertRaises(IndexingLimit):
            cost_micro_rub(True, 3222)

    def test_cny_and_rub_are_never_added(self):
        row = self.begin()
        self.ledger.finish(row, reply(), None)
        request = httpx.Request(
            "POST",
            "https://api.deepseek.com/chat/completions",
            json={
                "model": "deepseek-flash",
                "thinking": {"type": "disabled"},
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "synthetic"}],
            },
        )
        row = self.ledger.begin(request)
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
                },
            ),
            None,
        )
        state = self.ledger.accounting()
        self.assertEqual(state["totals"]["micro_rub"], 3222)
        self.assertEqual(state["totals"]["micro_cny"], 280)
        self.assertEqual(state["token_totals"]["embedding"], 100)
        reopened = Release470Ledger(self.root / "different-report", budget_path=self.path)
        self.assertEqual(reopened.accounting()["totals"], state["totals"])

    def test_unknown_usage_stops_all_providers_and_keeps_rub_reserve(self):
        row = self.begin()
        with self.assertRaises(IndexingLimit):
            self.ledger.finish(row, reply(usage={}), None)
        self.assertEqual(self.ledger.accounting()["totals"]["micro_rub"], 3222)
        with self.assertRaises(IndexingLimit):
            self.ledger.output_limit(1)

    def test_missing_total_usage_is_not_treated_as_zero_or_complete(self):
        row = self.begin()
        with self.assertRaises(IndexingLimit):
            self.ledger.finish(row, reply(usage={"prompt_tokens": 100}), None)
        self.assertTrue(self.ledger.accounting()["halted"])

    def test_default_cloud_policy_denies_before_transport(self):
        path = self.root / "default.sqlite3"
        BudgetStore.create(path)
        meter = Release470Ledger(self.root, budget_path=path)
        meter.context("pilot", "rag-1")
        with self.assertRaises(IndexingLimit):
            meter.begin(
                httpx.Request(
                    "POST",
                    "https://foundation-models.api.cloud.ru/v1/embeddings",
                    json=cloud_body(),
                )
            )
        self.assertEqual(meter.accounting()["attempts_started"], 0)

    def test_http_error_holds_reserve_even_with_usage(self):
        row = self.begin()
        response = reply()
        response.status_code = 500
        with self.assertRaises(IndexingLimit):
            self.ledger.finish(row, response, None)
        self.assertEqual(self.ledger.accounting()["totals"]["micro_rub"], 3222)
        self.assertTrue(self.ledger.accounting()["halted"])

    def test_invalid_vectors_and_model_stop_the_budget(self):
        row = self.begin()
        with self.assertRaises(IndexingLimit):
            self.ledger.finish(row, reply(data=[{"index": 0, "embedding": [1.0]}]), None)
        self.assertTrue(self.ledger.accounting()["halted"])

    def test_unknown_model_stops(self):
        row = self.begin()
        with self.assertRaises(IndexingLimit):
            self.ledger.finish(row, reply(model="other-model"), None)

    def test_rub_cap_is_checked_before_transport(self):
        path = self.root / "small-budget.sqlite3"
        BudgetStore.create(path, cloudru={**self.settings, "micro_rub_limit": 100})
        meter = Release470Ledger(self.root, budget_path=path, embedding_counter=Counter())
        meter.context("pilot", "rag-1")
        calls = []
        with (
            meter.capture(patch_provider=False),
            httpx.Client(
                transport=httpx.MockTransport(lambda r: (calls.append(r), reply())[1])
            ) as client,
        ):
            with self.assertRaises(IndexingLimit):
                client.post(
                    "https://foundation-models.api.cloud.ru/v1/embeddings",
                    json=cloud_body(),
                )
        self.assertEqual(calls, [])

    def test_sdk_embedding_and_wire_body_unchanged(self):
        calls = []

        def dispatch(request):
            calls.append(json.loads(request.content))
            return reply()

        with httpx.Client(transport=httpx.MockTransport(dispatch)) as http:
            sdk = OpenAI(
                api_key="synthetic",
                base_url="https://foundation-models.api.cloud.ru/v1",
                http_client=http,
            )
            with self.ledger.capture(patch_provider=False):
                result = sdk.embeddings.create(
                    model=MODEL, input=["synthetic"], encoding_format="float"
                )
        self.assertEqual(calls, [cloud_body()])
        self.assertEqual(len(result.data[0].embedding), 2048)
        self.assertEqual(self.ledger.accounting()["attempts_started"], 1)

    def test_tokenizer_binding_has_no_remote_code(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, processors

        tokenizer = Tokenizer(
            models.WordLevel(
                {"[UNK]": 0, "[BOS]": 1, "hello": 2, "world": 3}, unk_token="[UNK]"
            )
        )
        cast(Any, tokenizer).pre_tokenizer = pre_tokenizers.Whitespace()
        cast(Any, tokenizer).post_processor = processors.TemplateProcessing(
            single="[BOS] $A", pair="[BOS] $A $B", special_tokens=[("[BOS]", 1)]
        )
        tokenizer.enable_truncation(max_length=2)
        raw = tokenizer.to_str().encode()
        path = self.root / "cloudru-tokenizer.json"
        path.write_bytes(raw)
        settings = {**self.settings, "tokenizer_sha256": hashlib.sha256(raw).hexdigest()}
        counter = CloudCounter(path, settings)
        self.assertEqual(counter(cloud_body(input=["hello world", "hello"])), 5)
        with self.assertRaises(IndexingLimit):
            CloudCounter(path, self.settings)

    def test_missing_price_or_binding_cannot_initialize(self):
        for changes in ({"rate_hundredths": 0}, {"tokenizer_sha256": ""}):
            with self.assertRaises(IndexingLimit):
                BudgetStore.create(
                    self.root / "invalid.sqlite3", cloudru={**self.settings, **changes}
                )

    def test_query_and_other_model_are_refused(self):
        with self.assertRaises(IndexingLimit):
            self.begin(model="other")
        with self.assertRaises(IndexingLimit):
            self.ledger.begin(
                httpx.Request(
                    "POST",
                    "https://foundation-models.api.cloud.ru/v1/embeddings?x=1",
                    json=cloud_body(),
                )
            )


if __name__ == "__main__":
    unittest.main()
