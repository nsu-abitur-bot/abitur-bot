"""Offline accounting regressions: temporary SQLite, no app DB or network."""

import asyncio
import gzip
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import httpx
from openai import AsyncOpenAI

from tools.admission_usage import IndexingLimit
from tools.release470_budget import Release470Ledger
from tools.release470_gate import TransportGate
from tools.release470_runtime import instrument_question
from tools.release470_store import BudgetStore


class Counter:
    evidence = {"kind": "offline-synthetic"}

    def __call__(self, payload):
        return 100


def request(**changes):
    return httpx.Request(
        "POST",
        "https://api.deepseek.com/chat/completions",
        json={
            "model": "deepseek-flash",
            "max_tokens": 1024,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "synthetic"}],
            **changes,
        },
    )


def response(prompt=100, output=10, **changes):
    return httpx.Response(
        200,
        json={
            "model": "deepseek-flash",
            "id": "synthetic",
            "object": "chat.completion",
            "created": 0,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "synthetic answer"},
                }
            ],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": output,
                "total_tokens": prompt + output,
            },
            **changes,
        },
    )


class BudgetFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "fixed-budget.sqlite3"
        BudgetStore.create(self.path)
        self.ledger = self.attach()

    def attach(self, report_name="reports"):
        ledger = Release470Ledger(
            self.root / report_name, budget_path=self.path, counter=Counter()
        )
        ledger.context("pilot", "rag-1")
        return ledger


class ReleaseBudgetTests(BudgetFixture):
    def test_completed_spending_survives_restart_and_new_report_directory(self):
        row = self.ledger.begin(request())
        self.ledger.finish(row, response(), None)
        self.assertEqual(
            self.attach("different-directory").accounting()["token_totals"],
            {"input": 100, "output": 10, "embedding": 0},
        )

    def test_no_implicit_creation_or_reset(self):
        with self.assertRaises(FileExistsError):
            BudgetStore.create(self.path)
        with self.assertRaises(IndexingLimit):
            Release470Ledger(self.root, budget_path=self.root / "new.sqlite3")

    def test_crashed_pending_attempt_is_never_refunded(self):
        code = """
import sys,os
from pathlib import Path
from tools.release470_store import BudgetStore
from tools.release470_gate import TransportGate
p=Path(sys.argv[1]); gate=TransportGate(p); gate.acquire()
BudgetStore(p).reserve("rag-1","generation",
    {"input":100,"output":1024,"embedding":0,"micro_cny":8392},
    "synthetic-fingerprint",{"kind":"offline-synthetic"})
os._exit(0)
"""
        env = {
            k: os.environ[k]
            for k in ("SYSTEMROOT", "PATH", "TEMP", "TMP")
            if k in os.environ
        }
        env.update(
            PYTHONPATH=str(Path(__file__).resolve().parents[2]),
            PYTHON_DOTENV_DISABLED="1",
        )
        subprocess.run(
            [sys.executable, "-c", code, str(self.path)], env=env, check=True, timeout=30
        )
        gate = TransportGate(self.path)
        self.assertTrue(gate.attempt())
        gate.close()
        restarted = self.attach("restart")
        self.assertEqual(restarted.accounting()["token_totals"]["output"], 1024)
        with self.assertRaises(IndexingLimit):
            restarted.begin(request(max_tokens=1))

    def test_prompt_and_raw_provider_error_never_enter_journal(self):
        marker = "DO-NOT-PERSIST-PRIVATE-TEXT"
        row = self.ledger.begin(request(messages=[{"role": "user", "content": marker}]))
        with self.assertRaises(IndexingLimit):
            self.ledger.finish(
                row, httpx.Response(500, json={"error": {"message": marker}}), None
            )
        self.assertNotIn(marker, json.dumps(self.ledger.store.journal()))

    def test_real_provider_background_uses_immutable_question_plan(self):
        from unittest.mock import patch

        from langchain_core.messages import HumanMessage

        from llm.providers.deepseek import DeepSeekProvider

        text = "synthetic planned question"
        path = self.root / "planned.sqlite3"
        BudgetStore.create(
            path, case_plan={"rag-1": hashlib.sha256(text.encode()).hexdigest()}
        )
        ledger = Release470Ledger(self.root, budget_path=path, counter=Counter())
        ledger.context("setup", "setup")

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(lambda r: response())
            ) as http:
                provider = object.__new__(DeepSeekProvider)
                provider.model_name = "deepseek-flash"
                provider.client = AsyncOpenAI(
                    api_key="synthetic",
                    base_url="https://api.deepseek.com",
                    http_client=http,
                )

                async def question(message, session_id):
                    return asyncio.create_task(
                        provider.generate_with_usage([HumanMessage(content=message)])
                    )

                wrapped = instrument_question(question, ledger)
                with (
                    patch(
                        "llm.providers.deepseek.get_setting",
                        side_effect=lambda key, default=None: default,
                    ),
                    ledger.capture(),
                ):
                    task = await wrapped(text, "synthetic-session")
                    result = await task
                    self.assertEqual(result.text, "synthetic answer")
                    with self.assertRaises(IndexingLimit):
                        await wrapped("unplanned question", "synthetic-session")

        asyncio.run(run())
        self.assertEqual(ledger.accounting()["cases"]["rag-1"]["output"], 10)

    def test_final_journal_contains_final_totals_and_unknown_model_halt(self):
        row = self.ledger.begin(request())
        with self.assertRaises(IndexingLimit):
            self.ledger.finish(row, response(model="unknown"), None)
        state = self.ledger.accounting()
        final = self.ledger.store.journal()[-1]
        self.assertTrue(final["uncertain"])
        self.assertTrue(final["snapshot"]["halted"])
        self.assertEqual(final["snapshot"]["totals"], state["totals"])
        self.assertEqual(final["snapshot"]["pending_or_uncertain"], [1])

    def test_idempotent_final_does_not_charge_twice(self):
        row = self.ledger.begin(request())
        self.ledger.finish(row, response(), None)
        self.ledger.finish(row, response(), None)
        self.assertEqual(len(self.ledger.store.journal()), 3)
        self.assertEqual(self.ledger.accounting()["token_totals"]["output"], 10)

    def test_conflicting_final_is_rejected(self):
        row = self.ledger.begin(request())
        self.ledger.finish(row, response(), None)
        with self.assertRaises(IndexingLimit):
            self.ledger.finish(row, response(output=11), None)

    def test_missing_usage_holds_reserve_and_blocks_retry(self):
        row = self.ledger.begin(request())
        with self.assertRaises(IndexingLimit):
            self.ledger.finish(row, response(usage={}), None)
        self.assertEqual(self.ledger.accounting()["token_totals"]["output"], 1024)
        with self.assertRaises(IndexingLimit):
            self.attach().begin(request())

    def test_usage_above_input_bound_is_recorded_and_halts(self):
        row = self.ledger.begin(request())
        with self.assertRaises(IndexingLimit):
            self.ledger.finish(row, response(prompt=101), None)
        self.assertTrue(self.ledger.accounting()["halted"])
        self.assertEqual(self.ledger.accounting()["token_totals"]["input"], 101)

    def test_journal_and_state_corruption_are_detected(self):
        row = self.ledger.begin(request())
        self.ledger.finish(row, response(), None)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE attempts SET held=?", ("{}",))
        with self.assertRaises(Exception):
            self.attach()

    def test_unapproved_providers_oauth_and_models_refuse_before_transport(self):
        calls = []
        requests = [
            request(model="deepseek-chat"),
            httpx.Request(
                "POST",
                "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
                data={"scope": "GIGACHAT_API_PERS"},
            ),
            httpx.Request(
                "POST", "https://foundation-models.api.cloud.ru/v1/embeddings", json={}
            ),
        ]
        with (
            self.ledger.capture(patch_provider=False),
            httpx.Client(
                transport=httpx.MockTransport(lambda r: (calls.append(r), response())[1])
            ) as client,
        ):
            for item in requests:
                with self.assertRaises(IndexingLimit):
                    client.send(item)
        self.assertEqual(calls, [])
        self.assertEqual(self.ledger.accounting()["attempts_started"], 0)

    def test_output_clamped_on_actual_wire_payload(self):
        sent = []

        def dispatch(req):
            sent.append(json.loads(req.content)["max_tokens"])
            return response()

        with (
            self.ledger.capture(patch_provider=False),
            httpx.Client(transport=httpx.MockTransport(dispatch)) as client,
        ):
            client.send(request())
            client.send(request())
        self.assertEqual(sent, [1024, 1014])

    def test_unlabelled_background_fails_closed(self):
        ledger = Release470Ledger(self.root, budget_path=self.path, counter=Counter())
        with self.assertRaises(IndexingLimit):
            ledger.begin(request())

    def test_question_input_cap_refuses_before_reservation(self):
        class LargeCounter(Counter):
            def __call__(self, payload):
                return 32001

        self.ledger.counter = LargeCounter()
        with self.assertRaises(IndexingLimit):
            self.ledger.begin(request())
        self.assertEqual(self.ledger.accounting()["attempts_started"], 0)

    def test_child_background_task_keeps_its_question_context(self):
        async def run():
            async def child():
                await asyncio.sleep(0)
                row = self.ledger.begin(request())
                self.assertEqual(row["case_id"], "faq-1")
                self.ledger.finish(row, response(), None)

            with self.ledger.question("faq-1"):
                task = asyncio.create_task(child())
            self.ledger.context("pilot", "rag-2")
            await task

        asyncio.run(run())

    def test_other_billing_transports_are_refused(self):
        import urllib.request
        from unittest.mock import AsyncMock, patch

        import aiohttp
        import requests

        with (
            patch(
                "requests.adapters.HTTPAdapter.send",
                side_effect=AssertionError("network"),
            ),
            patch(
                "aiohttp.TCPConnector.connect",
                new=AsyncMock(side_effect=AssertionError("network")),
            ),
            self.ledger.capture(patch_provider=False),
        ):
            with self.assertRaises(IndexingLimit):
                requests.post("https://api.deepseek.com/chat/completions", json={})

            async def run():
                async with aiohttp.ClientSession() as client:
                    with self.assertRaises(IndexingLimit):
                        await client.post(
                            "https://api.deepseek.com/chat/completions", json={}
                        )

            asyncio.run(run())
            with self.assertRaises(IndexingLimit):
                urllib.request.urlopen("https://api.deepseek.com/chat/completions")

    def test_native_sdk_retry_is_blocked_before_second_physical_call(self):
        calls = []

        async def run():
            def dispatch(req):
                calls.append(req)
                return httpx.Response(503, json={})

            async with httpx.AsyncClient(transport=httpx.MockTransport(dispatch)) as http:
                sdk = AsyncOpenAI(
                    api_key="synthetic",
                    base_url="https://api.deepseek.com",
                    http_client=http,
                    max_retries=2,
                )
                with self.ledger.capture(patch_provider=False):
                    with self.assertRaises(Exception):
                        await sdk.chat.completions.create(
                            model="deepseek-flash",
                            messages=[{"role": "user", "content": "synthetic"}],
                            max_tokens=1024,
                            extra_body={"thinking": {"type": "disabled"}},
                        )

        asyncio.run(run())
        self.assertEqual(len(calls), 1)
        self.assertTrue(self.ledger.accounting()["halted"])

    def test_two_processes_share_output_cap(self):
        script = """
import os,sys,json
from pathlib import Path
import httpx
from tools.release470_budget import Release470Ledger
class Counter:
    evidence={"kind":"offline-synthetic"}
    def __call__(self,payload): return 100
ledger=Release470Ledger(Path(sys.argv[1]).parent/sys.argv[2],budget_path=Path(sys.argv[1]),counter=Counter())
ledger.context("pilot","rag-1")
def dispatch(req):
    n=min(600,json.loads(req.content)["max_tokens"])
    return httpx.Response(200,json={"model":"deepseek-flash","usage":{
        "prompt_tokens":100,"completion_tokens":n,"total_tokens":100+n}})
with ledger.capture(patch_provider=False),httpx.Client(
    transport=httpx.MockTransport(dispatch)) as client:
    client.post("https://api.deepseek.com/chat/completions",json={"model":"deepseek-flash","messages":[{"role":"user","content":"synthetic"}],"thinking":{"type":"disabled"},"max_tokens":1024})
"""
        env = {
            k: os.environ[k]
            for k in ("SYSTEMROOT", "PATH", "TEMP", "TMP")
            if k in os.environ
        }
        env.update(
            PYTHONPATH=str(Path(__file__).resolve().parents[2]),
            PYTHON_DOTENV_DISABLED="1",
        )
        children = [
            subprocess.Popen(
                [sys.executable, "-c", script, str(self.path), name],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for name in ("api-reports", "bot-reports")
        ]
        for child in children:
            out, err = child.communicate(timeout=30)
            self.assertEqual(child.returncode, 0, err.decode())
        state = self.attach().accounting()
        self.assertEqual(state["token_totals"]["output"], 1024)
        self.assertEqual(state["attempts_started"], 2)
        self.assertEqual(
            [
                e["held"]["output"]
                for e in self.ledger.store.journal()
                if e["event"] == "reserve"
            ],
            [1024, 424],
        )


class SSETests(BudgetFixture):
    def test_fragmented_gzip_and_final_usage(self):
        data = (
            b'data: {"model":"deepseek-flash","choices":[],"usage":'
            b'{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}}\n\n'
            b"data: [DONE]\n\n"
        )

        class Fragments(httpx.AsyncByteStream):
            async def __aiter__(self):
                compressed = gzip.compress(data)
                for offset in range(0, len(compressed), 3):
                    yield compressed[offset : offset + 3]

        async def run():
            def dispatch(req):
                return httpx.Response(
                    200,
                    headers={
                        "content-type": "text/event-stream",
                        "content-encoding": "gzip",
                    },
                    stream=Fragments(),
                )

            async with httpx.AsyncClient(
                transport=httpx.MockTransport(dispatch)
            ) as client:
                with self.ledger.capture(patch_provider=False):
                    async with client.stream(
                        "POST",
                        "https://api.deepseek.com/chat/completions",
                        json=json.loads(
                            request(
                                stream=True, stream_options={"include_usage": True}
                            ).content
                        ),
                    ) as reply:
                        self.assertEqual(await reply.aread(), data)

        asyncio.run(run())
        self.assertEqual(self.ledger.accounting()["token_totals"]["output"], 10)

    def test_usage_without_done_does_not_establish_success(self):
        data = (
            b'data: {"model":"deepseek-flash","usage":'
            b'{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}}\n\n'
        )

        async def run():
            def dispatch(req):
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=httpx.ByteStream(data),
                )

            async with httpx.AsyncClient(
                transport=httpx.MockTransport(dispatch)
            ) as client:
                with self.ledger.capture(patch_provider=False):
                    async with client.stream(
                        "POST",
                        "https://api.deepseek.com/chat/completions",
                        json=json.loads(
                            request(
                                stream=True, stream_options={"include_usage": True}
                            ).content
                        ),
                    ) as reply:
                        await reply.aread()

        with self.assertRaises(IndexingLimit):
            asyncio.run(run())
        self.assertTrue(self.ledger.accounting()["halted"])
        self.assertFalse(
            self.ledger.store.journal()[-1]["receipt"]["stream_evidence"]["done_seen"]
        )

    def test_sdk_stream_preserves_bytes_and_final_usage(self):
        async def run():
            frames = [
                {
                    "id": "synthetic",
                    "object": "chat.completion.chunk",
                    "model": "deepseek-flash",
                    "choices": [
                        {"index": 0, "delta": {"content": "hello"}, "finish_reason": None}
                    ],
                },
                {
                    "id": "synthetic",
                    "object": "chat.completion.chunk",
                    "model": "deepseek-flash",
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "total_tokens": 110,
                    },
                },
            ]
            data = (
                b"".join(b"data: " + json.dumps(f).encode() + b"\n\n" for f in frames)
                + b"data: [DONE]\n\n"
            )

            def dispatch(req):
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=httpx.ByteStream(data),
                )

            async with httpx.AsyncClient(transport=httpx.MockTransport(dispatch)) as http:
                sdk = AsyncOpenAI(
                    api_key="synthetic",
                    base_url="https://api.deepseek.com",
                    http_client=http,
                )
                with self.ledger.capture(patch_provider=False):
                    stream = await sdk.chat.completions.create(
                        model="deepseek-flash",
                        messages=[{"role": "user", "content": "synthetic"}],
                        max_tokens=1024,
                        stream=True,
                        stream_options={"include_usage": True},
                        extra_body={"thinking": {"type": "disabled"}},
                    )
                    text = ""
                    async for frame in stream:
                        if frame.choices:
                            text += frame.choices[0].delta.content or ""
                    self.assertEqual(text, "hello")

        asyncio.run(run())
        self.assertEqual(self.attach().accounting()["token_totals"]["output"], 10)

    def test_cancelled_stream_keeps_reservation(self):
        async def run():
            def dispatch(req):
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=httpx.ByteStream(b'data: {"choices":[]}\n\n'),
                )

            async with httpx.AsyncClient(transport=httpx.MockTransport(dispatch)) as http:
                with self.ledger.capture(patch_provider=False):
                    async with http.stream(
                        "POST",
                        "https://api.deepseek.com/chat/completions",
                        json=json.loads(
                            request(
                                stream=True, stream_options={"include_usage": True}
                            ).content
                        ),
                    ):
                        pass

        with self.assertRaises(IndexingLimit):
            asyncio.run(run())
        state = self.attach().accounting()
        self.assertTrue(state["halted"])
        self.assertEqual(state["token_totals"]["output"], 1024)


if __name__ == "__main__":
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    unittest.main()
