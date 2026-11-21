"""The executor must claim once and stop without exposing raw exceptions."""

import contextlib
import hashlib
import io
import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch

from tools import release470_pilot as pilot
from tools.release470_budget import Release470Ledger
from tools.release470_faq_api import api_call
from tools.release470_runtime import instrument_bot_core, instrument_question


class PilotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.old_logging = logging.root.manager.disable
        self.addCleanup(logging.disable, self.old_logging)

    def invoke(self, fake):
        with (
            patch.object(pilot, "REPORTS", self.root),
            patch.object(pilot, "BUDGET", self.root / "not-created.sqlite3"),
            patch.object(pilot, "run_case", fake),
            patch.dict(os.environ, {"PYTHON_DOTENV_DISABLED": "1"}),
            patch("sys.argv", ["pilot", "--case", "rag-1"]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            pilot.main()

    def test_completed_case_cannot_be_sent_twice(self):
        async def complete(case, result):
            result["completed"] = True

        fake = AsyncMock(side_effect=complete)
        self.invoke(fake)
        with self.assertRaises(FileExistsError):
            self.invoke(fake)
        self.assertEqual(fake.await_count, 1)

    def test_error_after_delivery_still_fails_and_remains_claimed(self):
        async def failure(case, result):
            result["completed"] = True
            raise RuntimeError("private-header-value")

        with self.assertRaises(SystemExit):
            self.invoke(AsyncMock(side_effect=failure))
        record = self.root / "pilot-rag-1-20261121.json"
        text = record.read_text()
        self.assertNotIn("private-header-value", text)
        self.assertEqual(json.loads(text)["error_type"], "RuntimeError")


class LocalAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_verified_botcore_prefix_is_bound_and_restored(self):
        question = "Approved synthetic question"
        calls = []

        async def pipeline(message):
            calls.append(message)

        ledger = cast(
            Release470Ledger,
            SimpleNamespace(
                store=SimpleNamespace(
                    policy={
                        "case_plan": {
                            "rag-1": hashlib.sha256(question.encode()).hexdigest()
                        }
                    }
                ),
                question=lambda case: contextlib.nullcontext(),
            ),
        )
        guarded_pipeline = instrument_question(pipeline, ledger)

        async def core(channel, external_user_id, session_id, user_name, user_text):
            await guarded_pipeline(f"[from {user_name}] {user_text}")

        guarded_core = instrument_bot_core(core, ledger, recipient="123")
        await guarded_core("telegram", "123", "123", "EVAL470", question)
        self.assertEqual(calls, ["[from EVAL470] " + question])
        # A previous authorized call cannot leave a binding behind.
        with self.assertRaisesRegex(RuntimeError, "outside the immutable"):
            await guarded_pipeline("[from EVAL470] " + question)
        for channel, uid, session, name, text in (
            ("max", "123", "123", "EVAL470", question),
            ("telegram", "456", "123", "EVAL470", question),
            ("telegram", "123", "other", "EVAL470", question),
            ("telegram", "123", "123", "forged]\n", question),
            ("telegram", "123", "123", "EVAL470", "unapproved question"),
        ):
            with self.assertRaises(RuntimeError):
                await guarded_core(channel, uid, session, name, text)
        self.assertEqual(len(calls), 1)

    async def test_botcore_cannot_change_the_formatted_question(self):
        question = "Approved synthetic question"
        ledger = cast(
            Release470Ledger,
            SimpleNamespace(
                store=SimpleNamespace(
                    policy={
                        "case_plan": {
                            "rag-1": hashlib.sha256(question.encode()).hexdigest()
                        }
                    }
                ),
                question=lambda case: contextlib.nullcontext(),
            ),
        )
        calls = []

        async def pipeline(message):
            calls.append(message)

        guarded_pipeline = instrument_question(pipeline, ledger)

        async def forged_core(
            channel, external_user_id, session_id, user_name, user_text
        ):
            await guarded_pipeline("[from forged] " + user_text)

        guarded_core = instrument_bot_core(forged_core, ledger, recipient="123")
        with self.assertRaisesRegex(RuntimeError, "changed the verified"):
            await guarded_core("telegram", "123", "123", "EVAL470", question)
        self.assertEqual(calls, [])

    async def test_botcore_prefix_is_rejected_by_the_current_question_binding(self):
        question = "Approved synthetic question"
        calls = []

        async def pipeline(message):
            calls.append(message)

        ledger = SimpleNamespace(
            store=SimpleNamespace(
                policy={
                    "case_plan": {"rag-1": hashlib.sha256(question.encode()).hexdigest()}
                }
            ),
            question=lambda case: contextlib.nullcontext(),
        )
        wrapped = instrument_question(pipeline, cast(Release470Ledger, ledger))
        await wrapped(question)
        with self.assertRaisesRegex(RuntimeError, "outside the immutable"):
            await wrapped("[from EVAL470] " + question)
        self.assertEqual(calls, [question])

    async def test_json_request_uses_asgi_without_an_http_transport(self):
        async def application(scope, receive, send):
            self.assertEqual(scope["path"], "/api/v1/faq")
            self.assertIn((b"authorization", b"Bearer synthetic"), scope["headers"])
            request = await receive()
            self.assertEqual(json.loads(request["body"]), {"question": "synthetic"})
            await send({"type": "http.response.start", "status": 201})
            await send({"type": "http.response.body", "body": b'{"ok":true}'})

        status, body = await api_call(
            application, "POST", "/api/v1/faq", {"question": "synthetic"}, "synthetic"
        )
        self.assertEqual((status, body), (201, {"ok": True}))

    async def test_empty_cleanup_response_is_supported(self):
        async def application(scope, receive, send):
            await send({"type": "http.response.start", "status": 204})
            await send({"type": "http.response.body", "body": b""})

        self.assertEqual(
            await api_call(application, "DELETE", "/api/v1/faq/test"), (204, {})
        )


if __name__ == "__main__":
    unittest.main()
