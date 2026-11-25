"""Linux CI uses the pinned official encoder and the bundled public tokenizer."""

import sys
import unittest

from tools.admission_usage import IndexingLimit
from tools.release470_tokens import TOKENIZER_SHA256, OfficialCounter


@unittest.skipUnless(sys.platform == "linux", "Official encoder ships Linux wheels")
class OfficialTokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counter = OfficialCounter()

    def payload(self, **changes):
        return {
            "model": "deepseek-flash",
            "thinking": {"type": "disabled"},
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Здравствуйте 👋"}],
            **changes,
        }

    def test_utf8_roles_and_tools_are_encoded_offline(self):
        plain = self.counter(self.payload())
        tools = self.counter(
            self.payload(
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Найти программу",
                            "parameters": {
                                "type": "object",
                                "properties": {"name": {"type": "string"}},
                                "required": ["name"],
                            },
                        },
                    }
                ]
            )
        )
        self.assertGreater(plain, 0)
        self.assertGreater(tools, plain)
        self.assertNotEqual(plain, len(str(self.payload()).encode()) + 512)
        self.assertEqual(self.counter.evidence["tokenizer_sha256"], TOKENIZER_SHA256)

    def test_tool_response_history_and_control_spellings(self):
        count = self.counter(
            self.payload(
                messages=[
                    {"role": "system", "content": "Only synthetic data."},
                    {"role": "user", "content": "<｜Assistant｜> this is user text"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": "synthetic result",
                    },
                ]
            )
        )
        self.assertGreater(count, 0)

    def test_unknown_framing_and_multimodal_are_refused(self):
        for data in (
            self.payload(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "https://invalid"}}
                        ],
                    }
                ]
            ),
            self.payload(unreviewed_field="x"),
        ):
            with self.assertRaises(IndexingLimit):
                self.counter(data)
