"""The retry retains the old owner limits and cannot become an SDK retry loop."""

import unittest

import httpx

from tools.release470_cloudru import HOST, MODEL
from tools.release470_cloudru_recheck import check_owner_caps, one_cloud_request


def state():
    return {
        "halted": False,
        "pending_or_uncertain": [],
        "totals": {"micro_rub": 12959, "micro_cny": 82058, "embedding": 402},
    }


class RetryTests(unittest.TestCase):
    def test_old_caps_are_enforced(self):
        self.assertEqual(check_owner_caps(state(), 94, 3222), 3029)
        changed = state()
        changed["totals"]["micro_rub"] = 999999
        with self.assertRaises(RuntimeError):
            check_owner_caps(changed, 94, 3222)
        changed = state()
        changed["totals"]["micro_cny"] = 3000001
        with self.assertRaises(RuntimeError):
            check_owner_caps(changed, 94, 3222)

    def test_only_one_physical_cloud_attempt(self):
        payload = {"model": MODEL, "input": ["synthetic"], "encoding_format": "float"}
        calls = []

        def dispatch(request):
            calls.append(request)
            return httpx.Response(
                200,
                json={
                    "model": MODEL,
                    "usage": {
                        "prompt_tokens": 94,
                        "total_tokens": 94,
                    },
                },
            )

        result = {}
        with httpx.Client(transport=httpx.MockTransport(dispatch)) as client:
            with one_cloud_request(payload, state, 94, 3222, result):
                with self.assertRaises(RuntimeError):
                    client.post("https://api.deepseek.com/chat/completions", json=payload)
                client.post(f"https://{HOST}/v1/embeddings", json=payload)
                with self.assertRaises(RuntimeError):
                    client.post(f"https://{HOST}/v1/embeddings", json=payload)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["physical_attempts"], 1)


if __name__ == "__main__":
    unittest.main()
