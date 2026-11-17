"""Additional pilot limits over the existing physical-attempt/SSE meter.

This module does not start a pilot or authorize spending. Transport integration,
account currency and conservative input bounds still require preflight review.
"""

import json
import threading
from pathlib import Path

from tools.admission_pilot import PilotLedger
from tools.admission_usage import IndexingLimit

CASES = {"setup", "faq-1", "faq-2", "rag-1", "rag-2", "tools-1", "tools-2"}


class Release470Ledger(PilotLedger):
    """Reserve limits before each physical attempt, including retries and setup."""

    def __init__(self, directory: Path) -> None:
        super().__init__(directory, max_attempts=512)
        self.deepseek_ceiling_micro_cny = 3_000_000
        self.budget_lock = threading.RLock()
        self.totals = {"input": 0, "output": 0, "embedding": 0}
        self.cases: dict[str, dict[str, int]] = {}

    def context(self, phase: str, case_id: str) -> None:
        if case_id not in CASES or phase not in {"setup", "pilot"}:
            raise IndexingLimit("Unknown release pilot context")
        super().context(phase, case_id)

    def begin(self, request) -> dict:
        with self.budget_lock:
            if self.case_id not in CASES:
                raise IndexingLimit("Missing release pilot context")
            payload = json.loads(request.content) if request.content else {}
            generation = request.url.host == "api.deepseek.com"
            embedding = request.url.path == "/v1/embeddings"
            reserve = {
                "input": len(request.content) + 512 if generation else 0,
                "output": payload.get("max_tokens", 1024) if generation else 0,
                "embedding": 512 if embedding else 0,
            }
            if generation:
                if (
                    type(payload.get("max_tokens")) is not int
                    or not 1 <= payload["max_tokens"] <= 1024
                    or payload.get("thinking") != {"type": "disabled"}
                ):
                    raise IndexingLimit("Explicit bounded non-thinking output required")
                # An input-byte check is an extra planning guard, not a certified
                # tokenizer bound. No live run until its adequacy is reviewed.
                if len(request.content) > 24000:
                    raise IndexingLimit("Release pilot input planning bound exceeded")
                if any(t.get("type") != "function" for t in payload.get("tools", [])):
                    raise IndexingLimit("Hosted provider tools are outside the budget")
            if embedding and len(request.content) > 256:
                raise IndexingLimit("Embedding planning bound exceeded")
            per = self.cases.setdefault(
                self.case_id,
                {
                    "input": 0,
                    "output": 0,
                    "embedding": 0,
                    "generations": 0,
                },
            )
            caps = {"input": 300000, "output": 10000, "embedding": 10000}
            per_caps = (
                caps
                if self.case_id == "setup"
                else {
                    "input": 32000,
                    "output": 1024,
                    "embedding": 512,
                }
            )
            if generation and per["generations"] >= 40:
                raise IndexingLimit("Question generation attempt limit reached")
            for kind, value in reserve.items():
                if (
                    self.totals[kind] + value > caps[kind]
                    or per[kind] + value > per_caps[kind]
                ):
                    raise IndexingLimit("Reserved release token limit reached")
            row = super().begin(request)
            row["release_reservation"] = reserve
            for kind, value in reserve.items():
                self.totals[kind] += value
                per[kind] += value
            per["generations"] += int(generation)
            return row

    def finish(self, row, response, error) -> None:
        with self.budget_lock:
            super().finish(row, response, error)
            reserve = row["release_reservation"]
            if not row["usage_complete"] or error is not None:
                self.halted = True
                return
            usage = row["usage"]
            actual = {
                "input": usage.get("prompt_tokens", 0)
                if row["provider"] == "deepseek"
                else 0,
                "output": usage.get("completion_tokens", 0),
                "embedding": usage.get("prompt_tokens", 0)
                if row["operation"] == "embedding"
                else 0,
            }
            if row["provider"] == "deepseek" and row.get("response_model") not in {
                "deepseek-flash",
                "DeepSeek-V4.1-Flash-1017",
            }:
                self.halted = True
            for kind, value in actual.items():
                if value > reserve[kind]:
                    self.halted = True
                self.totals[kind] += value - reserve[kind]
                self.cases[row["case_id"]][kind] += value - reserve[kind]

    def accounting(self) -> dict:
        return {
            **super().accounting(),
            "token_totals": dict(self.totals),
            "cases": self.cases,
            "live_integration_verified": False,
        }
