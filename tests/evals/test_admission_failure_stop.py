"""Technical errors must stop paid work, even with completely accounted HTTP."""

import asyncio
import json
from contextlib import nullcontext
from pathlib import Path

import httpx
import pytest
import yaml

from evals.checks import Observation
from evals.judge import parse_judgement
from evals.schema import load_scenarios
from tools import admission_pilot
from tools.admission_full_control import validate_result
from tools.admission_pilot import PilotLedger, Progress, evaluate_cases

DATA = Path(__file__).resolve().parents[2] / "docs/reports/release-388-full-data"


@pytest.mark.parametrize(
    "failure", ["judge_parse", "answer_timeout", "judge_timeout", "content"]
)
async def test_worker_saves_technical_error_and_never_starts_next_case(
    tmp_path, monkeypatch, failure
):
    path = tmp_path / "cases.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "scenarios": [
                    {
                        "id": name,
                        "question": name,
                        "reference_answer": "reference",
                        "topic": "test",
                        "source": "synthetic",
                        "approved_by": None,
                        "approved_at": None,
                        "requires_llm": True,
                        "judge_criteria": "Check reference",
                    }
                    for name in ("first", "second")
                ],
            }
        ),
        encoding="utf-8",
    )
    scenarios = {s.id: s for s in load_scenarios(path)}
    progress = Progress(tmp_path / "report", "with-kb", {}, case_ids=tuple(scenarios))
    monkeypatch.setattr(progress, "sample", lambda phase=None: {})
    monkeypatch.setattr(admission_pilot, "CASE_TIMEOUT", 0.02)
    ledger = PilotLedger(progress.path.parent)
    reached = []

    async def runner(case, session_id):
        reached.append(case.id)
        if failure == "answer_timeout":
            await asyncio.Event().wait()
        return Observation(text="answer")

    async def judge(**kwargs):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 2,
                        "total_tokens": 7,
                    }
                },
            )
        )
        with httpx.Client(transport=transport) as client:
            client.post(
                "https://api.deepseek.com/chat/completions",
                json={"model": "deepseek-flash", "max_tokens": 100},
            )
        if failure == "judge_timeout":
            await asyncio.Event().wait()
        if failure == "content":
            return {"score": 1, "reasoning": "Wrong answer"}
        # A successful HTTP response can still contain invalid judge JSON.
        return parse_judgement('{"score": "invalid", "reasoning": "test"}')

    with ledger.capture():
        with pytest.raises(RuntimeError) if failure != "content" else nullcontext():
            await evaluate_cases(
                progress, ledger, path, scenarios, tuple(scenarios), runner, judge
            )
    assert reached == (["first", "second"] if failure == "content" else ["first"])
    saved = json.loads(progress.path.read_text(encoding="utf-8"))
    assert len(saved["results"]) == len(reached)
    row = saved["results"][0]
    if failure == "content":
        assert all(not r.get("error") for r in saved["results"])
    else:
        assert row["error"] == (
            "ValueError" if failure == "judge_parse" else "TimeoutError"
        )
    assert row["passed"] is False
    assert saved["status"] != "completed"
    assert (
        ledger.sequence
        == ledger.completed
        == (0 if failure == "answer_timeout" else len(reached))
    )
    assert not ledger.uncertain_failure and not ledger.halted


def original_batch():
    report = json.loads(
        (DATA / "with-kb-full-20261110-02-batch-00-report.json").read_text()
    )
    usage = [
        json.loads(line)
        for line in (DATA / "with-kb-full-20261110-02-batch-00-usage.jsonl")
        .read_text()
        .splitlines()
    ]
    intents = [
        json.loads(line)
        for line in (DATA / "with-kb-full-20261110-02-batch-00-intents.jsonl")
        .read_text()
        .splitlines()
    ]
    return report, usage, intents, tuple(report["selected_ids"])


@pytest.mark.parametrize("error", ["ValueError", "TimeoutError"])
def test_controller_rejects_completed_batch_with_case_error(error):
    report, usage, intents, ids = original_batch()
    report["results"][0].update(error=error, passed=False)
    with pytest.raises(RuntimeError):
        validate_result(report, usage, intents, ids)


def test_controller_accepts_content_failure_and_original_evidence():
    report, usage, intents, ids = original_batch()
    validate_result(report, usage, intents, ids)
    report["results"][0].update(passed=False)
    validate_result(report, usage, intents, ids)
