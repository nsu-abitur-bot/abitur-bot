import asyncio
import json
from pathlib import Path

import pytest

from evals.checks import Observation, check_assertion
from evals.evaluator import PipelineEvaluator, compare_baseline
from evals.schema import Assertion, load_scenarios


@pytest.mark.parametrize(
    "kind,value,text,expected",
    [
        ("contains", "НГУ", "Ответ нгу", True),
        ("not_contains", "300", "246", True),
        ("regex", r"\d{3}", "Балл 246", True),
        ("refusal", True, "Не могу написать код", True),
        ("refusal", True, "Вот код", False),
        ("faq_match", True, "FAQ", False),
    ],
)
def test_assertion_types(kind, value, text, expected):
    assert (
        check_assertion(Assertion(kind=kind, value=value), Observation(text)) == expected
    )


@pytest.mark.parametrize(
    "kind,value",
    [
        ("regex", "["),
        ("tool_called", True),
        ("faq_match", "true"),
        ("contains", ""),
    ],
)
def test_invalid_assertions(kind, value):
    with pytest.raises(ValueError):
        Assertion(kind=kind, value=value)


def test_tool_assertion_uses_trace_not_answer_text():
    assertion = Assertion(kind="tool_called", value="get_admission_scores")
    assert not check_assertion(assertion, Observation(text="get_admission_scores"))
    assert check_assertion(
        assertion, Observation(text="246", tools_called=["get_admission_scores"])
    )


def test_source_contains_assertion():
    assertion = Assertion(kind="source_contains", value="https://mmf.nsu.ru")
    assert check_assertion(
        assertion, Observation(text="ok", sources=["https://mmf.nsu.ru/info"])
    )
    assert not check_assertion(
        assertion, Observation(text="ok", sources=["https://fit.nsu.ru/info"])
    )


def test_invalid_and_duplicate_scenarios_fail_closed(tmp_path):
    path = tmp_path / "cases.yaml"
    path.write_text("schema_version: 1\nscenarios: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_scenarios(path)


async def test_static_runner_is_bounded_and_sessions_are_unique(tmp_path):
    active = peak = 0
    sessions = []

    async def runner(scenario, session_id):
        nonlocal active, peak
        sessions.append(session_id)
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return Observation(text="ok")

    path = tmp_path / "cases.yaml"
    path.write_text(
        "schema_version: 1\nscenarios:\n"
        + "".join(
            f"  - id: case_{i}\n    question: Q\n    reference_answer: ok\n"
            "    topic: test\n    source: fixture\n    approved_by: null\n"
            "    approved_at: null\n    requires_llm: false\n"
            "    assertions: [{kind: contains, value: ok}]\n"
            for i in range(5)
        ),
        encoding="utf-8",
    )
    evaluator = PipelineEvaluator(path, runner=runner, concurrency=2)
    first = await evaluator.run(offline=True)
    second = await evaluator.run(offline=True)
    assert first["total"] == second["total"] == 5
    assert first["passed_perfectly"] == 5
    assert peak == 2
    assert len(set(sessions)) == 10
    assert all(s.startswith("eval:") for s in sessions)


def test_baseline_rejects_missing_cases_and_regressions():
    baseline = {"schema_version": 1, "cases": {"a": {"passed": True}}}
    assert compare_baseline({"results": []}, baseline)
    assert compare_baseline({"results": [{"id": "a", "passed": False}]}, baseline)
    assert not compare_baseline({"results": [{"id": "a", "passed": True}]}, baseline)


def test_repository_baseline_is_static():
    baseline = json.loads(Path("evals/baseline.json").read_text(encoding="utf-8"))
    scenarios = load_scenarios(Path("evals/scenarios.yaml"))
    assert set(baseline["cases"]) == {s.id for s in scenarios if not s.requires_llm}


async def test_real_offline_pipeline_passes_and_records_trace(monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    from sqlalchemy.ext.asyncio import AsyncEngine

    from evals.offline import offline_pipeline

    connections = []

    def forbid_connect(*args, **kwargs):
        connections.append(True)
        raise AssertionError("Unexpected database connection")

    monkeypatch.setattr(AsyncEngine, "connect", forbid_connect)

    with offline_pipeline() as runner:
        report = await PipelineEvaluator(runner=runner).run(offline=True)
    assert report["total"] == report["passed_perfectly"] == 7
    assert not connections
    by_id = {r["id"]: r for r in report["results"]}
    assert by_id["abbrev_1"]["faq_matched"]
    assert by_id["tool_1"]["tools_called"] == ["get_admission_scores"]
    assert not by_id["faq_negative_1"]["faq_matched"]
    assert by_id["crag_topic_switch_mmf"]["passed"]
    assert report["skipped_ids"] == ["guardrails", "hallucination_check"]


@pytest.mark.parametrize(
    "break_step,failed_case",
    [
        ("abbrev", "abbrev_1"),
        ("faq", "faq_1"),
        ("tools", "tool_1"),
        ("retrieval", "rag_1"),
    ],
)
async def test_pipeline_regressions_fail_gate(monkeypatch, break_step, failed_case):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    from evals.evaluator import gate_exit_code
    from evals.offline import offline_pipeline
    from llm.base import LLMResult
    from pipeline import llm_client

    with offline_pipeline() as runner:
        if break_step == "abbrev":
            monkeypatch.setattr(llm_client, "_expand_abbrevs", lambda text, sid: text)
        elif break_step == "faq":

            async def no_faq(*args):
                return None

            monkeypatch.setattr(llm_client, "_match_faq", no_faq)
        elif break_step == "retrieval":

            async def no_context(*args, **kwargs):
                return "", []

            monkeypatch.setattr(llm_client, "query_graph_with_crag", no_context)
        else:

            class NoToolsProvider:
                async def generate_with_tools(self, *args, **kwargs):
                    return LLMResult(text="Балл 246, get_admission_scores")

            monkeypatch.setattr(llm_client, "get_llm_provider", NoToolsProvider)
        report = await PipelineEvaluator(runner=runner).run(offline=True)
    baseline = json.loads(Path("evals/baseline.json").read_text(encoding="utf-8"))
    assert gate_exit_code(report, baseline, 0.0) == 1
    assert not next(r for r in report["results"] if r["id"] == failed_case)["passed"]


async def test_errors_do_not_leak_provider_details_or_pass(tmp_path):
    async def runner(scenario, session_id):
        raise RuntimeError("private-provider-url-and-token")

    evaluator = PipelineEvaluator(runner=runner)
    report = await evaluator.run(offline=True)
    assert report["passed_perfectly"] == 0
    assert "private-provider" not in json.dumps(report)


async def test_judge_runs_only_after_deterministic_checks(monkeypatch):
    called = []

    async def runner(scenario, session_id):
        return Observation("unrelated")

    async def judge(**kwargs):
        called.append(kwargs)
        return {"score": 5, "reasoning": "ok"}

    evaluator = PipelineEvaluator(runner=runner, judge=judge)
    report = await evaluator.run()
    assert all(c["question"] != "Напиши код калькулятора на Python." for c in called)
    assert report["total"] == 9


@pytest.mark.parametrize(
    "response",
    [
        '{"score":true,"reasoning":"x"}',
        '{"score":6,"reasoning":"x"}',
        '{"score":4}',
        'prefix {"score":5,"reasoning":"x"}',
    ],
)
def test_judge_rejects_malformed_responses(response):
    from evals.judge import parse_judgement

    with pytest.raises(ValueError):
        parse_judgement(response)


def test_judge_parses_nested_json_and_uses_zero_temperature():
    from evals.judge import JUDGE_PROFILE, parse_judgement

    response = '```json\n{"score":4,"reasoning":"{ok}","details":{"a":1}}\n```'
    assert parse_judgement(response)["score"] == 4
    assert JUDGE_PROFILE.temperature == 0


async def test_timeout_is_failed_case():
    async def runner(scenario, session_id):
        await asyncio.Event().wait()
        return Observation("unreachable")

    report = await PipelineEvaluator(runner=runner, timeout=0.01).run(offline=True)
    assert report["passed_perfectly"] == 0
    assert all(r["error"] == "TimeoutError" for r in report["results"])


def test_duplicate_ids_are_rejected(tmp_path):
    import yaml

    scenario = load_scenarios("evals/scenarios.yaml")[0].model_dump(mode="json")
    path = tmp_path / "duplicates.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "scenarios": [scenario, scenario],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_scenarios(path)


def test_gate_codes_and_score_regression():
    from evals.evaluator import gate_exit_code

    baseline = {"schema_version": 1, "cases": {"a": {"passed": True, "min_score": 4}}}
    report = {
        "pass_rate": 1.0,
        "results": [{"id": "a", "passed": True, "judge_score": 3}],
    }
    assert gate_exit_code(report, baseline, 1) == 1
    report["results"][0]["judge_score"] = 4
    assert gate_exit_code(report, baseline, 1) == 0
    report["pass_rate"] = 0.5
    assert gate_exit_code(report, baseline, 1) == 1
    with pytest.raises(ValueError):
        gate_exit_code(report, baseline, 2)


def test_invalid_cli_config_exits_before_running_pipeline(tmp_path, monkeypatch):
    from evals.evaluator import main

    path = tmp_path / "bad.json"
    path.write_text('{"schema_version":1,"cases":{"a":{"passed":"yes"}}}')
    monkeypatch.setattr("sys.argv", ["evals", "--baseline", str(path)])
    assert main() == 2


async def test_generator_only_creates_unapproved_drafts(monkeypatch):
    from evals import generate

    class Provider:
        async def generate(self, messages, profile):
            assert profile.temperature == 0
            return json.dumps(
                {
                    "question": "Вопрос?",
                    "reference_answer": "Ответ.",
                    "topic": "topic",
                    "judge_criteria": "Соответствует источнику",
                }
            )

    monkeypatch.setattr(generate, "get_llm_provider", Provider)
    excerpts = [{"source": "reviewed source", "content": "Text"}]
    first = await generate.generate_scenarios(excerpts)
    second = await generate.generate_scenarios(excerpts)
    assert first[0].id == second[0].id
    assert first[0].approved_by is None and first[0].approved_at is None
    assert first[0].requires_llm
    assert first[0].source == "reviewed source"
