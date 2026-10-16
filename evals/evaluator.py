"""Bounded evaluation of the product pipeline, with an optional calibrated judge."""

import argparse
import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import uuid4

from evals.checks import Observation, check_assertion
from evals.judge import evaluate_rag_answer
from evals.schema import Scenario, load_scenarios

Runner = Callable[[Scenario, str], Awaitable[Observation]]
Judge = Callable[..., Awaitable[dict]]


class PipelineEvaluator:
    def __init__(
        self,
        scenarios_path: str | Path = "evals/scenarios.yaml",
        *,
        runner: Runner | None = None,
        judge: Judge = evaluate_rag_answer,
        concurrency: int = 3,
        timeout: float = 180,
    ) -> None:
        if concurrency < 1 or timeout <= 0:
            raise ValueError("concurrency and timeout must be positive")
        self.scenarios = load_scenarios(scenarios_path)
        self.runner = runner
        self.judge = judge
        self.concurrency = concurrency
        self.timeout = timeout

    async def run(self, *, offline: bool = False) -> dict:
        selected = [s for s in self.scenarios if not offline or not s.requires_llm]
        if not selected:
            raise ValueError("No scenarios selected")
        runner = self.runner
        if runner is None:
            if offline:
                raise ValueError(
                    "Offline evaluation requires an isolated pipeline runner"
                )
            from evals.product import prepare_product, run_product

            await prepare_product()
            runner = run_product
        run_case: Runner = runner
        semaphore = asyncio.Semaphore(self.concurrency)
        run_id = uuid4().hex

        async def evaluate(scenario: Scenario) -> dict:
            session_id = f"eval:{run_id}:{scenario.id}"
            result: dict = {
                "id": scenario.id,
                "question": scenario.question,
                "session_id": session_id,
                "passed": False,
            }
            async with semaphore:
                try:
                    async with asyncio.timeout(self.timeout):
                        observation = await run_case(scenario, session_id)
                        checks = [
                            {"kind": a.kind, "passed": check_assertion(a, observation)}
                            for a in scenario.assertions
                        ]
                        result.update(
                            answer=observation.text,
                            tools_called=observation.tools_called,
                            faq_matched=observation.faq_matched,
                            checks=checks,
                        )
                        passed = all(c["passed"] for c in checks)
                        if passed and scenario.judge_criteria:
                            evaluation = await self.judge(
                                question=scenario.question,
                                reference=scenario.reference_answer,
                                criteria=scenario.judge_criteria,
                                actual_answer=observation.text,
                            )
                            result["judge_score"] = evaluation["score"]
                            result["judge_reasoning"] = evaluation["reasoning"]
                            passed = evaluation["score"] >= scenario.min_score
                        result["passed"] = passed
                except Exception as exc:
                    # Provider errors may contain credentials; don't echo their text.
                    result["error"] = type(exc).__name__
            return result

        results = await asyncio.gather(*(evaluate(s) for s in selected))
        faq_results = []
        rag_results = []
        for scenario, result in zip(selected, results):
            if any(a.kind == "faq_match" for a in scenario.assertions):
                faq_results.append(
                    {
                        **result,
                        "expected_match": next(
                            a.value for a in scenario.assertions if a.kind == "faq_match"
                        ),
                        "actual_match": result.get("faq_matched", False),
                        "faq_answer": result.get("answer"),
                    }
                )
            else:
                rag_results.append({**result, "rag_answer": result.get("answer", "")})
        passed = sum(r["passed"] for r in results)
        return {
            "schema_version": 1,
            "mode": "offline" if offline else "live",
            "total": len(results),
            "passed_perfectly": passed,
            "pass_rate": passed / len(results),
            "skipped_ids": [s.id for s in self.scenarios if s not in selected],
            "results": results,
            "rag_results": rag_results,
            "faq_results": faq_results,
        }


def compare_baseline(report: dict, baseline: dict) -> list[str]:
    if (
        not isinstance(baseline, dict)
        or baseline.get("schema_version") != 1
        or not isinstance(baseline.get("cases"), dict)
        or not baseline["cases"]
    ):
        raise ValueError("Expected a nonempty baseline schema_version=1")
    for expected in baseline["cases"].values():
        if (
            not isinstance(expected, dict)
            or type(expected.get("passed")) is not bool
            or set(expected) - {"passed", "min_score"}
        ):
            raise ValueError("Invalid baseline case")
        if "min_score" in expected and (
            type(expected["min_score"]) is not int or not 1 <= expected["min_score"] <= 5
        ):
            raise ValueError("Invalid baseline score")
    actual = {r["id"]: r for r in report["results"]}
    failures = []
    for case_id, expected in baseline["cases"].items():
        current = actual.get(case_id)
        if current is None:
            failures.append(f"{case_id}: missing")
        elif expected["passed"] and not current["passed"]:
            failures.append(f"{case_id}: regressed")
        elif "min_score" in expected and (
            current.get("judge_score", 0) < expected["min_score"]
        ):
            failures.append(f"{case_id}: judge score regressed")
    return failures


def gate_exit_code(report: dict, baseline: dict, min_pass_rate: float) -> int:
    if not 0 <= min_pass_rate <= 1:
        raise ValueError("min_pass_rate must be between 0 and 1")
    return int(
        bool(compare_baseline(report, baseline)) or report["pass_rate"] < min_pass_rate
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default="evals/scenarios.yaml")
    parser.add_argument("--baseline", default="evals/baseline.json")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    args = parser.parse_args()
    try:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        # Reject bad gates before making any live calls.
        compare_baseline({"results": []}, baseline)
        if not 0 <= args.min_pass_rate <= 1:
            raise ValueError("Invalid pass-rate threshold")
        if args.offline:
            from evals.offline import offline_pipeline

            with offline_pipeline() as runner:
                report = asyncio.run(
                    PipelineEvaluator(
                        args.scenarios, runner=runner, concurrency=args.concurrency
                    ).run(offline=True)
                )
        else:
            report = asyncio.run(
                PipelineEvaluator(args.scenarios, concurrency=args.concurrency).run()
            )
        code = gate_exit_code(report, baseline, args.min_pass_rate)
        report["regressions"] = compare_baseline(report, baseline)
        output = json.dumps(report, ensure_ascii=False, indent=2)
        if args.report:
            args.report.write_text(output + "\n", encoding="utf-8")
        print(output)
        return code
    except Exception as exc:
        print(f"Evaluation configuration failed: {type(exc).__name__}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
