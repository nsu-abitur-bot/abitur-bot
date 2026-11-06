"""Validate consultation coverage without counting unanswered questions as successes."""

import hashlib
import json
import re
from pathlib import Path

from evals.schema import load_scenarios

DATASETS = Path(__file__).resolve().parent / "datasets"


def validate_dataset(directory: Path = DATASETS) -> dict:
    """Check the runnable subset, evidence links and disposition of every note."""
    dataset = directory / "nsu_summer_2026.yaml"
    registry = json.loads(
        (directory / "nsu_summer_2026_registry.json").read_text(encoding="utf-8")
    )

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    require(registry["schema_version"] == 1, "Unsupported registry version")
    require(registry["dataset"] == dataset.name, "Wrong dataset")
    require(
        hashlib.sha256(dataset.read_bytes()).hexdigest() == registry["dataset_sha256"],
        "Dataset and evidence registry differ",
    )
    cases = registry["cases"]
    require(len({c["id"] for c in cases}) == len(cases), "Duplicate case id")
    sources = {s["id"]: s for s in registry["sources"]}
    require(len(sources) == len(registry["sources"]), "Duplicate source id")
    for source in sources.values():
        require(bool(re.fullmatch(r"[0-9a-f]{64}", source["sha256"])), "Invalid hash")
    scenarios = {s.id: s for s in load_scenarios(dataset)}
    checked = {c["id"] for c in cases if c["status"] == "source_checked"}
    require(set(scenarios) == checked, "Only source-checked cases may be evaluated")
    for case in cases:
        require(
            case["status"] in {"source_checked", "needs_source"}, "Unknown disposition"
        )
        if case["status"] == "needs_source":
            require(case["reference_answer"] is None, "A gap must not invent an answer")
            require(bool(case["follow_up"]["needed"]), "Missing follow-up")
            continue
        evidence = case["evidence"] or {}
        require(evidence.get("source_id") in sources, "Missing checked source")
        require(bool(evidence.get("section")), "Missing source section")
        scenario = scenarios[case["id"]]
        require(scenario.question == case["question"], "Question mismatch")
        require(scenario.history == case["history"], "Situation mismatch")
        require(
            scenario.source == sources[evidence["source_id"]]["url"], "Source mismatch"
        )
        require(scenario.reference_answer == case["reference_answer"], "Answer mismatch")
        require(bool(scenario.history), "Situation context must reach the bot")
        require(bool(scenario.judge_criteria), "Missing substantive answer criteria")
    return {
        "total": len(cases),
        "source_checked": len(checked),
        "needs_source": len(cases) - len(checked),
        "internal_confirmed": sum(bool(c["approval"]["internal"]) for c in cases),
        "expert_confirmed": sum(bool(c["approval"]["expert"]) for c in cases),
    }


if __name__ == "__main__":
    print(json.dumps(validate_dataset(), ensure_ascii=False, indent=2))
