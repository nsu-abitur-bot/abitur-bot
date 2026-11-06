"""Validate evidence and keep operational replies out of the factual answer suite."""

import hashlib
import json
import re
from datetime import date
from pathlib import Path

from evals.schema import load_scenarios

DATASETS = Path(__file__).resolve().parent / "datasets"


def validate_dataset(directory: Path = DATASETS) -> dict:
    """Check both suites, evidence, owner confirmations and excluded notes."""
    registry = json.loads(
        (directory / "nsu_summer_2026_registry.json").read_text(encoding="utf-8")
    )

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    require(registry["schema_version"] == 1, "Unsupported registry version")
    expected = {
        "factual": ("nsu_summer_2026.yaml", "dataset", "dataset_sha256"),
        "operational": (
            "nsu_summer_operational_2026.yaml",
            "operational_dataset",
            "operational_dataset_sha256",
        ),
    }
    scenarios = {}
    groups = {}
    for group, (name, key, digest) in expected.items():
        require(registry[key] == name, "Wrong dataset")
        path = directory / name
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == registry[digest],
            "Dataset and evidence registry differ",
        )
        for scenario in load_scenarios(path):
            require(scenario.id not in scenarios, "Duplicate id across suites")
            scenarios[scenario.id] = scenario
            groups[scenario.id] = group
    cases = registry["cases"]
    require(len({c["id"] for c in cases}) == len(cases), "Duplicate case id")
    sources = {s["id"]: s for s in registry["sources"]}
    require(len(sources) == len(registry["sources"]), "Duplicate source id")
    for source in sources.values():
        require(bool(re.fullmatch(r"[0-9a-f]{64}", source["sha256"])), "Invalid hash")
    ready = {"source_checked", "owner_confirmed"}
    checked = {c["id"] for c in cases if c["status"] in ready}
    require(set(scenarios) == checked, "Only evidenced cases may be evaluated")
    for case in cases:
        status = case["status"]
        require(status in ready | {"needs_source", "out_of_scope"}, "Unknown disposition")
        if status not in ready:
            require(case["reference_answer"] is None, "Inactive case has an answer")
            if status == "out_of_scope":
                require(bool(case["exclusion"]["reason"]), "Missing exclusion reason")
                require(bool(case["exclusion"]["decision_url"]), "Missing scope decision")
            else:
                require(bool(case["follow_up"]["needed"]), "Missing follow-up")
            continue
        evidence = case["evidence"] or {}
        require(evidence.get("source_id") in sources, "Missing checked source")
        source = sources[evidence["source_id"]]
        require(bool(evidence.get("section")), "Missing source section")
        scenario = scenarios[case["id"]]
        require(scenario.question == case["question"], "Question mismatch")
        require(scenario.history == case["history"], "Situation mismatch")
        require(scenario.source == source["url"], "Source mismatch")
        require(scenario.reference_answer == case["reference_answer"], "Answer mismatch")
        require(groups[case["id"]] == case["evaluation_group"], "Wrong evaluation suite")
        require(bool(scenario.history), "Situation context must reach the bot")
        require(bool(scenario.judge_criteria), "Missing substantive answer criteria")
        if status == "owner_confirmed":
            require(source.get("kind") == "owner_decision", "Missing owner source")
            approval = case["approval"]["internal"]
            require(isinstance(approval, dict), "Missing owner approval")
            require(bool(approval["reviewer"]), "Missing reviewer")
            date.fromisoformat(approval["reviewed_at"])
            require(approval["basis"] == scenario.source, "Wrong approval basis")
            require(
                approval["reference_sha256"]
                == hashlib.sha256(scenario.reference_answer.encode()).hexdigest(),
                "Changed answer needs renewed confirmation",
            )
        else:
            require(source.get("kind") != "owner_decision", "Not a document check")
    return {
        "total": len(cases),
        "active": len(scenarios),
        "source_checked": sum(c["status"] == "source_checked" for c in cases),
        "owner_confirmed": sum(c["status"] == "owner_confirmed" for c in cases),
        "out_of_scope": sum(c["status"] == "out_of_scope" for c in cases),
        "needs_source": sum(c["status"] == "needs_source" for c in cases),
        "factual": sum(g == "factual" for g in groups.values()),
        "operational": sum(g == "operational" for g in groups.values()),
        "internal_confirmed": sum(bool(c["approval"]["internal"]) for c in cases),
        "expert_confirmed": sum(bool(c["approval"]["expert"]) for c in cases),
    }


if __name__ == "__main__":
    print(json.dumps(validate_dataset(), ensure_ascii=False, indent=2))
