"""Check dataset integrity and review evidence without loading the application."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path

import yaml

from evals.schema import load_scenarios

DATASETS = Path(__file__).parent / "datasets"
QUOTA = {
    "PRE-FLOW": 4,
    "PRE-SCHOLAR": 3,
    "PRE-DORM": 3,
    "PRE-PARTNER": 2,
    "PRE-LANG": 4,
    "PRE-COMPARE": 4,
    "PRE-SPECIAL": 3,
    "PRE-CHANCE": 3,
    "PRE-BUILDING": 3,
    "PRE-GENERAL": 3,
    "POST-DORM": 4,
    "POST-ARRIVAL": 3,
    "POST-DOCS": 3,
    "PAID-CONTACT": 3,
    "PAID-STEPS": 5,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_dataset(directory: Path = DATASETS) -> dict:
    """Return coverage; an intact draft does not imply human confirmation."""
    path = directory / "nsu_admission_2026.yaml"
    registry = json.loads(
        (directory / "nsu_admission_2026_registry.json").read_text(encoding="utf-8")
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    require(registry["schema_version"] == 1, "Unexpected registry version")
    require(registry["dataset"] == path.name, "Wrong dataset reference")
    require(registry["dataset_sha256"] == digest, "Dataset changed: refresh registry")
    scenarios = load_scenarios(path)
    require(len(scenarios) == 50, "Expected 50 scenarios")
    require(Counter(s.topic for s in scenarios) == QUOTA, "Topic quota mismatch")
    require(registry["quota"] == QUOTA, "Registry quota mismatch")
    require(
        len({" ".join(s.question.casefold().split()) for s in scenarios}) == 50,
        "Duplicate questions",
    )
    cases = registry["cases"]
    require(len(cases) == 50, "Expected 50 review records")
    by_id = {case["id"]: case for case in cases}
    require(set(by_id) == {s.id for s in scenarios}, "Review IDs differ from dataset")
    sources = registry["sources"]
    groups = Counter(c.get("evaluation_group") for c in cases)
    require(groups == {"document": 45, "special": 5}, "Expected draft split 45/5")
    for source in sources.values():
        require(source["url"].startswith("https://"), "Expected HTTPS source")
        require(
            bool(source["section"] and source["edition_note"]), "Missing source context"
        )
        date.fromisoformat(source["checked_at"])

    reviewed = 0
    expert_reviewed = 0
    for scenario in scenarios:
        case = by_id[scenario.id]
        if case["evaluation_group"] == "document":
            require(bool(case.get("mandatory_facts")), f"{scenario.id}: missing facts")
            require(
                bool(case.get("behavior_constraints")),
                f"{scenario.id}: missing separate behavior constraints",
            )
            selection = case.get("fact_selection", {})
            require(
                selection.get("mode") in {"all", "at_least"},
                f"{scenario.id}: invalid fact selection",
            )
            if selection["mode"] == "at_least":
                minimum = selection.get("minimum")
                require(
                    type(minimum) is int and 0 < minimum <= len(case["mandatory_facts"]),
                    f"{scenario.id}: invalid fact minimum",
                )
            require(
                case["expected_action"] == "answer", f"{scenario.id}: fallback target"
            )
            require(
                all(sources[s].get("snapshot_sha256") for s in case["source_ids"]),
                f"{scenario.id}: missing source snapshot",
            )
        require(bool(case["context"]), f"{scenario.id}: missing campaign context")
        require(
            case["expected_action"] in {"answer", "clarify", "refer"},
            f"{scenario.id}: invalid expected action",
        )
        require(
            {"FR-1", "FR-5", "FR-8", "NFR-3"} <= set(case["requirements"]),
            f"{scenario.id}: missing requirements",
        )
        require(
            bool(case["required_checks"] and case["critical_failures"]),
            f"{scenario.id}: missing acceptance checks",
        )
        require(bool(case["source_ids"]), f"{scenario.id}: missing sources")
        require(
            set(case["source_sections"]) == set(case["source_ids"]),
            f"{scenario.id}: missing source sections",
        )
        require(
            all(sources[s]["url"] in scenario.source for s in case["source_ids"]),
            f"{scenario.id}: source links differ",
        )
        require(
            case["content_status"] in {"draft", "needs_clarification", "reviewed"},
            f"{scenario.id}: invalid content status",
        )
        require(
            (case["content_status"] == "needs_clarification")
            == bool(case["missing_evidence"]),
            f"{scenario.id}: clarification status differs from missing evidence",
        )
        for level, pending in (
            ("internal_review", "pending"),
            ("expert_review", "not_requested"),
        ):
            review = case[level]
            require(
                review["status"] in {pending, "confirmed"},
                f"{scenario.id}: invalid {level} status",
            )
            evidence = ("reviewer", "reviewed_at", "basis", "dataset_sha256")
            if review["status"] == "confirmed":
                require(
                    all(review[field] for field in evidence),
                    f"{scenario.id}: incomplete {level}",
                )
                date.fromisoformat(review["reviewed_at"])
                require(
                    review["dataset_sha256"] == digest,
                    f"{scenario.id}: stale {level}",
                )
            else:
                require(
                    all(review[field] is None for field in evidence),
                    f"{scenario.id}: pending review has confirmation fields",
                )
        internal = case["internal_review"]["status"] == "confirmed"
        expert = case["expert_review"]["status"] == "confirmed"
        require(
            (case["content_status"] == "reviewed") == internal,
            f"{scenario.id}: content and internal confirmation differ",
        )
        require(not expert or internal, f"{scenario.id}: expert review before internal")
        if expert:
            review = case["expert_review"]
            require(
                scenario.approved_by == review["reviewer"]
                and str(scenario.approved_at) == review["reviewed_at"],
                f"{scenario.id}: expert confirmation differs from YAML",
            )
        else:
            require(
                scenario.approved_by is None and scenario.approved_at is None,
                f"{scenario.id}: unrecorded expert confirmation",
            )
        reviewed += int(internal)
        expert_reviewed += int(expert)
    return {
        "total": len(scenarios),
        "evaluation_groups": dict(groups),
        "topics": dict(Counter(s.topic for s in scenarios)),
        "internal_confirmed": reviewed,
        "expert_confirmed": expert_reviewed,
        "needs_clarification": sum(
            c["content_status"] == "needs_clarification" for c in cases
        ),
        "dataset_sha256": digest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-reviewed", action="store_true")
    args = parser.parse_args()
    try:
        summary = validate_dataset()
    except (ValueError, KeyError, OSError, TypeError, yaml.YAMLError) as exc:
        print(f"Dataset invalid: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(args.require_reviewed and summary["internal_confirmed"] != 50)


if __name__ == "__main__":
    raise SystemExit(main())
