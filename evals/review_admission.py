"""Apply recorded owner decisions to immutable reports, without running models."""

import argparse
import hashlib
import json
from pathlib import Path

from evals.validate_admission import DATASETS, require, validate_dataset

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "docs/reports"
DATA = REPORTS / "release-388-full-data"
REVIEW = REPORTS / "388-owner-review-2026-11-11.json"


def calculate_review(review_path: Path = REVIEW) -> dict:
    """Return original and reviewed counts; never modify the stored evidence."""
    review_raw = review_path.read_bytes()
    review = json.loads(review_raw)
    dataset = validate_dataset()
    registry_raw = (DATASETS / "nsu_admission_2026_registry.json").read_bytes()
    registry = json.loads(registry_raw)
    require(review["schema_version"] == 1, "Unknown review schema")
    require(review["dataset_sha256"] == dataset["dataset_sha256"], "Stale dataset")
    require(
        bool(review["decision_source"] and review["reviewer"]), "Missing decision basis"
    )
    groups = {c["id"]: c["evaluation_group"] for c in registry["cases"]}
    original: dict = {}
    corrected: dict = {}
    inputs = {
        "review_sha256": hashlib.sha256(review_raw).hexdigest(),
        "registry_sha256": hashlib.sha256(registry_raw).hexdigest(),
    }
    for condition in ("with_kb", "empty_kb"):
        spec = review["reports"][condition]
        require(
            spec["file"] == f"{condition.replace('_', '-')}-aggregate.json",
            "Wrong report",
        )
        raw = (DATA / spec["file"]).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == spec["sha256"], "Report hash mismatch")
        inputs[condition + "_sha256"] = spec["sha256"]
        report = json.loads(raw)
        require(
            report["mode"] == "live" and report["condition"] == condition,
            "Wrong condition",
        )
        for key in ("dataset_sha256", "configuration_sha256", "image_revision"):
            require(report[key] == review[key], f"Report {key} differs")
        rows = report["results"]
        require(report["total"] == len(rows) == 50, "Incomplete report")
        require(len({r["id"] for r in rows}) == 50, "Duplicate result")
        require({r["id"] for r in rows} == set(groups), "Wrong result IDs")
        require(all(type(r["passed"]) is bool for r in rows), "Invalid outcome")
        original[condition] = {r["id"]: r for r in rows}
        corrected[condition] = {r["id"]: r["passed"] for r in rows}

    seen = set()
    critical = []
    for decision in review["decisions"]:
        condition, case_id = decision["condition"], decision["case_id"]
        require(condition in original and case_id in groups, "Unknown decision target")
        require((condition, case_id) not in seen, "Duplicate decision")
        seen.add((condition, case_id))
        row = original[condition][case_id]
        require(type(decision["automatic_passed"]) is bool, "Invalid original outcome")
        require(row["passed"] == decision["automatic_passed"], "Original outcome differs")
        require(type(decision["reviewed_passed"]) is bool, "Invalid reviewed outcome")
        require(bool(decision["reason"]), "Missing decision reason")
        require(
            decision["criticality"] in {"critical", "not_assessed"}, "Wrong criticality"
        )
        require(
            not row.get("error") or not decision["reviewed_passed"],
            "Cannot pass an error",
        )
        corrected[condition][case_id] = decision["reviewed_passed"]
        if decision["criticality"] == "critical":
            require(not decision["reviewed_passed"], "Critical answer cannot pass")
            critical.append({"condition": condition, "case_id": case_id})

    def counts(values: dict) -> dict:
        output = {}
        for group in ("document", "special"):
            ids = [key for key, value in groups.items() if value == group]
            output[group] = {"passed": sum(values[i] for i in ids), "total": len(ids)}
        output["total"] = {"passed": sum(values.values()), "total": len(values)}
        return output

    conditions = {
        condition: {
            "automatic": counts({key: row["passed"] for key, row in rows.items()}),
            "reviewed": counts(corrected[condition]),
            "changed_passes": [
                key
                for key, row in rows.items()
                if row["passed"] != corrected[condition][key]
            ],
        }
        for condition, rows in original.items()
    }
    targets = registry["quality_targets"]
    full = conditions["with_kb"]["reviewed"]
    return {
        "schema_version": 1,
        "run_id": review["run_id"],
        "dataset_sha256": dataset["dataset_sha256"],
        "decision_source": review["decision_source"],
        "scope": (
            "Only explicit owner decisions applied. "
            "All other automatic outcomes retained."
        ),
        "inputs": inputs,
        "conditions": conditions,
        "reviewed_document_delta_pp": round(
            100
            * (
                full["document"]["passed"]
                - conditions["empty_kb"]["reviewed"]["document"]["passed"]
            )
            / 45,
            6,
        ),
        "confirmed_critical_cases": critical,
        "targets": {
            "document_required": targets["document_passes"],
            "special_required": targets["special_passes"],
            "critical_allowed": targets["critical_failures_allowed"],
            "document_met": full["document"]["passed"] >= targets["document_passes"],
            "special_met": full["special"]["passed"] >= targets["special_passes"],
            "critical_absence_confirmed": False,
        },
        "release_readiness": "not established",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=REVIEW)
    args = parser.parse_args()
    print(json.dumps(calculate_review(args.review), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
