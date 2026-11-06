"""Compare two explicitly labelled live reports; never run the application."""

import argparse
import json
from pathlib import Path

from evals.validate_admission import DATASETS, validate_dataset


def compare_reports(with_kb: dict, empty_kb: dict) -> dict:
    summary = validate_dataset()
    for report, condition in ((with_kb, "with_kb"), (empty_kb, "empty_kb")):
        if report.get("mode") != "live" or report.get("condition") != condition:
            raise ValueError("Expected labelled live reports")
        if report.get("dataset_sha256") != summary["dataset_sha256"]:
            raise ValueError("Dataset versions differ")
        if not report.get("configuration_sha256") or not report.get("kb_manifest_sha256"):
            raise ValueError("Missing run provenance")
    if with_kb["configuration_sha256"] != empty_kb["configuration_sha256"]:
        raise ValueError("Model/prompt/configuration differs between runs")
    if with_kb["kb_manifest_sha256"] == empty_kb["kb_manifest_sha256"]:
        raise ValueError("Expected different knowledge conditions")
    registry = json.loads(
        (DATASETS / "nsu_admission_2026_registry.json").read_text("utf-8")
    )
    expected = {c["id"] for c in registry["cases"]}
    mappings = []
    for report in (with_kb, empty_kb):
        rows = report["results"]
        indexed = {r["id"]: r for r in rows}
        if len(rows) != len(indexed) or not set(indexed) <= expected:
            raise ValueError("Duplicate or foreign result IDs")
        if any(type(r.get("passed")) is not bool for r in rows):
            raise ValueError("Expected explicit boolean outcome")
        mappings.append(indexed)
    output = {}
    for group in ("document", "special"):
        ids = [c["id"] for c in registry["cases"] if c["evaluation_group"] == group]
        rates = [
            sum(m.get(i, {}).get("passed", False) for i in ids) / len(ids)
            for m in mappings
        ]
        output[group] = dict(
            total=len(ids),
            with_kb_pass_rate=rates[0],
            empty_kb_pass_rate=rates[1],
            delta=rates[0] - rates[1],
            missing_with_kb=[i for i in ids if i not in mappings[0]],
            missing_empty_kb=[i for i in ids if i not in mappings[1]],
        )
    return dict(groups=output, acceptance="pending agreed thresholds and human review")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("with_kb", type=Path)
    parser.add_argument("empty_kb", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            compare_reports(
                json.loads(args.with_kb.read_text("utf-8")),
                json.loads(args.empty_kb.read_text("utf-8")),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
