import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from evals.schema import load_scenarios
from evals.validate_summer import validate_dataset

DATASETS = Path(__file__).resolve().parents[2] / "evals" / "datasets"


def test_all_original_questions_have_a_disposition():
    registry = json.loads((DATASETS / "nsu_summer_2026_registry.json").read_text())
    cases = registry["cases"]
    assert sorted(c["original_position"] for c in cases) == list(range(1, 14))
    assert Counter(c["level"] for c in cases) == {
        "bachelor": 8,
        "master": 3,
        "phd": 2,
    }
    validate_dataset()


def test_unverified_questions_do_not_inflate_answer_quality():
    registry = json.loads((DATASETS / "nsu_summer_2026_registry.json").read_text())
    scenarios = load_scenarios(DATASETS / "nsu_summer_2026.yaml")
    ready = {c["id"] for c in registry["cases"] if c["status"] == "source_checked"}
    assert {s.id for s in scenarios} == ready
    for case in registry["cases"]:
        if case["status"] == "needs_source":
            assert case["reference_answer"] is None
            assert case["follow_up"]["needed"]
            assert case["follow_up"]["responsible_role"]
    assert all(s.history and s.judge_criteria for s in scenarios)
    assert all(s.approved_by is None and s.approved_at is None for s in scenarios)


def test_source_check_does_not_imply_human_approval():
    summary = validate_dataset()
    assert summary["source_checked"] > 0
    assert summary["internal_confirmed"] == summary["expert_confirmed"] == 0


@pytest.mark.parametrize(
    "corruption", ["dataset_hash", "promote_gap", "wrong_source", "wrong_question"]
)
def test_inconsistent_evidence_is_rejected(tmp_path, corruption):
    for name in ["nsu_summer_2026.yaml", "nsu_summer_2026_registry.json"]:
        (tmp_path / name).write_bytes((DATASETS / name).read_bytes())
    path = tmp_path / "nsu_summer_2026_registry.json"
    registry = copy.deepcopy(json.loads(path.read_text()))
    if corruption == "dataset_hash":
        registry["dataset_sha256"] = "0" * 64
    elif corruption == "promote_gap":
        gap = next(c for c in registry["cases"] if c["status"] == "needs_source")
        gap["status"] = "source_checked"
    elif corruption == "wrong_question":
        dataset = tmp_path / "nsu_summer_2026.yaml"
        document = yaml.safe_load(dataset.read_text())
        document["scenarios"][0]["question"] = "Привет!"
        dataset.write_text(yaml.safe_dump(document, allow_unicode=True))
        registry["dataset_sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
    else:
        ready = next(c for c in registry["cases"] if c["status"] == "source_checked")
        ready["evidence"]["source_id"] = "missing-source"
    path.write_text(json.dumps(registry))
    with pytest.raises(ValueError):
        validate_dataset(tmp_path)
