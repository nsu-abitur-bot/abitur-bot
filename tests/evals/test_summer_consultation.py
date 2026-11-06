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
    scenarios = load_scenarios(DATASETS / "nsu_summer_2026.yaml") + load_scenarios(
        DATASETS / "nsu_summer_operational_2026.yaml"
    )
    active = {s.id for s in scenarios}
    excluded = {c["id"] for c in cases if c["status"] == "out_of_scope"}
    assert len(active) == 11
    assert excluded == {c["id"] for c in cases if c["level"] == "phd"}
    assert not active & excluded
    assert all(c["exclusion"]["reason"] for c in cases if c["id"] in excluded)
    validate_dataset()


def test_only_evidenced_answers_are_runnable():
    registry = json.loads((DATASETS / "nsu_summer_2026_registry.json").read_text())
    scenarios = load_scenarios(DATASETS / "nsu_summer_2026.yaml") + load_scenarios(
        DATASETS / "nsu_summer_operational_2026.yaml"
    )
    ready = {
        c["id"]
        for c in registry["cases"]
        if c["status"] in {"source_checked", "owner_confirmed"}
    }
    assert {s.id for s in scenarios} == ready
    assert all(s.history and s.judge_criteria for s in scenarios)
    assert all(s.approved_by is None and s.approved_at is None for s in scenarios)


def test_owner_decisions_are_not_expert_approval_or_document_quality():
    summary = validate_dataset()
    assert summary["source_checked"] == 9
    assert summary["owner_confirmed"] == summary["internal_confirmed"] == 2
    assert summary["expert_confirmed"] == 0
    assert summary["factual"] == 10
    assert summary["operational"] == 1
    registry = json.loads((DATASETS / "nsu_summer_2026_registry.json").read_text())
    email = next(c for c in registry["cases"] if c["original_position"] == 11)
    assert email["evaluation_group"] == "operational"
    for case in registry["cases"]:
        if case["status"] == "source_checked":
            assert case["approval"]["internal"] is None


@pytest.mark.parametrize(
    "corruption",
    [
        "dataset_hash",
        "promote_excluded",
        "wrong_source",
        "wrong_question",
        "missing_owner_approval",
        "changed_owner_answer",
    ],
)
def test_inconsistent_evidence_is_rejected(tmp_path, corruption):
    for name in [
        "nsu_summer_2026.yaml",
        "nsu_summer_operational_2026.yaml",
        "nsu_summer_2026_registry.json",
    ]:
        (tmp_path / name).write_bytes((DATASETS / name).read_bytes())
    path = tmp_path / "nsu_summer_2026_registry.json"
    registry = copy.deepcopy(json.loads(path.read_text()))
    if corruption == "dataset_hash":
        registry["dataset_sha256"] = "0" * 64
    elif corruption == "promote_excluded":
        excluded = next(c for c in registry["cases"] if c["status"] == "out_of_scope")
        excluded["status"] = "source_checked"
    elif corruption == "missing_owner_approval":
        owner = next(c for c in registry["cases"] if c["status"] == "owner_confirmed")
        owner["approval"]["internal"] = None
    elif corruption in {"wrong_question", "changed_owner_answer"}:
        dataset = tmp_path / "nsu_summer_2026.yaml"
        document = yaml.safe_load(dataset.read_text())
        if corruption == "wrong_question":
            document["scenarios"][0]["question"] = "Привет!"
        else:
            owner = next(c for c in registry["cases"] if c["status"] == "owner_confirmed")
            scenario = next(s for s in document["scenarios"] if s["id"] == owner["id"])
            owner["reference_answer"] = scenario["reference_answer"] = "Конкурс общий."
        dataset.write_text(yaml.safe_dump(document, allow_unicode=True))
        registry["dataset_sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
    else:
        ready = next(c for c in registry["cases"] if c["status"] == "source_checked")
        ready["evidence"]["source_id"] = "missing-source"
    path.write_text(json.dumps(registry))
    with pytest.raises(ValueError):
        validate_dataset(tmp_path)


def test_unknown_email_timing_is_separate_from_factual_suite():
    factual = load_scenarios(DATASETS / "nsu_summer_2026.yaml")
    operational = load_scenarios(DATASETS / "nsu_summer_operational_2026.yaml")
    assert len(factual) == 10
    assert {s.id for s in operational} == {"nsu-summer-master-11"}
    assert not {s.id for s in factual} & {s.id for s in operational}
