"""Owner decisions remain a separate, hash-bound layer over the original run."""

import hashlib
import json
import shutil

import pytest

from evals import review_admission
from evals.review_admission import DATA, REVIEW, calculate_review


def test_recorded_decisions_reproduce_totals_without_mutating_evidence():
    files = [REVIEW, *DATA.glob("*")]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    result = calculate_review()
    recorded = REVIEW.parent / "388-reviewed-results-2026-11-12.json"
    assert result == json.loads(recorded.read_text(encoding="utf-8"))
    full = result["conditions"]["with_kb"]
    empty = result["conditions"]["empty_kb"]
    assert full["automatic"]["total"]["passed"] == 37
    assert full["reviewed"]["document"]["passed"] == 29
    assert full["reviewed"]["special"]["passed"] == 1
    assert full["reviewed"]["total"]["passed"] == 30
    assert empty["automatic"]["total"]["passed"] == 1
    assert empty["reviewed"]["total"]["passed"] == 0
    assert len(full["changed_passes"]) == 7
    assert len(empty["changed_passes"]) == 1
    # Price failures already existed, and must not be subtracted a second time.
    assert "nsu-paid-steps-01" not in full["changed_passes"]
    assert "nsu-paid-steps-02" not in full["changed_passes"]
    assert len(result["confirmed_critical_cases"]) == 3
    assert not result["targets"]["document_met"]
    assert not result["targets"]["special_met"]
    assert result["reviewed_document_delta_pp"] == pytest.approx(64.444444)
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


@pytest.fixture
def review_copy(tmp_path):
    path = tmp_path / "review.json"
    shutil.copyfile(REVIEW, path)
    return path


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate",
        "foreign_id",
        "stale_dataset",
        "original_outcome",
        "critical_pass",
        "missing_reason",
    ],
)
def test_invalid_decisions_cannot_silently_change_totals(review_copy, fault):
    review = json.loads(review_copy.read_text(encoding="utf-8"))
    first = review["decisions"][0]
    if fault == "duplicate":
        review["decisions"].append(first.copy())
    elif fault == "foreign_id":
        first["case_id"] = "foreign"
    elif fault == "stale_dataset":
        review["dataset_sha256"] = "0" * 64
    elif fault == "original_outcome":
        first["automatic_passed"] = False
    elif fault == "critical_pass":
        first.update(criticality="critical", reviewed_passed=True)
    else:
        first["reason"] = ""
    review_copy.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError):
        calculate_review(review_copy)


@pytest.mark.parametrize("fault", ["tamper", "missing", "duplicate", "configuration"])
def test_review_rejects_changed_or_incomplete_reports(review_copy, monkeypatch, fault):
    folder = review_copy.parent / "reports"
    folder.mkdir()
    for name in ("with-kb-aggregate.json", "empty-kb-aggregate.json"):
        shutil.copyfile(DATA / name, folder / name)
    monkeypatch.setattr(review_admission, "DATA", folder)
    review = json.loads(review_copy.read_text(encoding="utf-8"))
    report_path = folder / "with-kb-aggregate.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if fault in {"tamper", "configuration"}:
        report["configuration_sha256"] = "changed"
    elif fault == "missing":
        report["results"].pop()
    else:
        report["results"][0] = report["results"][1].copy()
    report_path.write_text(json.dumps(report), encoding="utf-8")
    if fault != "tamper":
        review["reports"]["with_kb"]["sha256"] = hashlib.sha256(
            report_path.read_bytes()
        ).hexdigest()
    review_copy.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError):
        calculate_review(review_copy)
