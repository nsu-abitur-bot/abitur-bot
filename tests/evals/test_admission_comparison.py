import pytest

from evals.compare_admission import compare_reports
from evals.validate_admission import validate_dataset


def reports():
    digest = validate_dataset()["dataset_sha256"]
    return [
        dict(
            mode="live",
            condition=condition,
            dataset_sha256=digest,
            configuration_sha256="synthetic-config",
            kb_manifest_sha256=condition,
            results=[],
        )
        for condition in ("with_kb", "empty_kb")
    ]


def test_missing_results_stay_in_denominator_and_groups_are_separate():
    full, empty = reports()
    full["results"] = [dict(id="nsu-pre-flow-01", passed=True)]
    result = compare_reports(full, empty)
    assert result["groups"]["document"]["total"] == 45
    assert result["groups"]["document"]["with_kb_pass_rate"] == 1 / 45
    assert result["groups"]["special"]["total"] == 5
    assert result["groups"]["special"]["with_kb_pass_rate"] == 0


def test_different_configuration_is_not_comparable():
    full, empty = reports()
    empty["configuration_sha256"] = "different-model"
    with pytest.raises(ValueError, match="configuration differs"):
        compare_reports(full, empty)


def test_duplicate_results_cannot_inflate_quality():
    full, empty = reports()
    full["results"] = [dict(id="nsu-pre-flow-01", passed=True)] * 2
    with pytest.raises(ValueError, match="Duplicate"):
        compare_reports(full, empty)


def test_offline_report_cannot_prove_live_improvement():
    full, empty = reports()
    full["mode"] = "offline"
    with pytest.raises(ValueError, match="live reports"):
        compare_reports(full, empty)
