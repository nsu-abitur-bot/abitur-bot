"""Review evidence must not silently survive changes to the acceptance dataset."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from evals.validate_admission import DATASETS, validate_dataset


@pytest.fixture
def dataset_copy(tmp_path: Path) -> Path:
    for name in ("nsu_admission_2026.yaml", "nsu_admission_2026_registry.json"):
        shutil.copyfile(DATASETS / name, tmp_path / name)
    return tmp_path


def update_registry(directory: Path, change) -> None:
    path = directory / "nsu_admission_2026_registry.json"
    registry = json.loads(path.read_text(encoding="utf-8"))
    change(registry)
    path.write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")


def test_committed_dataset_has_expected_coverage() -> None:
    summary = validate_dataset()
    assert summary["total"] == 50
    assert sum(v for k, v in summary["topics"].items() if k.startswith("PRE-")) == 32
    assert sum(v for k, v in summary["topics"].items() if k.startswith("POST-")) == 10
    assert sum(v for k, v in summary["topics"].items() if k.startswith("PAID-")) == 8


def test_changed_answer_invalidates_registry(dataset_copy: Path) -> None:
    path = dataset_copy / "nsu_admission_2026.yaml"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Dataset changed"):
        validate_dataset(dataset_copy)


def test_claimed_review_needs_evidence(dataset_copy: Path) -> None:
    update_registry(
        dataset_copy,
        lambda r: r["cases"][0]["internal_review"].update(status="confirmed"),
    )
    with pytest.raises(ValueError, match="incomplete internal_review"):
        validate_dataset(dataset_copy)


def test_missing_review_record_is_rejected(dataset_copy: Path) -> None:
    update_registry(dataset_copy, lambda r: r["cases"].pop())
    with pytest.raises(ValueError, match="50 review records"):
        validate_dataset(dataset_copy)


def test_unrecorded_expert_approval_is_rejected(dataset_copy: Path) -> None:
    path = dataset_copy / "nsu_admission_2026.yaml"
    text = path.read_text(encoding="utf-8").replace(
        "approved_by: null\n  approved_at: null",
        "approved_by: reviewer\n  approved_at: 2026-11-03",
        1,
    )
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    update_registry(dataset_copy, lambda r: r.update(dataset_sha256=digest))
    with pytest.raises(ValueError, match="unrecorded expert confirmation"):
        validate_dataset(dataset_copy)


def test_stale_review_digest_is_rejected(dataset_copy: Path) -> None:
    def change(registry: dict) -> None:
        case = registry["cases"][0]
        case["internal_review"].update(
            status="confirmed",
            reviewer="reviewer",
            reviewed_at="2026-11-03",
            basis="Review of source and answer",
            dataset_sha256="0" * 64,
        )

    update_registry(dataset_copy, change)
    with pytest.raises(ValueError, match="stale internal_review"):
        validate_dataset(dataset_copy)
