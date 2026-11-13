import copy
import hashlib
import json
import shutil
from decimal import Decimal
from pathlib import Path

import pytest

from tools.release_admission_metrics import calculate, summarize_usage


@pytest.fixture
def evidence_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from tools import release_admission_metrics

    source = release_admission_metrics.DATA
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for name in ["manifest.json", *(spec["file"] for spec in manifest["files"])]:
        assert Path(name).name == name
        shutil.copyfile(source / name, tmp_path / name)
    monkeypatch.setattr(release_admission_metrics, "DATA", tmp_path)
    return tmp_path


def update_test_hash(directory: Path, name: str) -> None:
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for spec in manifest["files"]:
        if spec["file"] == name:
            spec["sha256"] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_real_evidence_phases_reconcile_without_counting_judge_as_answers():
    result = calculate()
    assert len(result["verified_inputs"]) == 73
    total = result["total_usage"]
    assert total["http_attempts"] == 1826
    assert total["deepseek_generations"] == 1375
    assert total["deepseek_historical_cny"] == "0.74856644"
    assert total["gigachat_embedding_tokens"] == 18131
    assert total["gigachat_cost"] is None
    phases = [
        phase
        for condition in result["conditions"].values()
        for phase in condition["usage_by_phase"].values()
    ]
    assert sum(p["http_attempts"] for p in phases) == total["http_attempts"]
    assert sum(Decimal(p["deepseek_historical_cny"]) for p in phases) == Decimal(
        total["deepseek_historical_cny"]
    )
    assert (
        sum(
            c["usage_by_phase"]["judge"]["deepseek_generations"]
            for c in result["conditions"].values()
        )
        == 99
    )
    assert all(c["bot_only_latency"] is None for c in result["conditions"].values())
    assert result["quality"]["conditions"]["with_kb"]["reviewed"]["total"]["passed"] == 30


def test_missing_or_inconsistent_usage_cannot_become_zero_cost():
    row = {
        "usage_complete": True,
        "http_status": 200,
        "error_type": None,
        "operation": "generation",
        "provider": "deepseek",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "prompt_cache_hit_tokens": 4,
            "prompt_cache_miss_tokens": 6,
        },
    }
    assert summarize_usage([row])["deepseek_historical_cny"] == "0.00001408"
    for change in ("missing", "mismatch", "incomplete", "failed"):
        invalid = copy.deepcopy(row)
        if change == "missing":
            invalid["usage"].pop("prompt_cache_hit_tokens")
        elif change == "mismatch":
            invalid["usage"]["total_tokens"] = 11
        elif change == "incomplete":
            invalid["usage_complete"] = False
        else:
            invalid["http_status"] = 500
        with pytest.raises(ValueError):
            summarize_usage([invalid])


def test_modified_evidence_is_rejected_before_calculation(tmp_path, monkeypatch):
    from tools import release_admission_metrics

    (tmp_path / "manifest.json").write_text(
        json.dumps({"files": [{"file": "sample.json", "sha256": "0" * 64}]}),
        encoding="utf-8",
    )
    (tmp_path / "sample.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(release_admission_metrics, "DATA", tmp_path)
    with pytest.raises(ValueError, match="Evidence SHA mismatch"):
        calculate()


def test_duplicate_intent_with_updated_hash_is_rejected(evidence_copy: Path):
    name = "with-kb-full-20261110-02-batch-00-intents.jsonl"
    path = evidence_copy / name
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[0]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    update_test_hash(evidence_copy, name)
    with pytest.raises(ValueError, match="Duplicate intent"):
        calculate()


@pytest.mark.parametrize("kind", ["report.json", "usage.jsonl", "intents.jsonl"])
@pytest.mark.parametrize("keep_count", [False, True])
def test_required_file_outside_manifest_is_rejected(
    evidence_copy: Path, kind: str, keep_count: bool
):
    name = "with-kb-full-20261110-02-batch-00-" + kind
    path = evidence_copy / name
    if kind == "usage.jsonl":
        rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        for key in ("prompt_tokens", "total_tokens", "prompt_cache_miss_tokens"):
            rows[0]["usage"][key] += 1000
        path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
    else:
        text = path.read_text(encoding="utf-8")
        path.write_text(" " + text, encoding="utf-8")
    manifest_path = evidence_copy / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [s for s in manifest["files"] if s["file"] != name]
    if keep_count:
        extra = evidence_copy / "unrelated.json"
        extra.write_text("{}", encoding="utf-8")
        manifest["files"].append(
            {"file": extra.name, "sha256": hashlib.sha256(extra.read_bytes()).hexdigest()}
        )
        assert len(manifest["files"]) == 73
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert path.is_file()
    with pytest.raises(ValueError, match="Missing required evidence"):
        calculate()


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ("duplicate_usage", "Duplicate attempt"),
        ("different_ids", "Attempt ID sets differ"),
        ("boolean_id", "Invalid attempt ID"),
    ],
)
def test_invalid_attempt_mapping_is_rejected(
    evidence_copy: Path, change: str, expected: str
):
    kind = "usage" if change == "duplicate_usage" else "intents"
    name = f"with-kb-full-20261110-02-batch-00-{kind}.jsonl"
    path = evidence_copy / name
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if change == "duplicate_usage":
        rows[1] = rows[0]
    else:
        rows[0]["attempt"] = True if change == "boolean_id" else 999999
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    update_test_hash(evidence_copy, name)
    with pytest.raises(ValueError, match=expected):
        calculate()
