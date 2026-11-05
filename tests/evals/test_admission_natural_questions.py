import json
import re

import pytest

from evals.checks import Observation, check_assertion
from evals.schema import load_scenarios
from evals.validate_admission import DATASETS


def test_questions_do_not_hint_snapshot_or_document():
    cases = load_scenarios(DATASETS / "nsu_admission_2026.yaml")
    for case in cases:
        assert not re.search(
            r"сним[ок]|по обзору|на странице|по странице|по инструкции|по расписанию"
            r"|05\.11\.2026",
            case.question,
            re.IGNORECASE,
        ), case.id


@pytest.mark.parametrize("answer", ["Сбербанк", "ПАО Сбербанк", "Сбер"])
def test_short_bank_name_passes_factual_assertion_without_warning(answer):
    case = next(
        c
        for c in load_scenarios(DATASETS / "nsu_admission_2026.yaml")
        if c.id == "nsu-paid-contact-02"
    )
    assert all(check_assertion(a, Observation(answer)) for a in case.assertions)
    assert not all(check_assertion(a, Observation("Т-Банк")) for a in case.assertions)


def test_bank_facts_do_not_require_editorial_disclaimer():
    registry = json.loads(
        (DATASETS / "nsu_admission_2026_registry.json").read_text("utf-8")
    )
    case = next(c for c in registry["cases"] if c["id"] == "nsu-paid-contact-02")
    assert case["mandatory_facts"] == ["Сбербанк"]
    assert any("навязывать" in rule for rule in case["behavior_constraints"])


def test_example_questions_do_not_require_every_allowed_example():
    registry = json.loads(
        (DATASETS / "nsu_admission_2026_registry.json").read_text("utf-8")
    )
    case = next(c for c in registry["cases"] if c["id"] == "nsu-pre-partner-02")
    assert case["fact_selection"] == {"mode": "at_least", "minimum": 3}
    assert len(case["mandatory_facts"]) > 3
