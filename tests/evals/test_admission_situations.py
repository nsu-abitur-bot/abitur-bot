import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from evals.schema import load_scenarios
from evals.validate_admission import DATASETS


def data():
    scenarios = {s.id: s for s in load_scenarios(DATASETS / "nsu_admission_2026.yaml")}
    registry = json.loads(
        (DATASETS / "nsu_admission_2026_registry.json").read_text("utf-8")
    )
    return scenarios, {c["id"]: c for c in registry["cases"]}


def test_arrival_accepts_later_and_reserve_intervals_with_full_context():
    scenarios, registry = data()
    for suffix, later in (
        ("01", "2026-09-05"),
        ("02", "2026-09-07"),
        ("03", "2026-09-07"),
    ):
        key = "nsu-post-arrival-" + suffix
        dates = {row["date"] for row in registry[key]["acceptable_intervals"]}
        assert later in dates
        assert {f"2026-09-{day}" for day in range(10, 16)} <= dates
        context = " ".join(h["content"] for h in scenarios[key].history)
        assert "гражданин РФ" in context and "2026" in context
        assert "первый курс" in context and "ещё не оформил" in context
        assert "первая" not in scenarios[key].question


def test_pass_answer_describes_actions_instead_of_naming_the_object():
    scenarios, registry = data()
    key = "nsu-post-dorm-03"
    answer = scenarios[key].reference_answer.casefold()
    assert "заведующему" in answer and "чек" in answer and "двумя печатями" in answer
    assert len(registry[key]["mandatory_facts"]) == 3
    assert "финальном этапе" not in scenarios[key].question


def test_general_payment_schedule_is_not_mislabeled_calendar_year():
    scenarios, registry = data()
    case = scenarios["nsu-post-dorm-04"]
    assert "2026" not in case.question
    assert case.history and "общий график" in case.history[0]["content"]
    assert "2026" in case.history[0]["content"]
    assert "Требует согласования" in registry[case.id]["period_proposal"]


@pytest.mark.asyncio
async def test_live_adapter_passes_context_before_question(monkeypatch):
    from evals import product

    scenarios, _ = data()
    case = scenarios["nsu-post-arrival-01"]
    events = []

    async def add_message(session_id, entry):
        events.append((session_id, entry))

    redis = SimpleNamespace(add_message=add_message)
    monkeypatch.setattr(
        "pipeline.llm_client.get_redis_client", AsyncMock(return_value=redis)
    )

    async def ask(question, *, session_id, user_id):
        assert events == [(session_id, entry) for entry in case.history]
        assert question == case.question
        return SimpleNamespace(
            text="fixture",
            sources=[],
            trace=SimpleNamespace(tools_called=[], faq_matched=False),
        )

    monkeypatch.setattr(product, "ask_local_llm", ask)
    await product.run_product(case, "isolated-context-fixture")
