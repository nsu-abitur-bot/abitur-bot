"""Mirror the active ledger policy; retain operator accounting and transport gate."""

import inspect
from pathlib import Path
from typing import Any

from tools.admission_usage import IndexingLimit

FIELDS = ("input", "output", "embedding", "micro_cny", "micro_rub")


def policy_limits(policy: dict) -> tuple[dict, dict]:
    caps, question = policy.get("caps"), policy.get("question")
    if not isinstance(caps, dict) or not isinstance(question, dict):
        raise IndexingLimit("Invalid active ledger limits")
    for limits, required in (
        (caps, (*FIELDS, "attempts")),
        (question, ("input", "output", "embedding", "generations")),
    ):
        if not isinstance(limits, dict) or any(
            type(limits.get(key)) is not int or limits[key] < 0 for key in required
        ):
            raise IndexingLimit("Invalid active ledger limits")
    return caps, question


def check_reservation(
    state: dict, case: str, operation: str, held: dict, policy: dict
) -> None:
    caps, question = policy_limits(policy)
    if state["halted"] or state["pending_or_uncertain"]:
        raise IndexingLimit("Unresolved pilot accounting")
    if state["attempts_started"] >= caps["attempts"]:
        raise IndexingLimit("Owner attempt ceiling reached")
    per = state["cases"].get(case, {})
    limits = caps if case == "setup" else question
    for kind in FIELDS:
        value = held.get(kind, 0)
        if type(value) is not int or value < 0:
            raise IndexingLimit("Invalid owner reservation")
        if state["totals"][kind] + value > caps[kind]:
            raise IndexingLimit("Owner cumulative ceiling reached")
        if kind in limits:
            if per.get(kind, 0) + value > limits[kind]:
                raise IndexingLimit("Owner question ceiling reached")
    if (
        case != "setup"
        and operation == "generation"
        and per.get("generations", 0) >= question["generations"]
    ):
        raise IndexingLimit("Owner generation ceiling reached")


def attach_owner_guard(ledger: Any) -> None:
    """reserve is called inside the operator's physical transport gate.

    Never installs HTTP hooks, creates a budget, edits a policy, or refunds usage.
    The original reserve remains responsible for its transaction and journal.
    """
    if getattr(ledger, "_owner_guard470", False):
        raise IndexingLimit("Owner guard is already attached")
    policy_limits(ledger.store.policy)
    reserve = ledger.store.reserve
    output_limit = ledger.output_limit

    def bounded_reserve(case_id, operation, held, fingerprint, evidence):
        check_reservation(
            ledger.store.snapshot(), case_id, operation, held, ledger.store.policy
        )
        return reserve(case_id, operation, held, fingerprint, evidence)

    def bounded_output(requested):
        state = ledger.store.snapshot()
        case = ledger._case.get()
        caps, question = policy_limits(ledger.store.policy)
        cap = caps["output"] if case == "setup" else question["output"]
        remaining = min(
            cap - state["cases"].get(case, {}).get("output", 0),
            caps["output"] - state["totals"]["output"],
        )
        if remaining < 1:
            raise IndexingLimit("Owner output ceiling reached")
        return output_limit(min(requested, remaining))

    ledger.store.reserve = bounded_reserve
    ledger.output_limit = bounded_output
    ledger._owner_guard470 = True


def installed_ledger(guard: Any, plan: dict) -> Any:
    """Obtain the already installed guard's dependency, without another capture.

    Refuse unknown wrappers. No SDK/transport hook is removed or replaced here.
    """
    if (
        guard.__module__ != "pipeline.llm_client"
        or not hasattr(guard, "__wrapped__")
        or guard.__code__.co_name != "wrapped"
    ):
        raise IndexingLimit("Unknown operator question wrapper")
    values = inspect.getclosurevars(guard).nonlocals
    ledger = values.get("ledger")
    if (
        ledger is None
        or Path(ledger.store.path) != Path("/reports/release470-budget.sqlite3")
        or ledger.store.policy["case_plan"] != plan
        or values.get("plan") != plan
        or values.get("function") is not guard.__wrapped__
    ):
        raise IndexingLimit("Operator guard binding differs")
    return ledger
