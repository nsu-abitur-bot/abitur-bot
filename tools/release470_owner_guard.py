"""Additional owner ceilings; retain the installed operator accounting and gate."""

import inspect
from pathlib import Path

from tools.admission_usage import IndexingLimit

CAPS = {
    "input": 300000,
    "output": 10000,
    "embedding": 10000,
    "micro_cny": 3000000,
    "micro_rub": 1000000,
}
QUESTION = {"input": 32000, "output": 1024, "embedding": 512}


def check_reservation(state, case, operation, held):
    if state["halted"] or state["pending_or_uncertain"]:
        raise IndexingLimit("Unresolved pilot accounting")
    if state["attempts_started"] >= 512:
        raise IndexingLimit("Owner attempt ceiling reached")
    per = state["cases"].get(case, {})
    for kind, cap in CAPS.items():
        value = held.get(kind, 0)
        if type(value) is not int or value < 0:
            raise IndexingLimit("Invalid owner reservation")
        if state["totals"][kind] + value > cap:
            raise IndexingLimit("Owner cumulative ceiling reached")
        if case != "setup" and kind in QUESTION:
            if per.get(kind, 0) + value > QUESTION[kind]:
                raise IndexingLimit("Owner question ceiling reached")
    if case != "setup" and operation == "generation" and per.get("generations", 0) >= 40:
        raise IndexingLimit("Owner generation ceiling reached")


def attach_owner_guard(ledger):
    """reserve is called inside the operator's physical transport gate.

    Never installs HTTP hooks, creates a budget, edits a policy, or refunds usage.
    The original reserve remains responsible for its transaction and journal.
    """
    if getattr(ledger, "_owner_guard470", False):
        raise IndexingLimit("Owner guard is already attached")
    reserve = ledger.store.reserve
    output_limit = ledger.output_limit

    def bounded_reserve(case_id, operation, held, fingerprint, evidence):
        check_reservation(ledger.store.snapshot(), case_id, operation, held)
        return reserve(case_id, operation, held, fingerprint, evidence)

    def bounded_output(requested):
        state = ledger.store.snapshot()
        case = ledger._case.get()
        cap = 10000 if case == "setup" else 1024
        remaining = min(
            cap - state["cases"].get(case, {}).get("output", 0),
            10000 - state["totals"]["output"],
        )
        if remaining < 1:
            raise IndexingLimit("Owner output ceiling reached")
        return output_limit(min(requested, remaining))

    ledger.store.reserve = bounded_reserve
    ledger.output_limit = bounded_output
    ledger._owner_guard470 = True


def installed_ledger(guard, plan):
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
