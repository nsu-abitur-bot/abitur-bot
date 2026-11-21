"""Validate the approved pilot manifest, or explicitly initialize its fixed budget."""

import argparse
import hashlib
import json
from pathlib import Path

from tools.admission_usage import IndexingLimit
from tools.release470_cloudru import CloudCounter
from tools.release470_store import BudgetStore, canonical, cloud_policy, effective_policy

PLAN = Path("/reports/pilot-plan-approved-2026-11-21.json")
BUDGET = Path("/reports/release470-budget.sqlite3")
TOKENIZER = Path("/evaluation/cloudru-tokenizer.json")
CASES = {"faq-1", "faq-2", "rag-1", "rag-2", "tools-1", "tools-2"}


def validate_plan(plan: dict) -> tuple[dict, str]:
    if plan.get("status") != "owner_authorized_awaiting_operator_egress":
        raise IndexingLimit("An approved manifest is required")
    args = plan["budget_initialization"]
    if (
        set(args) != {"offline_only", "case_plan", "cloudru"}
        or args["offline_only"] is not False
    ):
        raise IndexingLimit("Unexpected initialization contract")
    questions, cases = plan["questions"], args["case_plan"]
    if set(questions) != CASES or set(cases) != CASES:
        raise IndexingLimit("Exactly six approved cases are required")
    expected = {
        key: hashlib.sha256(text.encode()).hexdigest() for key, text in questions.items()
    }
    if (
        cases != expected
        or plan["case_plan"] != expected
        or len(set(cases.values())) != 6
    ):
        raise IndexingLimit("Question hashes differ from the approved plan")
    cloud = cloud_policy(args["cloudru"])
    if (
        cloud["approved"] is not True
        or cloud["micro_rub_limit"] != 1000000
        or cloud["embedding_limit"] != 10000
        or cloud["rate_hundredths"] != 3222
    ):
        raise IndexingLimit("Cloud.ru limits differ from the owner authorization")
    policy = {**effective_policy(cloud), "offline_only": False, "case_plan": cases}
    digest = hashlib.sha256(canonical(policy).encode()).hexdigest()
    if digest != plan["expected_policy_sha256"]:
        raise IndexingLimit("Policy hash differs from the approved plan")
    return args, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize", action="store_true")
    options = parser.parse_args()
    if PLAN.is_symlink():
        raise IndexingLimit("A regular approved manifest is required")
    args, policy_hash = validate_plan(json.loads(PLAN.read_text(encoding="utf-8")))
    CloudCounter(TOKENIZER, args["cloudru"])
    if options.initialize:
        # Exclusive create only. An existing/uncertain budget is never reset.
        BudgetStore.create(BUDGET, **args)
    print(
        json.dumps({"policy_sha256": policy_hash, "budget_created": options.initialize})
    )


if __name__ == "__main__":
    main()
