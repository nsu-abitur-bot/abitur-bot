"""Pure validation for the explicitly authorized fifty-question comparison."""

import hashlib
import json
from pathlib import Path

BUDGET = 5_000_000
DATASET_SHA = "fa073f338684b73ee5947697c46b3fe876eccc9de1bb80ee3653b2ecbd3d2448"
IMAGE = "bfe4f8a088076f8b94056815ca27c613712e1a03"


def require(value: object) -> None:
    if not value:
        raise RuntimeError("Full evaluation validation failed")


def batch_ids(plan: dict, index: int) -> tuple[str, ...]:
    require(plan["mode"] == "full_50_comparison")
    require(plan["full_run_authorized"] is True)
    require(plan["dataset_sha256"] == DATASET_SHA)
    require(plan["image_revision"] == IMAGE)
    require(plan["budget_micro_cny"] == BUDGET)
    require(plan["batch_size"] == 10 and type(index) is int and 0 <= index < 5)
    ids = plan["case_ids"]
    require(len(ids) == len(set(ids)) == 50)
    require(set(plan["groups"]) == set(ids))
    require(list(plan["groups"].values()).count("document") == 45)
    require(list(plan["groups"].values()).count("special") == 5)
    return tuple(ids[index * 10 : (index + 1) * 10])


def reserve(rows: list[dict]) -> int:
    total = 0
    require(len({r["attempt"] for r in rows}) == len(rows))
    for row in rows:
        require(row["usage_complete"] and row["http_status"] == 200)
        require(row["error_type"] is None)
        if row["provider"] == "deepseek":
            u = row["usage"]
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                require(type(u[name]) is int and u[name] >= 0)
            require(u["prompt_tokens"] + u["completion_tokens"] == u["total_tokens"])
            total += u["prompt_tokens"] * 2 + u["completion_tokens"] * 8
    return total


def prior_cost(plan: dict, root: Path) -> int:
    total = 0
    seen = set()
    for evidence in plan.get("budget_evidence", []):
        relative = Path(evidence["path"])
        require(not relative.is_absolute() and ".." not in relative.parts)
        require(relative.parts[0] == "evidence" and relative.suffix == ".jsonl")
        require(str(relative) not in seen)
        seen.add(str(relative))
        raw = (root / relative).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == evidence["sha256"])
        total += reserve([json.loads(line) for line in raw.splitlines() if line])
    require(total < BUDGET)
    require(plan["remaining_micro_cny"] == BUDGET - total)
    return total
