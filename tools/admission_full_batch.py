"""One immutable ten-case slice of the authorized 50+50 evaluation."""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

RUNTIME = Path("/evaluation/full-runtime-20261110")
if RUNTIME.is_dir():
    sys.path.insert(0, str(RUNTIME))

from tools.admission_full_plan import batch_ids, prior_cost, require  # noqa: E402
from tools.admission_pilot import (  # noqa: E402
    PilotLedger,
    Progress,
    digest,
    evaluate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--condition", choices=("with-kb", "empty-kb"), required=True)
    parser.add_argument("--batch-index", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    require(args.execute and re.fullmatch(r"full-[0-9]{8}-[0-9]{2}", args.run_id))
    require(args.plan.parent == Path("/evaluation") / args.run_id / "plans")
    plan = json.loads(args.plan.read_text())
    require(plan["condition"] == args.condition and plan["run_id"] == args.run_id)
    ids = batch_ids(plan, args.batch_index)
    for relative, expected in plan["instrumentation_files"].items():
        require(digest((Path("/evaluation") / relative).read_bytes()) == expected)
    prior = prior_cost(plan, args.plan.parent.parent)
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.umask(0o027)
    sys.path.insert(0, "/app")
    logging.disable(logging.CRITICAL)
    require(not args.inspect_only or args.batch_index == 0)
    label = (
        f"{args.run_id}-inspect"
        if args.inspect_only
        else f"{args.run_id}-batch-{args.batch_index:02}"
    )
    progress = Progress(
        Path("/reports") / label,
        args.condition,
        plan,
        case_ids=ids,
        full_run=not args.inspect_only,
    )
    progress.data.update(
        batch_index=args.batch_index,
        run_id=args.run_id,
        root_run_id=os.environ.get("EVAL388_RUN_ID"),
    )
    progress.data["inspect_only"] = args.inspect_only
    ledger = PilotLedger(
        progress.path.parent,
        prior_reserve_micro_cny=prior,
        max_attempts=512,
        inspect_only=args.inspect_only,
    )
    observer = threading.Thread(target=progress.observe, daemon=True)
    observer.start()
    code = 1
    try:
        from evals.schema import load_scenarios

        dataset = Path("/evaluation/dataset/nsu_admission_2026.yaml")
        require([s.id for s in load_scenarios(dataset)] == plan["case_ids"])
        with ledger.capture():
            asyncio.run(
                evaluate(
                    progress, ledger, plan, case_ids=ids, inspect_only=args.inspect_only
                )
            )
        code = 0
    except Exception as exc:
        progress.data["status"] = "failed"
        progress.data["error_type"] = type(exc).__name__
    finally:
        progress.stop.set()
        observer.join(timeout=2)
        progress.data["accounting"] = ledger.accounting()
        progress.data["finished_at_utc"] = datetime.now(UTC).isoformat()
        progress.data["unexecuted_ids"] = [
            i for i in ids if i not in {r["id"] for r in progress.data["results"]}
        ]
        progress.sample("finished")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
