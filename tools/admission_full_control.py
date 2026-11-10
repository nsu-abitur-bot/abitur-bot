"""Finite host controller; starts only the established isolated CLI wrapper."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from tools.admission_full_plan import BUDGET, batch_ids, require, reserve

BASE = Path("/home/mihail/abitur-evals/388-20261107")
REPORTS = Path("/home/mihail/abitur-evals/388-reports")


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def validate_result(
    report: dict, usage: list[dict], intents: list[dict], ids: tuple
) -> None:
    require(report["status"] == "completed")
    require(report["selected_ids"] == list(ids))
    require([r["id"] for r in report["results"]] == list(ids))
    require(not report["unexecuted_ids"])
    require(len({r["attempt"] for r in intents}) == len(intents) == len(usage))
    require({r["attempt"] for r in intents} == {r["attempt"] for r in usage})
    require(report["accounting"]["attempts_started"] == len(intents))
    require(not report["accounting"]["uncertain_failure"])
    require(not report["accounting"]["halted"])
    reserve(usage)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    require(args.execute and re.fullmatch(r"full-[0-9]{8}-[0-9]{2}", args.run_id))
    os.umask(0o027)
    work = BASE / args.run_id
    output = REPORTS / f"{args.run_id}-controller"
    output.mkdir()
    (work / "plans").mkdir(exist_ok=True)
    (work / "evidence").mkdir(exist_ok=True)
    plan = json.loads((work / "base-plan.json").read_text())
    for name, sha in plan["instrumentation_files"].items():
        require(hashlib.sha256((BASE / name).read_bytes()).hexdigest() == sha)
    sessions = {
        c: list(plan["prior_state"][c]["log_sessions"]) for c in ("with-kb", "empty-kb")
    }
    evidence = []
    used = 0
    state: dict = {
        "status": "running",
        "run_id": args.run_id,
        "completed_batches": [],
        "budget_micro_cny": BUDGET,
    }
    save(output / "progress.json", state)
    try:
        for condition in ("with-kb", "empty-kb"):
            for index in range(5):
                ids = batch_ids(plan, index)
                current: dict = {
                    **plan,
                    "condition": condition,
                    "run_id": args.run_id,
                    "budget_evidence": list(evidence),
                    "remaining_micro_cny": BUDGET - used,
                    "prior_state": {
                        condition: {
                            "log_sessions": sessions[condition],
                            "log_count": None,
                        }
                    },
                }
                require(current["remaining_micro_cny"] > 0)
                name = f"{condition}-{index:02}"
                file = work / "plans" / f"{name}.json"
                with file.open("x") as stream:
                    json.dump(current, stream, ensure_ascii=False, indent=2)
                state.update(active_batch=name, used_micro_cny=used)
                save(output / "progress.json", state)
                command = [
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=120s",
                    "1700s",
                    "sudo",
                    "-n",
                    "/usr/local/sbin/abitur-eval388",
                    condition,
                    "cli-run",
                    "--profile",
                    "pilot",
                    "/evaluation/admission_full_batch_v2.py",
                    "--execute",
                    "--condition",
                    condition,
                    "--batch-index",
                    str(index),
                    "--run-id",
                    args.run_id,
                    "--plan",
                    f"/evaluation/{args.run_id}/plans/{name}.json",
                ]
                controller_file = output / f"{name}-controller.json"
                with controller_file.open("x") as stdout:
                    result = subprocess.run(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=subprocess.DEVNULL,
                    )
                require(result.returncode == 0)
                control = json.loads(controller_file.read_text())
                require(control["cleanup_ok"] and control["worker_exit_code"] == 0)
                folder = REPORTS / condition / f"{args.run_id}-batch-{index:02}"
                report = json.loads((folder / "report.json").read_text())
                raw = (folder / "usage.jsonl").read_bytes()
                usage = [json.loads(line) for line in raw.splitlines() if line]
                intents = [
                    json.loads(line)
                    for line in (folder / "intents.jsonl").read_bytes().splitlines()
                    if line
                ]
                validate_result(report, usage, intents, ids)
                used += reserve(usage)
                require(used <= BUDGET)
                saved = work / "evidence" / f"{name}.jsonl"
                with saved.open("xb") as stream:
                    stream.write(raw)
                evidence.append(
                    {
                        "path": f"evidence/{name}.jsonl",
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
                sessions[condition].extend(r["session_id"] for r in report["results"])
                state["completed_batches"].append(name)
                state.update(active_batch=None, used_micro_cny=used)
                save(output / "progress.json", state)
        state["status"] = "completed"
        return 0
    except Exception as exc:
        state.update(status="failed", error_type=type(exc).__name__)
        return 1
    finally:
        save(output / "progress.json", state)


if __name__ == "__main__":
    sys.exit(main())
