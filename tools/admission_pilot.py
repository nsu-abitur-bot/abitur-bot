"""Only the fixed ten-case pilot. No full-dataset execution mode exists here."""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evals.schema import Scenario

RUNTIME = Path("/evaluation/pilot-runtime-20261110-v2")
if RUNTIME.is_dir():
    sys.path.insert(0, str(RUNTIME))

from tools.admission_evaluation_usage import EvaluationUsageLedger  # noqa: E402
from tools.admission_usage import IndexingLimit  # noqa: E402

CASE_IDS = (
    "nsu-pre-flow-01",
    "nsu-pre-scholar-01",
    "nsu-pre-compare-01",
    "nsu-pre-compare-02",
    "nsu-post-dorm-03",
    "nsu-pre-building-03",
    "nsu-paid-steps-01",
    "nsu-pre-general-02",
    "nsu-pre-compare-04",
    "nsu-pre-chance-01",
)
DATASET_SHA = "fa073f338684b73ee5947697c46b3fe876eccc9de1bb80ee3653b2ecbd3d2448"
IMAGE_REVISION = "bfe4f8a088076f8b94056815ca27c613712e1a03"
CASE_TIMEOUT = 90


def require(value: object) -> None:
    if not value:
        raise RuntimeError("Pilot safety check failed")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def object_digest(value: object) -> str:
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True).encode())


def validate_plan(plan: dict) -> None:
    require(plan["mode"] == "ten_case_pilot")
    require(plan["case_ids"] == list(CASE_IDS))
    require(plan["dataset_sha256"] == DATASET_SHA)
    require(plan["image_revision"] == IMAGE_REVISION)
    require(plan["full_run_authorized"] is False)


class PilotLedger(EvaluationUsageLedger):
    def __init__(
        self,
        directory: Path,
        *,
        inspect_only: bool = False,
        prior_reserve_micro_cny: int = 0,
    ) -> None:
        require(
            type(prior_reserve_micro_cny) is int
            and 0 <= prior_reserve_micro_cny < 5_000_000
        )
        super().__init__(
            directory / "usage.jsonl",
            batch=True,
            max_attempts=0 if inspect_only else 256,
            embedding_byte_limit=1_000_000,
            deepseek_ceiling_micro_cny=5_000_000 - prior_reserve_micro_cny,
        )
        self.intent_path = directory / "intents.jsonl"
        self.path.touch(exist_ok=False)
        self.intent_path.touch(exist_ok=False)
        self.phase = "setup"
        self.case_id = "setup"
        self.completed = 0
        self.halted = False
        self.prior_reserve_micro_cny = prior_reserve_micro_cny

    def context(self, phase: str, case_id: str) -> None:
        require(
            self.completed == self.sequence
            and not self.uncertain_failure
            and not self.halted
        )
        self.phase, self.case_id = phase, case_id

    def begin(self, request) -> dict:
        if self.halted:
            raise IndexingLimit("Pilot was halted")
        try:
            row = super().begin(request)
        except IndexingLimit:
            self.halted = True
            raise
        row.update(phase=self.phase, case_id=self.case_id)
        with self.lock:
            with self.intent_path.open("a", encoding="utf-8") as output:
                json.dump(
                    {k: v for k, v in row.items() if k != "started_monotonic"}, output
                )
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
        return row

    def finish(self, row, response, error) -> None:
        super().finish(row, response, error)
        with self.lock:
            self.completed += 1

    def accounting(self) -> dict:
        return {
            "attempts_started": self.sequence,
            "attempts_finished": self.completed,
            "operation_counts": dict(self.operation_counts),
            "uncertain_failure": self.uncertain_failure,
            "halted": self.halted,
            "prior_reserve_micro_cny": self.prior_reserve_micro_cny,
            "reserved_peak_no_cache_micro_cny": self.reserved_deepseek_micro_cny,
            "embedding_input_bytes": self.reserved_embedding_bytes,
        }


class Progress:
    def __init__(self, directory: Path, condition: str, plan: dict) -> None:
        directory.mkdir()
        self.path = directory / "report.json"
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.data: dict = {
            "schema_version": 1,
            "mode": "live_pilot",
            "status": "started",
            "condition": condition,
            "dataset_sha256": DATASET_SHA,
            "plan_sha256": object_digest(plan),
            "image_revision": IMAGE_REVISION,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "total": 10,
            "selected_ids": list(CASE_IDS),
            "results": [],
            "calibration": [],
            "phases": [],
            "phase": "preflight",
            "case_timeout_seconds": CASE_TIMEOUT,
            "concurrency": 1,
            "case_retries": 0,
            "full_run_started": False,
            "peak_sampled_swap_bytes": 0,
            "min_host_available_kib": 2**63,
        }
        self.save()

    def save(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, self.path)

    def sample(self, phase: str | None = None) -> dict:
        from tools.admission_memory_probe import snapshot

        value = snapshot()
        with self.lock:
            self.data["latest_memory"] = value
            if phase:
                self.data["phase"] = phase
                self.data["phases"].append({"phase": phase, **value})
            self.data["peak_sampled_swap_bytes"] = max(
                self.data["peak_sampled_swap_bytes"], value["memory.swap.current"]
            )
            self.data["min_host_available_kib"] = min(
                self.data["min_host_available_kib"], value["host_available_kib"]
            )
            if value["host_available_kib"] < 300 * 1024:
                self.data["status"] = "aborted_low_host_memory"
                self.save()
                os._exit(75)
            self.save()
        return value

    def observe(self) -> None:
        while not self.stop.wait(0.5):
            try:
                self.sample()
            except Exception:
                os._exit(76)


async def inspect_condition(
    condition: str, *, prior_log_sessions: tuple[str, ...] = (), prior_log_count: int = 0
) -> dict:
    from sqlalchemy import bindparam, text

    from db.postgres.db import engine

    require(engine.url.host == "postgres" and engine.url.database == "eval388")
    mount = f"/abitur-eval388-{condition}_files/_data /app/data "
    require(mount in Path("/proc/self/mountinfo").read_text())
    async with engine.connect() as connection:
        counts = {
            table: await connection.scalar(text(f'SELECT count(*) FROM "{table}"'))
            for table in (
                "document",
                "content_proposal",
                "faq_entry",
                "user",
                "message",
                "message_logs",
            )
        }
        require(all(counts[t] == 0 for t in ("faq_entry", "user", "message")))
        require(counts["message_logs"] == prior_log_count)
        if prior_log_count:
            require(bool(prior_log_sessions))
            known_logs = await connection.scalar(
                text(
                    "SELECT count(*) FROM message_logs WHERE session_id IN :sessions"
                ).bindparams(bindparam("sessions", expanding=True)),
                {"sessions": list(prior_log_sessions)},
            )
            require(known_logs == prior_log_count)
        rows = [
            dict(row)
            for row in (
                await connection.execute(
                    text(
                        "SELECT d.id,d.rag_doc_id,d.source_url,d.content_hash,d.graph_id,"
                        "d.last_indexed_at,p.status AS proposal_status,"
                        "p.document_id AS proposal_document_id "
                        "FROM document d JOIN content_proposal p "
                        "ON p.id=d.source_proposal_id"
                    )
                )
            ).mappings()
        ]
    from db.redis.client import RedisClient

    redis = RedisClient()
    try:
        require(await redis.client.dbsize() == 0)
    finally:
        await redis.client.aclose()
    root = Path("/app/data/lightrag")
    statuses = root / "abitur_kb/kv_store_doc_status.json"
    records = json.loads(statuses.read_text()) if statuses.exists() else {}
    if condition == "with-kb":
        sources = json.loads(Path("/evaluation/batch-sources.json").read_text())[
            "sources"
        ]
        require(counts["document"] == counts["content_proposal"] == len(records) == 9)
        by_url = {source["url"]: source for source in sources}
        require({row["source_url"] for row in rows} == set(by_url))
        for row in rows:
            source = by_url[row["source_url"]]
            require(row["content_hash"] == source["raw_sha256"])
            require(row["graph_id"] == "abitur_kb" and row["last_indexed_at"] is not None)
            require(
                row["proposal_status"] == "approved"
                and row["proposal_document_id"] == row["id"]
            )
            require(
                digest((Path("/evaluation/inputs") / source["filename"]).read_bytes())
                == source["text_sha256"]
            )
        require(set(records) == {row["rag_doc_id"] for row in rows})
        require(all(value["status"] == "processed" for value in records.values()))
        for row in rows:
            require(records[row["rag_doc_id"]]["file_path"] == row["source_url"])
    else:
        require(counts["document"] == counts["content_proposal"] == 0)
        require(not records)
        for file in root.glob("**/*.json"):
            require(not file.is_symlink())
            data = json.loads(file.read_text())
            require(not (data.get("data") if file.name.startswith("vdb_") else data))
        for file in root.glob("**/*.graphml"):
            from xml.etree import ElementTree

            require(
                not list(
                    ElementTree.parse(file).iter(
                        "{http://graphml.graphdrawing.org/xmlns}node"
                    )
                )
            )
    manifest = {
        "counts": counts,
        "documents": [
            {k: str(v) if v is not None else None for k, v in row.items()} for row in rows
        ],
        "graph_documents": len(records),
        "redis_keys": 0,
    }
    return {"manifest": manifest, "sha256": object_digest(manifest)}


def calibrations(scenarios: dict[str, Scenario]) -> list[tuple[str, Scenario, str, bool]]:
    fact = scenarios["nsu-pre-flow-02"]
    price = scenarios["nsu-paid-steps-01"]
    choice = scenarios["nsu-pre-compare-04"]
    return [
        ("correct", fact, fact.reference_answer, True),
        (
            "wrong-category",
            price,
            price.reference_answer
            + " Это гарантированно действует для всех курсов и гражданств "
            "в 2030/31 году.",
            False,
        ),
        ("referral", fact, "Уточните в приёмной комиссии.", False),
        (
            "decision",
            choice,
            "Выбирай ПИиКН. Это лучший вариант для тебя, "
            "остальные варианты рассматривать не нужно.",
            False,
        ),
    ]


async def evaluate(
    progress: Progress, ledger: PilotLedger, plan: dict, *, inspect_only: bool = False
) -> None:
    from bootstrap.runtime_settings import refresh_settings
    from config.instance import SPECS, get_setting
    from db.postgres.db import engine

    try:
        await refresh_settings()
        require(get_setting("LLM_PROVIDER") == "deepseek")
        require(get_setting("DEEPSEEK_MODEL") == "deepseek-flash")
        require(get_setting("LIGHTRAG_EMBEDDING_PROVIDER") == "gigachat")
        require(get_setting("GIGACHAT_EMBEDDING_MODEL") == "EmbeddingsGigaR")
        require(get_setting("LIGHTRAG_STORAGE", "json") == "json")
        for relative, expected in plan["application_files"].items():
            require(digest((Path("/app") / relative).read_bytes()) == expected)
        for relative, expected in plan["instrumentation_files"].items():
            require(digest((Path("/evaluation") / relative).read_bytes()) == expected)
        condition = progress.data["condition"]
        prior_state = plan.get("prior_state", {}).get(condition, {})
        kb = await inspect_condition(
            condition,
            prior_log_sessions=tuple(prior_state.get("log_sessions", [])),
            prior_log_count=prior_state.get("log_count", 0),
        )
        config = {
            name: get_setting(name, spec.default)
            for name, spec in SPECS.items()
            if not spec.secret
        }
        progress.data.update(
            configuration_sha256=object_digest(
                {
                    "settings": config,
                    "code": plan["application_files"],
                    "case_timeout": CASE_TIMEOUT,
                }
            ),
            kb_manifest_sha256=kb["sha256"],
            knowledge_manifest=kb["manifest"],
            providers={
                key: get_setting(key)
                for key in (
                    "LLM_PROVIDER",
                    "DEEPSEEK_MODEL",
                    "DEEPSEEK_GRAPH_MODEL",
                    "LIGHTRAG_LLM_PROVIDER",
                    "LIGHTRAG_EMBEDDING_PROVIDER",
                    "GIGACHAT_EMBEDDING_MODEL",
                    "GIGACHAT_SCOPE",
                )
            },
        )
        from evals.evaluator import PipelineEvaluator
        from evals.judge import evaluate_rag_answer
        from evals.product import prepare_product, run_product
        from evals.schema import load_scenarios

        path = Path("/evaluation/dataset/nsu_admission_2026.yaml")
        require(digest(path.read_bytes()) == DATASET_SHA)
        scenarios = load_scenarios(path)
        require(len(scenarios) == 50)
        by_id = {s.id: s for s in scenarios}
        require(set(CASE_IDS) <= set(by_id))
        progress.sample("preflight_complete")
        if plan.get("expected_configuration_sha256"):
            require(
                progress.data["configuration_sha256"]
                == plan["expected_configuration_sha256"]
            )
        if inspect_only:
            progress.data["status"] = "inspected"
            return
        gc.collect()
        await prepare_product()
        for name, scenario, answer, expected in calibrations(by_id):
            ledger.context("calibration", name)
            started = time.monotonic()
            async with asyncio.timeout(50):
                result = await evaluate_rag_answer(
                    question=scenario.question,
                    reference=scenario.reference_answer,
                    criteria=scenario.judge_criteria or "",
                    actual_answer=answer,
                )
            entry = {
                "id": name,
                "expected_pass": expected,
                "matches": (result["score"] >= scenario.min_score) == expected,
                "score": result["score"],
                "reasoning": result["reasoning"],
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            with progress.lock:
                progress.data["calibration"].append(entry)
                progress.data["accounting"] = ledger.accounting()
                progress.save()
            require(entry["matches"])
        ledger.context("answer", "setup")
        progress.sample("calibration_completed")
        for case_id in CASE_IDS:
            scenario = by_id[case_id]
            before = progress.sample(case_id)
            started = time.monotonic()

            async def answer_case(case, session_id):
                ledger.context("answer", case.id)
                return await run_product(case, session_id)

            async def judge_case(**kwargs):
                ledger.context("judge", case_id)
                return await evaluate_rag_answer(**kwargs)

            evaluator = PipelineEvaluator(
                path,
                runner=answer_case,
                judge=judge_case,
                concurrency=1,
                timeout=CASE_TIMEOUT,
            )
            evaluator.scenarios = [scenario]
            report = await evaluator.run()
            row = report["results"][0]
            row.update(
                elapsed_seconds=round(time.monotonic() - started, 3),
                memory_before=before,
                memory_after=progress.sample(),
            )
            with progress.lock:
                progress.data["results"].append(row)
                progress.data["accounting"] = ledger.accounting()
                progress.save()
            require(
                not ledger.uncertain_failure
                and not ledger.halted
                and ledger.sequence == ledger.completed
            )
            gc.collect()
        progress.data["status"] = "completed"
    finally:
        module = sys.modules.get("rag.graph_memory")
        memory = getattr(module, "_graph_memory_instance", None)
        if memory is not None:
            await memory.cleanup()
        await engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--condition", choices=("with-kb", "empty-kb"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    require(args.execute and re.fullmatch(r"pilot-[0-9]{8}-[0-9]{2}", args.label))
    require(
        args.plan.parent == Path("/evaluation")
        and args.plan.name.startswith("pilot-plan-")
    )
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.umask(0o027)
    sys.path.insert(0, "/app")
    logging.disable(logging.CRITICAL)
    progress = Progress(Path("/reports") / args.label, args.condition, plan)
    progress.data["inspect_only"] = args.inspect_only
    progress.data["root_run_id"] = os.environ.get("EVAL388_RUN_ID")
    prior = plan.get("prior_reserve_micro_cny", {}).get(args.condition, 0)
    if prior:
        records = plan.get("prior_usage_files", {}).get(args.condition)
        if records:
            for record in records:
                require(
                    re.fullmatch(
                        r"(?:pilot|stream-diagnostic)-[0-9]{8}-[0-9]{2}/usage.jsonl",
                        record["path"],
                    )
                )
                require(
                    digest((Path("/reports") / record["path"]).read_bytes())
                    == record["sha256"]
                )
        else:
            require(args.condition == "with-kb")
            require(
                digest(Path("/reports/pilot-20261110-01/usage.jsonl").read_bytes())
                == plan["prior_usage_sha256"]
            )
    ledger = PilotLedger(
        progress.path.parent,
        inspect_only=args.inspect_only,
        prior_reserve_micro_cny=prior,
    )
    observer = threading.Thread(target=progress.observe, daemon=True)
    observer.start()
    code = 1
    try:
        with ledger.capture():
            asyncio.run(evaluate(progress, ledger, plan, inspect_only=args.inspect_only))
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
            i for i in CASE_IDS if i not in {r["id"] for r in progress.data["results"]}
        ]
        progress.sample("finished")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
