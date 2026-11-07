"""Sequentially index the remaining pinned #388 snapshots in with-kb only."""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
from sqlalchemy import text

from tools.admission_index_one import claim_reports, require
from tools.admission_usage import UsageLedger


def progress(value: dict) -> None:
    """A lost SSH output channel must not interrupt a completed document."""
    try:
        print(json.dumps(value, ensure_ascii=False), flush=True)
    except BrokenPipeError:
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), sys.stdout.fileno())


def restore_after_disconnect(
    ledger: UsageLedger, directory: Path, recovery_path: Path
) -> set[str]:
    recovery = json.loads(recovery_path.read_text())
    payload = (directory / "usage.jsonl").read_bytes()
    rows = [
        json.loads(line) for line in (directory / "usage.jsonl").read_text().splitlines()
    ]
    require(
        recovery["status"] == "recovered"
        and recovery["prior_usage_sha256"] == hashlib.sha256(payload).hexdigest()
        and [row["attempt"] for row in rows] == list(range(1, len(rows) + 1))
        and all(row["http_status"] == 200 and row["usage_complete"] for row in rows)
        and len(recovery["completed_source_ids"]) == 6,
        "Prior batch requires manual recovery",
    )
    ledger.sequence = len(rows)
    ledger.operation_counts = dict(Counter(row["operation"] for row in rows))
    # Preserve a separate reserve for the unobserved interruption tail. This is
    # a spending guard, not a claim about provider billing or complete usage.
    ledger.reserved_deepseek_micro_cny = 5_000_000 + sum(
        row["usage"]["prompt_tokens"] * 2 + row["usage"]["completion_tokens"] * 8
        for row in rows
        if row["operation"] == "generation"
    )
    ledger.reserved_embedding_bytes = 5_000_000
    ledger.operation_counts["generation"] = (
        ledger.operation_counts.get("generation", 0) + 4
    )
    ledger.operation_counts["embedding"] = ledger.operation_counts.get("embedding", 0) + 1
    ledger.operation_counts["oauth"] = ledger.operation_counts.get("oauth", 0) + 1
    return set(recovery["completed_source_ids"])


def ready_sources(rows: list[dict], sources: list[dict]) -> set[str]:
    """Refuse partial, duplicate or foreign documents instead of replacing them."""
    by_url = {source["url"]: source for source in sources}
    ready = set()
    for row in rows:
        require(row["source_url"] in by_url, "Foreign evaluation document")
        source = by_url[row["source_url"]]
        require(row["source_url"] not in ready, "Duplicate evaluation document")
        require(
            row["content_hash"] == source["raw_sha256"]
            and row["graph_id"] == "abitur_kb"
            and row["last_indexed_at"] is not None
            and row["proposal_status"] == "approved"
            and row["proposal_document_id"] == row["id"],
            "Prior document is incomplete or differs from pinned source",
        )
        ready.add(row["source_url"])
    return ready


def read_sources(base: Path) -> list[dict]:
    manifest = json.loads((base / "batch-sources.json").read_text())
    registry = json.loads((base / "dataset/nsu_admission_2026_registry.json").read_text())
    sources = manifest["sources"]
    require(len(sources) == 9, "Expected nine sources")
    require({s["id"] for s in sources} == set(registry["sources"]), "Source IDs differ")
    for source in sources:
        registered = registry["sources"][source["id"]]
        require(
            source["url"] == registered["url"]
            and source["raw_sha256"] == registered["snapshot_sha256"]
            and source["filename"] == source["id"] + ".txt",
            "Source identity mismatch",
        )
        payload = (base / "inputs" / source["filename"]).read_bytes()
        require(
            hashlib.sha256(payload).hexdigest() == source["text_sha256"],
            "Text digest mismatch",
        )
        require(0 < len(payload) <= 150_000, "Unexpected source size")
    return sources


async def index_remaining(
    account: dict,
    ledger: UsageLedger,
    report: Path,
    expected_ready: set[str] | None = None,
) -> dict:
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    sys.path.insert(0, "/app")
    logging.disable(logging.CRITICAL)
    from api.main import app
    from config.instance import get_setting
    from db.postgres.db import engine

    require(
        engine.url.host == "postgres" and engine.url.database == "eval388",
        "Unexpected database",
    )
    require(
        "/abitur-eval388-with-kb_files/_data /app/data "
        in Path("/proc/self/mountinfo").read_text(),
        "Unexpected data mount",
    )
    base = Path("/evaluation")
    sources = read_sources(base)
    result: dict = {"status": "started", "sources": []}

    def save() -> None:
        result.update(
            attempts=ledger.sequence,
            uncertain_failure=ledger.uncertain_failure,
            peak_no_cache_reserve_cny=ledger.reserved_deepseek_micro_cny / 1e6,
            embedding_input_bytes=ledger.reserved_embedding_bytes,
        )
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    async def inspect() -> set[str]:
        async with engine.connect() as conn:
            for table in ("faq_entry", "user", "message", "message_logs"):
                require(
                    await conn.scalar(text(f'SELECT count(*) FROM "{table}"')) == 0,
                    "Unexpected user or FAQ data",
                )
            rows = (
                (
                    await conn.execute(
                        text(
                            "SELECT d.id, d.rag_doc_id, d.source_url, d.content_hash, "
                            "d.graph_id, "
                            "d.last_indexed_at, p.status AS proposal_status, "
                            "p.document_id AS proposal_document_id FROM document d "
                            "LEFT JOIN content_proposal p ON p.id=d.source_proposal_id"
                        )
                    )
                )
                .mappings()
                .all()
            )
            require(
                await conn.scalar(text("SELECT count(*) FROM content_proposal"))
                == len(rows),
                "Unexpected proposal state",
            )
        graph = json.loads(
            Path("/app/data/lightrag/abitur_kb/kv_store_doc_status.json").read_text()
        )
        require(
            set(graph) == {row["rag_doc_id"] for row in rows}
            and all(record["status"] == "processed" for record in graph.values()),
            "Graph and SQL differ",
        )
        return ready_sources([dict(row) for row in rows], sources)

    try:
        async with app.router.lifespan_context(app):
            for name, expected in {
                "LIGHTRAG_LLM_PROVIDER": "deepseek",
                "DEEPSEEK_GRAPH_MODEL": "deepseek-flash",
                "LIGHTRAG_EMBEDDING_PROVIDER": "gigachat",
                "GIGACHAT_LOCK_DIRECTORY": "/app/provider-locks",
                "LIGHTRAG_STORAGE": "json",
            }.items():
                require(get_setting(name) == expected, "Unexpected indexing settings")
            ready = await inspect()
            if expected_ready is None:
                require(len(ready) == 1, "Expected only the completed pilot source")
            else:
                require(
                    {source["id"] for source in sources if source["url"] in ready}
                    == expected_ready,
                    "SQL differs from completed sources in prior batch",
                )
            require(sources[0]["url"] in ready, "Completed pilot source differs")
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://eval388"
            ) as client:
                response = await client.post("/api/v1/auth/login", json=account)
                require(response.status_code == 200, "API authentication failed")
                client.headers["Authorization"] = (
                    "Bearer " + response.json()["access_token"]
                )
                account.clear()
                for source in sources:
                    if source["url"] in ready:
                        result["sources"].append(
                            {"id": source["id"], "status": "already_indexed"}
                        )
                        save()
                        continue
                    row = {"id": source["id"], "url": source["url"], "status": "started"}
                    result["sources"].append(row)
                    save()
                    progress(row)
                    item = {
                        "source": "parser",
                        "title": source["title"],
                        "text": (base / "inputs" / source["filename"])
                        .read_text()
                        .strip(),
                        "source_url": source["url"],
                        "source_hash": source["raw_sha256"],
                        "graph_id": "abitur_kb",
                    }
                    response = await client.post("/api/v1/content-proposals", json=item)
                    require(response.status_code == 201, "API upload failed")
                    row["proposal_id"] = response.json()["id"]
                    save()
                    started = time.monotonic()
                    async with asyncio.timeout(1800):
                        response = await client.post(
                            f"/api/v1/content-proposals/{row['proposal_id']}/approve"
                        )
                    row.update(
                        approval_http_status=response.status_code,
                        elapsed_seconds=round(time.monotonic() - started, 3),
                    )
                    require(
                        response.status_code == 200 and not ledger.uncertain_failure,
                        "Indexing failed; inspect before retry",
                    )
                    ready = await inspect()
                    require(source["url"] in ready, "Document not indexed")
                    row.update(
                        status="indexed", document_id=response.json()["document_id"]
                    )
                    save()
                    progress(row)
                require(len(ready) == 9, "Expected nine indexed documents")
                result["status"] = "completed"
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__)
    finally:
        account.clear()
        save()
    return result


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continue-after-disconnect", action="store_true")
    args = parser.parse_args()
    directory = Path(
        "/reports/remaining-eight-02"
        if args.continue_after_disconnect
        else "/reports/remaining-eight-01"
    )
    directory.mkdir(mode=0o750)
    report, usage = directory / "result.json", directory / "usage.jsonl"
    claim_reports(report, usage)
    account = json.loads(sys.stdin.read())
    ledger = UsageLedger(usage, max_attempts=8720, batch=True)
    try:
        expected = (
            restore_after_disconnect(
                ledger,
                Path("/reports/remaining-eight-01"),
                Path("/reports/recovery-math.json"),
            )
            if args.continue_after_disconnect
            else None
        )
        with ledger.capture():
            result = await index_remaining(account, ledger, report, expected)
    except Exception as exc:
        account.clear()
        result = {"status": "failed", "error_type": type(exc).__name__}
        report.write_text(json.dumps(result) + "\n")
    progress(result)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
