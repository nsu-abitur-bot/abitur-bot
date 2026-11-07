"""Resume the interrupted math graph through LightRAG's normal pending queue."""

import asyncio
import gc
import hashlib
import json
import logging
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from tools.admission_graph_integrity import atomic_vector_writes, graph_integrity
from tools.admission_index_one import SafetyCheckFailed, claim_reports, require
from tools.admission_index_remaining import read_sources
from tools.admission_usage import UsageLedger


def carry_usage(ledger: UsageLedger, reports: Path = Path("/reports")) -> int:
    rows = []
    for name in ("remaining-eight-01/usage.jsonl", "vector-repair-01-usage.jsonl"):
        records = [json.loads(line) for line in (reports / name).read_text().splitlines()]
        require(
            all(row["http_status"] == 200 and row["usage_complete"] for row in records),
            "Prior usage needs review",
        )
        rows += records
    ledger.sequence = len(rows)
    ledger.operation_counts = dict(Counter(row["operation"] for row in rows))
    ledger.operation_counts["generation"] = (
        ledger.operation_counts.get("generation", 0) + 4
    )
    ledger.operation_counts["embedding"] = ledger.operation_counts.get("embedding", 0) + 1
    ledger.operation_counts["oauth"] = ledger.operation_counts.get("oauth", 0) + 1
    ledger.reserved_deepseek_micro_cny = 5_000_000 + sum(
        row["usage"]["prompt_tokens"] * 2 + row["usage"]["completion_tokens"] * 8
        for row in rows
        if row["operation"] == "generation"
    )
    repair = json.loads((reports / "vector-repair-01-result.json").read_text())
    require(
        repair["check"] == "Missing graph nodes" and not repair["uncertain_failure"],
        "Unexpected vector repair result",
    )
    ledger.reserved_embedding_bytes = 5_000_000 + repair["embedding_input_bytes"]
    return len(rows)


async def replay() -> dict:
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    sys.path.insert(0, "/app")
    logging.disable(logging.CRITICAL)
    from bootstrap.runtime_settings import refresh_settings
    from config.instance import get_setting
    from db.postgres.db import engine
    from rag.graph_memory import get_graph_memory

    require(
        engine.url.host == "postgres" and engine.url.database == "eval388",
        "Unexpected database",
    )
    require(
        "/abitur-eval388-with-kb_files/_data /app/data "
        in Path("/proc/self/mountinfo").read_text(),
        "Unexpected data mount",
    )
    await refresh_settings()
    require(
        get_setting("LIGHTRAG_LLM_PROVIDER") == "deepseek"
        and get_setting("DEEPSEEK_GRAPH_MODEL") == "deepseek-flash"
        and get_setting("LIGHTRAG_EMBEDDING_PROVIDER") == "gigachat",
        "Unexpected graph configuration",
    )
    sources = read_sources(Path("/evaluation"))
    root = Path("/app/data/lightrag/abitur_kb")
    identifier = "01a1b62e-d624-7c20-a21f-d3cecb390bce"
    source = next(s for s in sources if s["id"] == "DOC-770b0c6ebb172dbe")
    expected = (Path("/evaluation/inputs") / source["filename"]).read_text().strip()
    status = json.loads((root / "kv_store_doc_status.json").read_text())
    documents = json.loads((root / "kv_store_full_docs.json").read_text())
    require(
        len(status) == 6 and all(row["status"] == "processed" for row in status.values()),
        "Unexpected status records",
    )
    require(
        documents[identifier]["content"] == expected
        and status[identifier]["file_path"] == source["url"],
        "Math source differs",
    )
    backup = Path("/app/data/recovery388-graph-replay-01")
    backup.mkdir(mode=0o750)
    for name in (
        "kv_store_doc_status.json",
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json",
        "kv_store_full_entities.json",
        "kv_store_full_relations.json",
        "kv_store_entity_chunks.json",
        "kv_store_relation_chunks.json",
        "kv_store_llm_response_cache.json",
        "vdb_chunks.json",
        "vdb_entities.json",
        "vdb_relationships.json",
        "graph_chunk_entity_relation.graphml",
    ):
        shutil.copyfile(root / name, backup / name)
    memory = get_graph_memory()
    async with memory._use_graph("abitur_kb") as rag:
        record = await rag.doc_status.get_by_id(identifier)
        require(record["status"] == "processed", "Math status changed concurrently")
        await rag.doc_status.upsert({identifier: {**record, "status": "pending"}})
        await rag.doc_status.index_done_callback()
        await rag.apipeline_process_enqueue_documents()
        completed = await rag.doc_status.get_by_id(identifier)
        require(completed["status"] == "processed", "Math replay did not complete")
    await memory.cleanup("abitur_kb")
    gc.collect()
    integrity = graph_integrity(root, sources, Path("/evaluation/inputs"))
    require(integrity["documents"] == 6, "Replay changed source count")
    return {
        "status": "completed",
        "source_id": source["id"],
        "rag_doc_id": identifier,
        "integrity": integrity,
        "backup_directory": str(backup),
        "source_sha256": hashlib.sha256(expected.encode()).hexdigest(),
    }


async def main() -> int:
    report, usage = (
        Path("/reports/graph-replay-01-result.json"),
        Path("/reports/graph-replay-01-usage.jsonl"),
    )
    claim_reports(report, usage)
    ledger = UsageLedger(usage, max_attempts=8720, batch=True)
    prior = carry_usage(ledger)
    try:
        with ledger.capture(), atomic_vector_writes():
            async with asyncio.timeout(1800):
                result = await replay()
    except Exception as exc:
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "check": str(exc) if isinstance(exc, SafetyCheckFailed) else None,
        }
    result.update(
        prior_attempts=prior,
        attempts=ledger.sequence,
        uncertain_failure=ledger.uncertain_failure,
        embedding_input_bytes=ledger.reserved_embedding_bytes,
        peak_no_cache_reserve_cny=ledger.reserved_deepseek_micro_cny / 1e6,
    )
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
