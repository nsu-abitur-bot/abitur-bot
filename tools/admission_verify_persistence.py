"""Read-only verification of all nine persisted #388 evaluation sources."""

import asyncio
import hashlib
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import text

from tools.admission_graph_integrity import graph_integrity
from tools.admission_index_one import SafetyCheckFailed, require
from tools.admission_index_remaining import read_sources, ready_sources


async def verify() -> dict:
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    sys.path.insert(0, "/app")
    logging.disable(logging.CRITICAL)
    require(
        "/abitur-eval388-with-kb_files/_data /app/data "
        in Path("/proc/self/mountinfo").read_text(),
        "Unexpected data mount",
    )
    from db.postgres.db import engine

    require(
        engine.url.host == "postgres" and engine.url.database == "eval388",
        "Unexpected database",
    )
    sources = read_sources(Path("/evaluation"))
    root = Path("/app/data/lightrag/abitur_kb")
    integrity = graph_integrity(root, sources, Path("/evaluation/inputs"))
    require(
        set(integrity["processed_source_ids"]) == {source["id"] for source in sources},
        "Expected all nine sources",
    )
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    text(
                        "SELECT d.id,d.rag_doc_id,d.source_url,d.content_hash,d.graph_id,"
                        "d.last_indexed_at,p.status AS proposal_status,"
                        "p.document_id AS proposal_document_id FROM document d "
                        "JOIN content_proposal p ON p.id=d.source_proposal_id"
                    )
                )
            )
            .mappings()
            .all()
        )
        counts = {
            table: await conn.scalar(text(f'SELECT count(*) FROM "{table}"'))
            for table in (
                "document",
                "content_proposal",
                "faq_entry",
                "user",
                "message",
                "message_logs",
            )
        }
    require(
        counts["document"] == counts["content_proposal"] == 9
        and all(
            counts[name] == 0 for name in ("faq_entry", "user", "message", "message_logs")
        ),
        "Unexpected record counts",
    )
    require(
        ready_sources([dict(row) for row in rows], sources)
        == {source["url"] for source in sources},
        "SQL source hashes or approvals differ",
    )
    status = json.loads((root / "kv_store_doc_status.json").read_text())
    require(set(status) == {row["rag_doc_id"] for row in rows}, "Graph IDs differ")
    require(
        all(status[row["rag_doc_id"]]["file_path"] == row["source_url"] for row in rows),
        "SQL and graph source mapping differs",
    )
    registry = json.loads(
        Path("/evaluation/dataset/nsu_admission_2026_registry.json").read_text()
    )
    dataset_sha = hashlib.sha256(
        Path("/evaluation/dataset/nsu_admission_2026.yaml").read_bytes()
    ).hexdigest()
    require(dataset_sha == registry["dataset_sha256"], "Dataset digest differs")
    await engine.dispose()
    return {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "row_counts": counts,
        "integrity": integrity,
        "sql_source_hashes_verified": True,
        "dataset_sha256": dataset_sha,
    }


async def main() -> int:
    try:
        result = await verify()
        os.umask(0o027)
        with Path("/reports/graph-integrity-after-nine.json").open("x") as output:
            json.dump(result, output, ensure_ascii=False, indent=2)
            output.write("\n")
        with Path("/reports/runtime-after-nine.json").open("xb") as output:
            output.write(Path("/reports/runtime-after.json").read_bytes())
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "check": str(exc) if isinstance(exc, SafetyCheckFailed) else None,
                }
            )
        )
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
