"""Finish the SQL link for the exact processed math document in #388 only."""

import asyncio
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import httpx
from sqlalchemy import text

from tools.admission_index_one import claim_reports, require
from tools.admission_index_remaining import read_sources, ready_sources
from tools.admission_usage import UsageLedger


async def recover(account: dict) -> dict:
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    sys.path.insert(0, "/app")
    logging.disable(logging.CRITICAL)
    from api.main import app
    from db.postgres.db import AsyncSessionLocal, engine
    from db.postgres.services.document import DocumentService

    require(
        engine.url.host == "postgres" and engine.url.database == "eval388",
        "Unexpected database",
    )
    require(
        "/abitur-eval388-with-kb_files/_data /app/data "
        in Path("/proc/self/mountinfo").read_text(),
        "Unexpected mount",
    )
    source_id = "DOC-770b0c6ebb172dbe"
    document_id = "01a1b62e-d61d-751d-a1d8-d74b9f67da68"
    proposal_id = "01a1b62e-d58f-71db-b2a3-ba3fd47b12e2"
    rag_id = "01a1b62e-d624-7c20-a21f-d3cecb390bce"
    sources = read_sources(Path("/evaluation"))
    source = next(s for s in sources if s["id"] == source_id)
    expected = (Path("/evaluation/inputs") / source["filename"]).read_text().strip()
    root = Path("/app/data/lightrag/abitur_kb")
    graph = json.loads((root / "kv_store_doc_status.json").read_text())
    docs = json.loads((root / "kv_store_full_docs.json").read_text())
    chunks = json.loads((root / "kv_store_text_chunks.json").read_text())
    vector_ids = {
        row["__id__"]
        for row in json.loads((root / "vdb_chunks.json").read_text())["data"]
    }
    record = graph[rag_id]
    require(
        len(graph) == 6 and all(r["status"] == "processed" for r in graph.values()),
        "Graph not complete",
    )
    require(
        docs[rag_id]["content"] == expected and record["file_path"] == source["url"],
        "Processed text differs",
    )
    require(
        record["chunks_list"] and set(record["chunks_list"]) <= set(chunks) & vector_ids,
        "Missing chunks or vectors",
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://eval388"
        ) as client:
            response = await client.post("/api/v1/auth/login", json=account)
            require(response.status_code == 200, "Authentication failed")
            client.headers["Authorization"] = "Bearer " + response.json()["access_token"]
            account.clear()
            response = await client.get("/api/v1/content-proposals")
            require(response.status_code == 200, "Cannot inspect pending proposal")
            pending = response.json()
            require(
                len(pending) == 1 and pending[0]["id"] == proposal_id,
                "Unexpected pending proposal",
            )
            proposal = pending[0]
            require(
                proposal["text"] == expected
                and proposal["source_url"] == source["url"]
                and proposal["source_hash"] == source["raw_sha256"],
                "Proposal differs",
            )
            async with AsyncSessionLocal() as session:
                service = DocumentService(session)
                document = await service.get_by_id(document_id)
                if document is None:
                    raise RuntimeError("Missing evaluation document")
                require(
                    document is not None and document.last_indexed_at is None,
                    "Unexpected SQL state",
                )
                require(
                    document.source_proposal_id == proposal_id
                    and document.source_url == source["url"]
                    and document.graph_id == "abitur_kb",
                    "Unexpected SQL source",
                )
                rows = (
                    (
                        await session.execute(
                            text(
                                "SELECT d.id, d.source_url, d.content_hash, d.graph_id, "
                                "d.last_indexed_at, p.status AS proposal_status, "
                                "p.document_id AS proposal_document_id, d.rag_doc_id "
                                "FROM document d JOIN content_proposal p "
                                "ON p.id=d.source_proposal_id WHERE d.id != :id"
                            ),
                            {"id": document_id},
                        )
                    )
                    .mappings()
                    .all()
                )
                require(len(rows) == 5, "Unexpected SQL document count")
                ready_sources([dict(row) for row in rows], sources)
                require(
                    set(graph) - {row["rag_doc_id"] for row in rows} == {rag_id},
                    "Unexpected orphan graph record",
                )
                await service.mark_indexed(
                    document_id,
                    content_hash=source["raw_sha256"],
                    content_length=len(expected),
                    title=proposal["title"],
                    source_url=source["url"],
                    rag_doc_id=rag_id,
                    source_proposal_id=proposal_id,
                )
            response = await client.post(
                f"/api/v1/content-proposals/{proposal_id}/approve"
            )
            require(
                response.status_code == 200
                and response.json()["document_id"] == document_id,
                "API approval failed",
            )
    prior = Path("/reports/remaining-eight-01/usage.jsonl")
    return {
        "status": "recovered",
        "source_id": source_id,
        "document_id": document_id,
        "rag_doc_id": rag_id,
        "proposal_id": proposal_id,
        "completed_source_ids": [
            s["id"]
            for s in sources
            if s["url"] in {row["source_url"] for row in rows} | {source["url"]}
        ],
        "prior_usage_sha256": hashlib.sha256(prior.read_bytes()).hexdigest(),
        "interruption_tail_uncertain": True,
        "note": (
            "Processed text, URL, chunks and vectors matched. SQL metadata was "
            "linked and authenticated API approval completed. No reindexing "
            "or provider calls."
        ),
    }


async def main() -> int:
    report, usage = (
        Path("/reports/recovery-math.json"),
        Path("/reports/recovery-math-usage.jsonl"),
    )
    claim_reports(report, usage)
    account = json.loads(sys.stdin.read())
    try:
        with UsageLedger(usage, max_attempts=0).capture():
            result = await recover(account)
    except Exception as exc:
        result = {"status": "failed", "error_type": type(exc).__name__}
    finally:
        account.clear()
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "recovered" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
