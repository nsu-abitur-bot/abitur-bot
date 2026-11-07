"""Index one remaining source without importing a second FastAPI application."""

import argparse
import asyncio
import gc
import json
import logging
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
from sqlalchemy import text
from uuid6 import uuid7

from tools.admission_graph_integrity import atomic_vector_writes, graph_integrity
from tools.admission_index_one import SafetyCheckFailed, claim_reports, require
from tools.admission_index_remaining import read_sources, ready_sources
from tools.admission_usage import UsageLedger

REMAINING = ("DOC-83001f87d00915c7", "DOC-be8ca7262223f1f8", "DOC-de8ef06e37fe496b")
PREVIOUS = (
    "DOC-3d63df62af354868",
    "DOC-03ae446000cfbc45",
    "DOC-2338f0807b9bc328",
    "DOC-317c53d5b5ae2baf",
    "DOC-6674be71e05b972c",
    "DOC-770b0c6ebb172dbe",
)


def carry_previous(
    ledger: UsageLedger, source_id: str, reports: Path = Path("/reports")
) -> int:
    paths = [
        "remaining-eight-01/usage.jsonl",
        "vector-repair-01-usage.jsonl",
        "graph-replay-01-usage.jsonl",
        "remaining-eight-02/usage.jsonl",
    ]
    byte_upper_bound = 10_000_000
    for previous in REMAINING[: REMAINING.index(source_id)]:
        result = json.loads(
            (reports / f"remaining-eight-03/{previous}-result.json").read_text()
        )
        require(
            result["status"] == "completed" and not result["uncertain_failure"],
            "Prior source requires review",
        )
        paths.append(f"remaining-eight-03/{previous}-usage.jsonl")
        byte_upper_bound = result["embedding_input_bytes"]
    rows = []
    for name in paths:
        records = [json.loads(line) for line in (reports / name).read_text().splitlines()]
        require(
            all(row["http_status"] == 200 and row["usage_complete"] for row in records),
            "Prior usage requires review",
        )
        rows += records
    ledger.sequence = len(rows)
    ledger.operation_counts = dict(Counter(row["operation"] for row in rows))
    for name, reserve in (("generation", 8), ("embedding", 2), ("oauth", 2)):
        ledger.operation_counts[name] = ledger.operation_counts.get(name, 0) + reserve
    # Two interrupted processes have an unobserved tail. Reserve 5 CNY for each.
    ledger.reserved_deepseek_micro_cny = 10_000_000 + sum(
        row["usage"]["prompt_tokens"] * 2 + row["usage"]["completion_tokens"] * 8
        for row in rows
        if row["operation"] == "generation"
    )
    ledger.reserved_embedding_bytes = byte_upper_bound
    return len(rows)


def approval_ready(row: dict, source: dict, proposal_id: str, rag_id: str) -> bool:
    return bool(
        row["last_indexed_at"] is not None
        and row["content_hash"] == source["raw_sha256"]
        and row["source_url"] == source["url"]
        and row["source_proposal_id"] == proposal_id
        and row["rag_doc_id"] == rag_id
        and row["graph_id"] == "abitur_kb"
    )


async def index_source(
    account: dict, ledger: UsageLedger, source_id: str, report: Path
) -> dict:
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    sys.path.insert(0, "/app")
    logging.disable(logging.CRITICAL)
    from bootstrap.runtime_settings import refresh_settings
    from config.instance import get_setting
    from db.postgres.db import AsyncSessionLocal, engine
    from db.postgres.services.content_proposal import ContentProposalService
    from db.postgres.services.document import DocumentService

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
    for key, value in {
        "LIGHTRAG_LLM_PROVIDER": "deepseek",
        "DEEPSEEK_GRAPH_MODEL": "deepseek-flash",
        "LIGHTRAG_EMBEDDING_PROVIDER": "gigachat",
        "GIGACHAT_EMBEDDING_MODEL": "EmbeddingsGigaR",
    }.items():
        require(get_setting(key) == value, "Unexpected model configuration")
    sources = read_sources(Path("/evaluation"))
    source = next(s for s in sources if s["id"] == source_id)
    expected_ids = set(PREVIOUS + REMAINING[: REMAINING.index(source_id)])
    expected_urls = {s["url"] for s in sources if s["id"] in expected_ids}
    root = Path("/app/data/lightrag/abitur_kb")
    payload = (Path("/evaluation/inputs") / source["filename"]).read_text().strip()

    async def rows() -> list[dict]:
        async with engine.connect() as conn:
            for table in ("faq_entry", "user", "message", "message_logs"):
                require(
                    await conn.scalar(text(f'SELECT count(*) FROM "{table}"')) == 0,
                    "Unexpected evaluation data",
                )
            result = await conn.execute(
                text(
                    "SELECT d.id,d.rag_doc_id,d.source_url,d.content_hash,d.graph_id,"
                    "d.last_indexed_at,d.source_proposal_id,p.status AS proposal_status,"
                    "p.document_id AS proposal_document_id FROM document d "
                    "LEFT JOIN content_proposal p ON p.id=d.source_proposal_id"
                )
            )
            values = [dict(row) for row in result.mappings()]
            count = await conn.scalar(text("SELECT count(*) FROM content_proposal"))
            require(count == len(values), "Unexpected proposal count")
            return values

    existing = await rows()
    previous = [row for row in existing if row["source_url"] != source["url"]]
    require(
        ready_sources(previous, sources) == expected_urls,
        "Unexpected completed SQL sources",
    )
    require(len(existing) - len(previous) <= 1, "Duplicate source document")
    state = json.loads((root / "kv_store_doc_status.json").read_text())
    require(
        {row["file_path"] for row in state.values()} <= expected_urls | {source["url"]},
        "Unexpected graph source",
    )
    require(
        all(
            row["rag_doc_id"] in state
            and state[row["rag_doc_id"]]["file_path"] == row["source_url"]
            for row in previous
        ),
        "SQL graph mapping differs",
    )
    base_integrity = graph_integrity(
        root, sources, Path("/evaluation/inputs"), expected_ids
    )
    require(
        set(base_integrity["processed_source_ids"]) == expected_ids,
        "Prior graph incomplete",
    )

    # Administrative HTTP calls are outside provider metering. Create/pending
    # routes do not index. Approval is used only after the completed-row guard
    # establishes the application's no-indexing fast path.
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8000",
        trust_env=False,
        timeout=30,
        follow_redirects=False,
    ) as client:
        response = await client.post("/api/v1/auth/login", json=account)
        require(response.status_code == 200, "API authentication failed")
        client.headers["Authorization"] = "Bearer " + response.json()["access_token"]
        account.clear()
        response = await client.get("/api/v1/content-proposals")
        require(response.status_code == 200, "Admin access failed")
        pending = response.json()
        item = {
            "source": "parser",
            "title": source["title"],
            "text": payload,
            "source_url": source["url"],
            "source_hash": source["raw_sha256"],
            "graph_id": "abitur_kb",
        }
        if pending:
            require(
                len(pending) == 1
                and all(pending[0][key] == value for key, value in item.items()),
                "Unexpected pending proposal",
            )
            proposal_id = pending[0]["id"]
        else:
            require(
                not any(row["source_url"] == source["url"] for row in existing),
                "Source already has a SQL document",
            )
            response = await client.post("/api/v1/content-proposals", json=item)
            require(response.status_code == 201, "Proposal creation failed")
            proposal_id = response.json()["id"]
        report.write_text(
            json.dumps(
                {"status": "started", "source_id": source_id, "proposal_id": proposal_id}
            )
            + "\n"
        )

        async def process_graph(identifier: str, replay: bool) -> None:
            from rag.graph_memory import get_graph_memory

            memory = get_graph_memory()
            async with memory._use_graph("abitur_kb") as rag:
                if replay:
                    record = await rag.doc_status.get_by_id(identifier)
                    if record is None:
                        raise SafetyCheckFailed("Missing replay document")
                    await rag.doc_status.upsert(
                        {identifier: {**record, "status": "pending"}}
                    )
                    await rag.doc_status.index_done_callback()
                    await rag.apipeline_process_enqueue_documents()
                else:
                    await rag.ainsert(payload, ids=identifier, file_paths=source["url"])
                record = await rag.doc_status.get_by_id(identifier)
                require(
                    record is not None and record["status"] == "processed",
                    "Graph indexing incomplete",
                )
            await memory.cleanup("abitur_kb")
            del rag
            gc.collect()

        async with AsyncSessionLocal() as review_session:
            async with review_session.begin():
                proposal = await ContentProposalService(review_session).lock_pending(
                    proposal_id
                )
                if proposal is None:
                    raise SafetyCheckFailed("Proposal is not pending")
                require(
                    all(getattr(proposal, key) == value for key, value in item.items()),
                    "Locked proposal differs",
                )
                matching = [
                    identifier
                    for identifier, row in state.items()
                    if row["file_path"] == source["url"]
                ]
                require(len(matching) <= 1, "Duplicate graph source")
                rag_id = matching[0] if matching else str(uuid7())
                complete = False
                if matching:
                    docs = json.loads((root / "kv_store_full_docs.json").read_text())
                    require(docs[rag_id]["content"] == payload, "Graph text differs")
                    try:
                        integrity = graph_integrity(
                            root, sources, Path("/evaluation/inputs")
                        )
                        complete = set(
                            integrity["processed_source_ids"]
                        ) == expected_ids | {source_id}
                    except SafetyCheckFailed:
                        complete = False
                with ledger.capture(), atomic_vector_writes():
                    if not complete:
                        backup = Path("/app/data/recovery388-light") / source_id
                        backup.mkdir(parents=True, mode=0o750)
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
                        await process_graph(rag_id, bool(matching))
                    require(
                        not ledger.uncertain_failure,
                        "Inspect provider failure before approval",
                    )
                    integrity = graph_integrity(root, sources, Path("/evaluation/inputs"))
                    require(
                        set(integrity["processed_source_ids"])
                        == expected_ids | {source_id},
                        "Graph source count differs",
                    )
                async with AsyncSessionLocal() as session:
                    service = DocumentService(session)
                    document = await service.get_by_source_proposal_id(proposal_id)
                    if document is None:
                        document = await service.create_or_update_for_source(
                            graph_id="abitur_kb",
                            title=source["title"],
                            source_url=source["url"],
                            source_proposal_id=proposal_id,
                        )
                    require(
                        document.last_indexed_at is None
                        and document.source_url == source["url"]
                        and document.graph_id == "abitur_kb",
                        "Unexpected SQL document state",
                    )
                    document = await service.mark_indexed(
                        document.id,
                        content_hash=source["raw_sha256"],
                        content_length=len(payload),
                        title=source["title"],
                        source_url=source["url"],
                        rag_doc_id=rag_id,
                        source_proposal_id=proposal_id,
                    )
                    if document is None:
                        raise SafetyCheckFailed("Indexed SQL document missing")
                    document_id = document.id
                indexed = [row for row in await rows() if row["id"] == document_id]
                require(
                    len(indexed) == 1
                    and approval_ready(indexed[0], source, proposal_id, rag_id),
                    "Approval fast path is not established",
                )
        response = await client.post(f"/api/v1/content-proposals/{proposal_id}/approve")
        require(
            response.status_code == 200
            and response.json()["status"] == "approved"
            and response.json()["document_id"] == document_id,
            "API approval failed",
        )
    require(
        ready_sources(await rows(), sources) == expected_urls | {source["url"]},
        "Final SQL source state differs",
    )
    return {
        "status": "completed",
        "source_id": source_id,
        "document_id": document_id,
        "rag_doc_id": rag_id,
        "proposal_id": proposal_id,
        "mode": "link_persisted_graph" if complete else "lightweight_graph_indexing",
        "integrity": integrity,
        "admin_approval_http_status": 200,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", choices=REMAINING, required=True)
    args = parser.parse_args()
    directory = Path("/reports/remaining-eight-03")
    directory.mkdir(mode=0o750, exist_ok=True)
    report, usage = (
        directory / f"{args.source_id}-result.json",
        directory / f"{args.source_id}-usage.jsonl",
    )
    claim_reports(report, usage)
    account = json.loads(sys.stdin.read())
    ledger = UsageLedger(
        usage, max_attempts=8720, batch=True, embedding_byte_limit=20_000_000
    )
    prior = carry_previous(ledger, args.source_id)
    started = time.monotonic()
    result: dict
    try:
        async with asyncio.timeout(1800):
            result = await index_source(account, ledger, args.source_id, report)
    except Exception as exc:
        result = {
            "status": "failed",
            "source_id": args.source_id,
            "error_type": type(exc).__name__,
            "check": str(exc) if isinstance(exc, SafetyCheckFailed) else None,
        }
    finally:
        account.clear()
    result.update(
        prior_attempts=prior,
        attempts=ledger.sequence,
        uncertain_failure=ledger.uncertain_failure,
        embedding_input_bytes=ledger.reserved_embedding_bytes,
        peak_no_cache_reserve_cny=ledger.reserved_deepseek_micro_cny / 1e6,
        elapsed_seconds=round(time.monotonic() - started, 3),
        peak_process_rss_kib=next(
            (
                int(line.split()[1])
                for line in Path("/proc/self/status").read_text().splitlines()
                if line.startswith("VmHWM:")
            ),
            None,
        ),
    )
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
