"""Repair the interrupted #388 vector persistence using pinned Giga embeddings."""

import asyncio
import base64
import gc
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import zlib
from pathlib import Path
from typing import cast

import numpy as np

from tools.admission_graph_integrity import graph_integrity, matrix_valid, vector_metadata
from tools.admission_index_one import SafetyCheckFailed, claim_reports, require
from tools.admission_index_remaining import read_sources
from tools.admission_usage import UsageLedger


async def repair(ledger: UsageLedger) -> dict:
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    sys.path.insert(0, "/app")
    logging.disable(logging.CRITICAL)
    from nano_vectordb import NanoVectorDB
    from nano_vectordb.dbs import Data

    from bootstrap.runtime_settings import refresh_settings
    from config.instance import get_setting
    from llm.providers.gigachat import GigaChatEmbedding

    require(
        "/abitur-eval388-with-kb_files/_data /app/data "
        in Path("/proc/self/mountinfo").read_text(),
        "Unexpected data mount",
    )
    await refresh_settings()
    require(
        get_setting("LIGHTRAG_EMBEDDING_PROVIDER") == "gigachat"
        and get_setting("GIGACHAT_EMBEDDING_MODEL") == "EmbeddingsGigaR",
        "Unexpected embedding settings",
    )
    root = Path("/app/data/lightrag/abitur_kb")
    backup = Path("/app/data/recovery388-vectors-01")
    backup.mkdir(mode=0o750)
    sources = read_sources(Path("/evaluation"))
    status = json.loads((root / "kv_store_doc_status.json").read_text())
    documents = json.loads((root / "kv_store_full_docs.json").read_text())
    by_url = {source["url"]: source for source in sources}
    require(set(status) == set(documents), "Unexpected full document records")
    for identifier, row in status.items():
        source = by_url[row["file_path"]]
        require(
            documents[identifier]["content"]
            == (Path("/evaluation/inputs") / source["filename"]).read_text().strip(),
            "Graph source differs",
        )
    chunks = json.loads((root / "kv_store_text_chunks.json").read_text())
    require(
        len(status) == 6 and all(row["status"] == "processed" for row in status.values()),
        "Unexpected document state",
    )
    required_chunks = set().union(*(set(row["chunks_list"]) for row in status.values()))
    changes = []
    adapter = GigaChatEmbedding(model="EmbeddingsGigaR")
    try:
        for kind in ("entities", "relationships", "chunks"):
            path = root / f"vdb_{kind}.json"
            original = path.read_bytes()
            digest = hashlib.sha256(original).hexdigest()
            shutil.copyfile(path, backup / path.name)
            metadata = vector_metadata(original.decode())
            require(
                metadata["embedding_dim"] == adapter.embedding_dim == 2560,
                "Unexpected dimension",
            )
            try:
                valid = matrix_valid(json.loads(original))
            except json.JSONDecodeError:
                valid = False
            existing = {row["__id__"] for row in metadata["data"]}
            require(len(existing) == len(metadata["data"]), "Duplicate vector IDs")
            entries = [] if valid else metadata["data"]
            if kind == "chunks":
                require(valid, "Chunk store must have a valid prior matrix")
                for identifier in required_chunks - existing:
                    row = chunks[identifier]
                    require(
                        row["full_doc_id"] in status
                        and row["content"] in documents[row["full_doc_id"]]["content"],
                        "Chunk content differs",
                    )
                    entries.append(
                        {
                            "__id__": identifier,
                            "__created_at__": int(time.time()),
                            "content": row["content"],
                            "full_doc_id": row["full_doc_id"],
                            "file_path": row["file_path"],
                        }
                    )
            require(
                len(entries) <= 2048
                and sum(len(row["content"].encode()) for row in entries) <= 2_000_000,
                "Unexpected repair size",
            )
            if kind == "entities":
                expected_entities = set().union(
                    *(
                        set(row["entity_names"])
                        for row in json.loads(
                            (root / "kv_store_full_entities.json").read_text()
                        ).values()
                    )
                )
                require(
                    all(
                        row["entity_name"] in expected_entities
                        and row["__id__"]
                        == "ent-" + hashlib.md5(row["entity_name"].encode()).hexdigest()
                        for row in entries
                    ),
                    "Unexpected entity record",
                )
            if not entries:
                changes.append(
                    {"store": kind, "status": "unchanged", "rows": len(existing)}
                )
                continue
            temporary = root / f"vdb_{kind}.repair388-tmp"
            require(not temporary.exists(), "Prior repair temp exists")
            if valid:
                temporary.write_bytes(original)
            store = NanoVectorDB(2560, storage_file=str(temporary))
            vectors = np.asarray(
                await adapter([row["content"] for row in entries]), dtype=np.float32
            )
            require(
                vectors.shape == (len(entries), 2560) and np.isfinite(vectors).all(),
                "Invalid embeddings",
            )
            records = []
            for row, vector in zip(entries, vectors, strict=True):
                records.append(
                    {
                        **row,
                        "vector": base64.b64encode(
                            zlib.compress(vector.astype(np.float16).tobytes())
                        ).decode(),
                        "__vector__": vector,
                    }
                )
            store.upsert(cast(list[Data], records))
            store.save()
            updated = json.loads(temporary.read_text())
            require(matrix_valid(updated), "Repaired matrix invalid")
            expected = existing | {row["__id__"] for row in entries}
            require(
                {row["__id__"] for row in updated["data"]} == expected,
                "Repair changed record identities",
            )
            require(
                hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                "Concurrent vector change",
            )
            with temporary.open("rb") as file:
                os.fsync(file.fileno())
            os.replace(temporary, path)
            changes.append(
                {
                    "store": kind,
                    "status": "repaired",
                    "embedded_rows": len(entries),
                    "rows": len(expected),
                    "before_sha256": digest,
                    "after_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            del vectors, records, updated, store, metadata, original
            gc.collect()
    finally:
        await adapter.aclose()
    integrity = graph_integrity(root, sources, Path("/evaluation/inputs"))
    return {
        "status": "completed",
        "stores": changes,
        "integrity": integrity,
        "attempts": ledger.sequence,
        "embedding_input_bytes": ledger.reserved_embedding_bytes,
        "uncertain_failure": ledger.uncertain_failure,
        "backup_directory": str(backup),
    }


async def main() -> int:
    report, usage = (
        Path("/reports/vector-repair-01-result.json"),
        Path("/reports/vector-repair-01-usage.jsonl"),
    )
    claim_reports(report, usage)
    ledger = UsageLedger(usage, max_attempts=256, batch=True)
    ledger.operation_counts["generation"] = 512
    try:
        with ledger.capture():
            async with asyncio.timeout(900):
                result = await repair(ledger)
    except Exception as exc:
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "check": str(exc) if isinstance(exc, SafetyCheckFailed) else None,
            "attempts": ledger.sequence,
            "embedding_input_bytes": ledger.reserved_embedding_bytes,
            "uncertain_failure": ledger.uncertain_failure,
        }
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
