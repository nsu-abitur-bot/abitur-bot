"""Index one fixed public snapshot through authenticated application routes."""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

import httpx
from sqlalchemy import text

from tools.admission_usage import UsageLedger


class SafetyCheckFailed(RuntimeError):
    """The pilot cannot establish its required isolation or initial state."""


def require(condition: object, message: str) -> None:
    """Keep safety checks active even when Python is run with optimization."""
    if not condition:
        raise SafetyCheckFailed(message)


def claim_reports(report: Path, usage_path: Path) -> None:
    """Create permanent run markers before any login or provider call."""
    require(not usage_path.exists(), "Inspect prior usage before retry")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    # Preserve the administrator's inherited read ACL for the host report owner.
    # These files contain only sanitized result/usage, never credentials.
    descriptor = os.open(report, flags, 0o640)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write('{"status": "started"}\n')
    descriptor = os.open(usage_path, flags, 0o640)
    os.close(descriptor)


def pending_resume(records: list[dict], item: dict, proposal_id: str) -> str:
    """Only resume the same pinned, pending source after inspecting the prior run."""
    matching = [record for record in records if record["id"] == proposal_id]
    require(len(matching) == 1, "Expected pending proposal not found")
    record = matching[0]
    require(record["status"] == "pending", "Proposal already completed")
    require(
        all(record[name] == item[name] for name in item),
        "Pending proposal differs from pinned source",
    )
    return proposal_id


async def index_one(
    account: dict, ledger: UsageLedger, resume: str | None = None
) -> dict:
    """Run the API upload/approval with real authentication and real providers."""
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/evaluation")
    logging.disable(logging.CRITICAL)
    from api.main import app
    from config.instance import get_setting
    from db.postgres.db import engine

    require(
        engine.url.host == "postgres" and engine.url.database == "eval388",
        "Unexpected database",
    )
    mounts = Path("/proc/self/mountinfo").read_text()
    require(
        "/abitur-eval388-with-kb_files/_data /app/data " in mounts,
        "Unexpected data mount",
    )
    source_path = Path("/evaluation/inputs/nsu-plus.txt")
    source_bytes = source_path.read_bytes()
    manifest = json.loads(Path("/evaluation/manifest.json").read_text())
    require(
        hashlib.sha256(source_bytes).hexdigest()
        == manifest["source"]["extracted_utf8_sha256"],
        "Source digest mismatch",
    )
    async with app.router.lifespan_context(app):
        for name, expected in {
            "LIGHTRAG_LLM_PROVIDER": "deepseek",
            "DEEPSEEK_GRAPH_MODEL": "deepseek-flash",
            "LIGHTRAG_EMBEDDING_PROVIDER": "gigachat",
            "GIGACHAT_LOCK_DIRECTORY": "/app/provider-locks",
        }.items():
            require(get_setting(name) == expected, "Unexpected indexing configuration")
        async with engine.connect() as connection:
            for table in ("document", "faq_entry", "content_proposal"):
                maximum = 1 if resume and table in {"document", "content_proposal"} else 0
                require(
                    await connection.scalar(text(f'SELECT count(*) FROM "{table}"'))
                    <= maximum,
                    "Unexpected records in the evaluation instance",
                )
            if resume:
                rows = (
                    await connection.execute(
                        text("SELECT source_proposal_id, last_indexed_at FROM document")
                    )
                ).all()
                require(
                    all(row[0] == resume and row[1] is None for row in rows),
                    "Resume would replace a different or completed document",
                )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://eval388"
        ) as client:
            response = await client.post("/api/v1/auth/login", json=account)
            require(response.status_code == 200, "Test API authentication failed")
            token = response.json()["access_token"]
            client.headers["Authorization"] = "Bearer " + token
            del token
            account.clear()
            # Use the original-byte hash to retain the pinned HTML edition. This
            # suppresses a second fetch of the live page during approval.
            item = {
                "source": "parser",
                "title": "NSU+: сохранённая страница от 05.11.2026",
                "text": source_bytes.decode("utf-8").strip(),
                "source_url": manifest["source"]["url"],
                "source_hash": manifest["source"]["snapshot_raw_sha256"],
                "graph_id": "abitur_kb",
            }
            if resume:
                response = await client.get("/api/v1/content-proposals")
                require(response.status_code == 200, "Cannot inspect pending proposal")
                proposal_id = pending_resume(response.json(), item, resume)
            else:
                response = await client.post("/api/v1/content-proposals", json=item)
                require(response.status_code == 201, "Test API upload failed")
                proposal_id = response.json()["id"]
            started = time.monotonic()
            response = await client.post(
                f"/api/v1/content-proposals/{proposal_id}/approve"
            )
            result = {
                "proposal_id": proposal_id,
                "approval_http_status": response.status_code,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "source_chars": len(source_bytes.decode("utf-8")),
                "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "attempts": ledger.sequence,
                "uncertain_failure": ledger.uncertain_failure,
                "planning_reserve_deepseek_cny": ledger.reserved_deepseek_micro_cny / 1e6,
            }
            if response.status_code == 200:
                record = response.json()
                result.update(status=record["status"], document_id=record["document_id"])
            return result


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-proposal", type=lambda value: str(uuid.UUID(value)))
    args = parser.parse_args()
    report = Path("/reports/index-one-result.json")
    usage_path = Path("/reports/index-one-usage.jsonl")
    claim_reports(report, usage_path)
    # Authentication data is supplied via stdin, never from an environment dump
    # or a model key. The caller must separately authorize the test account file.
    account = json.loads(sys.stdin.read())
    ledger = UsageLedger(usage_path)
    if args.resume_proposal:
        prior = json.loads(Path("/reports/attempt-01/index-one-result.json").read_text())
        require(
            prior["proposal_id"] == args.resume_proposal
            and prior["approval_http_status"] == 503
            and not prior["uncertain_failure"],
            "Prior attempt needs manual review",
        )
        ledger.reserved_deepseek_micro_cny = round(
            prior["planning_reserve_deepseek_cny"] * 1e6
        )
    try:
        with ledger.capture():
            async with asyncio.timeout(900):
                result = await index_one(account, ledger, args.resume_proposal)
    except Exception as exc:
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "attempts": ledger.sequence,
            "uncertain_failure": ledger.uncertain_failure,
        }
    finally:
        account.clear()
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("approval_http_status") == 200 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
