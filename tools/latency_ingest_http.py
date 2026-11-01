"""Ingest the approved official FIT page through normal authenticated HTTP routes.

Read admin credentials in memory; never print credentials, bearer tokens, parsed
content, HTTP error bodies, or full exception messages. Invoke exactly one parse
and one confirm. A timeout requires inspection, never an automatic resubmission.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx

SOURCE_URL = "https://education.nsu.ru/apply_fit"


def emit(event: str, **fields: Any) -> None:
    """Emit deliberately selected operational metadata only."""
    print(json.dumps({"event": event, **fields}), flush=True)


async def run(base_url: str, credentials: Path) -> None:
    logging.disable(logging.CRITICAL)
    secrets = json.loads(credentials.read_text())
    async with httpx.AsyncClient(
        base_url=base_url, timeout=600, trust_env=False
    ) as client:
        response = await client.post(
            "/api/v1/auth/login",
            json={"username": secrets["username"], "password": secrets["password"]},
        )
        response.raise_for_status()
        client.headers["Authorization"] = "Bearer " + response.json()["access_token"]
        del secrets
        before = await client.get("/api/v1/rag/docs")
        before.raise_for_status()
        if any(doc.get("url") == SOURCE_URL for doc in before.json()["documents"]):
            emit("ingestion_stopped", reason="approved_source_already_exists")
            return
        started = perf_counter()
        parsed_response = await client.post(
            "/api/v1/rag/parse", params={"url": SOURCE_URL}
        )
        parsed_response.raise_for_status()
        parsed = parsed_response.json()
        emit(
            "parse_complete",
            chars=len(parsed["text"]),
            wall_ms=round((perf_counter() - started) * 1000, 3),
        )
        if not 100 <= len(parsed["text"]) <= 25000:
            raise RuntimeError("Parsed source outside size guard")
        # Copy the standard parse output verbatim. Do not ingest linked documents.
        payload = {key: parsed[key] for key in ("title", "url", "text")}
        payload["documents"] = []
        started = perf_counter()
        try:
            confirmed = await client.post("/api/v1/rag/confirm", json=payload)
            emit(
                "confirm_response",
                http_status=confirmed.status_code,
                wall_ms=round((perf_counter() - started) * 1000, 3),
            )
            confirmed.raise_for_status()
        except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            emit("confirm_uncertain", error_type=type(exc).__name__)
        # Check actual status even after a timeout; never blindly resubmit confirm.
        documents = await client.get("/api/v1/rag/docs")
        documents.raise_for_status()
        selected = [
            doc for doc in documents.json()["documents"] if doc.get("url") == SOURCE_URL
        ]
        if len(selected) != 1:
            emit("ingestion_not_verified", matching_documents=len(selected))
            return
        diagnostic = await client.get(
            "/api/v1/rag/docs/" + selected[0]["id"] + "/diagnostics"
        )
        diagnostic.raise_for_status()
        data = diagnostic.json()
        emit(
            "ingestion_diagnostics",
            **{
                key: value
                for key, value in data.items()
                if key
                in {
                    "postgres_status",
                    "lightrag_status",
                    "status",
                    "chunks_count",
                    "entities_count",
                    "relations_count",
                    "content_length",
                }
            },
        )


def main() -> None:
    """Require explicit source authorization and an operator-supplied API address."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-approved-source", action="store_true", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    args = parser.parse_args()
    try:
        asyncio.run(run(args.base_url, args.credentials))
    except Exception as exc:
        emit("ingestion_error", error_type=type(exc).__name__)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
