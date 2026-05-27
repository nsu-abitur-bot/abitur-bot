"""Backfill documents from LightRAG metadata

Revision ID: b1c2d3e4f5a6
Revises: 0a1b2c3d4e5f
Create Date: 2026-05-27 01:00:00.000000

"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import insert as pg_insert
from uuid6 import uuid7

revision: str = "b1c2d3e4f5a6"
down_revision: Union[str, Sequence[str], None] = "0a1b2c3d4e5f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

DEFAULT_GRAPH_ID = "abitur_kb"


document_table = sa.table(
    "document",
    sa.column("id", sa.String(36)),
    sa.column("title", sa.String(500)),
    sa.column("source_url", sa.String(1000)),
    sa.column("graph_id", sa.String(100)),
    sa.column("rag_doc_id", sa.String(500)),
    sa.column("content_hash", sa.String(64)),
    sa.column("content_length", sa.Integer()),
    sa.column("status", sa.String(50)),
    sa.column("last_indexed_at", sa.DateTime()),
    sa.column("created_at", sa.DateTime()),
    sa.column("updated_at", sa.DateTime()),
)


def upgrade() -> None:
    docs = _load_lightrag_docs(DEFAULT_GRAPH_ID)
    if not docs:
        return

    now = datetime.now()
    rows = []
    for rag_doc in docs:
        rag_doc_id = _string_or_none(rag_doc.get("id"))
        if not rag_doc_id:
            continue

        rows.append(
            {
                "id": str(uuid7()),
                "title": rag_doc_id,
                "source_url": _string_or_none(rag_doc.get("url")),
                "graph_id": DEFAULT_GRAPH_ID,
                "rag_doc_id": rag_doc_id,
                "content_hash": None,
                "content_length": _int_or_none(rag_doc.get("content_length")),
                "status": "active",
                "last_indexed_at": None,
                "created_at": _parse_datetime(rag_doc.get("created_at")) or now,
                "updated_at": now,
            }
        )

    if not rows:
        return

    bind = op.get_bind()
    stmt = pg_insert(document_table).values(rows)
    bind.execute(stmt.on_conflict_do_nothing(constraint="uq_document_graph_rag_doc_id"))


def downgrade() -> None:
    pass


def _load_lightrag_docs(graph_id: str) -> list[dict[str, Any]]:
    workspace_base = Path(os.getenv("LIGHTRAG_WORKSPACE_BASE", "./data/lightrag"))
    workspace_path = workspace_base / graph_id
    status_data = _read_json_object(workspace_path / "kv_store_doc_status.json")
    if not status_data:
        return []

    full_docs_data = _read_json_object(workspace_path / "kv_store_full_docs.json")

    docs = []
    for doc_id, raw_info in status_data.items():
        if not isinstance(raw_info, dict):
            continue
        docs.append(
            {
                "id": doc_id,
                "url": _extract_url(doc_id, raw_info, full_docs_data),
                "content_length": raw_info.get("content_length"),
                "created_at": raw_info.get("created_at"),
            }
        )

    docs.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return docs


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}

    return value if isinstance(value, dict) else {}


def _extract_url(
    doc_id: str, info: dict[str, Any], full_docs_data: dict[str, Any]
) -> str | None:
    url = _string_or_none(
        info.get("url") or info.get("file_path") or info.get("file_paths_str")
    )
    if url:
        return url

    doc_full = full_docs_data.get(doc_id, {})
    if isinstance(doc_full, dict):
        metadata = doc_full.get("metadata", {})
        if isinstance(metadata, dict):
            url = _string_or_none(
                metadata.get("file_paths")
                or metadata.get("file_path")
                or metadata.get("url")
            )
            if url:
                return url

        url = _string_or_none(
            doc_full.get("file_paths") or doc_full.get("file_path") or doc_full.get("url")
        )
        if url:
            return url

    return doc_id if doc_id.startswith("http") else None


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str):
        return value or None
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item:
                return item
        return None
    if value is None:
        return None
    return str(value)


def _int_or_none(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None
