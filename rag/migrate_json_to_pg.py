"""Разовая миграция KV-хранилищ LightRAG из JSON-файлов в PostgreSQL.

Запускается при переходе на LIGHTRAG_STORAGE=postgres, если база знаний
уже наполнялась в файловом режиме:

    uv run python -m rag.migrate_json_to_pg                # все графы
    uv run python -m rag.migrate_json_to_pg --graph-id abitur_kb
    uv run python -m rag.migrate_json_to_pg --dry-run

Переносит kv_store_*.json → таблицы lightrag_* (workspace = graph_id):
статусы документов, полные тексты, чанки, кэш LLM, сущности и связи.
Векторные индексы (vdb_*.json) и граф (graphml) остаются файловыми —
они не мигрируют. Скрипт идемпотентен: повторный запуск обновит те же
строки (ON CONFLICT DO UPDATE), а не задублирует их.
"""

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import text

logger = logging.getLogger("migrate_json_to_pg")


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Не удалось прочитать %s: %s", path, exc)
        return {}


def _parse_ts(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (ValueError, OSError):
            return None
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _dump(value: Any) -> str | None:
    """JSONB-значение для вставки (SQLAlchemy + asyncpg отдаёт jsonb строками)."""
    if value is None:
        return None
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value if item is not None)
    return str(value)


def _doc_status_rows(workspace: str, data: dict) -> list[dict]:
    rows = []
    for doc_id, info in data.items():
        if not isinstance(info, dict):
            continue
        metadata = info.get("metadata") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        file_path = _normalize_text(
            info.get("file_path")
            or info.get("file_paths_str")
            or info.get("url")
            or metadata.get("file_paths")
            or metadata.get("file_path")
            or metadata.get("url")
        )
        rows.append(
            {
                "ws": workspace,
                "id": doc_id,
                "content_summary": info.get("content_summary"),
                "content_length": info.get("content_length"),
                "chunks_count": info.get("chunks_count"),
                "status": info.get("status"),
                "file_path": file_path,
                "chunks_list": _dump(info.get("chunks_list") or []),
                "track_id": info.get("track_id"),
                "metadata": _dump(metadata or {}),
                "error_msg": info.get("error_msg") or info.get("error"),
                "created_at": _parse_ts(info.get("created_at")),
                "updated_at": _parse_ts(info.get("updated_at")),
            }
        )
    return rows


def _full_docs_rows(workspace: str, data: dict) -> list[dict]:
    return [
        {
            "ws": workspace,
            "id": doc_id,
            "doc_name": (info or {}).get("doc_name"),
            "content": (info or {}).get("content"),
            "meta": _dump((info or {}).get("meta")),
        }
        for doc_id, info in data.items()
        if isinstance(info, dict)
    ]


def _text_chunks_rows(workspace: str, data: dict) -> list[dict]:
    return [
        {
            "ws": workspace,
            "id": chunk_id,
            "full_doc_id": info.get("full_doc_id"),
            "chunk_order_index": info.get("chunk_order_index"),
            "tokens": info.get("tokens"),
            "content": info.get("content"),
            "file_path": info.get("file_path"),
            "llm_cache_list": _dump(info.get("llm_cache_list") or []),
        }
        for chunk_id, info in data.items()
        if isinstance(info, dict)
    ]


def _llm_cache_rows(workspace: str, data: dict) -> list[dict]:
    rows = []
    for key, info in data.items():
        if not isinstance(info, dict):
            continue
        # Ключ формата {mode}:{cache_type}:{hash} — lightrag.utils.generate_cache_key.
        parts = str(key).split(":")
        cache_type = info.get("cache_type") or (parts[1] if len(parts) == 3 else None)
        queryparam = info.get("queryparam")
        if isinstance(queryparam, str):
            try:
                queryparam = json.loads(queryparam)
            except json.JSONDecodeError:
                queryparam = None
        rows.append(
            {
                "ws": workspace,
                "id": key,
                "original_prompt": info.get("original_prompt"),
                "return_value": info.get("return") or info.get("return_value"),
                "chunk_id": info.get("chunk_id"),
                "cache_type": cache_type,
                "queryparam": _dump(queryparam),
                "create_time": _parse_ts(info.get("create_time")),
                "update_time": _parse_ts(info.get("update_time")),
            }
        )
    return rows


def _counted_rows(workspace: str, data: dict, values_field: str) -> list[dict]:
    return [
        {
            "ws": workspace,
            "id": key,
            "values": _dump(info.get(values_field) or []),
            "count": info.get("count"),
        }
        for key, info in data.items()
        if isinstance(info, dict)
    ]


# (json-файл, таблица, сборщик строк, INSERT)
_STORES: list[tuple[str, str, Any, str]] = [
    (
        "kv_store_doc_status.json",
        "lightrag_doc_status",
        _doc_status_rows,
        """
        INSERT INTO lightrag_doc_status (
            workspace, id, content_summary, content_length, chunks_count,
            status, file_path, chunks_list, track_id, metadata, error_msg,
            created_at, updated_at
        ) VALUES (
            :ws, :id, :content_summary, :content_length, :chunks_count,
            :status, :file_path, :chunks_list::jsonb, :track_id,
            :metadata::jsonb, :error_msg, :created_at, :updated_at
        )
        ON CONFLICT (workspace, id) DO UPDATE SET
            content_summary = EXCLUDED.content_summary,
            content_length = EXCLUDED.content_length,
            chunks_count = EXCLUDED.chunks_count,
            status = EXCLUDED.status,
            file_path = EXCLUDED.file_path,
            chunks_list = EXCLUDED.chunks_list,
            track_id = EXCLUDED.track_id,
            metadata = EXCLUDED.metadata,
            error_msg = EXCLUDED.error_msg,
            created_at = EXCLUDED.created_at,
            updated_at = EXCLUDED.updated_at
        """,
    ),
    (
        "kv_store_full_docs.json",
        "lightrag_doc_full",
        _full_docs_rows,
        """
        INSERT INTO lightrag_doc_full (workspace, id, doc_name, content, meta)
        VALUES (:ws, :id, :doc_name, :content, :meta::jsonb)
        ON CONFLICT (workspace, id) DO UPDATE SET
            doc_name = EXCLUDED.doc_name,
            content = EXCLUDED.content,
            meta = EXCLUDED.meta
        """,
    ),
    (
        "kv_store_text_chunks.json",
        "lightrag_doc_chunks",
        _text_chunks_rows,
        """
        INSERT INTO lightrag_doc_chunks (
            workspace, id, full_doc_id, chunk_order_index, tokens, content,
            file_path, llm_cache_list
        ) VALUES (
            :ws, :id, :full_doc_id, :chunk_order_index, :tokens, :content,
            :file_path, :llm_cache_list::jsonb
        )
        ON CONFLICT (workspace, id) DO UPDATE SET
            full_doc_id = EXCLUDED.full_doc_id,
            chunk_order_index = EXCLUDED.chunk_order_index,
            tokens = EXCLUDED.tokens,
            content = EXCLUDED.content,
            file_path = EXCLUDED.file_path,
            llm_cache_list = EXCLUDED.llm_cache_list
        """,
    ),
    (
        "kv_store_llm_response_cache.json",
        "lightrag_llm_cache",
        _llm_cache_rows,
        """
        INSERT INTO lightrag_llm_cache (
            workspace, id, original_prompt, return_value, chunk_id,
            cache_type, queryparam, create_time, update_time
        ) VALUES (
            :ws, :id, :original_prompt, :return_value, :chunk_id,
            :cache_type, :queryparam::jsonb, :create_time, :update_time
        )
        ON CONFLICT (workspace, id) DO UPDATE SET
            original_prompt = EXCLUDED.original_prompt,
            return_value = EXCLUDED.return_value,
            chunk_id = EXCLUDED.chunk_id,
            cache_type = EXCLUDED.cache_type,
            queryparam = EXCLUDED.queryparam,
            create_time = EXCLUDED.create_time,
            update_time = EXCLUDED.update_time
        """,
    ),
    (
        "kv_store_full_entities.json",
        "lightrag_full_entities",
        lambda ws, data: _counted_rows(ws, data, "entity_names"),
        """
        INSERT INTO lightrag_full_entities (workspace, id, entity_names, count)
        VALUES (:ws, :id, :values::jsonb, :count)
        ON CONFLICT (workspace, id) DO UPDATE SET
            entity_names = EXCLUDED.entity_names,
            count = EXCLUDED.count
        """,
    ),
    (
        "kv_store_full_relations.json",
        "lightrag_full_relations",
        lambda ws, data: _counted_rows(ws, data, "relation_pairs"),
        """
        INSERT INTO lightrag_full_relations (workspace, id, relation_pairs, count)
        VALUES (:ws, :id, :values::jsonb, :count)
        ON CONFLICT (workspace, id) DO UPDATE SET
            relation_pairs = EXCLUDED.relation_pairs,
            count = EXCLUDED.count
        """,
    ),
]


async def migrate_graph(session, workspace: str, base: str, dry_run: bool) -> None:
    graph_dir = os.path.join(base, workspace)
    if not os.path.isdir(graph_dir):
        logger.warning("Каталог графа не найден: %s — пропуск", graph_dir)
        return

    migrated_any = False
    for filename, table, build_rows, insert_sql in _STORES:
        data = _load_json(os.path.join(graph_dir, filename))
        if not data:
            continue
        rows = build_rows(workspace, data)
        logger.info(
            "%s/%s: %d строк → %s%s",
            workspace,
            filename,
            len(rows),
            table,
            " (dry-run)" if dry_run else "",
        )
        if dry_run or not rows:
            continue
        await session.execute(text(insert_sql), rows)
        migrated_any = True

    if migrated_any:
        await session.commit()
        logger.info("%s: миграция зафиксирована в БД", workspace)
    else:
        logger.info("%s: переносить нечего", workspace)


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-id",
        default=None,
        help="Мигрировать только указанный граф (по умолчанию все в workspace-base)",
    )
    parser.add_argument(
        "--workspace-base",
        default=os.getenv("LIGHTRAG_WORKSPACE_BASE", "./data/lightrag"),
        help="База каталогов графов (по умолчанию LIGHTRAG_WORKSPACE_BASE)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать, что будет перенесено, без записи",
    )
    args = parser.parse_args()

    from db.postgres.db import AsyncSessionLocal
    from rag.pg_storage import ensure_lightrag_tables

    await ensure_lightrag_tables()

    if args.graph_id:
        graph_ids = [args.graph_id]
    else:
        graph_ids = sorted(
            name
            for name in os.listdir(args.workspace_base)
            if os.path.isdir(os.path.join(args.workspace_base, name))
            and not name.startswith(".")
        )

    if not graph_ids:
        logger.info("Графы для миграции не найдены в %s", args.workspace_base)
        return

    async with AsyncSessionLocal() as session:
        for graph_id in graph_ids:
            await migrate_graph(session, graph_id, args.workspace_base, args.dry_run)

    logger.info("Миграция завершена.")


if __name__ == "__main__":
    asyncio.run(main())
