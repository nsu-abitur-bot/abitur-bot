"""PostgreSQL-хранилища LightRAG (KV + doc_status) вместо JSON-файлов.

Режим включается переменной окружения ``LIGHTRAG_STORAGE=postgres``:
KV-стораджи (полные тексты, чанки, кэш LLM, сущности/связи) и статусы
документов переезжают в PostgreSQL-таблицы LightRAG (JSONB-колонки),
векторный индекс и граф остаются файловыми (nano-vectordb + graphml).

Именованные таблицы LightRAG создаются здесь заранее: ``check_tables``
библиотеки создаёт их сам только при отсутствии, а её DDL для
``lightrag_vdb_*`` требует расширение pgvector. Предсоздавая VDB-таблицы
заглушками (векторный стор файловый и в них не пишет), мы обходимся без
pgvector на сервере.

Графы разделяются колонкой ``workspace`` (= graph_id), поэтому один
инстанс PostgreSQL обслуживает несколько баз знаний.
"""

import json
import logging
import os
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

PG_STORAGE_ENV = "LIGHTRAG_STORAGE"


def is_pg_storage_enabled() -> bool:
    """Включён ли режим PostgreSQL-хранилищ LightRAG."""
    return os.getenv(PG_STORAGE_ENV, "json").strip().lower() == "postgres"


def apply_postgres_env() -> None:
    """Проецирует DB_* на POSTGRES_* для lightrag-клиента, если их не задали.

    LightRAG читает POSTGRES_HOST/PORT/USER/PASSWORD/DATABASE напрямую;
    чтобы БД настраивалась одним .env, берём значения из DB_*.
    POSTGRES_WORKSPACE не выставляем: он глобально перекрыл бы workspace
    отдельных графов (graph_id) и смешал бы базы знаний в одну.
    """
    mapping = {
        "POSTGRES_HOST": "DB_HOST",
        "POSTGRES_PORT": "DB_PORT",
        "POSTGRES_USER": "DB_USER",
        "POSTGRES_PASSWORD": "DB_PASSWORD",
        "POSTGRES_DATABASE": "DB_NAME",
    }
    for pg_var, db_var in mapping.items():
        if not os.environ.get(pg_var) and os.environ.get(db_var):
            os.environ[pg_var] = os.environ[db_var]
    if os.environ.get("POSTGRES_WORKSPACE"):
        logger.warning(
            "POSTGRES_WORKSPACE задан — все графы LightRAG будут писать в одно "
            "workspace; для изоляции баз знаний его нужно снять"
        )


# DDL повторяет lightrag.kg.postgres_impl.TABLES для используемых таблиц.
# VDB-таблицы — заглушки без VECTOR-колонки: векторный стор файловый.
_TABLES_DDL: dict[str, str] = {
    "lightrag_doc_full": """
        CREATE TABLE IF NOT EXISTS lightrag_doc_full (
            id VARCHAR(255),
            workspace VARCHAR(255),
            doc_name VARCHAR(1024),
            content TEXT,
            meta JSONB,
            create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
            update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT lightrag_doc_full_pk PRIMARY KEY (workspace, id)
        )""",
    "lightrag_doc_chunks": """
        CREATE TABLE IF NOT EXISTS lightrag_doc_chunks (
            id VARCHAR(255),
            workspace VARCHAR(255),
            full_doc_id VARCHAR(256),
            chunk_order_index INTEGER,
            tokens INTEGER,
            content TEXT,
            file_path TEXT NULL,
            llm_cache_list JSONB NULL DEFAULT '[]'::jsonb,
            create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
            update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT lightrag_doc_chunks_pk PRIMARY KEY (workspace, id)
        )""",
    "lightrag_llm_cache": """
        CREATE TABLE IF NOT EXISTS lightrag_llm_cache (
            workspace VARCHAR(255) NOT NULL,
            id VARCHAR(255) NOT NULL,
            original_prompt TEXT,
            return_value TEXT,
            chunk_id VARCHAR(255) NULL,
            cache_type VARCHAR(32),
            queryparam JSONB NULL,
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT lightrag_llm_cache_pk PRIMARY KEY (workspace, id)
        )""",
    "lightrag_doc_status": """
        CREATE TABLE IF NOT EXISTS lightrag_doc_status (
            workspace VARCHAR(255) NOT NULL,
            id VARCHAR(255) NOT NULL,
            content_summary VARCHAR(255) NULL,
            content_length INT4 NULL,
            chunks_count INT4 NULL,
            status VARCHAR(64) NULL,
            file_path TEXT NULL,
            chunks_list JSONB NULL DEFAULT '[]'::jsonb,
            track_id VARCHAR(255) NULL,
            metadata JSONB NULL DEFAULT '{}'::jsonb,
            error_msg TEXT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT lightrag_doc_status_pk PRIMARY KEY (workspace, id)
        )""",
    "lightrag_full_entities": """
        CREATE TABLE IF NOT EXISTS lightrag_full_entities (
            id VARCHAR(255),
            workspace VARCHAR(255),
            entity_names JSONB,
            count INTEGER,
            create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
            update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT lightrag_full_entities_pk PRIMARY KEY (workspace, id)
        )""",
    "lightrag_full_relations": """
        CREATE TABLE IF NOT EXISTS lightrag_full_relations (
            id VARCHAR(255),
            workspace VARCHAR(255),
            relation_pairs JSONB,
            count INTEGER,
            create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
            update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT lightrag_full_relations_pk PRIMARY KEY (workspace, id)
        )""",
    "lightrag_entity_chunks": """
        CREATE TABLE IF NOT EXISTS lightrag_entity_chunks (
            id VARCHAR(512),
            workspace VARCHAR(255),
            chunk_ids JSONB,
            count INTEGER,
            create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
            update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT lightrag_entity_chunks_pk PRIMARY KEY (workspace, id)
        )""",
    "lightrag_relation_chunks": """
        CREATE TABLE IF NOT EXISTS lightrag_relation_chunks (
            id VARCHAR(512),
            workspace VARCHAR(255),
            chunk_ids JSONB,
            count INTEGER,
            create_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
            update_time TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT lightrag_relation_chunks_pk PRIMARY KEY (workspace, id)
        )""",
    # Заглушки: нужны только чтобы check_tables библиотеки не пыталась
    # создать их сама (её DDL требует pgvector). Векторный стор файловый.
    "lightrag_vdb_chunks": """
        CREATE TABLE IF NOT EXISTS lightrag_vdb_chunks (
            id VARCHAR(255),
            workspace VARCHAR(255),
            CONSTRAINT lightrag_vdb_chunks_pk PRIMARY KEY (workspace, id)
        )""",
    "lightrag_vdb_entity": """
        CREATE TABLE IF NOT EXISTS lightrag_vdb_entity (
            id VARCHAR(255),
            workspace VARCHAR(255),
            CONSTRAINT lightrag_vdb_entity_pk PRIMARY KEY (workspace, id)
        )""",
    "lightrag_vdb_relation": """
        CREATE TABLE IF NOT EXISTS lightrag_vdb_relation (
            id VARCHAR(255),
            workspace VARCHAR(255),
            CONSTRAINT lightrag_vdb_relation_pk PRIMARY KEY (workspace, id)
        )""",
}

async def ensure_lightrag_tables() -> None:
    """Идемпотентно создаёт таблицы LightRAG (без pgvector)."""
    from db.postgres.db import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        for ddl in _TABLES_DDL.values():
            await session.execute(text(ddl))
        await session.commit()
    logger.info("LightRAG PostgreSQL-таблицы готовы (%d шт.)", len(_TABLES_DDL))


def _json(value: Any, default: Any) -> Any:
    """JSONB-колонки через asyncpg приходят строками — приводим к объектам."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value if value is not None else default


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value if item is not None)
    return str(value)


def doc_status_row_to_dict(row: Any) -> dict:
    """Строка lightrag_doc_status → формат get_list_docs (совместим с JSON-режимом)."""
    metadata = _json(getattr(row, "metadata", None), {})
    if not isinstance(metadata, dict):
        metadata = {}
    url = _normalize_text(
        getattr(row, "file_path", None)
        or metadata.get("file_paths")
        or metadata.get("file_path")
        or metadata.get("url")
    )
    if not url and str(getattr(row, "id", "")).startswith("http"):
        url = str(row.id)
    created_at = getattr(row, "created_at", None)
    return {
        "id": getattr(row, "id", None),
        "url": url,
        "status": getattr(row, "status", None),
        "content_summary": getattr(row, "content_summary", None),
        "content_length": getattr(row, "content_length", None),
        "created_at": created_at.isoformat() if created_at else None,
    }


async def list_docs(workspace: str) -> list[dict]:
    """Список документов из lightrag_doc_status (аналог JSON-версии get_list_docs)."""
    from db.postgres.db import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text(
                "SELECT id, status, content_summary, content_length, file_path,"
                " metadata, created_at FROM lightrag_doc_status WHERE workspace = :ws"
            ),
            {"ws": workspace},
        )
        rows = result.mappings().all()
    docs = [doc_status_row_to_dict(r) for r in rows]
    docs.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return docs


async def get_full_doc(workspace: str, doc_id: str) -> str | None:
    """Полный текст документа из lightrag_doc_full."""
    from db.postgres.db import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text(
                "SELECT COALESCE(content, '') AS content FROM lightrag_doc_full"
                " WHERE workspace = :ws AND id = :doc_id"
            ),
            {"ws": workspace, "doc_id": doc_id},
        )
        row = result.mappings().first()
    if row is None:
        return None
    content = row["content"]
    return content if content else None


async def doc_status_signature(workspace: str) -> tuple[int, Any]:
    """Дешёвая сигнатура doc_status для инвалидации кэшей: count + max(updated_at)."""
    from db.postgres.db import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text(
                "SELECT COUNT(*) AS cnt, MAX(updated_at) AS max_updated"
                " FROM lightrag_doc_status WHERE workspace = :ws"
            ),
            {"ws": workspace},
        )
        row = result.mappings().first()
    if row is None:
        return 0, None
    return int(row["cnt"] or 0), row["max_updated"]


async def doc_diagnostics(workspace: str, doc_id: str) -> dict:
    """Диагностика документа (аналог JSON-версии get_doc_diagnostics)."""
    from db.postgres.db import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        status_result = await session.execute(
            text(
                "SELECT status, content_length, chunks_count, error_msg, metadata"
                " FROM lightrag_doc_status WHERE workspace = :ws AND id = :doc_id"
            ),
            {"ws": workspace, "doc_id": doc_id},
        )
        status_row = status_result.mappings().first()
        if status_row is None:
            return {
                "lightrag_status": "not_found",
                "chunks_count": 0,
                "entities_count": 0,
                "relations_count": 0,
                "content_length": None,
                "error": None,
            }

        entities_result = await session.execute(
            text(
                "SELECT count FROM lightrag_full_entities"
                " WHERE workspace = :ws AND id = :doc_id"
            ),
            {"ws": workspace, "doc_id": doc_id},
        )
        entities_row = entities_result.mappings().first()
        relations_result = await session.execute(
            text(
                "SELECT count FROM lightrag_full_relations"
                " WHERE workspace = :ws AND id = :doc_id"
            ),
            {"ws": workspace, "doc_id": doc_id},
        )
        relations_row = relations_result.mappings().first()

        metadata = _json(status_row.get("metadata"), {})
        if not isinstance(metadata, dict):
            metadata = {}
        error = (
            status_row.get("error_msg")
            or metadata.get("error")
            or metadata.get("error_msg")
        )

        return {
            "lightrag_status": status_row.get("status") or "unknown",
            "chunks_count": status_row.get("chunks_count") or 0,
            "entities_count": (entities_row or {}).get("count") or 0,
            "relations_count": (relations_row or {}).get("count") or 0,
            "content_length": status_row.get("content_length"),
            "error": error,
        }


async def dump_kv_stores(workspace: str) -> dict[str, str]:
    """Дамп KV-таблиц в JSON-строки (для export_stores в PG-режиме).

    Возвращает {имя_файла: json_строка}; формат — {id: {поля}} как у
    исходных kv_store_*.json, чтобы экспорт оставался читаемым.
    """
    from db.postgres.db import AsyncSessionLocal

    tables = {
        "kv_store_doc_status.json": (
            "SELECT id, content_summary, content_length, chunks_count, status,"
            " file_path, chunks_list, track_id, metadata, error_msg,"
            " created_at, updated_at FROM lightrag_doc_status WHERE workspace = :ws"
        ),
        "kv_store_full_docs.json": (
            "SELECT id, doc_name, content, meta FROM lightrag_doc_full"
            " WHERE workspace = :ws"
        ),
        "kv_store_text_chunks.json": (
            "SELECT id, full_doc_id, chunk_order_index, tokens, content,"
            " file_path, llm_cache_list FROM lightrag_doc_chunks"
            " WHERE workspace = :ws"
        ),
        "kv_store_llm_response_cache.json": (
            "SELECT id, original_prompt, return_value, chunk_id, cache_type,"
            " queryparam, create_time, update_time FROM lightrag_llm_cache"
            " WHERE workspace = :ws"
        ),
        "kv_store_full_entities.json": (
            "SELECT id, entity_names, count FROM lightrag_full_entities"
            " WHERE workspace = :ws"
        ),
        "kv_store_full_relations.json": (
            "SELECT id, relation_pairs, count FROM lightrag_full_relations"
            " WHERE workspace = :ws"
        ),
    }

    dump: dict[str, str] = {}
    async with AsyncSessionLocal() as session:
        for filename, query in tables.items():
            result = await session.execute(text(query), {"ws": workspace})
            rows = result.mappings().all()
            payload = {}
            for row in rows:
                item = dict(row)
                for key, value in list(item.items()):
                    item[key] = _json(value, value)
                item.pop("id", None)
                payload[str(row["id"])] = item
            dump[filename] = json.dumps(payload, ensure_ascii=False, default=str)
    return dump
