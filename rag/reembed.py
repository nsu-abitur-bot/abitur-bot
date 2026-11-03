"""Recompute every LightRAG vector in a staging copy, preserving canonical data."""

import base64
import json
import shutil
import zlib
from pathlib import Path
from typing import Protocol, cast

import networkx as nx
import numpy as np
from nano_vectordb import NanoVectorDB
from nano_vectordb.dbs import Data
from networkx.readwrite.graphml import read_graphml


class Embedder(Protocol):
    embedding_dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


def rebuild_workspace(
    source: Path, destination: Path, embedder: Embedder
) -> dict[str, int]:
    """Source must be quiescent; never changes or deletes its files."""
    if destination.exists():
        raise ValueError("Каталог пересчёта уже существует")
    if source.is_symlink() or any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Символические ссылки в индексе не поддерживаются")
    files = sorted(source.glob("vdb_*.json"))
    allowed = {"vdb_chunks.json", "vdb_entities.json", "vdb_relationships.json"}
    if any(path.name not in allowed for path in files):
        raise ValueError("Неизвестный формат векторного индекса")
    chunks_path = source / "kv_store_text_chunks.json"
    chunks = json.loads(chunks_path.read_text()) if chunks_path.exists() else {}
    graph_path = source / "graph_chunk_entity_relation.graphml"
    graph = read_graphml(graph_path) if graph_path.exists() else nx.Graph()
    stores = {path.name: json.loads(path.read_text()) for path in files}
    canonical = {
        "vdb_chunks.json": set(chunks),
        "vdb_entities.json": set(graph.nodes),
        "vdb_relationships.json": {frozenset((src, tgt)) for src, tgt in graph.edges},
    }
    for name, expected in canonical.items():
        rows = stores.get(name, {}).get("data", [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Некорректный векторный индекс")
        if name == "vdb_chunks.json":
            actual = {row["__id__"] for row in rows}
        elif name == "vdb_entities.json":
            actual = {row["entity_name"] for row in rows}
        else:
            actual = {frozenset((row["src_id"], row["tgt_id"])) for row in rows}
        if actual != expected or len(actual) != len(rows):
            raise ValueError("Исходный векторный индекс неполный или несогласован")
    shutil.copytree(source, destination)
    response_cache = destination / "kv_store_llm_response_cache.json"
    if response_cache.exists():
        response_cache.write_text("{}")
    counts: dict[str, int] = {}
    for path in files:
        original = stores[path.name]
        rows = original["data"]
        if not isinstance(rows, list):
            raise ValueError("Некорректный векторный индекс")
        texts: list[str] = []
        for row in rows:
            if path.name == "vdb_chunks.json":
                content = chunks[row["__id__"]]["content"]
            elif path.name == "vdb_entities.json":
                name = row["entity_name"]
                content = f"{name}\n{graph.nodes[name]['description']}"
            else:
                src, tgt = row["src_id"], row["tgt_id"]
                edge = graph.edges[src, tgt]
                content = f"{edge['keywords']}\t{src}\n{tgt}\n{edge['description']}"
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Отсутствует исходный текст для пересчёта")
            texts.append(content)
        target = destination / path.name
        target.unlink()
        database = NanoVectorDB(embedder.embedding_dim, storage_file=str(target))
        for start in range(0, len(rows), 16):
            vectors = embedder.embed_documents(texts[start : start + 16])
            batch = rows[start : start + 16]
            if len(vectors) != len(batch):
                raise ValueError("Получен неполный набор эмбеддингов")
            records: list[Data] = []
            for row, vector in zip(batch, vectors):
                array = np.asarray(vector, dtype=np.float32)
                norm = np.linalg.norm(array)
                if (
                    array.shape != (embedder.embedding_dim,)
                    or not np.isfinite(array).all()
                    or not np.isfinite(norm)
                    or norm <= 0
                ):
                    raise ValueError("Получен некорректный вектор")
                compressed = base64.b64encode(
                    zlib.compress(array.astype(np.float16).tobytes())
                ).decode()
                records.append(
                    cast(Data, {**row, "__vector__": array, "vector": compressed})
                )
            database.upsert(records)
        if "additional_data" in original:
            database.store_additional_data(**original["additional_data"])
        database.save()
        counts[path.name] = len(rows)
    return counts
