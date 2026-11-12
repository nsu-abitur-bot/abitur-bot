"""Validate persisted #388 graph files without returning source text or vectors."""

import base64
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from xml.etree import ElementTree

import numpy as np

from tools.admission_index_one import require


@contextmanager
def atomic_vector_writes(
    root: Path = Path("/app/data/lightrag/abitur_kb"),
) -> Generator[None, None, None]:
    """Keep a complete prior vector file if the indexing process is interrupted."""
    from nano_vectordb import NanoVectorDB

    original = NanoVectorDB.save

    def save(self: NanoVectorDB) -> None:
        destination = Path(self.storage_file)
        require(
            destination.parent == root,
            "Unexpected vector destination",
        )
        descriptor, name = tempfile.mkstemp(prefix=".vector388-", dir=destination.parent)
        os.close(descriptor)
        temporary = Path(name)
        self.storage_file = str(temporary)
        try:
            original(self)
            with temporary.open("r+b") as file:
                os.fsync(file.fileno())
            os.replace(temporary, destination)
        finally:
            self.storage_file = str(destination)
            if temporary.exists():
                temporary.unlink()

    NanoVectorDB.save = save
    try:
        yield
    finally:
        NanoVectorDB.save = original


def vector_metadata(raw: str) -> dict:
    """Recover only complete metadata preceding a possibly truncated matrix."""
    decoder = json.JSONDecoder()
    require(raw.startswith("{"), "Unexpected vector file")
    position, result = 1, {}
    while position < len(raw):
        while raw[position].isspace() or raw[position] == ",":
            position += 1
        key, position = decoder.raw_decode(raw, position)
        require(key not in result, "Duplicate vector file key")
        while raw[position].isspace() or raw[position] == ":":
            position += 1
        if key == "matrix":
            require(
                set(result) == {"embedding_dim", "data"}, "Incomplete vector metadata"
            )
            require(isinstance(result["data"], list), "Invalid vector records")
            return result
        value, position = decoder.raw_decode(raw, position)
        result[key] = value
    raise ValueError("Matrix key not found")


def matrix_valid(record: dict) -> bool:
    try:
        dimension = record["embedding_dim"]
        array = np.frombuffer(
            base64.b64decode(record["matrix"], validate=True), dtype=np.float32
        )
        return bool(
            dimension == 2560
            and array.size == len(record["data"]) * dimension
            and np.isfinite(array).all()
            and (np.linalg.norm(array.reshape(-1, dimension), axis=1) > 0).all()
        )
    except (KeyError, TypeError, ValueError):
        return False


def graph_integrity(
    root: Path, sources: list[dict], inputs: Path, only_sources: set[str] | None = None
) -> dict:
    status = json.loads((root / "kv_store_doc_status.json").read_text())
    docs = json.loads((root / "kv_store_full_docs.json").read_text())
    chunks = json.loads((root / "kv_store_text_chunks.json").read_text())
    entities = json.loads((root / "kv_store_full_entities.json").read_text())
    relations = json.loads((root / "kv_store_full_relations.json").read_text())
    vectors = {}
    for kind in ("chunks", "entities", "relationships"):
        record = json.loads((root / f"vdb_{kind}.json").read_text())
        require(matrix_valid(record), "Invalid vector matrix")
        vectors[kind] = {row["__id__"] for row in record["data"]}
        require(len(vectors[kind]) == len(record["data"]), "Duplicate vector IDs")
    xml = ElementTree.parse(root / "graph_chunk_entity_relation.graphml")
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    nodes = {node.attrib["id"] for node in xml.findall(".//g:node", ns)}
    edges = {
        frozenset((edge.attrib["source"], edge.attrib["target"]))
        for edge in xml.findall(".//g:edge", ns)
    }
    by_url = {source["url"]: source for source in sources}
    completed = []
    for identifier, record in status.items():
        source = by_url[record["file_path"]]
        if only_sources is not None and source["id"] not in only_sources:
            continue
        require(record["status"] == "processed", "Unfinished graph document")
        require(
            docs[identifier]["content"]
            == (inputs / source["filename"]).read_text().strip(),
            "Graph text differs from source",
        )
        ids = set(record["chunks_list"])
        require(
            ids and ids <= set(chunks) & vectors["chunks"],
            "Missing text chunks or vectors",
        )
        names = entities[identifier]["entity_names"]
        pairs = relations[identifier]["relation_pairs"]
        require(set(names) <= nodes, "Missing graph nodes")
        require(
            all(
                "ent-" + hashlib.md5(name.encode()).hexdigest() in vectors["entities"]
                for name in names
            ),
            "Missing entity vectors",
        )
        require(all(frozenset(pair) in edges for pair in pairs), "Missing graph edges")
        require(
            all(
                "rel-" + hashlib.md5((a + b).encode()).hexdigest()
                in vectors["relationships"]
                or "rel-" + hashlib.md5((b + a).encode()).hexdigest()
                in vectors["relationships"]
                for a, b in pairs
            ),
            "Missing relation vectors",
        )
        completed.append(source["id"])
    require(len(set(completed)) == len(completed), "Duplicate graph sources")
    return {
        "processed_source_ids": completed,
        "documents": len(completed),
        "vector_counts": {name: len(ids) for name, ids in vectors.items()},
        "graph_nodes": len(nodes),
        "graph_edges": len(edges),
    }
