import json
from typing import Any, cast

import networkx as nx
import numpy as np
import pytest
from nano_vectordb import NanoVectorDB
from nano_vectordb.dbs import Data
from networkx.readwrite.graphml import write_graphml

from rag.reembed import rebuild_workspace


class Embeddings:
    embedding_dim = 3

    def __init__(self):
        self.texts = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [[1.0, 2.0, 3.0] for _ in texts]


def fixture(source):
    source.mkdir()
    (source / "kv_store_text_chunks.json").write_text(
        json.dumps({"chunk": {"content": "Документ вуза"}})
    )
    graph = nx.Graph()
    graph.add_node("ВУЗ", description="Университет")
    graph.add_node("ФАК", description="Факультет")
    graph.add_edge("ВУЗ", "ФАК", keywords="включает", description="Факультет вуза")
    write_graphml(graph, source / "graph_chunk_entity_relation.graphml")
    rows = {
        "chunks": {"__id__": "chunk", "full_doc_id": "doc"},
        "entities": {"__id__": "entity", "entity_name": "ВУЗ"},
        "relationships": {"__id__": "edge", "src_id": "ВУЗ", "tgt_id": "ФАК"},
    }
    for name, row in rows.items():
        database = NanoVectorDB(2, storage_file=str(source / f"vdb_{name}.json"))
        database.upsert([cast(Data, {**row, "__vector__": np.array([1.0, 0.5])})])
        database.save()


def test_rebuilds_all_vectors_preserving_ids_and_source(tmp_path):
    source, target = tmp_path / "source", tmp_path / "stage"
    fixture(source)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    embedder = Embeddings()
    counts = rebuild_workspace(source, target, embedder)
    assert counts == {
        "vdb_chunks.json": 1,
        "vdb_entities.json": 1,
        "vdb_relationships.json": 1,
    }
    assert set(embedder.texts) == {
        "Документ вуза",
        "ВУЗ\nУниверситет",
        "включает\tВУЗ\nФАК\nФакультет вуза",
    }
    for path in source.iterdir():
        assert path.read_bytes() == before[path.name]
        if not path.name.startswith("vdb_"):
            assert (target / path.name).read_bytes() == before[path.name]
    database = NanoVectorDB(3, storage_file=str(target / "vdb_chunks.json"))
    assert cast(Any, database.get(["chunk"])[0])["full_doc_id"] == "doc"
    assert database.query(np.array([1, 2, 3]), top_k=1)[0]["__id__"] == "chunk"


def test_provider_failure_never_changes_active_index(tmp_path):
    source = tmp_path / "source"
    fixture(source)
    before = {p.name: p.read_bytes() for p in source.iterdir()}

    class Failing(Embeddings):
        def embed_documents(self, texts):
            raise RuntimeError("quota")

    with pytest.raises(RuntimeError, match="quota"):
        rebuild_workspace(source, tmp_path / "failed-stage", Failing())
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


def test_missing_canonical_text_does_not_publish_partial_index(tmp_path):
    source = tmp_path / "source"
    fixture(source)
    (source / "kv_store_text_chunks.json").write_text("{}")
    with pytest.raises(KeyError):
        rebuild_workspace(source, tmp_path / "stage", Embeddings())
    assert json.loads((source / "vdb_chunks.json").read_text())["embedding_dim"] == 2
