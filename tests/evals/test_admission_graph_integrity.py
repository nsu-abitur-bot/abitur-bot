import base64
import json
from pathlib import Path

import numpy as np
import pytest

from tools.admission_graph_integrity import (
    atomic_vector_writes,
    matrix_valid,
    vector_metadata,
)


def test_complete_metadata_can_be_read_when_only_matrix_was_interrupted():
    source = {
        "embedding_dim": 2560,
        "data": [{"__id__": "one", "content": "source"}],
        "matrix": "unfinished",
    }
    raw = json.dumps(source)
    broken = raw[:-4]
    result = vector_metadata(broken)
    assert result == {"embedding_dim": 2560, "data": source["data"]}
    with pytest.raises(ValueError):
        vector_metadata(raw[: raw.index("source") + 2])


def test_matrix_dimensions_and_values_must_match_metadata():
    matrix = np.ones((2, 2560), dtype=np.float32)
    record = {
        "embedding_dim": 2560,
        "data": [{}, {}],
        "matrix": base64.b64encode(matrix.tobytes()).decode(),
    }
    assert matrix_valid(record)
    assert not matrix_valid({**record, "data": [{}]})
    assert not matrix_valid({**record, "matrix": record["matrix"][:-8]})
    matrix[0, 1] = np.nan
    assert not matrix_valid(
        {**record, "matrix": base64.b64encode(matrix.tobytes()).decode()}
    )


def test_failed_vector_save_preserves_the_last_complete_file(tmp_path, monkeypatch):
    from nano_vectordb import NanoVectorDB

    target = tmp_path / "vdb_chunks.json"
    store = NanoVectorDB(2560, storage_file=str(target))
    store.upsert([{"__id__": "one", "__vector__": np.ones(2560, dtype=np.float32)}])
    store.save()
    original = target.read_bytes()

    def interrupted_save(self):
        Path(self.storage_file).write_text('{"partial":')
        raise OSError("interrupted write")

    monkeypatch.setattr(NanoVectorDB, "save", interrupted_save)
    with pytest.raises(OSError), atomic_vector_writes(tmp_path):
        store.save()
    assert target.read_bytes() == original
    assert store.storage_file == str(target)
    assert list(tmp_path.iterdir()) == [target]
