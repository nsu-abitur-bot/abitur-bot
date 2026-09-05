import json
from datetime import datetime, timezone

import pytest

from rag import pg_storage
from rag.pg_storage import doc_status_row_to_dict


class _Row:
    """Минимальная имитация строки lightrag_doc_status."""

    def __init__(self, **kwargs):
        defaults = {
            "id": "doc-1",
            "status": "processed",
            "content_summary": "сводка",
            "content_length": 123,
            "file_path": None,
            "metadata": None,
            "created_at": datetime(2026, 9, 4, 12, 0, 0),
        }
        defaults.update(kwargs)
        for key, value in defaults.items():
            setattr(self, key, value)


def test_is_pg_storage_enabled_default_off(monkeypatch):
    monkeypatch.delenv(pg_storage.PG_STORAGE_ENV, raising=False)
    assert pg_storage.is_pg_storage_enabled() is False


def test_is_pg_storage_enabled_postgres(monkeypatch):
    monkeypatch.setenv(pg_storage.PG_STORAGE_ENV, "postgres")
    assert pg_storage.is_pg_storage_enabled() is True


def test_doc_status_row_url_from_file_path():
    row = _Row(file_path="https://nsu.ru/page")
    result = doc_status_row_to_dict(row)
    assert result["url"] == "https://nsu.ru/page"
    assert result["status"] == "processed"
    assert result["created_at"] == "2026-09-04T12:00:00"


def test_doc_status_row_url_from_metadata_json_string():
    row = _Row(metadata=json.dumps({"url": "https://nsu.ru/meta"}))
    assert doc_status_row_to_dict(row)["url"] == "https://nsu.ru/meta"


def test_doc_status_row_url_from_metadata_file_paths_list():
    row = _Row(metadata={"file_paths": ["https://nsu.ru/a", "https://nsu.ru/b"]})
    assert doc_status_row_to_dict(row)["url"] == "https://nsu.ru/a, https://nsu.ru/b"


def test_doc_status_row_url_fallback_to_http_doc_id():
    row = _Row(id="https://nsu.ru/doc", file_path=None)
    assert doc_status_row_to_dict(row)["url"] == "https://nsu.ru/doc"


def test_doc_status_row_no_url():
    assert doc_status_row_to_dict(_Row())["url"] is None


def test_json_parses_strings_and_passes_objects():
    assert pg_storage._json('{"a": 1}', {}) == {"a": 1}
    assert pg_storage._json(["x"], []) == ["x"]
    assert pg_storage._json("{broken", {"default": True}) == {"default": True}
    assert pg_storage._json(None, []) == []


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2026-09-04T10:00:00", datetime(2026, 9, 4, 10, 0, 0)),
        (
            "2026-09-04T10:00:00Z",
            datetime(2026, 9, 4, 10, 0, 0, tzinfo=timezone.utc),
        ),
        (1725444000, datetime(2024, 9, 4, 10, 0, 0, tzinfo=timezone.utc)),
        (None, None),
        ("not-a-date", None),
    ],
)
def test_parse_ts(raw, expected):
    from rag.migrate_json_to_pg import _parse_ts

    assert _parse_ts(raw) == expected
