import pytest

from api.services import url_to_rag


@pytest.mark.asyncio
async def test_parse_and_save_url_expands_before_saving(monkeypatch):
    saved_payload = {}

    async def fake_process_url(url: str) -> str:
        return "Документы принимает НГУ."

    async def fake_add_texts_async(texts, source_ids, file_paths):  # noqa: ANN001
        saved_payload["texts"] = texts
        saved_payload["source_ids"] = source_ids
        saved_payload["file_paths"] = file_paths
        return 1

    monkeypatch.setattr(url_to_rag, "process_url", fake_process_url)
    monkeypatch.setattr(url_to_rag, "add_texts_async", fake_add_texts_async)

    result = await url_to_rag.parse_and_save_url("https://example.test/doc", "Документ")

    assert result is True
    assert saved_payload["texts"] == [
        "Документы принимает НГУ (Новосибирский государственный университет)."
    ]
