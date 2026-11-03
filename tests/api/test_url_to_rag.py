import pytest

from abbrev.expander import AbbrevExpander
from api.services import url_to_rag


@pytest.mark.asyncio
async def test_parse_and_save_url_queues_expanded_text(monkeypatch):
    saved = {}
    expander = AbbrevExpander()
    expander.load_items(
        [{"short": "НГУ", "full": "Новосибирский государственный университет"}]
    )
    monkeypatch.setattr(url_to_rag, "get_abbrev_expander", lambda: expander)

    async def fake_process_url(url: str) -> str:
        return "Документы принимает НГУ."

    class FakeSessionContext:
        async def __aenter__(self):  # noqa: ANN001
            return object()

        async def __aexit__(self, exc_type, exc, traceback):  # noqa: ANN001
            return None

    class FakeProposalService:
        def __init__(self, session):  # noqa: ANN001
            pass

        async def create(self, **kwargs):  # noqa: ANN001
            saved.update(kwargs)

    monkeypatch.setattr(url_to_rag, "process_url", fake_process_url)
    monkeypatch.setattr(url_to_rag, "AsyncSessionLocal", lambda: FakeSessionContext())
    monkeypatch.setattr(url_to_rag, "ContentProposalService", FakeProposalService)

    assert await url_to_rag.parse_and_save_url("https://example.test/doc", "Документ")
    assert saved == {
        "source": "parser",
        "graph_id": "abitur_kb",
        "title": "Документ",
        "source_url": "https://example.test/doc",
        "text": "Документы принимает НГУ (Новосибирский государственный университет).",
    }


@pytest.mark.asyncio
async def test_parse_and_save_url_skips_empty_content(monkeypatch):
    async def fake_process_url(url: str) -> str:
        return ""

    monkeypatch.setattr(url_to_rag, "process_url", fake_process_url)
    assert not await url_to_rag.parse_and_save_url("https://example.test/doc", "Документ")
