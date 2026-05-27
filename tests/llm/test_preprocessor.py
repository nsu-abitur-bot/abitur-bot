import pytest

from abbrev.expander import AbbrevExpander
from llm import preprocessor


class _FakeLLM:
    async def generate(self, messages, profile):  # noqa: ANN001
        return "Документы принимает НГУ и ФИТ."


@pytest.mark.asyncio
async def test_clean_and_structure_text_reexpands_llm_response(monkeypatch):
    expander = AbbrevExpander()
    expander.load_items(
        [
            {"short": "НГУ", "full": "Новосибирский государственный университет"},
            {"short": "ФИТ", "full": "Факультет информационных технологий"},
        ]
    )

    monkeypatch.setattr(preprocessor, "get_llm_provider", lambda: _FakeLLM())
    monkeypatch.setattr(preprocessor, "get_abbrev_expander", lambda: expander)

    result = await preprocessor.clean_and_structure_text("Документы принимает НГУ.")

    assert "НГУ (Новосибирский государственный университет)" in result
    assert "ФИТ (Факультет информационных технологий)" in result
