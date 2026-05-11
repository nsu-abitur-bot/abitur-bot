import pytest

from abbrev.expander import AbbrevExpander

SAMPLE_ABBREVS = [
    {"short": "НГУ", "full": "Новосибирский государственный университет"},
    {"short": "СО РАН", "full": "Сибирское отделение Российской академии наук"},
    {"short": "ФИТ", "full": "Факультет информационных технологий"},
]


@pytest.fixture
def expander() -> AbbrevExpander:
    e = AbbrevExpander()
    e.load_items(SAMPLE_ABBREVS)
    return e


def test_expand_is_idempotent_for_existing_injection(expander: AbbrevExpander):
    text = "НГУ (Новосибирский государственный университет) принимает документы."
    assert expander.expand(text) == text


def test_expand_keeps_multi_word_abbreviation(expander: AbbrevExpander):
    assert (
        expander.expand("Партнеры СО РАН")
        == "Партнеры СО РАН (Сибирское отделение Российской академии наук)"
    )
