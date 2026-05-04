from abbrev.expander import get_abbrev_expander


def test_expand_is_idempotent_for_existing_injection():
    expander = get_abbrev_expander()

    text = "НГУ (Новосибирский государственный университет) принимает документы."

    assert expander.expand(text) == text


def test_expand_keeps_multi_word_abbreviation():
    expander = get_abbrev_expander()

    assert (
        expander.expand("Партнеры СО РАН")
        == "Партнеры СО РАН (Сибирское отделение Российской академии наук)"
    )
