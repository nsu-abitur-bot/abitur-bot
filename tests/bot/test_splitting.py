"""Нарезка длинного ответа на сообщения в пределах лимита мессенджера."""

import re
from html.parser import HTMLParser

from bot.splitting import MESSAGE_CHUNK_LIMIT, split_message


class _TagBalance(HTMLParser):
    """Проверяет, что каждая часть — самостоятельная валидная разметка."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.stack: list[str] = []
        self.broken = False

    def handle_starttag(self, tag, attrs):
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.broken = True
            return
        self.stack.pop()


def assert_balanced(chunk: str) -> None:
    parser = _TagBalance()
    parser.feed(chunk)
    parser.close()
    assert not parser.broken, chunk
    assert parser.stack == [], chunk
    # Ни тега, ни сущности, разрезанных пополам.
    assert not re.search(r"<[^>]*$", chunk), chunk
    assert not re.search(r"^[^<]*>", chunk), chunk
    assert not re.search(r"&[#\w]*$", chunk), chunk


def test_short_text_is_one_chunk():
    assert split_message("Привет") == ["Привет"]


def test_empty_text_gives_no_chunks():
    assert split_message("") == []
    assert split_message("   \n\n  ") == []


def test_default_limit_fits_telegram_and_max():
    # Telegram: 4096, MAX: строго меньше 4000.
    assert MESSAGE_CHUNK_LIMIT < 4000


def test_splits_on_paragraph_boundaries():
    paragraphs = [f"Абзац {i}. " + "слово " * 30 for i in range(20)]
    text = "\n\n".join(p.strip() for p in paragraphs)

    chunks = split_message(text, limit=500)

    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    # Каждая часть — целые абзацы: разрезов внутри абзаца нет.
    for chunk in chunks:
        for part in chunk.split("\n\n"):
            assert part.startswith("Абзац "), part
    assert "\n\n".join(chunks) == text


def test_falls_back_to_lines_then_spaces_then_hard_cut():
    lines = "\n".join("строка " * 10 for _ in range(30))
    for chunk in split_message(lines, limit=200):
        assert len(chunk) <= 200
        assert not chunk.endswith("стро")

    words = "слово " * 200
    word_chunks = split_message(words, limit=100)
    assert all(len(c) <= 100 for c in word_chunks)
    assert all(re.fullmatch(r"(слово ?)+", c) for c in word_chunks)

    solid = "x" * 250
    assert split_message(solid, limit=100) == ["x" * 100, "x" * 100, "x" * 50]


def test_no_empty_chunks():
    text = "\n\n\n\n".join("a" * 90 for _ in range(10)) + "\n\n\n   \n"
    chunks = split_message(text, limit=100)
    assert chunks
    assert all(c.strip() for c in chunks)


def test_html_tags_are_closed_and_reopened_across_chunks():
    text = "<b>" + "жирный текст " * 50 + "</b> конец"

    chunks = split_message(text, limit=200, html=True)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 200
        assert_balanced(chunk)
    assert chunks[1].startswith("<b>")
    assert chunks[-1].endswith("конец")


def test_html_link_keeps_href_when_split_inside():
    text = '<a href="https://www.nsu.ru/x">' + "длинная ссылка " * 40 + "</a>"

    chunks = split_message(text, limit=150, html=True)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 150
        assert_balanced(chunk)
        assert chunk.startswith('<a href="https://www.nsu.ru/x">')


def test_html_never_cuts_inside_tag_or_entity():
    # Без пробелов: разрез только жёсткий, и он не должен попасть в разметку.
    unit = '&lt;<i>x</i>&amp;<a href="https://e.x/a">y</a>'
    text = unit * 60

    chunks = split_message(text, limit=97, html=True)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 97
        assert_balanced(chunk)


def test_realistic_telegram_answer_is_split_and_valid():
    body = "\n\n".join(
        f"<b>Пункт {i}.</b> Документы подаются через <i>Госуслуги</i> "
        f"или лично &laquo;в приёмной комиссии&raquo;. " + "Подробности. " * 25
        for i in range(30)
    )
    text = body + '\n\n<b>Источники:</b>\n<a href="https://www.nsu.ru/">Приём</a>'

    chunks = split_message(text, html=True)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= MESSAGE_CHUNK_LIMIT
        assert_balanced(chunk)
