"""Форматирование ответа под канал: Telegram получает HTML, MAX — обычный текст."""

from bot.formatting import (
    format_for_channel,
    render_sources_html,
    render_sources_plain,
    sanitize_telegram_html,
    to_plain_text,
)

# Как отвечает модель: разметка вольная, теги не из набора Telegram.
LLM_TEXT = "<p>Приём идёт до <strong>25 июля</strong>.</p><p>Подробнее ниже.</p>"
SOURCES = [
    {"url": "https://www.nsu.ru/postuplenie", "title": "Приём 2026"},
    {"url": "https://www.nsu.ru/itogi", "title": "Итоги приёма"},
]


class TestMaxChannel:
    def test_max_gets_no_tags(self):
        """Баг #288: в MAX уходили сырые <b> и <a href=...>."""
        result = format_for_channel(LLM_TEXT, SOURCES, "max")
        assert "<" not in result and ">" not in result

    def test_max_keeps_source_urls(self):
        result = format_for_channel(LLM_TEXT, SOURCES, "max")
        assert "Источники:" in result
        assert "Приём 2026 — https://www.nsu.ru/postuplenie" in result

    def test_max_keeps_text(self):
        result = format_for_channel(LLM_TEXT, SOURCES, "max")
        assert "Приём идёт до 25 июля." in result
        assert "Подробнее ниже." in result

    def test_link_inside_text_keeps_address(self):
        """Ссылка в тексте не должна потерять адрес там, где нет разметки."""
        plain = to_plain_text('Смотри <a href="https://nsu.ru/x">тут</a>.')
        assert plain == "Смотри тут (https://nsu.ru/x)."

    def test_entities_are_decoded(self):
        assert to_plain_text("Иванов &amp; Петров") == "Иванов & Петров"


class TestTelegramChannel:
    def test_unsupported_tags_are_converted(self):
        result = sanitize_telegram_html(LLM_TEXT)
        assert "<strong>" not in result and "<p>" not in result
        assert "<b>25 июля</b>" in result

    def test_paragraphs_become_blank_lines(self):
        assert sanitize_telegram_html(LLM_TEXT) == (
            "Приём идёт до <b>25 июля</b>.\n\nПодробнее ниже."
        )

    def test_sources_are_links(self):
        result = format_for_channel(LLM_TEXT, SOURCES, "telegram")
        assert "<b>Источники:</b>" in result
        assert '<a href="https://www.nsu.ru/postuplenie">Приём 2026</a>' in result

    def test_url_with_space_is_encoded(self):
        result = sanitize_telegram_html('<a href="https://nsu.ru/a b">тут</a>')
        assert result == '<a href="https://nsu.ru/a%20b">тут</a>'

    def test_br_becomes_newline(self):
        assert sanitize_telegram_html("первая<br>вторая") == "первая\nвторая"


class TestSources:
    def test_no_sources_gives_empty_block(self):
        assert render_sources_html([]) == ""
        assert render_sources_plain([]) == ""

    def test_at_most_three_sources(self):
        many = [{"url": f"https://nsu.ru/{i}", "title": f"Т{i}"} for i in range(10)]
        assert render_sources_plain(many).count("https://") == 3

    def test_source_without_url_is_skipped(self):
        sources = [{"url": "", "title": "Без ссылки"}, SOURCES[0]]
        result = render_sources_plain(sources)
        assert "Без ссылки" not in result
        assert "Приём 2026" in result

    def test_source_without_title_gets_default(self):
        result = render_sources_plain([{"url": "https://nsu.ru/x"}])
        assert "Источник информации — https://nsu.ru/x" in result
