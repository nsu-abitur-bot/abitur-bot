"""Форматирование ответа под конкретный канал.

Пайплайн (`pipeline.llm_client.ask_local_llm`) возвращает текст без разметки канала
и отдельно список источников. Разметку выбирает адаптер канала, потому что
Telegram и MAX понимают её по-разному: Telegram принимает ограниченный набор
HTML-тегов, MAX получает обычный текст (`bot/core.py` ставит ему
`parse_mode=None`). Раньше пайплайн безусловно готовил Telegram-HTML, и
пользователи MAX видели в ответах сырые `<b>` и `<a href=...>`.
"""

import logging
import re
from html import escape, unescape

from bot.utils import normalize_url_for_messaging

logger = logging.getLogger(__name__)

MAX_SOURCES = 3
DEFAULT_SOURCE_TITLE = "Источник информации"

# Теги, которые Telegram понимает без атрибутов.
_TELEGRAM_SIMPLE_TAGS = {"b", "i", "u", "s", "code", "pre"}
_HREF_RE = re.compile(r"href\s*=\s*[\"\']([^\"\']+)[\"\']", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_ANCHOR_RE = re.compile(
    r"<a\s+[^>]*href\s*=\s*[\"\']([^\"\']+)[\"\'][^>]*>(.*?)</a>",
    re.IGNORECASE | re.DOTALL,
)


def sanitize_telegram_html(text: str) -> str:
    """Оставляет только те теги, которые понимает Telegram.

    Модель отвечает разметкой вольно: `<p>`, `<strong>`, `<br>`. Telegram на
    незнакомом теге отвечает ошибкой разбора, поэтому приводим к его набору.
    """
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>\s*<p>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?p>", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"</?strong>",
        lambda m: "</b>" if m.group(0).startswith("</") else "<b>",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"</?em>",
        lambda m: "</i>" if m.group(0).startswith("</") else "<i>",
        text,
        flags=re.IGNORECASE,
    )

    def replace_tag(match: re.Match[str]) -> str:
        raw_tag = match.group(0)
        tag = raw_tag.strip("<>").strip()
        is_closing = tag.startswith("/")
        tag_body = tag[1:].strip() if is_closing else tag
        tag_name = tag_body.split()[0].lower() if tag_body else ""

        if tag_name in _TELEGRAM_SIMPLE_TAGS:
            return f"</{tag_name}>" if is_closing else f"<{tag_name}>"

        if tag_name == "a":
            if is_closing:
                return "</a>"
            href_match = _HREF_RE.search(tag_body)
            if href_match:
                url = normalize_url_for_messaging(href_match.group(1))
                return f'<a href="{escape(url, quote=True)}">'
            return ""

        return ""

    return _TAG_RE.sub(replace_tag, text).strip()


def _anchor_to_text(match: re.Match[str]) -> str:
    """`<a href=U>T</a>` -> `T (U)`, чтобы адрес не потерялся в тексте без разметки."""
    url = normalize_url_for_messaging(match.group(1))
    label = _TAG_RE.sub("", match.group(2)).strip()
    if not label:
        return url
    return f"{label} ({url})"


def to_plain_text(text: str) -> str:
    """Снимает разметку целиком — для каналов без HTML (MAX)."""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>\s*<p>", "\n\n", text, flags=re.IGNORECASE)
    text = _ANCHOR_RE.sub(_anchor_to_text, text)
    text = _TAG_RE.sub("", text)
    return unescape(text).strip()


def _visible_sources(sources: list[dict]) -> list[tuple[str, str]]:
    """Пары (название, ссылка) для показа: не больше MAX_SOURCES, без пустых."""
    out = []
    for source in sources[:MAX_SOURCES]:
        url = str(source.get("url") or "")
        if not url:
            continue
        title = str(source.get("title") or DEFAULT_SOURCE_TITLE)
        out.append((title, url))
    return out


def render_sources_html(sources: list[dict]) -> str:
    """Блок «Источники» ссылками — для Telegram."""
    visible = _visible_sources(sources)
    if not visible:
        return ""
    links = []
    for title, url in visible:
        href = escape(normalize_url_for_messaging(url), quote=True)
        links.append(f'<a href="{href}">{escape(title)}</a>')
    return "\n\n<b>Источники:</b>\n" + "\n".join(links)


def render_sources_plain(sources: list[dict]) -> str:
    """Блок «Источники» текстом — для каналов без разметки."""
    visible = _visible_sources(sources)
    if not visible:
        return ""
    lines = [f"{title} — {normalize_url_for_messaging(url)}" for title, url in visible]
    return "\n\nИсточники:\n" + "\n".join(lines)


def format_for_channel(text: str, sources: list[dict], channel: str) -> str:
    """Готовый текст ответа для канала: разметка плюс блок источников."""
    if channel == "telegram":
        return sanitize_telegram_html(text) + render_sources_html(sources)
    return to_plain_text(text) + render_sources_plain(sources)
