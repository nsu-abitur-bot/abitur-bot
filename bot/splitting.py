"""Нарезка длинного ответа на сообщения, которые примет мессенджер.

Telegram не принимает сообщение длиннее 4096 символов, MAX — от 4000 и выше
(проверка в `maxapi.methods.send_message`). Ответ модели с источниками и
футером бывает длиннее, и раньше такой ответ просто не доходил до человека.

Режем с запасом до `MESSAGE_CHUNK_LIMIT`, предпочитая границы покрупнее:
абзац, строка, конец предложения, пробел — и только в крайнем случае рвём
посреди слова. Для Telegram-HTML каждая часть должна разбираться сама по
себе: разрез не попадает внутрь тега или HTML-сущности, а теги, открытые на
границе, закрываются в конце части и открываются заново в начале следующей.
"""

import re

# Один запас на оба канала: меньше лимита Telegram (4096) и MAX (< 4000),
# с местом под закрывающие теги на границе.
MESSAGE_CHUNK_LIMIT = 3900

# От крупной границы к мелкой. Разрез ставим после разделителя.
_SEPARATORS = ("\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ")

_TAG_RE = re.compile(r"<(/?)([a-zA-Z]+)[^>]*>")
_ENTITY_RE = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")


def _inside_markup(text: str, cut: int) -> int | None:
    """Если разрез в `cut` попадает внутрь тега или сущности — их начало."""
    head = text[:cut]
    last_lt = head.rfind("<")
    if last_lt > head.rfind(">"):
        return last_lt
    last_amp = head.rfind("&")
    if last_amp != -1 and ";" not in head[last_amp:]:
        entity = _ENTITY_RE.match(text, last_amp)
        if entity is not None and entity.end() > cut:
            return last_amp
    return None


def _find_cut(text: str, budget: int, html: bool) -> int:
    """Позиция разреза не дальше `budget`: по самой крупной доступной границе.

    Граница в самом начале куска не годится — получилась бы крошечная часть,
    поэтому ищем её не левее четверти бюджета.
    """
    floor = budget // 4
    for sep in _SEPARATORS:
        pos = text.rfind(sep, 0, budget - len(sep) + 1)
        while pos > floor:
            cut = pos + len(sep)
            if not html or _inside_markup(text, cut) is None:
                return cut
            pos = text.rfind(sep, 0, pos)

    cut = budget
    if html:
        start = _inside_markup(text, cut)
        if start is not None and start > 0:
            cut = start
    return cut


def _open_tags(html: str) -> list[tuple[str, str]]:
    """Теги, открытые к концу `html`: пары (имя, открывающий тег как есть)."""
    stack: list[tuple[str, str]] = []
    for match in _TAG_RE.finditer(html):
        name = match.group(2).lower()
        if match.group(1):
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == name:
                    del stack[i]
                    break
        else:
            stack.append((name, match.group(0)))
    return stack


def split_message(
    text: str, limit: int = MESSAGE_CHUNK_LIMIT, html: bool = False
) -> list[str]:
    """Режет `text` на части не длиннее `limit` символов, без пустых частей.

    `html=True` — текст в Telegram-HTML: каждая часть остаётся валидной
    разметкой сама по себе.
    """
    rest = text.strip()
    chunks: list[str] = []
    while rest:
        if len(rest) <= limit:
            chunks.append(rest)
            break

        budget = limit
        while True:
            cut = _find_cut(rest, budget, html)
            head = rest[:cut].rstrip()
            tail = rest[cut:].lstrip()
            if not html:
                break
            stack = _open_tags(head)
            closers = "".join(f"</{name}>" for name, _ in reversed(stack))
            overflow = len(head) + len(closers) - limit
            # Вложенных тегов не бывает столько, чтобы съесть полбюджета; если
            # всё же так — режем без них, Telegram-fallback отправит текстом.
            if overflow > 0 and budget - overflow > limit // 2:
                budget -= overflow
                continue
            if overflow <= 0:
                reopened = "".join(tag for _, tag in stack) + tail
                head += closers
                # Повторно открытые теги не должны мешать продвижению вперёд.
                if len(reopened) < len(rest):
                    tail = reopened
            break

        if head:
            chunks.append(head)
        rest = tail
    return chunks
