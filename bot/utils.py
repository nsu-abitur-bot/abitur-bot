import re

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_DOUBLE_ENCODED_OCTET_RE = re.compile(r"%25([0-9A-Fa-f]{2})")


def normalize_links_for_messaging(text: str) -> str:
    """Исправляет двойное URL-кодирование в ссылках внутри текста.

    Некоторые клиенты могут повторно кодировать `%` и ломать URL вида `%20`,
    превращая их в `%2520`. Преобразуем только `%25XX -> %XX`, не меняя
    остальную структуру ссылки.
    """

    def _fix_url(match: re.Match[str]) -> str:
        url = match.group(0)
        trimmed_url = url.rstrip(")]>,.!?;:\"'")
        trailing = url[len(trimmed_url) :]

        if not _DOUBLE_ENCODED_OCTET_RE.search(trimmed_url):
            return url

        fixed = _DOUBLE_ENCODED_OCTET_RE.sub(r"%\1", trimmed_url)
        return f"{fixed}{trailing}"

    return _URL_RE.sub(_fix_url, text)
