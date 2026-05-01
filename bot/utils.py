import re
from urllib.parse import quote, splitport, splituser, urlsplit, urlunsplit

_URL_RE = re.compile(r"https?://[^\s<>'\"]+", re.IGNORECASE)
_DOUBLE_ENCODED_OCTET_RE = re.compile(r"%25([0-9A-Fa-f]{2})")


def _encode_url_parts(url: str) -> str:
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return url

    userinfo, hostport = splituser(parts.netloc)
    if hostport is None:
        hostport = parts.netloc
        userinfo = None

    host, port = splitport(hostport)
    if host is None:
        host = hostport

    if host.startswith("[") and host.endswith("]"):
        safe_host = host
    else:
        try:
            safe_host = host.encode("idna").decode("ascii")
        except Exception:
            safe_host = host

    safe_netloc = safe_host
    if port:
        safe_netloc = f"{safe_netloc}:{port}"
    if userinfo:
        safe_netloc = f"{userinfo}@{safe_netloc}"

    safe_path = quote(parts.path, safe="/%:@&=+$,;~*'()!-._")
    safe_query = quote(parts.query, safe="=&%+;:@/?-._~")
    safe_fragment = quote(parts.fragment, safe="%!-._~")

    return urlunsplit((parts.scheme, safe_netloc, safe_path, safe_query, safe_fragment))


def normalize_url_for_messaging(url: str) -> str:
    trimmed_url = url.rstrip(")]>,.!?;:\"'")
    trailing = url[len(trimmed_url) :]

    if not trimmed_url:
        return url

    fixed = _DOUBLE_ENCODED_OCTET_RE.sub(r"%\1", trimmed_url)
    try:
        fixed = _encode_url_parts(fixed)
    except Exception:
        return url

    return f"{fixed}{trailing}"


def normalize_links_for_messaging(text: str) -> str:
    """Исправляет двойное URL-кодирование в ссылках внутри текста.

    Некоторые клиенты могут повторно кодировать `%` и ломать URL вида `%20`,
    превращая их в `%2520`. Преобразуем только `%25XX -> %XX`, не меняя
    остальную структуру ссылки.
    """

    def _fix_url(match: re.Match[str]) -> str:
        return normalize_url_for_messaging(match.group(0))

    return _URL_RE.sub(_fix_url, text)
