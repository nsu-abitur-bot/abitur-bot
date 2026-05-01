import hashlib
import os
from urllib.parse import urljoin, urlsplit


def extract_sources(text: str, fallback: str) -> tuple[str, str]:
    """Ищет 'Источники: <url1>, <url2>' или 'Источник: <url>' в начале текста.

    Возвращает (primary_url, all_urls_joined) для использования в ids и file_paths.
    """
    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("источники:"):
            urls_str = stripped.split(":", 1)[1].strip()
            urls = [u.strip() for u in urls_str.split(",") if u.strip()]
            if urls:
                return urls[0], ", ".join(urls)
        elif lower.startswith("источник:"):
            url = stripped.split(":", 1)[1].strip()
            if url:
                return url, url
    return fallback, fallback


def extract_documents_metadata(
    soup, base_url: str, allowed_extensions=(".pdf",)
) -> list[dict]:
    """Находит все документы на странице и возвращает их названия и ссылки."""
    docs = []
    seen_urls = set()

    links = soup.find_all("a", href=True)

    for link in links:
        href = link["href"]
        full_url = urljoin(base_url, href)
        path = urlsplit(full_url).path.lower()

        if any(path.endswith(ext) for ext in allowed_extensions):
            if full_url in seen_urls:
                continue

            seen_urls.add(full_url)

            name = link.get_text(separator=" ", strip=True)
            if not name:
                name = os.path.basename(path)

            ext = next(ext for ext in allowed_extensions if path.endswith(ext))
            docs.append({"name": name, "url": full_url, "extension": ext})

    return docs


def calculate_page_hash(html_content: str) -> str:
    """Вычисляет SHA-256 хеш HTML контента."""
    return hashlib.sha256(html_content.encode("utf-8")).hexdigest()


def parse_page_header(soup, data):
    title = soup.title.text if soup.title else "Не найдено"
    data["title"] = title

    meta_desc = soup.find("meta", attrs={"name": "description"})
    data["description"] = meta_desc["content"] if meta_desc else ""

    meta_keywords = soup.find("meta", attrs={"name": "keywords"})
    data["keywords"] = meta_keywords["content"] if meta_keywords else ""


def parse_content_blocks(soup, data, style=None, attr=None, k_blocks=0) -> int:
    if style and attr:
        selector = f"{style}.{attr}"
    elif style:
        selector = style
    else:
        return k_blocks

    content_blocks = soup.select(selector)
    blocks = (
        [block.text for block in content_blocks if block.text and block.text.strip()]
        if content_blocks
        else []
    )

    if "content_blocks" not in data:
        data["content_blocks"] = blocks
    else:
        data["content_blocks"].extend(blocks)

    return k_blocks + len(blocks)
