import hashlib
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlsplit

import fitz  # PyMuPDF
import requests


def parse_pdf_from_url(url: str, headers: dict) -> str:
    """Загружает PDF по ссылке (сохраняя во временный файл)
    и извлекает текст с помощью PyMuPDF."""
    tmp_name = ""
    try:
        # Добавлена небольшая задержка, чтобы не спамить запросами
        time.sleep(1)
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        # Сохраняем во временный файл,
        # delete=False для корректного переоткрытия в Windows
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)
            tmp_name = tmp_file.name

        # Читаем через PyMuPDF (fitz) *вне* блока with (когда файл закрыт для записи)
        doc = fitz.open(tmp_name)
        text_blocks = []
        for page in doc:
            text = page.get_text("text")
            if isinstance(text, str) and text.strip():
                text_blocks.append(text)
        doc.close()

        return "\n".join(text_blocks)
    except Exception as e:
        print(f"Ошибка при парсинге PDF ({url}): {e}")
        return ""
    finally:
        # Вручную удаляем временный файл
        if tmp_name and os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass


def parse_document_from_url(url: str, headers: dict, ext: str) -> str:
    """Обертка для загрузки различных форматов."""
    if ext == ".pdf":
        return parse_pdf_from_url(url, headers)
    # Здесь можно добавить обработчики для .docx, .xlsx и т.д.
    print(f"Парсинг формата {ext} пока не поддерживается (Url: {url}).")
    return ""


def download_and_parse_documents(documents: list[dict], headers: dict) -> list[dict]:
    """Скачивает и парсит выбранные документы.
    Возвращает список словарей с текстами:
    [{"name": "...", "url": "...", "text": "...", "extension": ".pdf"}, ...]
    """
    results = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_doc = {}
        for doc in documents:
            url = doc["url"]
            ext = doc["extension"]
            print(f"Постановка в очередь документа: {url}")
            future = executor.submit(parse_document_from_url, url, headers, ext)
            future_to_doc[future] = doc

        for future in as_completed(future_to_doc):
            doc = future_to_doc[future]
            try:
                text = future.result()
                if text and text.strip():
                    results.append(
                        {
                            "name": doc["name"],
                            "url": doc["url"],
                            "extension": doc["extension"],
                            "text": text.strip(),
                        }
                    )
            except Exception as e:
                print(f"Исключение при парсинге {doc['url']}: {e}")

    return results


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

        # Проверяем, соответствует ли файл разрешенным расширениям
        if any(path.endswith(ext) for ext in allowed_extensions):
            if full_url in seen_urls:
                continue

            seen_urls.add(full_url)

            # Пытаемся получить адекватное название из текста ссылки
            # Заменит все теги и переносы строк внутри на пробелы
            name = link.get_text(separator=" ", strip=True)
            if not name:
                # Если текст пустой, берем имя файла из URL
                name = os.path.basename(path)

            ext = next(ext for ext in allowed_extensions if path.endswith(ext))
            docs.append({"name": name, "url": full_url, "extension": ext})

    return docs


def calculate_page_hash(html_content: str) -> str:
    """Вычисляет SHA-256 хеш HTML контента.

    Args:
        html_content: HTML контент в виде строки

    Returns:
        Шестнадцатеричное представление хеша
    """
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
