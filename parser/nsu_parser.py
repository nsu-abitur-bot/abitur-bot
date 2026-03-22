import requests
from bs4 import BeautifulSoup

from parser.config.load_config import get_parser_config
from parser.utils import (
    calculate_page_hash,
    extract_documents_metadata,
    parse_content_blocks,
    parse_page_header,
)


def parse_page(url: str, selectors: list[dict], headers: dict) -> dict:
    """Парсит произвольную страницу по указанным селекторам."""
    if headers is None:
        headers = {}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        html_content = response.text
        page_hash = calculate_page_hash(html_content)

        soup = BeautifulSoup(html_content, "html.parser")
        data = {"page_hash": page_hash, "total_blocks": 0, "documents": []}

        parse_page_header(soup, data)

        # Извлекаем ссылки и названия документов для последующего скачивания
        # Можно легко добавить '.docx', '.xlsx' в этот список
        docs_metadata = extract_documents_metadata(
            soup, url, allowed_extensions=(".pdf",)
        )
        if docs_metadata:
            data["documents"] = docs_metadata

        k_blocks = 0

        if not selectors:
            print("Список селекторов в конфигурации пуст!")
            return data

        for selector in selectors:
            style = selector.get("style")
            attr = selector.get("attr")

            k_blocks = parse_content_blocks(
                soup,
                data,
                style=style,
                attr=attr,
                k_blocks=k_blocks,
            )

        data["total_blocks"] = k_blocks

        return data

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе: {e}")
        return {}
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        return {}


def parse_nsu_faculty(faculty) -> dict:
    """Парсит страницу факультета НГУ на основе конфигурации."""
    try:
        PARSER_CONFIG = get_parser_config()

        base_url = PARSER_CONFIG.get("url", "")
        headers = PARSER_CONFIG.get("headers", {})
        selectors = PARSER_CONFIG.get("selectors", [])

        if not base_url:
            print("Базовый URL не найден в конфигурации!")
            return {}

        full_page_url = f"{base_url}{faculty}"

        return parse_page(full_page_url, selectors, headers)

    except Exception as e:
        print(f"Ошибка при получении конфигурации или парсинге факультета: {e}")
        return {}
