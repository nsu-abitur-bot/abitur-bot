import requests
from bs4 import BeautifulSoup

from parser.config.load_config import get_parser_config
from parser.utils import calculate_page_hash, parse_content_blocks, parse_header_faculty


def parse_nsu_faculty(faculty) -> dict:
    try:
        PARSER_CONFIG = get_parser_config()

        response = requests.get(
            f"{PARSER_CONFIG['url']}{faculty}", headers=PARSER_CONFIG["headers"]
        )
        response.raise_for_status()

        html_content = response.text
        page_hash = calculate_page_hash(html_content)

        soup = BeautifulSoup(html_content, "html.parser")
        data = {"page_hash": page_hash, "total_blocks": 0}

        parse_header_faculty(soup, data)

        k_blocks = 0

        selectors = PARSER_CONFIG.get("selectors", [])

        if not selectors:
            print("Список селекторов в конфигурации пуст!")
            return {}

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
