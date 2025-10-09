import requests
from bs4 import BeautifulSoup

from parser.config.load_config import get_parser_config
from parser.utils import parse_content_blocks, parse_header_faculty


def parse_nsu_faculty(faculty) -> dict:
    try:
        PARSER_CONFIG = get_parser_config()

        response = requests.get(
            f"{PARSER_CONFIG['url']}{faculty}", headers=PARSER_CONFIG["headers"]
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        data = {}

        parse_header_faculty(soup, data)

        k_blocks = 0

        k_blocks = parse_content_blocks(
            soup, data, style="div", attr="tn-atom", k_blocks=k_blocks
        )
        k_blocks = parse_content_blocks(
            soup, data, style="div", attr="t585__text", k_blocks=k_blocks
        )
        # ещё парсинг блоков

        data["total_blocks"] = k_blocks

        return data

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе: {e}")
        return {}
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        return {}
