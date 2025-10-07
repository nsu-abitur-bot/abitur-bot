import requests
from bs4 import BeautifulSoup
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from parser.utils import parse_header_faculty, parse_content_blocks
from parser.resources.config import NSU_CONFIG

def parse_nsu_faculty(faculty) -> dict:
    try:
        response = requests.get(f"{NSU_CONFIG["url"]}{faculty}", headers=NSU_CONFIG["headers"])
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        data = {}

        parse_header_faculty(soup, data)

        k_blocks = 0

        k_blocks = parse_content_blocks(soup, data, style="div", attr="tn-atom", k_blocks=k_blocks)
        k_blocks = parse_content_blocks(soup, data, style="div", attr="t585__text", k_blocks=k_blocks)
        # ещё парсинг блоков

        data["total_blocks"] = k_blocks

        return data

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе: {e}")
        return {}
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        return {}