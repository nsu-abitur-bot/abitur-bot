import requests
from bs4 import BeautifulSoup

from utils import parse_header_faculty, parse_content_blocks
from config import NSU_CONFIG

def parse_nsu(faculty) -> dict:
    try:
        response = requests.get(f"{NSU_CONFIG["url"]}{faculty}", headers=NSU_CONFIG["headers"])
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        data = {}

        parse_header_faculty(soup, data)

        parse_content_blocks(soup, data, style="div", attr="tn-atom")

        return data

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе: {e}")
        return {}
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        return {}