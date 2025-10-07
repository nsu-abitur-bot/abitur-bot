from bs4 import BeautifulSoup
import requests
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from parser.resources.config import NSU_CONFIG
from parser.utils import parse_header_faculty, parse_content_blocks

class TestUtils:
    def test_parse_header_faculty(self):
        response = requests.get(f"{NSU_CONFIG["url"]}information-technologies/", headers=NSU_CONFIG["headers"])
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        data = {}

        parse_header_faculty(soup, data)

        assert data["title"] == "Информационные технологии НГУ"
        assert data["description"] == (
            "Станьте IT-специалистом с глубокими знаниями в информатике, математике и физике. "
            "Работа в высокотехнологичных компаниях и научных центрах.")
        assert data[
                   "keywords"] == "информационные технологии, НГУ, программирование, математика, физика, IT-специалист, высокотехнологичные компании"

    def test_parse_content_blocks(self):
        response = requests.get(f"{NSU_CONFIG["url"]}information-technologies/", headers=NSU_CONFIG["headers"])
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        data = {}

        k_blocks = 0
        k_blocks = parse_content_blocks(soup, data, style="div", attr="tn-atom", k_blocks=k_blocks)

        data["total_blocks"] = k_blocks

        assert len(data["content_blocks"]) == 81
        assert data["content_blocks"][0] == "БАКАЛАВРИАТ"
        assert data["content_blocks"][80] == "Мы в соцсетях"
        assert data["total_blocks"] == 81