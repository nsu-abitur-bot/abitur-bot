import requests
from bs4 import BeautifulSoup

from parser.config.load_config import get_parser_config
from parser.utils import parse_content_blocks, parse_header_faculty


class TestUtils:
    def test_parse_header_fit(self):
        PARSER_CONFIG = get_parser_config()

        response = requests.get(
            f"{PARSER_CONFIG['url']}information-technologies/",
            headers=PARSER_CONFIG["headers"],
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        data = {}

        parse_header_faculty(soup, data)

        assert data["title"] == "Информационные технологии НГУ"
        assert (
            data["description"]
            == "Станьте IT-специалистом с глубокими знаниями в информатике, математике и физике. Работа в высокотехнологичных компаниях и научных центрах."  # noqa: E501
        )
        assert (
            data["keywords"]
            == "информационные технологии, НГУ, программирование, математика, физика, IT-специалист, высокотехнологичные компании"  # noqa: E501
        )
        assert "content_blocks" not in data

    def test_parse_content_blocks_fit(self):
        PARSER_CONFIG = get_parser_config()

        response = requests.get(
            f"{PARSER_CONFIG['url']}information-technologies/",
            headers=PARSER_CONFIG["headers"],
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        data = {}

        k_blocks = 0

        k_blocks = parse_content_blocks(
            soup, data, style="div", attr="tn-atom", k_blocks=k_blocks
        )

        data["total_blocks"] = k_blocks

        assert len(data["content_blocks"]) == data["total_blocks"] > 0
