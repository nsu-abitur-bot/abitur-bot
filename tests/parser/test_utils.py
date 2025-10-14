from parser.utils import parse_content_blocks, parse_header_faculty


class TestUtils:
    def test_parse_header_fit(self, load_html_fixture):
        soup = load_html_fixture("it.html")
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

    def test_parse_content_blocks_fit(self, load_html_fixture):
        soup = load_html_fixture("it.html")
        data = {}

        k_blocks = 0

        k_blocks = parse_content_blocks(
            soup, data, style="div", attr="tn-atom", k_blocks=k_blocks
        )

        data["total_blocks"] = k_blocks

        assert len(data["content_blocks"]) == data["total_blocks"] > 0
        assert data["total_blocks"] == 3
