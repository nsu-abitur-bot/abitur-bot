from parser.utils import calculate_page_hash, parse_content_blocks, parse_page_header


class TestUtils:
    def test_parse_header_fit(self, load_html_fixture):
        soup = load_html_fixture("it.html")
        data = {}

        parse_page_header(soup, data)

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

    def test_calculate_page_hash(self):
        # Тест с простым контентом
        content1 = "<html><body>Test content</body></html>"
        hash1 = calculate_page_hash(content1)

        # Хеш должен быть шестнадцатеричной строкой длиной 64 (SHA-256)
        assert isinstance(hash1, str)
        assert len(hash1) == 64
        assert all(c in "0123456789abcdef" for c in hash1)

        # Одинаковый контент должен давать одинаковый хеш
        hash1_duplicate = calculate_page_hash(content1)
        assert hash1 == hash1_duplicate

        # Разный контент должен давать разный хеш
        content2 = "<html><body>Different content</body></html>"
        hash2 = calculate_page_hash(content2)
        assert hash1 != hash2
