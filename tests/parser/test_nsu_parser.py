from unittest.mock import MagicMock, patch

from parser.nsu_parser import parse_nsu_faculty


class TestNsuParser:
    @patch("parser.nsu_parser.requests.get")
    def test_parse_nsu_fit(self, mock_get, html_fixture_path):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with open(html_fixture_path / "it.html", "r", encoding="utf-8") as file:
            mock_response.text = file.read()

        mock_get.return_value = mock_response

        data = parse_nsu_faculty("information-technologies/")

        assert "title" in data
        assert data["title"] == "Информационные технологии НГУ"
        assert data["total_blocks"] > 0
