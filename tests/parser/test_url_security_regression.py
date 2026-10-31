from unittest.mock import MagicMock, patch

from parser.nsu import parse_page


def test_parse_page_rejects_loopback_before_network():
    with patch(
        "requests.get", return_value=MagicMock(text="<title>internal</title>")
    ) as get:
        assert parse_page("http://127.0.0.1/", [], {}) == {}
        get.assert_not_called()


async def test_scores_rejects_loopback_before_network():
    from parser.scores import parse_scores

    with patch("requests.get", return_value=MagicMock(text="")) as get:
        assert await parse_scores("http://127.0.0.1/") == []
        get.assert_not_called()
