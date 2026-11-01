from unittest.mock import patch

from parser.nsu import parse_page
from parser.scores import parse_scores


def test_parse_page_rejects_loopback_before_network():
    with patch("httpx.HTTPTransport.handle_request") as send:
        assert parse_page("http://127.0.0.1/", [], {}) == {}
        send.assert_not_called()


async def test_scores_rejects_loopback_before_network():
    with patch("httpx.AsyncHTTPTransport.handle_async_request") as send:
        assert await parse_scores("http://127.0.0.1/") == []
        send.assert_not_called()
