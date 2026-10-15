"""Загрузка списков для планировщика: ошибка отличима от пустого списка (#286).

HTTP-сессия подменяется MagicMock — к сайту НГУ тесты не ходят.
"""

import json
from unittest.mock import MagicMock

import pytest
import requests

import parser.rating as rating_module
from parser.rating import (
    ListFetchResult,
    RatingListFetcher,
    parse_rating_url_with_session,
)

URL = "https://abiturient.nsu.ru/bachelor?faculty=8&direction=7&condition=10&type=30"


def _response(payload=None, status_error: Exception | None = None, text=None):
    response = MagicMock()
    response.text = text if text is not None else json.dumps(payload)
    response.json.return_value = payload
    response.raise_for_status = MagicMock(side_effect=status_error)
    return response


def _session_returning(*responses) -> MagicMock:
    session = MagicMock()
    session.post.side_effect = list(responses)
    return session


class TestParseRatingUrlWithSession:
    def test_success_is_ok(self, mock_json_response):
        session = _session_returning(_response(mock_json_response))
        result = parse_rating_url_with_session(session, "token", URL)
        assert result.ok
        assert len(result.entries) == 5
        assert result.page_hash

    def test_genuinely_empty_list_is_ok(self):
        session = _session_returning(_response({"items": []}))
        result = parse_rating_url_with_session(session, "token", URL)
        assert result.ok
        assert result.entries == []

    def test_http_500_is_error_not_empty(self):
        error = requests.exceptions.HTTPError("500 Server Error")
        session = _session_returning(_response({}, status_error=error))
        result = parse_rating_url_with_session(session, "token", URL)
        assert not result.ok
        assert "500" in (result.error or "")

    def test_connection_error_is_error(self):
        session = MagicMock()
        session.post.side_effect = requests.exceptions.ConnectionError("reset")
        result = parse_rating_url_with_session(session, "token", URL)
        assert not result.ok

    def test_non_json_response_is_error(self):
        response = _response(text="<html>Сессия истекла</html>")
        response.json.side_effect = ValueError("not json")
        result = parse_rating_url_with_session(_session_returning(response), "t", URL)
        assert not result.ok

    def test_unexpected_shape_is_error(self):
        session = _session_returning(_response({"status": "error"}))
        result = parse_rating_url_with_session(session, "token", URL)
        assert not result.ok
        assert "items" in (result.error or "")

    def test_bad_url_is_not_retryable(self):
        result = parse_rating_url_with_session(MagicMock(), "t", "https://x/?faculty=abc")
        assert not result.ok
        assert not result.retryable


class TestRatingListFetcher:
    @pytest.fixture
    def sessions(self, monkeypatch):
        """Каждый create_rating_session отдаёт следующую подготовленную сессию."""
        queue: list[MagicMock | None] = []
        created: list[MagicMock] = []

        def fake_create():
            session = queue.pop(0)
            if session is None:
                return None, None
            created.append(session)
            return session, f"token-{len(created)}"

        monkeypatch.setattr(rating_module, "create_rating_session", fake_create)
        return queue, created

    def test_refreshes_session_and_retries_once_on_error(
        self, sessions, mock_json_response
    ):
        queue, created = sessions
        expired = _session_returning(
            _response({}, status_error=requests.exceptions.HTTPError("400 csrf"))
        )
        fresh = _session_returning(_response(mock_json_response))
        queue.extend([expired, fresh])

        fetcher = RatingListFetcher(max_refreshes=3)
        assert fetcher.open()
        result = fetcher.fetch(URL)

        assert result.ok
        assert len(result.entries) == 5
        assert created == [expired, fresh]
        expired.close.assert_called_once()
        # Повтор ушёл с новым токеном.
        sent = fresh.post.call_args.kwargs["data"]["_csrf-frontend"]
        assert sent == "token-2"

    def test_retry_failure_is_reported_as_error(self, sessions):
        queue, _ = sessions
        down = requests.exceptions.HTTPError("503")
        queue.extend(
            [
                _session_returning(_response({}, status_error=down)),
                _session_returning(_response({}, status_error=down)),
            ]
        )
        fetcher = RatingListFetcher(max_refreshes=3)
        fetcher.open()
        result = fetcher.fetch(URL)
        assert not result.ok
        assert fetcher.refreshes_left == 2

    def test_refresh_budget_is_limited(self, sessions):
        queue, created = sessions
        down = requests.exceptions.HTTPError("503")
        queue.extend(
            [
                _session_returning(*[_response({}, status_error=down)] * 2),
                _session_returning(*[_response({}, status_error=down)] * 3),
            ]
        )
        fetcher = RatingListFetcher(max_refreshes=1)
        fetcher.open()
        assert not fetcher.fetch(URL).ok
        assert not fetcher.fetch(URL).ok
        assert not fetcher.fetch(URL).ok
        # Одна стартовая сессия и одна перезапрошенная — дальше без перезапросов.
        assert len(created) == 2

    def test_no_token_on_refresh_stops_further_refreshes(self, sessions):
        queue, _ = sessions
        down = requests.exceptions.HTTPError("503")
        queue.extend([_session_returning(*[_response({}, status_error=down)] * 2), None])
        fetcher = RatingListFetcher(max_refreshes=3)
        fetcher.open()
        result = fetcher.fetch(URL)
        assert not result.ok
        assert fetcher.refreshes_left == 0

    def test_bad_url_is_not_retried(self, sessions):
        queue, created = sessions
        queue.append(MagicMock())
        fetcher = RatingListFetcher(max_refreshes=3)
        fetcher.open()
        assert not fetcher.fetch("https://x/?faculty=abc").ok
        assert len(created) == 1


def test_failure_helper():
    result = ListFetchResult.failure("boom")
    assert not result.ok and result.entries == [] and result.page_hash == ""
