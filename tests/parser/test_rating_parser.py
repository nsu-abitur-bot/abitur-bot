from unittest.mock import MagicMock, patch

import pytest

from db.postgres.dto import RatingEntry
from parser.rating import (
    _extract_entries,
    _extract_params_from_url,
    _get_csrf_token,
    build_leaderboard_url,
    find_entry_by_identifier,
    parse_rating_page,
)

# ─── build_leaderboard_url ────────────────────────────────────────────────────


class TestBuildLeaderboardUrl:
    def test_returns_correct_url(self):
        url = build_leaderboard_url(faculty=8, direction=7, condition=10, type_=0)
        assert "faculty=8" in url
        assert "direction=7" in url
        assert "condition=10" in url
        assert "type=0" in url

    def test_default_type_is_zero(self):
        url = build_leaderboard_url(faculty=8, direction=7, condition=10)
        assert "type=0" in url

    def test_custom_type(self):
        url = build_leaderboard_url(faculty=1, direction=2, condition=3, type_=5)
        assert "type=5" in url


# ─── _extract_params_from_url ─────────────────────────────────────────────────


class TestExtractParamsFromUrl:
    def test_extracts_all_params(self):
        url = "https://abiturient.nsu.ru/bachelor?faculty=8&direction=7&condition=10&type=0"
        params = _extract_params_from_url(url)
        assert params["faculty"] == 8
        assert params["direction"] == 7
        assert params["condition"] == 10
        assert params["type"] == 0

    def test_uses_defaults_when_params_missing(self):
        url = "https://abiturient.nsu.ru/bachelor"
        params = _extract_params_from_url(url)
        assert params["faculty"] == 8
        assert params["direction"] == 7
        assert params["condition"] == 10
        assert params["type"] == 0

    def test_returns_int_types(self):
        url = "https://abiturient.nsu.ru/bachelor?faculty=3&direction=5&condition=2&type=1"
        params = _extract_params_from_url(url)
        for value in params.values():
            assert isinstance(value, int)


# ─── _extract_entries ─────────────────────────────────────────────────────────


class TestExtractEntries:
    def test_extracts_correct_count(self, mock_json_response):
        entries = _extract_entries(mock_json_response)
        assert len(entries) == 5

    def test_extracts_competition_type(self, mock_json_response):
        entries = _extract_entries(mock_json_response)
        types = {e.competition_type for e in entries}
        assert "отдельная квота" in types
        assert "общий конкурс" in types

    def test_empty_title_becomes_bez_nazvaniya(self, mock_json_response):
        entries = _extract_entries(mock_json_response)
        no_name = [e for e in entries if e.competition_type == "без названия"]
        assert len(no_name) == 1

    def test_extracts_identifier_and_place(self, mock_json_response):
        entries = _extract_entries(mock_json_response)
        first = next(e for e in entries if e.identifier == "4305351")
        assert first.place == 1
        assert first.status == "Зачислен"

    def test_empty_response(self):
        entries = _extract_entries({})
        assert entries == []

    def test_empty_items(self):
        entries = _extract_entries({"items": []})
        assert entries == []

    def test_skips_invalid_rows(self):
        data = {
            "items": [
                {
                    "title": "общий конкурс",
                    "table": [
                        {"number": "не число", "name": "123"},  # невалидный number
                        {"number": 1, "name": "456", "status": "Подано"},
                    ],
                }
            ]
        }
        entries = _extract_entries(data)
        assert len(entries) == 1
        assert entries[0].identifier == "456"

    def test_skips_zero_place(self):
        data = {
            "items": [
                {
                    "title": "общий конкурс",
                    "table": [
                        {"number": 0, "name": "123", "status": "Подано"},
                        {"number": 1, "name": "456", "status": "Подано"},
                    ],
                }
            ]
        }
        entries = _extract_entries(data)
        assert len(entries) == 1


# ─── find_entry_by_identifier ─────────────────────────────────────────────────


class TestFindEntryByIdentifier:
    @pytest.fixture
    def sample_entries(self):
        return [
            RatingEntry(
                identifier="4305351",
                place=1,
                status="Зачислен",
                competition_type="отдельная квота",
            ),
            RatingEntry(
                identifier="3945082",
                place=1,
                status="Зачислен",
                competition_type="общий конкурс",
            ),
            RatingEntry(
                identifier="4953537",
                place=1,
                status="Подано",
                competition_type="без названия",
            ),
        ]

    def test_finds_existing_entry(self, sample_entries):
        result = find_entry_by_identifier(sample_entries, "3945082")
        assert result is not None
        assert result.identifier == "3945082"
        assert result.competition_type == "общий конкурс"

    def test_returns_none_for_missing(self, sample_entries):
        result = find_entry_by_identifier(sample_entries, "9999999")
        assert result is None

    def test_returns_none_for_empty_list(self):
        result = find_entry_by_identifier([], "4305351")
        assert result is None


# ─── _get_csrf_token ──────────────────────────────────────────────────────────


class TestGetCsrfToken:
    def test_returns_token_when_found(self, mock_html_with_csrf):
        session = MagicMock()
        response = MagicMock()
        response.text = mock_html_with_csrf
        response.raise_for_status = MagicMock()
        session.get.return_value = response

        token = _get_csrf_token(session, {})
        assert token == "test-csrf-token-12345"

    def test_returns_none_when_no_meta(self, mock_html_without_csrf):
        session = MagicMock()
        response = MagicMock()
        response.text = mock_html_without_csrf
        response.raise_for_status = MagicMock()
        session.get.return_value = response

        token = _get_csrf_token(session, {})
        assert token is None

    def test_returns_none_on_request_error(self):
        import requests

        session = MagicMock()
        session.get.side_effect = requests.exceptions.ConnectionError()

        token = _get_csrf_token(session, {})
        assert token is None


# ─── parse_rating_page ────────────────────────────────────────────────────────


class TestParseRatingPage:
    URL = "https://abiturient.nsu.ru/bachelor?faculty=8&direction=7&condition=10&type=0"

    @patch("parser.rating.requests.Session")
    def test_returns_entries_on_success(
        self, mock_session_cls, mock_json_response, mock_html_with_csrf
    ):
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)

        # GET для CSRF
        get_response = MagicMock()
        get_response.text = mock_html_with_csrf
        get_response.raise_for_status = MagicMock()

        # POST для данных
        post_response = MagicMock()
        post_response.json.return_value = mock_json_response
        post_response.text = '{"items": []}'
        post_response.raise_for_status = MagicMock()

        mock_session.get.return_value = get_response
        mock_session.post.return_value = post_response

        entries, page_hash, direction_name = parse_rating_page(self.URL)

        assert len(entries) == 5
        assert page_hash != ""

    @patch("parser.rating.requests.Session")
    def test_returns_empty_when_no_csrf(self, mock_session_cls, mock_html_without_csrf):
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)

        get_response = MagicMock()
        get_response.text = mock_html_without_csrf
        get_response.raise_for_status = MagicMock()
        mock_session.get.return_value = get_response

        entries, page_hash, direction_name = parse_rating_page(self.URL)

        assert entries == []
        assert page_hash == ""
        assert direction_name == ""

    @patch("parser.rating.requests.Session")
    def test_returns_empty_on_post_error(self, mock_session_cls, mock_html_with_csrf):
        import requests

        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)

        get_response = MagicMock()
        get_response.text = mock_html_with_csrf
        get_response.raise_for_status = MagicMock()
        mock_session.get.return_value = get_response
        mock_session.post.side_effect = requests.exceptions.ConnectionError()

        entries, page_hash, direction_name = parse_rating_page(self.URL)

        assert entries == []
        assert page_hash == ""
        assert direction_name == ""
