"""Тесты парсера итогов приёма (по HTML-фикстуре, без сети)."""

from unittest.mock import MagicMock, patch

import pytest

from parser.scores import parse_scores, parse_scores_html


@pytest.fixture
def scores_html(html_fixture_path):
    return (html_fixture_path / "nsu_scores.html").read_text(encoding="utf-8")


@pytest.fixture
def scores_rows(scores_html):
    return parse_scores_html(scores_html)


def _find(rows, program, year, form):
    for row in rows:
        if row.program_name == program and row.year == year and row.form == form:
            return row
    return None


class TestParseScores:
    def test_parses_many_rows(self, scores_rows):
        assert len(scores_rows) > 100

    def test_faculties_and_forms(self, scores_rows):
        faculties = {r.faculty_name for r in scores_rows}
        assert "Факультет информационных технологий" in faculties
        assert {r.form for r in scores_rows} == {"budget", "paid"}
        assert {r.level for r in scores_rows} == {"bachelor", "specialist"}

    def test_fit_programmnaya_inzheneriya(self, scores_rows):
        """ФИТ / «Программная инженерия и компьютерные науки» (09.03.01)."""
        prog = "Программная инженерия и компьютерные науки"
        b2023 = _find(scores_rows, prog, 2023, "budget")
        b2024 = _find(scores_rows, prog, 2024, "budget")
        p2024 = _find(scores_rows, prog, 2024, "paid")

        assert b2023 is not None and b2023.passing_score == 245
        assert b2024 is not None and b2024.passing_score == 246
        assert p2024 is not None and p2024.passing_score == 191

        assert b2024.faculty_name == "Факультет информационных технологий"
        assert b2024.code == "09.03.01"
        assert b2024.level == "bachelor"

    def test_fit_kompyuternye_nauki(self, scores_rows):
        """ФИТ / «Компьютерные науки и системотехника» (09.03.01)."""
        prog = "Компьютерные науки и системотехника"
        assert _find(scores_rows, prog, 2023, "budget").passing_score == 258
        assert _find(scores_rows, prog, 2024, "budget").passing_score == 260
        assert _find(scores_rows, prog, 2024, "paid").passing_score == 231

    def test_average_score_is_float(self, scores_rows):
        row = _find(
            scores_rows,
            "Программная инженерия и компьютерные науки",
            2024,
            "budget",
        )
        assert isinstance(row.average_score, float)
        assert row.average_score == pytest.approx(260.9)

    def test_zero_and_dash_cells_skipped(self, scores_rows):
        """Проходной/средний балл никогда не равны 0; годы без данных не попадают."""
        for row in scores_rows:
            assert row.passing_score is None or row.passing_score > 0
            assert row.average_score is None or row.average_score > 0
            # Полностью пустые (оба None) строки не эмитятся.
            assert row.passing_score is not None or row.average_score is not None


class TestParseScoresAsync:
    @patch("parser.scores.requests.get")
    def test_parse_scores_fetches_and_parses(self, mock_get, scores_html):
        import asyncio

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.text = scores_html
        mock_get.return_value = mock_response

        rows = asyncio.run(parse_scores("http://example.test/itogi/"))
        assert len(rows) > 100
        mock_get.assert_called_once()

    @patch("parser.scores.requests.get")
    def test_parse_scores_network_error_returns_empty(self, mock_get):
        import asyncio

        import requests

        mock_get.side_effect = requests.exceptions.RequestException("boom")
        rows = asyncio.run(parse_scores("http://example.test/itogi/"))
        assert rows == []
