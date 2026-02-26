import json
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_path():
    return _FIXTURES_DIR


@pytest.fixture(scope="session")
def mock_json_response():
    json_path = _FIXTURES_DIR / "json" / "mock_rating_response.json"
    return json.loads(json_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def mock_html_with_csrf():
    return (_FIXTURES_DIR / "html" / "csrf_present.html").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def mock_html_without_csrf():
    return (_FIXTURES_DIR / "html" / "csrf_absent.html").read_text(encoding="utf-8")


@pytest.fixture
def html_fixture_path(fixtures_path):
    return fixtures_path / "html"


@pytest.fixture
def load_html_fixture(html_fixture_path):
    def _loader(filename):
        filepath = html_fixture_path / filename
        with open(filepath, "r", encoding="utf-8") as file:
            html_content = file.read()
        return BeautifulSoup(html_content, "html.parser")

    return _loader
