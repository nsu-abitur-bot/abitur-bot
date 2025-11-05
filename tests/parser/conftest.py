from pathlib import Path

import pytest
from bs4 import BeautifulSoup


@pytest.fixture
def fixtures_path():
    return Path(__file__).parent / "fixtures"


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
