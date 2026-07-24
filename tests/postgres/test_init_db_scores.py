"""Тесты автозаливки проходных баллов при старте.

Без реальной БД и без сети: подменяем parse_scores, AsyncSessionLocal,
AdmissionScoreService и SettingsService — проверяем управляющую логику.
"""

import contextlib
from datetime import UTC, datetime, timedelta

import pytest

from db.postgres import init_db


class _DummySession:
    pass


@contextlib.asynccontextmanager
async def _dummy_session():
    yield _DummySession()


class _FakeScoreService:
    upserted: list | None = None

    def __init__(self, session):
        pass

    async def upsert_from_rows(self, rows):
        _FakeScoreService.upserted = list(rows)
        return {"created": len(_FakeScoreService.upserted), "updated": 0, "skipped": 0}


class _FakeSettings:
    """Фейковые настройки: отметка времени последней заливки."""

    stored: str | None = None
    written: str | None = None

    def __init__(self, session):
        pass

    async def get_value(self, key):
        assert key == init_db.LAST_SCORES_IMPORT_KEY
        return _FakeSettings.stored

    async def set_value(self, key, value, description=""):
        assert key == init_db.LAST_SCORES_IMPORT_KEY
        _FakeSettings.written = value


@pytest.fixture(autouse=True)
def _reset():
    _FakeScoreService.upserted = None
    _FakeSettings.stored = None
    _FakeSettings.written = None
    yield


def _patch(monkeypatch, parse_result=None, parse_error=None):
    """Подменяет всё внешнее. Возвращает список URL, с которыми звали парсер."""
    calls: list[str] = []

    async def fake_parse_scores(url):
        calls.append(url)
        if parse_error is not None:
            raise parse_error
        return parse_result if parse_result is not None else []

    monkeypatch.setattr("db.postgres.db.AsyncSessionLocal", _dummy_session)
    monkeypatch.setattr(
        "db.postgres.services.admission_score.AdmissionScoreService", _FakeScoreService
    )
    monkeypatch.setattr("db.postgres.services.settings.SettingsService", _FakeSettings)
    monkeypatch.setattr("parser.scores.parse_scores", fake_parse_scores)
    for var in (
        "SEED_ADMISSION_SCORES",
        "SEED_ADMISSION_SCORES_FORCE",
        "ADMISSION_SCORES_MAX_AGE_HOURS",
        "ADMISSION_SCORES_URL",
    ):
        monkeypatch.delenv(var, raising=False)
    return calls


def _iso_ago(**kwargs) -> str:
    return (datetime.now(UTC).replace(tzinfo=None) - timedelta(**kwargs)).isoformat()


@pytest.mark.asyncio
async def test_disabled_by_env_skips_everything(monkeypatch):
    calls = _patch(monkeypatch, parse_result=["row"])
    monkeypatch.setenv("SEED_ADMISSION_SCORES", "0")

    await init_db.seed_admission_scores()

    assert calls == []
    assert _FakeScoreService.upserted is None


@pytest.mark.asyncio
async def test_recent_import_skips_fetch(monkeypatch):
    """Недавно заливали — сайт НГУ не дёргаем (защита от рестарт-цикла)."""
    calls = _patch(monkeypatch, parse_result=["row"])
    _FakeSettings.stored = _iso_ago(hours=1)

    await init_db.seed_admission_scores()

    assert calls == []
    assert _FakeScoreService.upserted is None


@pytest.mark.asyncio
async def test_stale_import_triggers_refresh(monkeypatch):
    calls = _patch(monkeypatch, parse_result=["row-1", "row-2"])
    _FakeSettings.stored = _iso_ago(hours=48)

    await init_db.seed_admission_scores()

    assert len(calls) == 1
    assert _FakeScoreService.upserted == ["row-1", "row-2"]
    assert _FakeSettings.written is not None  # отметка обновлена


@pytest.mark.asyncio
async def test_first_run_imports_and_records_marker(monkeypatch):
    calls = _patch(monkeypatch, parse_result=["row"])
    _FakeSettings.stored = None  # отметки ещё нет

    await init_db.seed_admission_scores()

    assert len(calls) == 1
    assert _FakeScoreService.upserted == ["row"]
    assert _FakeSettings.written is not None


@pytest.mark.asyncio
async def test_force_ignores_freshness(monkeypatch):
    calls = _patch(monkeypatch, parse_result=["row"])
    _FakeSettings.stored = _iso_ago(minutes=1)
    monkeypatch.setenv("SEED_ADMISSION_SCORES_FORCE", "1")

    await init_db.seed_admission_scores()

    assert len(calls) == 1
    assert _FakeScoreService.upserted == ["row"]


@pytest.mark.asyncio
async def test_corrupted_marker_is_treated_as_stale(monkeypatch):
    calls = _patch(monkeypatch, parse_result=["row"])
    _FakeSettings.stored = "не-дата"

    await init_db.seed_admission_scores()

    assert len(calls) == 1
    assert _FakeScoreService.upserted == ["row"]


@pytest.mark.asyncio
async def test_empty_parse_result_does_not_upsert(monkeypatch):
    """Сайт недоступен (парсер вернул []) — ничего не пишем и отметку не двигаем."""
    calls = _patch(monkeypatch, parse_result=[])

    await init_db.seed_admission_scores()

    assert len(calls) == 1
    assert _FakeScoreService.upserted is None
    assert _FakeSettings.written is None


@pytest.mark.asyncio
async def test_parser_exception_is_not_fatal(monkeypatch):
    """Исключение внутри заливки не должно ронять старт бота."""
    calls = _patch(monkeypatch, parse_error=RuntimeError("boom"))

    await init_db.seed_admission_scores()  # не бросает

    assert len(calls) == 1
    assert _FakeScoreService.upserted is None


@pytest.mark.asyncio
async def test_custom_url_from_env(monkeypatch):
    calls = _patch(monkeypatch, parse_result=["row"])
    monkeypatch.setenv("ADMISSION_SCORES_URL", "https://example.test/scores")

    await init_db.seed_admission_scores()

    assert calls == ["https://example.test/scores"]
