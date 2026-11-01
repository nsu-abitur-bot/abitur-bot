import pytest

from api.services.instance_settings import ConfigurationError, merge_changes
from config import instance
from db.postgres.services.instance_settings import InstanceSnapshot
from llm.profiles import LLMProfiles


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(instance, "_values", None)
    monkeypatch.setattr(instance, "_revision", 0)


def test_database_snapshot_wins_even_for_cleared_secrets(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "old-env-token")
    instance.install({"BOT_TOKEN": ""}, 3)
    assert instance.get_setting("BOT_TOKEN") is None
    assert instance.revision() == 3
    assert not instance.enabled("TELEGRAM_ENABLED")


@pytest.mark.parametrize(
    "key,value",
    [
        ("UNKNOWN", "bad"),
        ("OPENAI_TEMPERATURE", "NaN"),
        ("OPENAI_TEMPERATURE", "3"),
        ("RATING_POLL_MIN_SECONDS", "0"),
        ("TELEGRAM_ENABLED", "yes"),
        ("OPENAI_BASE_URL", "file:///tmp/key"),
        ("OPENAI_BASE_URL", "https://user:pass@example.test"),
        ("RATING_MAX_SESSION_REFRESHES", "1.5"),
    ],
)
def test_invalid_changes_rejected(key, value):
    with pytest.raises(ConfigurationError):
        merge_changes(InstanceSnapshot(0, {}), {key: value})


def test_profiles_follow_runtime_changes_without_reimport():
    instance.install({"PROFILE_CHAT_MAX_TOKENS": "512"}, 1)
    assert LLMProfiles.CHAT.max_tokens == 512
    assert LLMProfiles.VISION.max_tokens == 8192


def test_formatter_redacts_current_retired_and_urlencoded_secrets():
    import logging
    from urllib.parse import quote

    from logging_config import SecretSafeFormatter

    instance.install({"BOT_TOKEN": "old:secret", "OPENAI_API_KEY": "current:key"}, 1)
    instance.install({"BOT_TOKEN": "new:secret"}, 2)
    raw = "old:secret new:secret " + quote("current:key", safe="")
    record = logging.LogRecord("test", logging.ERROR, "", 0, raw, (), None)
    formatted = SecretSafeFormatter().format(record)
    assert "secret" not in formatted and "current" not in formatted
    assert formatted.count("[REDACTED]") == 3


@pytest.mark.parametrize("value", ["1.0", "1e3"])
def test_integer_settings_reject_float_notation_used_by_integer_consumers(value):
    with pytest.raises(ConfigurationError):
        merge_changes(InstanceSnapshot(0, {}), {"RATING_MAX_SESSION_REFRESHES": value})
