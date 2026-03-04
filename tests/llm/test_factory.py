from unittest.mock import Mock, patch

import pytest

from llm import factory
from llm.base import BaseLLMProvider


# Фикстура для сброса синглтона перед каждым тестом
@pytest.fixture(autouse=True)
def reset_provider():
    factory._provider_instance = None
    yield
    factory._provider_instance = None


# Мокаем классы, чтобы не делать реальных запросов и не требовать установки пакетов, если они опциональны  # noqa: E501
@pytest.fixture(autouse=True)
def mock_deps():
    with (
        patch("llm.providers.gigachat.GigaChat"),
        patch("llm.providers.cerebras._load_cerebras_class") as load_cerebras,
    ):
        load_cerebras.return_value = Mock()
        yield


def test_get_gigachat_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gigachat")
    monkeypatch.setenv("GIGACHAT_API_KEY", "fake_key")

    provider = factory.get_llm_provider()

    # Проверяем по имени класса
    assert type(provider).__name__ == "GigaChatProvider"
    assert isinstance(provider, BaseLLMProvider)


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unknown_provider")

    with pytest.raises(ValueError, match="Неизвестный LLM провайдер"):
        factory.get_llm_provider()


def test_default_provider_is_cerebras(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("CEREBRAS_API_KEY", "fake_key")

    provider = factory.get_llm_provider()

    assert type(provider).__name__ == "CerebrasProvider"


def test_get_cerebras_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "cerebras")
    monkeypatch.setenv("CEREBRAS_API_KEY", "fake_key")

    provider = factory.get_llm_provider()

    assert type(provider).__name__ == "CerebrasProvider"
    assert isinstance(provider, BaseLLMProvider)


def test_singleton_behavior(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gigachat")
    monkeypatch.setenv("GIGACHAT_API_KEY", "fake_key")

    provider1 = factory.get_llm_provider()
    provider2 = factory.get_llm_provider()

    assert provider1 is provider2
