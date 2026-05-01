import pytest

from llm import factory
from llm.base import BaseLLMProvider


@pytest.fixture(autouse=True)
def reset_provider():
    factory._provider_instance = None
    yield
    factory._provider_instance = None


def test_get_gemini_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")

    provider = factory.get_llm_provider()

    assert type(provider).__name__ == "GeminiProvider"
    assert isinstance(provider, BaseLLMProvider)


def test_get_openai_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")

    provider = factory.get_llm_provider()

    assert type(provider).__name__ == "OpenAIProvider"
    assert isinstance(provider, BaseLLMProvider)


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "unknown_provider")

    with pytest.raises(ValueError, match="Неизвестный LLM провайдер"):
        factory.get_llm_provider()


def test_default_provider_is_gemini(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")

    provider = factory.get_llm_provider()

    assert type(provider).__name__ == "GeminiProvider"


def test_singleton_behavior(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")

    provider1 = factory.get_llm_provider()
    provider2 = factory.get_llm_provider()

    assert provider1 is provider2
