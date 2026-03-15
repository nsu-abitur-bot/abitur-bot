import logging
import os

from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

_provider_instance: BaseLLMProvider | None = None


def get_llm_provider() -> BaseLLMProvider:
    """Создаёт LLM провайдер на основе переменной окружения LLM_PROVIDER.

    Поддерживаемые значения:
        - "gigachat" (по умолчанию)
        - "cerebras"
        - "deepseek"
        - "openai"
        - "gemini"
    """
    global _provider_instance
    if _provider_instance is not None:
        return _provider_instance

    provider_name = os.getenv("LLM_PROVIDER", "gigachat").lower()

    if provider_name == "gigachat":
        from llm.providers.gigachat import GigaChatProvider

        _provider_instance = GigaChatProvider()
    elif provider_name == "cerebras":
        from llm.providers.cerebras import CerebrasProvider

        _provider_instance = CerebrasProvider()
    elif provider_name == "deepseek":
        from llm.providers.deepseek import DeepSeekProvider

        _provider_instance = DeepSeekProvider()
    elif provider_name == "openai":
        from llm.providers.openai import OpenAIProvider

        _provider_instance = OpenAIProvider()
    elif provider_name == "gemini":
        from llm.providers.gemini import GeminiProvider

        _provider_instance = GeminiProvider()
    else:
        raise ValueError(
            f"Неизвестный LLM провайдер: '{provider_name}'. "
            f"Допустимые значения: gigachat, cerebras, deepseek, openai, gemini"
        )

    logger.info("LLM провайдер инициализирован: %s", provider_name)
    return _provider_instance
