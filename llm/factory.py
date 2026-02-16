import logging
import os

from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

_provider_instance: BaseLLMProvider | None = None


def get_llm_provider() -> BaseLLMProvider:
    """Создаёт LLM провайдер на основе переменной окружения LLM_PROVIDER.

    Поддерживаемые значения:
        - "gigachat" (по умолчанию)
    """
    global _provider_instance
    if _provider_instance is not None:
        return _provider_instance

    provider_name = os.getenv("LLM_PROVIDER", "gigachat").lower()

    if provider_name == "gigachat":
        from llm.providers.gigachat import GigaChatProvider

        _provider_instance = GigaChatProvider()
    else:
        raise ValueError(
            f"Неизвестный LLM провайдер: '{provider_name}'. "
            f"Допустимые значения: gigachat, local"
        )

    logger.info("LLM провайдер инициализирован: %s", provider_name)
    return _provider_instance
