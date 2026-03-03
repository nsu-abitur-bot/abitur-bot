import logging
import os

from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

_provider_instance: BaseLLMProvider | None = None


def get_llm_provider() -> BaseLLMProvider:
    """Создаёт LLM провайдер на основе переменной окружения LLM_PROVIDER.

    Поддерживаемые значения:
        - "cerebras" (по умолчанию)
        - "gigachat"
    """
    global _provider_instance
    if _provider_instance is not None:
        return _provider_instance

    provider_name = os.getenv("LLM_PROVIDER", "cerebras").lower()

    if provider_name == "gigachat":
        from llm.providers.gigachat import GigaChatProvider

        _provider_instance = GigaChatProvider()
    elif provider_name == "cerebras":
        from llm.providers.cerebras import CerebrasProvider

        _provider_instance = CerebrasProvider()
    else:
        raise ValueError(
            f"Неизвестный LLM провайдер: '{provider_name}'. "
            f"Допустимые значения: gigachat, cerebras"
        )

    logger.info("LLM провайдер инициализирован: %s", provider_name)
    return _provider_instance
