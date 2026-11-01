import logging

from config.instance import get_setting
from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

_provider_instance: BaseLLMProvider | None = None
_provider_signature: tuple = ()


def get_llm_provider() -> BaseLLMProvider:
    """Создаёт LLM провайдер на основе переменной окружения LLM_PROVIDER.

    Поддерживаемые значения:
        - "openai"
        - "gemini" (по умолчанию)
    """
    global _provider_instance, _provider_signature
    signature = tuple(
        get_setting(k)
        for k in (
            "LLM_PROVIDER",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_SOCKS5_PROXY",
            "OPENAI_MODEL",
            "OPENAI_TEMPERATURE",
            "OPENAI_MAX_TOKENS",
            "OPENAI_TIMEOUT_SECONDS",
            "GEMINI_API_KEY",
            "GEMINI_BASE_URL",
            "GEMINI_MODEL",
            "GEMINI_TEMPERATURE",
        )
    )
    if _provider_instance is not None and signature == _provider_signature:
        return _provider_instance

    provider_name = get_setting("LLM_PROVIDER", "gemini").lower()

    if provider_name == "openai":
        from llm.providers.openai import OpenAIProvider

        _provider_instance = OpenAIProvider()
    elif provider_name == "gemini":
        from llm.providers.gemini import GeminiProvider

        _provider_instance = GeminiProvider()
    else:
        raise ValueError(
            f"Неизвестный LLM провайдер: '{provider_name}'. "
            f"Допустимые значения: openai, gemini"
        )

    _provider_signature = signature
    logger.info("LLM провайдер инициализирован: %s", provider_name)
    return _provider_instance


_faq_provider: BaseLLMProvider | None = None
_faq_signature: tuple = ()


def get_faq_provider() -> BaseLLMProvider:
    global _faq_provider, _faq_signature
    name = get_setting("FAQ_EMBEDDING_PROVIDER", get_setting("LLM_PROVIDER", "gemini"))
    if name == get_setting("LLM_PROVIDER", "gemini"):
        return get_llm_provider()
    signature = (
        name,
        *(
            get_setting(k)
            for k in (
                "OPENAI_API_KEY",
                "OPENAI_BASE_URL",
                "OPENAI_SOCKS5_PROXY",
                "GEMINI_API_KEY",
                "GEMINI_BASE_URL",
            )
        ),
    )
    if _faq_provider is None or signature != _faq_signature:
        if name == "openai":
            from llm.providers.openai import OpenAIProvider

            _faq_provider = OpenAIProvider()
        else:
            from llm.providers.gemini import GeminiProvider

            _faq_provider = GeminiProvider()
        _faq_signature = signature
    return _faq_provider
