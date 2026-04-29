import logging
import os
from typing import Any, List, Optional

from langchain_core.messages import BaseMessage
from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings import GigaChatEmbeddings

from llm.base import BaseLLMProvider
from llm.profiles import LLMProfile

logger = logging.getLogger(__name__)


class GigaChatProvider(BaseLLMProvider):
    def __init__(self):
        # TODO
        # Настроить проверку SSL-сертификата через переменную окружения.
        # По умолчанию проверка включена для безопасности.

        api_key = os.getenv("GIGACHAT_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Переменная окружения GIGACHAT_API_KEY не установлена. "
                "Пожалуйста, настройте её для использования провайдера GigaChat."
            )
        model_name = os.getenv("GIGACHAT_MODEL", "GigaChat")

        self.llm = GigaChat(
            credentials=os.getenv("GIGACHAT_API_KEY"),
            scope="GIGACHAT_API_PERS",
            model=model_name,
            verify_ssl_certs=False,
        )
        logger.info(
            "GigaChat провайдер инициализирован (модель: %s)",
            model_name,
        )

    async def generate(
        self,
        messages: List[BaseMessage],
        profile: Optional[LLMProfile] = None,
        **kwargs,
    ) -> str:
        # LangChain LLM instances are immutable/Pydantic-based often,
        # so we pass config overrides via bind or config directly to ainvoke.
        kwargs_invoke = {}
        if profile:
            if profile.temperature is not None:
                kwargs_invoke['temperature'] = profile.temperature
            if profile.max_tokens is not None:
                kwargs_invoke['max_tokens'] = profile.max_tokens
            if profile.timeout is not None:
                kwargs_invoke['timeout'] = profile.timeout
        
        # Merge other kwargs
        kwargs_invoke.update(kwargs)

        llm_binded = self.llm.bind(**kwargs_invoke) if kwargs_invoke else self.llm

        response = await llm_binded.ainvoke(messages)
        if isinstance(response.content, str):
            content = response.content.strip()
        else:
            content = str(response.content) if response.content else ""
        return content

    def get_embeddings_model(self) -> Any:
        return GigaChatEmbeddings(
            credentials=os.getenv("GIGACHAT_API_KEY"),
            scope="GIGACHAT_API_PERS",
            model=os.getenv("GIGACHAT_EMBEDDING_MODEL", "Embeddings"),
            verify_ssl_certs=False,
        )
