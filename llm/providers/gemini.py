import logging
import os
from typing import Any, List, Optional

from google import genai
from google.genai import types
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Переменная окружения GEMINI_API_KEY не установлена.")

        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
        self.temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))

        self.client = genai.Client(api_key=api_key)

        logger.info(
            "Gemini провайдер инициализирован (модель: %s)",
            self.model_name,
        )

    async def generate(self, messages: List[BaseMessage]) -> str:
        system_instruction = None
        gemini_messages = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_instruction = str(msg.content)
            elif isinstance(msg, AIMessage):
                gemini_messages.append(
                    {"role": "model", "parts": [{"text": str(msg.content)}]}
                )
            elif isinstance(msg, HumanMessage):
                gemini_messages.append(
                    {"role": "user", "parts": [{"text": str(msg.content)}]}
                )
            else:
                gemini_messages.append(
                    {"role": "user", "parts": [{"text": str(msg.content)}]}
                )

        config = types.GenerateContentConfig(
            temperature=self.temperature,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=gemini_messages,
            config=config,
        )

        content = response.text
        return content.strip() if isinstance(content, str) else str(content or "")

    def get_embeddings_model(self) -> Any:
        return GeminiEmbeddings(self.client)


class GeminiEmbeddings:
    """Воркер для эмбеддингов Gemini, совместимый с FAQMatcher."""

    def __init__(self, client: genai.Client):
        self.client = client
        self.model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # google-genai SDK поддерживает пакетную обработку
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
        )
        # response.embeddings — список объектов с полем values
        if not response.embeddings:
            return []
        return [list(e.values or []) for e in response.embeddings]
