import logging
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings import GigaChatEmbeddings

logger = logging.getLogger(__name__)


class GigaChatLLM:
    def __init__(
        self,
        credentials: str,
        scope: str = "GIGACHAT_API_PERS",
        model: str = "GigaChat",
    ):
        self.credentials = credentials
        self.scope = scope
        self.model = model

    async def __call__(self, prompt: str, **kwargs) -> str:
        try:
            chat = GigaChat(
                credentials=self.credentials,
                scope=self.scope,
                model=self.model,
                verify_ssl_certs=False,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 2000),
            )

            messages = []
            if "system_prompt" in kwargs and kwargs["system_prompt"]:
                messages.append(SystemMessage(content=kwargs["system_prompt"]))

            messages.append(HumanMessage(content=prompt))  # type: ignore

            resp = await chat.ainvoke(messages)
            return str(resp.content)
        except Exception as e:
            logger.error(f"Error in GigaChatLLM: {e}")
            return ""


class GigaChatEmbedding:
    def __init__(
        self,
        credentials: str,
        scope: str = "GIGACHAT_API_PERS",
        model: str = "Embeddings",
        embedding_dim: int = 1024,
    ):
        self.credentials = credentials
        self.scope = scope
        self.model = model
        self.embedding_dim = embedding_dim

    async def __call__(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = GigaChatEmbeddings(
                credentials=self.credentials,
                scope=self.scope,
                model=self.model,
                verify_ssl_certs=False,
            )
            return await embeddings.aembed_documents(texts)
        except Exception as e:
            logger.error(f"Error in GigaChatEmbedding: {e}")
            return [[0.0] * self.embedding_dim for _ in texts]
