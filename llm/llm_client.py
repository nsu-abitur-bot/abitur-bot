import httpx
import re
import logging

LM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "Llama-3.2-3B-Instruct-Q4_K_S.gguf"

SYSTEM_PROMPT = """
ТЫ LLM помощник для поступления в НГУ (Новосибирский государственный университет), отвечай только на вопросы связанные с университетом и поступлением.
Отвечай коротко, долго не думай
"""

logger = logging.getLogger(__name__)


async def ask_local_llm(message: str, session_id: str) -> str:
    """
    message - сообщение от пользователя
    session_id - идентификатор переписки, необходим для сохранения контекста (сейчас не используется)
    """
    try:
        prompt = SYSTEM_PROMPT
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": message},
            ],
            "temperature": 0.8,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(LM_API_URL, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"]
        logger.debug(f"LLM response: {content}")

        # Удаляем <think>...</think>, если есть
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

        logger.debug(f"answer: {content}")
        if not content:
            logger.warning("No messages found in response")

        return content

    except httpx.RequestError as e:
        logger.error(f"LLM request error: {e}")
        return "Что-то пошло не так"
    except Exception as e:
        logger.error(f"LLM response processing error: {e}")
        return "Что-то пошло не так"
