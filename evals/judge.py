import json
import re

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from llm.factory import get_llm_provider

# Системный промпт-судья
JUDGE_PROMPT = """Ты — беспристрастный судья. Оцени ответ RAG-системы по шкале
от 1 (полный провал) до 5 (идеально). Ответь СТРОГО в формате JSON без markdown-кавычек:
{"score": 5, "reasoning": "краткое объяснение"}"""


async def evaluate_rag_answer(
    question: str, reference: str, criteria: str, actual_answer: str
) -> dict:
    """Вызывает LLM для оценки ответа."""

    # Собираем данные для проверки
    human_text = f"""
    Вопрос абитуриента: {question}
    Эталонный ответ: {reference}
    Критерий оценки: {criteria}
    
    Фактический ответ RAG-системы: {actual_answer}
    """
    messages: list[BaseMessage] = [
        SystemMessage(content=JUDGE_PROMPT),
        HumanMessage(content=human_text),
    ]

    # 1. Получаем текущую LLM из .env
    provider = get_llm_provider()

    # 2. Делаем запрос
    llm_response = await provider.generate(messages)

    # 3. Достаём JSON из ответа через регулярку (на случай если LLM добавит ```json)
    try:
        match = re.search(r"\{.*?\}", llm_response, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {"score": 1, "reasoning": "Ошибка парсинга ответа LLM-судьи"}
    except json.JSONDecodeError:
        return {"score": 1, "reasoning": f"Невалидный JSON: {llm_response}"}
