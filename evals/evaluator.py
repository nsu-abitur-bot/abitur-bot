import asyncio
import json
import logging
import os
import random

import yaml
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

import rag.loader
import rag.retriever
from evals.judge import evaluate_rag_answer
from faq.matcher import get_faq_matcher
from llm.factory import get_llm_provider

logger = logging.getLogger(__name__)


class PipelineEvaluator:
    def __init__(self, scenarios_path: str = "evals/scenarios.yaml"):
        self.scenarios = []

        # Подгружаем статические сценарии (гардрейлы, галлюцинации и т.д.)
        if os.path.exists(scenarios_path):
            with open(scenarios_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                self.scenarios.extend(data.get("scenarios", []))

    async def generate_dynamic_scenarios(self, count: int = 3):
        """Автоматически генерирует вопросы на основе случайных документов из реальной базы."""
        chunk_file = "./data/lightrag/abitur_kb/kv_store_text_chunks.json"

        if not os.path.exists(chunk_file):
            logger.warning(
                "База данных графа (chunks) не найдена. Динамическая генерация пропущена."
            )
            return

        with open(chunk_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = list(data.values())
        if not chunks:
            return

        # Берем случайные куски достаточной длины
        valid_chunks = [ch for ch in chunks if len(ch.get("content", "")) > 100]
        selected = random.sample(valid_chunks, min(count, len(valid_chunks)))

        llm = get_llm_provider()

        for chunk in selected:
            content = chunk.get("content", "")
            prompt: list[BaseMessage] = [
                SystemMessage(
                    content="Ты — проверяющий эксперт. Твоя задача сгенерировать 1 конкретный тестовый вопрос (question) по предоставленному тексту, "
                    "написать эталонный краткий ответ (reference_answer) и критерии оценки (criteria). "
                    "Вопрос должен проверять, сможет ли RAG-система найти этот факт. "
                    "Верни ТОЛЬКО валидный JSON. Ключи: question, reference_answer, criteria."
                ),
                HumanMessage(content=f"Текст из базы НГУ:\n{content}"),
            ]

            resp = await llm.generate(prompt)
            try:
                clean_json = resp.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean_json)
                parsed["id"] = f"dynamic_rag_{random.randint(1000, 9999)}"
                parsed["type"] = "rag"
                self.scenarios.append(parsed)
                logger.info(f"Добавлен динамический сценарий: {parsed['question']}")
            except Exception as e:
                logger.error(
                    f"Ошибка парсинга динамического сценария: {e} | Ответ: {resp}"
                )

            await asyncio.sleep(5)  # Задержка, чтобы не спамить API при генерации

    async def run(self) -> dict:
        # Генерируем 3 случайных вопроса из боевой RAG базы перед стартом
        logger.info("Генерация тестовых RAG сценариев на основе текущих документов...")
        await self.generate_dynamic_scenarios(count=3)

        report = {
            "total": len(self.scenarios),
            "passed_perfectly": 0,
            "rag_results": [],
            "faq_results": []
        }

        faq_matcher = get_faq_matcher()

        for scenario in self.scenarios:
            scenario_type = scenario.get("type", "rag")

            if scenario_type == "faq":
                # --- ЛОГИКА ОЦЕНКИ FAQ ---
                question = scenario["question"]
                expected_match = scenario.get("expected_match", True)
                
                # Вызываем FAQ Matcher
                faq_response = faq_matcher.match(question)
                is_matched = bool(faq_response)

                passed = (is_matched == expected_match)
                if passed:
                    report["passed_perfectly"] += 1

                report["faq_results"].append({
                    "id": scenario["id"],
                    "question": question,
                    "expected_match": expected_match,
                    "actual_match": is_matched,
                    "passed": passed,
                    "faq_answer": faq_response if is_matched else None
                })
            else:
                # --- ЛОГИКА ОЦЕНКИ RAG ---
                # 1. Спрашиваем НАПРЯМУЮ RAG (LightRAG), минуя бота и FAQ
                try:
                    rag_answer = await rag.retriever.query_graph(
                        query=scenario["question"], mode="hybrid"
                    )
                except Exception as e:
                    logger.error(f"Error querying RAG graph directly: {str(e)}")
                    rag_answer = f"Error: {str(e)}"

                # 2. Отдаём LLM-судье на оценку
                evaluation = await evaluate_rag_answer(
                    question=scenario["question"],
                    reference=scenario.get("reference_answer", ""),
                    criteria=scenario.get("criteria", ""),
                    actual_answer=rag_answer,
                )

                score = evaluation.get("score", 0)
                if score >= 4:
                    report["passed_perfectly"] += 1

                report["rag_results"].append(
                    {
                        "id": scenario["id"],
                        "question": scenario["question"],
                        "rag_answer": rag_answer,
                        "judge_score": score,
                        "judge_reasoning": evaluation.get("reasoning", ""),
                    }
                )

                # Добавляем паузу между сценариями, чтобы не превысить лимит (обычно 15 RPM для бесплатных LLM)
                logger.info(
                    "Ожидание 15 секунд перед следующим сценарием для обхода лимитов LLM (15 RPM)..."
                )
                await asyncio.sleep(15)

        return report


if __name__ == "__main__":
    import asyncio
    import json

    async def main():
        logging.basicConfig(level=logging.INFO)
        evaluator = PipelineEvaluator("evals/scenarios.yaml")
        result = await evaluator.run()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    asyncio.run(main())
