import asyncio
import logging
import os
import uuid

import yaml

import rag.loader
import rag.retriever
from db.redis_client import RedisClient
from evals.judge import evaluate_bot_answer
from llm.llm_client import ask_local_llm
from rag.graph_memory import get_graph_memory
from rag.loader import add_texts_async

logger = logging.getLogger(__name__)


class PipelineEvaluator:
    def __init__(self, scenarios_path: str = "evals/scenarios.yaml"):
        # ПЕРЕХВАТ ПУТИ ДЛЯ RAG: заставляем LightRAG смотреть в тестовую папку
        os.environ["LIGHTRAG_WORKSPACE_BASE"] = "./data/lightrag_eval"

        self.session_id = f"eval_session_{uuid.uuid4()}"
        # Передаем user_id=0, в llm_client.py логика сохранения часто пропускает запись,
        # если id 0 или None. Если это не поможет,
        # в будущем можно добавить mock для postgres
        self.test_user_id = 0

        with open(scenarios_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            self.test_documents = data.get("test_documents", [])
            self.scenarios = data.get("scenarios", [])

    async def _setup_test_graph(self):
        """Инжектим выдуманный текст в базу."""
        await add_texts_async(self.test_documents, graph_id="eval_kb")
        logger.info(
            f"Тестовые документы ({len(self.test_documents)} шт.) загружены в eval_kb"
        )

    async def run(self) -> dict:
        report = {"total": len(self.scenarios), "passed_perfectly": 0, "results": []}

        try:
            for scenario in self.scenarios:
                # 1. Спрашиваем реального бота
                # Тут потребуется так де временно подменить graph_id в retriever'е,
                # если ask_local_llm использует дефолтный abitur_kb
                original_graph = rag.retriever.DEFAULT_GRAPH_ID
                setattr(rag.retriever, "DEFAULT_GRAPH_ID", "eval_kb")
                setattr(rag.loader, "DEFAULT_GRAPH_ID", "eval_kb")

                try:
                    bot_answer = await ask_local_llm(
                        message=scenario["question"],
                        session_id=self.session_id,
                        user_id=self.test_user_id,
                    )
                except Exception as e:
                    logger.error(f"Error asking local LLM: {str(e)}")
                    bot_answer = f"Error: {str(e)}"
                finally:
                    rag.retriever.DEFAULT_GRAPH_ID = original_graph
                    rag.loader.DEFAULT_GRAPH_ID = original_graph

                # 2. Отдаём LLM-судье на оценку
                evaluation = await evaluate_bot_answer(
                    question=scenario["question"],
                    reference=scenario["reference_answer"],
                    criteria=scenario["criteria"],
                    actual_answer=bot_answer,
                )

                score = evaluation.get("score", 0)
                if score >= 4:
                    report["passed_perfectly"] += 1

                report["results"].append(
                    {
                        "id": scenario["id"],
                        "question": scenario["question"],
                        "bot_answer": bot_answer,
                        "judge_score": score,
                        "judge_reasoning": evaluation.get("reasoning", ""),
                    }
                )

                # Если сценарий требует добавления документов в граф ПОСЛЕ проверки
                if scenario.get("load_documents_after"):
                    await self._setup_test_graph()

                # Добавляем паузу между сценариями, чтобы не превысить лимит (обычно 15 RPM для бесплатных LLM)
                logger.info(
                    "Ожидание 15 секунд перед следующим сценарием для обхода лимитов LLM (15 RPM)..."
                )
                await asyncio.sleep(15)

        finally:
            # 3. Удаляем мусор из Redis
            redis = RedisClient()
            await redis.clear_history(self.session_id)
            logger.info("Тестовая история Redis очищена")

            # 4. Удаляем добавленные документы из LightRAG графа
            try:
                gm = get_graph_memory()
                docs = await gm.get_list_docs("eval_kb")
                deleted_count = 0
                for doc in docs:
                    doc_id = doc.get("id")
                    if doc_id:
                        success = await gm.delete_doc("eval_kb", doc_id)
                        if success:
                            deleted_count += 1
                        else:
                            logger.warning(
                                f"Не удалось удалить документ {doc_id} из eval_kb"
                            )
                logger.info(f"Удалено документов из графа eval_kb: {deleted_count}")
            except Exception as e:
                logger.error(f"Ошибка при очистке документов графа: {e}")

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
