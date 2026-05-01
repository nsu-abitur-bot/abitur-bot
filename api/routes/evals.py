import json
import logging
import os

from fastapi import APIRouter, BackgroundTasks

from db.redis.client import RedisClient
from evals.evaluator import PipelineEvaluator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evals", tags=["Evaluations"])


async def get_eval_status() -> dict:
    """Получает текущий статус оценки из Redis."""
    redis = RedisClient()
    val = await redis.client.get("eval_status")
    if val:
        return json.loads(val)
    return {"is_running": False, "last_result": None, "error": None}


async def set_eval_status(status: dict):
    """Сохраняет текущий статус оценки в Redis (на 1 неделю)."""
    redis = RedisClient()
    await redis.client.set("eval_status", json.dumps(status), ex=7 * 24 * 60 * 60)


async def background_eval_task():
    status = await get_eval_status()
    status["is_running"] = True
    status["error"] = None
    status["last_result"] = None
    await set_eval_status(status)

    try:
        # Запускаем твой обновленный пайплайн
        logger.info("Запуск пайплайна оценки...")
        evaluator = PipelineEvaluator("evals/scenarios.yaml")
        result = await evaluator.run()

        status = await get_eval_status()
        status["last_result"] = result
        await set_eval_status(status)

        # Сохраняем в файл на случай перезагрузки сервера
        try:
            os.makedirs("evals", exist_ok=True)
            with open("evals/eval_results.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Не удалось сохранить результаты оценки в файл: {e}")

    except Exception as e:
        logger.error(f"Ошибка при выполнении оценки: {e}")
        status = await get_eval_status()
        status["error"] = str(e)
        await set_eval_status(status)
    finally:
        status = await get_eval_status()
        status["is_running"] = False
        await set_eval_status(status)


@router.post("/run")
async def run_evaluation(background_tasks: BackgroundTasks):
    status = await get_eval_status()
    if status["is_running"]:
        return {"message": "Оценка уже запущена. Пожалуйста, подождите."}

    # Запускаем фоновую задачу для выполнения оценки
    background_tasks.add_task(background_eval_task)
    return {"message": "Оценка запущена в фоновом режиме."}


@router.get("/status")
async def route_get_evaluation_status():
    return await get_eval_status()
