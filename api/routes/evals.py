import logging

from fastapi import APIRouter, BackgroundTasks

from evals.evaluator import PipelineEvaluator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evals", tags=["Evaluations"])

# Временное хранилище статуса в памяти (для продакшена лучше Redis)
eval_status = {"is_running": False, "last_result": None, "error": None}


async def background_eval_task():
    eval_status["is_running"] = True
    try:
        # Запускаем твой обновленный пайплайн
        logger.info("Запуск пайплайна оценки...")
        evaluator = PipelineEvaluator("evals/scenarios.yaml")
        result = await evaluator.run()
        eval_status["last_result"] = result
    except Exception as e:
        logger.error(f"Ошибка при выполнении оценки: {e}")
        eval_status["error"] = str(e)
    finally:
        eval_status["is_running"] = False


@router.post("/run")
async def run_evaluation(background_tasks: BackgroundTasks):
    if eval_status["is_running"]:
        return {"message": "Оценка уже запущена. Пожалуйста, подождите."}

    # Запускаем фоновую задачу для выполнения оценки
    background_tasks.add_task(background_eval_task)
    return {"message": "Оценка запущена в фоновом режиме."}


@router.get("/status")
async def get_evaluation_status():
    return eval_status
