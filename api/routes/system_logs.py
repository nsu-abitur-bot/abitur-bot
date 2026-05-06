from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth.dependencies import require_admin

router = APIRouter(prefix="/system-logs", tags=["System Logs"])

LOG_DIR = Path("logs")


@router.get("/", dependencies=[Depends(require_admin)])
async def get_system_logs(
    filename: str = Query("abitur_bot.log", description="Имя лог файла"),
    lines: int = Query(100, ge=1, le=1000, description="Количество последних строк"),
) -> List[str]:
    """Получает последние N строк из указанного файла логов."""
    log_file = LOG_DIR / filename

    if not log_file.exists():
        raise HTTPException(status_code=404, detail="Файл логов не найден")

    # Защита от выхода за пределы директории logs
    try:
        log_file.resolve().relative_to(LOG_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Доступ запрещен")

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            # Простой способ получения последних строк
            # (может быть неэффективным для огромных файлов, но у нас есть ротация файлов)
            content = f.readlines()
            return content[-lines:]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка чтения логов: {str(e)}")


@router.get("/files", dependencies=[Depends(require_admin)])
async def list_log_files() -> List[str]:
    """Возвращает список доступных файлов логов."""
    if not LOG_DIR.exists():
        return []
    return [f.name for f in LOG_DIR.glob("*.log*") if f.is_file()]
