"""Реестр инструментов и дефолтный диспетчер вызовов."""

import logging

from llm.base import ToolExecutor, ToolSpec
from llm.tools.admission_scores import (
    ADMISSION_SCORES_TOOL,
    execute_admission_scores,
)

logger = logging.getLogger(__name__)


# Имя инструмента -> (описание, исполнитель).
TOOLS: dict[str, tuple[ToolSpec, ToolExecutor]] = {
    ADMISSION_SCORES_TOOL.name: (ADMISSION_SCORES_TOOL, execute_admission_scores),
}


async def default_tool_executor(name: str, arguments: dict) -> str:
    """Единая точка исполнения вызовов инструментов из реестра.

    Возвращает текст результата. Если инструмент неизвестен — короткий маркер
    ошибки (модель не должна выдумывать за него ответ).
    """
    entry = TOOLS.get(name)
    if entry is None:
        logger.warning("Запрошен неизвестный инструмент: %r", name)
        return f"Инструмент «{name}» недоступен."
    _, executor = entry
    return await executor(arguments)
