"""Инструменты (function calling), доступные основной LLM-генерации бота."""

from pipeline.tools.admission_scores import (
    ADMISSION_SCORES_TOOL,
    execute_admission_scores,
)
from pipeline.tools.registry import TOOLS, default_tool_executor

__all__ = [
    "ADMISSION_SCORES_TOOL",
    "execute_admission_scores",
    "TOOLS",
    "default_tool_executor",
]
