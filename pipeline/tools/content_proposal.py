"""Инструмент модели для предложения редактору нового материала базы знаний."""

import logging

from db.postgres.db import AsyncSessionLocal
from db.postgres.services.content_proposal import ContentProposalService
from llm.base import ToolSpec
from rag.loader import DEFAULT_GRAPH_ID

logger = logging.getLogger(__name__)

CONTENT_PROPOSAL_TOOL = ToolSpec(
    name="propose_content",
    description=(
        "Предложи редактору новый материал для базы знаний, если обнаружил "
        "проверенный факт, которого в ней нет. Предложение не публикуется "
        "без проверки редактора. Не вызывай для обычного ответа на вопрос."
    ),
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Название материала"},
            "text": {
                "type": "string",
                "description": "Точный текст предлагаемого материала",
            },
            "source_url": {
                "type": "string",
                "description": "URL источника факта, если он известен",
            },
        },
        "required": ["title", "text"],
        "additionalProperties": False,
    },
)


async def execute_content_proposal(arguments: dict) -> str:
    title = arguments.get("title")
    content = arguments.get("text")
    if not isinstance(title, str) or not isinstance(content, str):
        return "Предложение не создано: нужны название и текст."
    source_url = arguments.get("source_url")
    if source_url is not None and not isinstance(source_url, str):
        return "Предложение не создано: некорректная ссылка на источник."
    try:
        async with AsyncSessionLocal() as session:
            proposal = await ContentProposalService(session).create(
                source="model",
                graph_id=DEFAULT_GRAPH_ID,
                title=title,
                text=content,
                source_url=source_url,
            )
    except ValueError:
        return "Предложение не создано: название или текст пусты."
    except Exception:
        logger.exception("Не удалось поставить предложение модели в очередь")
        return "Не удалось поставить предложение в очередь редактора."
    return f"Предложение {proposal.id} отправлено редактору на проверку."
