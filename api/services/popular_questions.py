"""Кластеризация популярных вопросов по смыслу.

Сервис доступа к данным (`db/postgres/services/message_log.py`) отдаёт группы
по точному совпадению текста, а близкие формулировки («когда подавать
документы?» и «до какого числа принимаете документы») объединяются здесь: для
этого нужны эмбеддинги, то есть вызов модели, и слою данных он не принадлежит.

Раньше этот код жил в сервисе БД и сам доставал провайдера через
`get_llm_provider()`. Из-за этого `tests/postgres/`, работающие с реальной
базой, могли по неудачной ветке уйти в настоящие запросы за эмбеддингами —
и это была половина кольцевого импорта `db <-> llm` (#310).
"""

import asyncio
import logging
import math
import os
from typing import Any, List, TypedDict

from db.postgres.services.message_log import MessageLogService, QuestionGroup
from llm.factory import get_llm_provider

logger = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.86
DEFAULT_RAW_LIMIT = 500
MAX_VARIANTS = 5


class PopularQuestion(TypedDict):
    question: str
    count: int
    variants: list[str]


def _embedder_cache_identity(embedder: Any) -> tuple[str, str]:
    """Провайдер и модель эмбеддера — ключ кэша векторов."""
    module_name = embedder.__class__.__module__.lower()
    class_name = embedder.__class__.__name__.lower()
    if "openai" in module_name or "openai" in class_name:
        provider = "openai"
    elif "gemini" in module_name or "gemini" in class_name:
        provider = "gemini"
    else:
        provider = os.getenv("LLM_PROVIDER", "") or class_name

    model = (
        getattr(embedder, "model", None)
        or getattr(embedder, "model_name", None)
        or getattr(embedder, "model_id", None)
        or "default"
    )
    return str(provider).lower(), str(model)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _rows_without_clustering(
    groups: List[QuestionGroup],
    limit: int,
) -> List[PopularQuestion]:
    """Ответ по точным группам — когда эмбеддинги недоступны."""
    return [
        {
            "question": group["question"],
            "count": group["count"],
            "variants": group["variants"][:MAX_VARIANTS],
        }
        for group in groups[:limit]
    ]


async def _vectors_for_groups(
    log_service: MessageLogService,
    groups: List[QuestionGroup],
    embedder: Any,
) -> dict[str, list[float]] | None:
    """Векторы по каноническому тексту: из кэша, недостающие — считаем и кэшируем.

    None означает «посчитать не удалось» — вызывающий отвечает точными группами.
    """
    provider, model = _embedder_cache_identity(embedder)
    canonicals = [group["canonical"] for group in groups]
    vectors = await log_service.get_cached_question_vectors(provider, model, canonicals)

    missing = [group for group in groups if group["canonical"] not in vectors]
    if not missing:
        return vectors

    computed = await asyncio.to_thread(
        embedder.embed_documents,
        [group["question"] for group in missing],
    )
    if len(computed) != len(missing):
        logger.warning(
            "Embeddings count mismatch for popular questions: "
            "%d vectors for %d questions",
            len(computed),
            len(missing),
        )
        return None

    normalized = [[float(value) for value in vector] for vector in computed]
    for group, vector in zip(missing, normalized):
        vectors[group["canonical"]] = vector

    await log_service.store_question_vectors(
        provider,
        model,
        [(group["canonical"], vector) for group, vector in zip(missing, normalized)],
    )
    return vectors


def _cluster(
    groups: List[QuestionGroup],
    vectors: dict[str, list[float]],
    similarity_threshold: float,
) -> list[dict[str, Any]]:
    """Жадная кластеризация по центроиду: группы идут от частых к редким."""
    clusters: list[dict[str, Any]] = []
    for group in groups:
        vector = vectors[group["canonical"]]
        best_cluster: dict[str, Any] | None = None
        best_similarity = similarity_threshold
        for cluster in clusters:
            similarity = _cosine_similarity(vector, cluster["centroid"])
            if similarity >= best_similarity:
                best_cluster = cluster
                best_similarity = similarity

        if best_cluster is None:
            clusters.append(
                {
                    "question": group["question"],
                    "count": group["count"],
                    "variants": list(group["variants"]),
                    "centroid": list(vector),
                    "representative_count": group["count"],
                }
            )
            continue

        old_count = best_cluster["count"]
        new_count = old_count + group["count"]
        best_cluster["centroid"] = [
            ((old_value * old_count) + (new_value * group["count"])) / new_count
            for old_value, new_value in zip(best_cluster["centroid"], vector)
        ]
        best_cluster["count"] = new_count
        if group["count"] > best_cluster["representative_count"]:
            best_cluster["question"] = group["question"]
            best_cluster["representative_count"] = group["count"]
        for variant in [group["question"], *group["variants"]]:
            if variant not in best_cluster["variants"]:
                best_cluster["variants"].append(variant)

    clusters.sort(key=lambda item: item["count"], reverse=True)
    return clusters


async def get_popular_questions(
    log_service: MessageLogService,
    limit: int = 10,
    raw_limit: int = DEFAULT_RAW_LIMIT,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    embedder: Any | None = None,
) -> List[PopularQuestion]:
    """Популярные вопросы с объединением близких формулировок.

    Без эмбеддера отвечаем группами по точному совпадению: это хуже, но
    работает, и лучше, чем отдать ошибку.
    """
    groups = await log_service.get_question_groups(raw_limit=max(limit, raw_limit))
    if not groups:
        return []

    if embedder is None:
        embedder = get_llm_provider().get_embeddings_model()
    if embedder is None:
        logger.warning("LLM provider does not expose embeddings; using exact groups")
        return _rows_without_clustering(groups, limit)

    vectors = await _vectors_for_groups(log_service, groups, embedder)
    if vectors is None:
        return _rows_without_clustering(groups, limit)

    clusters = _cluster(groups, vectors, similarity_threshold)
    return [
        {
            "question": cluster["question"],
            "count": cluster["count"],
            "variants": cluster["variants"][:MAX_VARIANTS],
        }
        for cluster in clusters[:limit]
    ]
