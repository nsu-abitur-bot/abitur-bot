"""
FAQ Matcher — модуль для поиска готовых ответов на типовые вопросы.

Использует embeddings через текущий LLM provider
и косинусное сходство для определения, подходит ли заготовленный ответ.

Если сходство вопроса пользователя с одним из FAQ-вопросов
превышает порог (SIMILARITY_THRESHOLD), возвращается готовый ответ
без обращения к LLM.
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import yaml

from llm.factory import get_llm_provider

logger = logging.getLogger(__name__)

# Порог косинусного сходства для срабатывания FAQ.
# 0.80 — баланс между ложными срабатываниями на размытых вопросах
# и пропуском перефразированных конкретных вопросов.
SIMILARITY_THRESHOLD = 0.95

# Слова-паразиты / приветствия, которые не несут смысловой нагрузки
# и мешают семантическому сопоставлению с FAQ.
_FILLER_WORDS = [
    "привет",
    "здравствуйте",
    "здравствуй",
    "добрый день",
    "добрый вечер",
    "доброе утро",
    "хай",
    "хей",
    "hello",
    "hi",
    "подскажи",
    "подскажите",
    "скажи",
    "скажите",
    "расскажи",
    "расскажите",
    "ответь",
    "ответьте",
    "пожалуйста",
    "плз",
    "плиз",
    "а ",
    "ну ",
    "так ",
    "вот ",
]

# Паттерн для удаления префикса [from username]
_FROM_PREFIX_RE = re.compile(r"^\[from\s+[^\]]*\]\s*", re.IGNORECASE)


def clean_user_input(text: str) -> str:
    """Очищает пользовательский ввод от мусора для FAQ-сопоставления.

    Удаляет:
      - Префикс [from username]
      - Приветствия и слова-паразиты
      - Лишние пробелы и пунктуацию в начале
    """
    # Убираем [from ...]
    text = _FROM_PREFIX_RE.sub("", text)

    # Убираем приветствия / филлеры (без учёта регистра)
    lower = text.lower()
    for filler in _FILLER_WORDS:
        if lower.startswith(filler):
            text = text[len(filler) :]
            lower = text.lower()

    # Убираем ведущие запятые, точки, пробелы
    text = text.lstrip(" ,.:;!?-—")

    return text.strip()


# Путь к файлу с FAQ-данными
FAQ_DATA_PATH = Path(__file__).parent / "faq_data.yaml"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Вычисляет косинусное сходство вектора `a` со всеми строками матрицы `b`."""
    # a: (dim,)  b: (n, dim) → результат: (n,)
    dot = b @ a
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b, axis=1)
    return dot / (norm_a * norm_b + 1e-10)


class FAQMatcher:
    """
    Сопоставляет пользовательский вопрос с базой заготовленных FAQ.

    При инициализации:
      1. Загружает FAQ из YAML-файла.
      2. Вычисляет embeddings для всех вопросов и их alias'ов.

    При вызове match():
      1. Вычисляет embedding вопроса пользователя.
      2. Находит ближайший FAQ-вопрос по косинусному сходству.
      3. Если сходство >= порога, возвращает готовый ответ.
    """

    def __init__(
        self,
        faq_path: Optional[Path] = None,
        threshold: float = SIMILARITY_THRESHOLD,
        embedder: Optional["EmbeddingClient"] = None,
    ):
        self._threshold = threshold
        self._faq_path = faq_path or FAQ_DATA_PATH

        self._embedder = embedder or self._create_embedder()

        # Загружаем FAQ
        self._questions: list[str] = []  # все формулировки (question + aliases)
        self._answers: list[str] = []  # ответ для каждой формулировки
        self._embeddings: Optional[np.ndarray] = None

        self._load_faq()

    def _create_embedder(self) -> Optional["EmbeddingClient"]:
        try:
            provider = get_llm_provider()
            embedder = provider.get_embeddings_model()
            if embedder:
                provider_name = provider.__class__.__name__
                logger.info("FAQ embedder initialized using %s", provider_name)
                return embedder
            else:
                logger.warning(
                    "LLM provider '%s' does not support embeddings for FAQ. "
                    "FAQ semantic matching disabled.",
                    provider.__class__.__name__,
                )
                return None
        except Exception as exc:
            logger.warning(
                "Failed to initialize LLM provider for FAQ embeddings: %s. "
                "FAQ semantic matching disabled.",
                exc,
            )
            return None

    def _load_faq(self) -> None:
        """Загружает FAQ из YAML (используется в тестах)."""
        if self._embedder is None:
            logger.warning("FAQ embedder не инициализирован, FAQ matching отключен")
            return

        if not self._faq_path.exists():
            logger.warning(f"FAQ file not found: {self._faq_path}")
            return

        with open(self._faq_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        faq_items = data.get("faq", [])
        if not faq_items:
            logger.warning("FAQ file is empty")
            return

        self._process_items(faq_items)

    def _process_items(self, faq_items: list[dict]) -> None:
        """Строит in-memory индекс из списка FAQ-элементов."""
        if self._embedder is None:
            return

        self._questions.clear()
        self._answers.clear()
        self._embeddings = None

        for item in faq_items:
            question = (item.get("question") or "").strip()
            answer = (item.get("answer") or "").strip()
            aliases = item.get("aliases") or []

            if not question or not answer:
                continue

            self._questions.append(question)
            self._answers.append(answer)

            for alias in aliases:
                alias = alias.strip()
                if alias:
                    self._questions.append(alias)
                    self._answers.append(answer)

        if not self._questions:
            logger.warning("No valid FAQ entries found")
            return

        import pickle

        cache_path = self._faq_path.with_name("faq_cache.pkl")
        cache = {}
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cache = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load FAQ embeddings cache: {e}")

        needed_phrases = list(set(self._questions))
        to_embed_phrases = [p for p in needed_phrases if p not in cache]

        if to_embed_phrases:
            logger.info(f"FAQ: Embedding {len(to_embed_phrases)} new phrases...")
            try:
                new_embeddings = self._embedder.embed_documents(to_embed_phrases)
            except Exception as e:
                logger.error(f"FAQ: Failed to embed new phrases: {e}")
                raise

            for phrase, emb in zip(to_embed_phrases, new_embeddings):
                cache[phrase] = emb

            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(cache, f)
            except Exception as e:
                logger.warning(f"Failed to save FAQ embeddings cache: {e}")

        vectors_list = [cache[phrase] for phrase in self._questions]
        vectors = np.array(vectors_list, dtype=float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self._embeddings = vectors / np.maximum(norms, 1e-10)
        logger.info(
            f"FAQ loaded: {len(faq_items)} entries, "
            f"{len(self._questions)} total phrases "
            f"(cached {len(self._questions) - len(to_embed_phrases)})"
        )

    def load_items(self, items: list[dict]) -> None:
        """Загружает FAQ из списка словарей (из БД)."""
        self._process_items(items)

    def match(self, user_question: str) -> Optional[str]:
        """
        Проверяет, подходит ли пользовательский вопрос под один из FAQ.

        Перед сопоставлением очищает текст от приветствий, префикса
        [from username] и слов-паразитов.

        Args:
            user_question: Текст вопроса пользователя (может содержать
                           префикс [from ...], приветствия и т.д.).

        Returns:
            Готовый ответ, если сходство >= порога, иначе None.
        """
        if self._embeddings is None or len(self._questions) == 0:
            return None

        # Очищаем вопрос от мусора
        cleaned = clean_user_input(user_question)
        if not cleaned:
            return None

        logger.info(f"[FAQ] Input cleaned: '{user_question}' → '{cleaned}'")

        # Вычисляем embedding очищенного вопроса
        if self._embedder is None:
            return None

        query_vec = np.array(self._embedder.embed_documents([cleaned])[0], dtype=float)
        query_embedding = query_vec / max(float(np.linalg.norm(query_vec)), 1e-10)

        # Косинусное сходство со всеми FAQ-фразами
        similarities = _cosine_similarity(query_embedding, self._embeddings)

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        logger.info(
            f"[FAQ] Match result: best='{self._questions[best_idx]}' "
            f"score={best_score:.4f} threshold={self._threshold}"
        )

        if best_score >= self._threshold:
            logger.info(
                f"[FAQ] HIT: '{cleaned}' → '{self._questions[best_idx]}' "
                f"(score={best_score:.4f})"
            )
            return self._answers[best_idx]

        return None

    async def match_async(self, user_question: str) -> Optional[str]:
        """Async-вариант match: выносит синхронный embed_documents в отдельный поток,
        чтобы не блокировать event loop."""
        if self._embeddings is None or len(self._questions) == 0:
            return None

        cleaned = clean_user_input(user_question)
        if not cleaned:
            return None

        logger.info(f"[FAQ] Input cleaned: '{user_question}' → '{cleaned}'")

        if self._embedder is None:
            return None

        embed_result = await asyncio.to_thread(
            self._embedder.embed_documents, [cleaned]
        )
        query_vec = np.array(embed_result[0], dtype=float)
        query_embedding = query_vec / max(float(np.linalg.norm(query_vec)), 1e-10)

        similarities = _cosine_similarity(query_embedding, self._embeddings)

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        logger.info(
            f"[FAQ] Match result: best='{self._questions[best_idx]}' "
            f"score={best_score:.4f} threshold={self._threshold}"
        )

        if best_score >= self._threshold:
            logger.info(
                f"[FAQ] HIT: '{cleaned}' → '{self._questions[best_idx]}' "
                f"(score={best_score:.4f})"
            )
            return self._answers[best_idx]

        return None

    def reload(self) -> None:
        """Перезагружает FAQ из файла (для тестов с faq_path)."""
        self._load_faq()

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value

    @property
    def size(self) -> int:
        """Количество FAQ-фраз (question + aliases)."""
        return len(self._questions)


class EmbeddingClient(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


# ── Singleton ────────────────────────────────────────────────────────────

_faq_matcher: Optional[FAQMatcher] = None


def get_faq_matcher() -> FAQMatcher:
    """Возвращает глобальный экземпляр FAQMatcher (lazy singleton)."""
    global _faq_matcher
    if _faq_matcher is None:
        _faq_matcher = FAQMatcher()
    return _faq_matcher
