"""
FAQ Matcher — модуль для поиска готовых ответов на типовые вопросы.

Использует ту же embedding-модель (all-MiniLM-L6-v2), что и RAG,
и косинусное сходство для определения, подходит ли заготовленный ответ.

Если сходство вопроса пользователя с одним из FAQ-вопросов
превышает порог (SIMILARITY_THRESHOLD), возвращается готовый ответ
без обращения к LLM.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Порог косинусного сходства для срабатывания FAQ.
# 0.85 — достаточно строгий, чтобы не давать ложных срабатываний,
# но и достаточно мягкий для перефразированных вопросов.
SIMILARITY_THRESHOLD = 0.85

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
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self._threshold = threshold
        self._faq_path = faq_path or FAQ_DATA_PATH
        self._model = SentenceTransformer(model_name)

        # Загружаем FAQ
        self._questions: list[str] = []  # все формулировки (question + aliases)
        self._answers: list[str] = []  # ответ для каждой формулировки
        self._embeddings: Optional[np.ndarray] = None

        self._load_faq()

    def _load_faq(self) -> None:
        """Загружает FAQ из YAML и вычисляет embeddings."""
        if not self._faq_path.exists():
            logger.warning(f"FAQ file not found: {self._faq_path}")
            return

        with open(self._faq_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        faq_items = data.get("faq", [])
        if not faq_items:
            logger.warning("FAQ file is empty")
            return

        for item in faq_items:
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            aliases = item.get("aliases", [])

            if not question or not answer:
                continue

            # Основной вопрос
            self._questions.append(question)
            self._answers.append(answer)

            # Альтернативные формулировки (aliases)
            for alias in aliases:
                alias = alias.strip()
                if alias:
                    self._questions.append(alias)
                    self._answers.append(answer)

        if self._questions:
            self._embeddings = self._model.encode(
                self._questions, normalize_embeddings=True
            )
            logger.info(
                f"FAQ loaded: {len(faq_items)} entries, "
                f"{len(self._questions)} total phrases"
            )
        else:
            logger.warning("No valid FAQ entries found")

    def match(self, user_question: str) -> Optional[str]:
        """
        Проверяет, подходит ли пользовательский вопрос под один из FAQ.

        Args:
            user_question: Текст вопроса пользователя.

        Returns:
            Готовый ответ, если сходство >= порога, иначе None.
        """
        if self._embeddings is None or len(self._questions) == 0:
            return None

        # Вычисляем embedding вопроса пользователя
        query_embedding = self._model.encode(
            user_question, normalize_embeddings=True
        )

        # Косинусное сходство со всеми FAQ-фразами
        similarities = _cosine_similarity(query_embedding, self._embeddings)

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        logger.debug(
            f"FAQ match: best='{self._questions[best_idx]}' "
            f"score={best_score:.4f} threshold={self._threshold}"
        )

        if best_score >= self._threshold:
            logger.info(
                f"FAQ HIT: '{user_question}' → '{self._questions[best_idx]}' "
                f"(score={best_score:.4f})"
            )
            return self._answers[best_idx]

        return None

    def reload(self) -> None:
        """Перезагружает FAQ из файла (hot-reload)."""
        self._questions.clear()
        self._answers.clear()
        self._embeddings = None
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


# ── Singleton ────────────────────────────────────────────────────────────

_faq_matcher: Optional[FAQMatcher] = None


def get_faq_matcher() -> FAQMatcher:
    """Возвращает глобальный экземпляр FAQMatcher (lazy singleton)."""
    global _faq_matcher
    if _faq_matcher is None:
        _faq_matcher = FAQMatcher()
    return _faq_matcher
