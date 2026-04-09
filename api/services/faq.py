import threading
from typing import List

import yaml

from api.schemas.faq import FaqItem
from faq.faq_matcher import FAQ_DATA_PATH, get_faq_matcher


class FaqService:
    """Сервис для управления FAQ (чтение, добавление, обновление, удаление)."""

    def __init__(self):
        # Используем lock, чтобы избежать конфликтов при одновременной записи в файл
        self._lock = threading.Lock()

    def get_all(self) -> List[FaqItem]:
        """Возвращает список всех FAQ из файла."""
        with self._lock:
            data = self._read_data()
            return [FaqItem(**item) for item in data.get("faq", [])]

    def create(self, item: FaqItem) -> FaqItem:
        """Создает новый вопрос FAQ и перезагружает matcher."""
        with self._lock:
            data = self._read_data()
            if "faq" not in data:
                data["faq"] = []

            data["faq"].append(item.model_dump())
            self._write_data(data)

        self._reload_matcher()
        return item

    def create_many(self, items: List[FaqItem]) -> List[FaqItem]:
        """Создает несколько вопросов FAQ и один раз перезагружает matcher."""
        with self._lock:
            data = self._read_data()
            if "faq" not in data:
                data["faq"] = []

            for item in items:
                data["faq"].append(item.model_dump())
            self._write_data(data)

        self._reload_matcher()
        return items

    def update(self, index: int, item: FaqItem) -> FaqItem:
        """Обновляет существующий FAQ элемент по индексу."""
        with self._lock:
            data = self._read_data()
            if "faq" not in data or index < 0 or index >= len(data["faq"]):
                raise IndexError(f"FAQ item with index {index} not found")

            data["faq"][index] = item.model_dump()
            self._write_data(data)

        self._reload_matcher()
        return item

    def delete(self, index: int) -> None:
        """Удаляет существующий FAQ элемент по индексу."""
        with self._lock:
            data = self._read_data()
            if "faq" not in data or index < 0 or index >= len(data["faq"]):
                raise IndexError(f"FAQ item with index {index} not found")

            del data["faq"][index]
            self._write_data(data)

        self._reload_matcher()

    def _read_data(self) -> dict:
        """Читает YAML файл с FAQ. Возвращает dict."""
        if not FAQ_DATA_PATH.exists():
            return {"faq": []}

        try:
            with open(FAQ_DATA_PATH, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data or {"faq": []}
        except Exception:
            # При ошибке чтения можно вернуть пустой или бросить дальше
            return {"faq": []}

    def _write_data(self, data: dict) -> None:
        """Записывает dict обратно в YAML файл."""
        # Используем allow_unicode для сохранения кириллицы
        # yaml.dump стандартный стирает комментарии
        with open(FAQ_DATA_PATH, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    def _reload_matcher(self) -> None:
        """Оповещает FAQMatcher о необходимости перезагрузить данные из файла."""
        get_faq_matcher().reload()
