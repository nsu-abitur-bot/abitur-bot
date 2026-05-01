import threading
from typing import List

import yaml

from abbrev.abbrev_expander import ABBREV_DATA_PATH, get_abbrev_expander
from api.schemas.abbrev import AbbrevItem


class AbbrevService:
    """Управление словарём аббревиатур: чтение, добавление, обновление, удаление."""

    def __init__(self):
        self._lock = threading.Lock()

    def get_all(self) -> List[AbbrevItem]:
        with self._lock:
            data = self._read_data()
            return [AbbrevItem(**item) for item in data.get("abbreviations", [])]

    def create(self, item: AbbrevItem) -> AbbrevItem:
        with self._lock:
            data = self._read_data()
            data.setdefault("abbreviations", [])
            data["abbreviations"].append(item.model_dump())
            self._write_data(data)
        self._reload_expander()
        return item

    def create_many(self, items: List[AbbrevItem]) -> List[AbbrevItem]:
        with self._lock:
            data = self._read_data()
            data.setdefault("abbreviations", [])
            for item in items:
                data["abbreviations"].append(item.model_dump())
            self._write_data(data)
        self._reload_expander()
        return items

    def update(self, index: int, item: AbbrevItem) -> AbbrevItem:
        with self._lock:
            data = self._read_data()
            items = data.get("abbreviations", [])
            if index < 0 or index >= len(items):
                raise IndexError(f"Abbreviation at index {index} not found")
            items[index] = item.model_dump()
            self._write_data(data)
        self._reload_expander()
        return item

    def delete(self, index: int) -> None:
        with self._lock:
            data = self._read_data()
            items = data.get("abbreviations", [])
            if index < 0 or index >= len(items):
                raise IndexError(f"Abbreviation at index {index} not found")
            del items[index]
            self._write_data(data)
        self._reload_expander()

    def _read_data(self) -> dict:
        if not ABBREV_DATA_PATH.exists():
            return {"abbreviations": []}
        try:
            with open(ABBREV_DATA_PATH, encoding="utf-8") as f:
                return yaml.safe_load(f) or {"abbreviations": []}
        except Exception:
            return {"abbreviations": []}

    def _write_data(self, data: dict) -> None:
        with open(ABBREV_DATA_PATH, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)

    def _reload_expander(self) -> None:
        get_abbrev_expander().reload()
