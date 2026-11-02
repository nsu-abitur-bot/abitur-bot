from dataclasses import dataclass
from typing import Any, Optional

from config.instance import get_setting


@dataclass
class LLMProfile:
    max_tokens: Optional[int] = None
    timeout: Optional[float] = None
    temperature: Optional[float] = None


class ConfiguredProfile:
    def __init__(self, name: str, profile: LLMProfile) -> None:
        self.name = name
        self.profile = profile

    def __get__(self, instance: object, owner: type) -> LLMProfile:
        values: dict[str, Any] = {}
        for field in ("max_tokens", "timeout", "temperature"):
            fallback = getattr(self.profile, field)
            raw = get_setting(f"PROFILE_{self.name}_{field.upper()}")
            values[field] = (
                (int(raw) if field == "max_tokens" else float(raw))
                if raw is not None
                else fallback
            )
        return LLMProfile(**values)


class LLMProfiles:
    # 1. Чат-бот
    CHAT = ConfiguredProfile(
        "CHAT", LLMProfile(max_tokens=1500, timeout=45, temperature=0.2)
    )
    # 2. Граф (LightRAG)
    GRAPH = ConfiguredProfile(
        "GRAPH", LLMProfile(max_tokens=8192, timeout=180, temperature=0.1)
    )
    # 3. Препроцессинг HTML
    PARSER = ConfiguredProfile(
        "PARSER", LLMProfile(max_tokens=4000, timeout=90, temperature=0.1)
    )
    # 4. Парсинг PDF (Vision)
    VISION = ConfiguredProfile(
        "VISION", LLMProfile(max_tokens=8192, timeout=240, temperature=0.1)
    )
    # 5. Роутинг
    INTENT = ConfiguredProfile(
        "INTENT", LLMProfile(max_tokens=50, timeout=15, temperature=0.0)
    )
    # 6. Генерация названий
    TITLE = ConfiguredProfile(
        "TITLE", LLMProfile(max_tokens=100, timeout=20, temperature=0.2)
    )
    # 7. Эмбеддинги (только timeout, так как max_tokens не применяется)
    EMBEDDING = ConfiguredProfile("EMBEDDING", LLMProfile(timeout=60))
