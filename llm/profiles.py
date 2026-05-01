from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMProfile:
    max_tokens: Optional[int] = None
    timeout: Optional[float] = None
    temperature: Optional[float] = None


class LLMProfiles:
    # 1. Чат-бот
    CHAT = LLMProfile(max_tokens=1500, timeout=45, temperature=0.2)
    # 2. Граф (LightRAG)
    GRAPH = LLMProfile(max_tokens=8192, timeout=180, temperature=0.1)
    # 3. Препроцессинг HTML
    PARSER = LLMProfile(max_tokens=4000, timeout=90, temperature=0.1)
    # 4. Парсинг PDF (Vision)
    VISION = LLMProfile(max_tokens=8192, timeout=240, temperature=0.1)
    # 5. Роутинг
    INTENT = LLMProfile(max_tokens=50, timeout=15, temperature=0.0)
    # 6. Генерация названий
    TITLE = LLMProfile(max_tokens=100, timeout=20, temperature=0.2)
    # 7. Эмбеддинги (только timeout, так как max_tokens не применяется)
    EMBEDDING = LLMProfile(timeout=60)
