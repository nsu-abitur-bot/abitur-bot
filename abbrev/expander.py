import re
import threading
from pathlib import Path

import yaml

logger = __import__("logging").getLogger(__name__)

ABBREV_DATA_PATH = Path(__file__).parent / "abbrev_data.yaml"


class AbbrevExpander:
    """
    Расширяет аббревиатуры в тексте:
    НГУ → НГУ (Новосибирский государственный университет).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._abbrev: dict[str, str] = {}
        self._pattern: re.Pattern | None = None
        self._load()

    def _load(self) -> None:
        if not ABBREV_DATA_PATH.exists():
            self._abbrev = {}
            self._pattern = None
            return

        try:
            with open(ABBREV_DATA_PATH, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Ошибка загрузки словаря аббревиатур: {e}")
            return

        abbrev: dict[str, str] = {}
        for item in data.get("abbreviations", []):
            short = (item.get("short") or "").strip()
            full = (item.get("full") or "").strip()
            if short and full:
                abbrev[short] = full

        self._abbrev = abbrev
        if abbrev:
            # Сортируем по убыванию длины, чтобы более длинные аббревиатуры
            # матчились раньше более коротких (ФЕН перед ФЕ)
            pattern_str = "|".join(
                re.escape(k) for k in sorted(abbrev, key=len, reverse=True)
            )
            # (?<!\w) и (?!\w) работают как границы слова для кириллицы
            self._pattern = re.compile(r"(?<!\w)(" + pattern_str + r")(?!\w)")
        else:
            self._pattern = None

    def reload(self) -> None:
        with self._lock:
            self._load()

    def expand(self, text: str) -> str:
        """Заменяет аббревиатуры в тексте на форму «АББР (полная расшифровка)»."""
        if not self._pattern or not text:
            return text

        def _replace(match: re.Match) -> str:
            short = match.group(1)
            full = self._abbrev.get(short, "")
            return f"{short} ({full})" if full else short

        try:
            return self._pattern.sub(_replace, text)
        except Exception as e:
            logger.error(f"Ошибка расширения аббревиатур: {e}")
            return text

    @property
    def abbreviations(self) -> dict[str, str]:
        return dict(self._abbrev)


_expander: AbbrevExpander | None = None
_expander_lock = threading.Lock()


def get_abbrev_expander() -> AbbrevExpander:
    global _expander
    if _expander is None:
        with _expander_lock:
            if _expander is None:
                _expander = AbbrevExpander()
    return _expander
