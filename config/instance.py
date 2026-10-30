"""Process-local, immutable view of application settings; no database dependencies."""

import os
from dataclasses import dataclass
from typing import overload


@dataclass(frozen=True)
class SettingSpec:
    label: str
    group: str
    default: str = ""
    secret: bool = False
    kind: str = "text"
    choices: tuple[str, ...] = ()


SPECS: dict[str, SettingSpec] = {}


def define(
    name: str,
    label: str,
    group: str,
    default: str = "",
    *,
    secret: bool = False,
    kind: str = "text",
    choices: tuple[str, ...] = (),
) -> None:
    SPECS[name] = SettingSpec(label, group, default, secret, kind, choices)


for provider in ("OPENAI", "GEMINI"):
    group = "OpenAI" if provider == "OPENAI" else "Gemini"
    define(f"{provider}_API_KEY", "API-ключ", group, secret=True)
    define(f"{provider}_BASE_URL", "Адрес API (пусто — официальный)", group, kind="url")
    for suffix, label, default in (
        (
            "MODEL",
            "Ответы и обработка текста",
            "gpt-4o-mini" if provider == "OPENAI" else "gemini-3.1-flash-lite-preview",
        ),
        (
            "GRAPH_MODEL",
            "Построение графа",
            "gpt-4o-mini" if provider == "OPENAI" else "gemini-3.1-flash-lite-preview",
        ),
        (
            "MODEL_VISION",
            "Обработка PDF и изображений",
            "gpt-5.4-mini" if provider == "OPENAI" else "gemini-2.5-flash",
        ),
        (
            "EMBEDDING_MODEL",
            "Эмбеддинги FAQ",
            "text-embedding-3-small"
            if provider == "OPENAI"
            else "gemini-embedding-2-preview",
        ),
        (
            "GRAPH_EMBEDDING_MODEL",
            "Эмбеддинги графа",
            "text-embedding-3-small"
            if provider == "OPENAI"
            else "gemini-embedding-2-preview",
        ),
        ("TEMPERATURE", "Температура", "0.2"),
        ("TIMEOUT_SECONDS", "Таймаут, секунд", "60"),
        ("MAX_TOKENS", "Максимум токенов по умолчанию", "1024"),
    ):
        if provider == "GEMINI" and suffix == "MAX_TOKENS":
            continue
        define(
            f"{provider}_{suffix}",
            label,
            group,
            default,
            kind="number"
            if suffix in {"TEMPERATURE", "TIMEOUT_SECONDS", "MAX_TOKENS"}
            else "text",
        )
# Proxy credentials are secrets too.
define("OPENAI_SOCKS5_PROXY", "Прокси OpenAI", "OpenAI", secret=True, kind="proxy")
for name, label, default in (
    ("LLM_PROVIDER", "Ответы и обработка текста", "gemini"),
    ("FAQ_EMBEDDING_PROVIDER", "Эмбеддинги FAQ", "gemini"),
    ("LIGHTRAG_LLM_PROVIDER", "Построение графа", "gemini"),
    ("LIGHTRAG_EMBEDDING_PROVIDER", "Эмбеддинги графа", "gemini"),
    ("PDF_PARSER_PROVIDER", "Документы и изображения", "openai"),
):
    define(name, label, "Назначение моделей", default, choices=("openai", "gemini"))
for name, label, group in (
    ("BOT_TOKEN", "Токен бота", "Telegram"),
    ("MAX_BOT_TOKEN", "Токен бота", "MAX"),
):
    define(name, label, group, secret=True)
    define(
        "TELEGRAM_ENABLED" if group == "Telegram" else "MAX_ENABLED",
        "Канал включён",
        group,
        "false",
        kind="boolean",
    )
define("TELEGRAM_SOCKS5_PROXY", "Прокси Telegram", "Telegram", secret=True, kind="proxy")
for name, label, default, kind in (
    (
        "SEED_ADMISSION_SCORES",
        "Загрузка проходных баллов НГУ при старте",
        "true",
        "boolean",
    ),
    ("SEED_FACULTIES", "Загрузка справочника НГУ при старте", "true", "boolean"),
    (
        "SEED_ADMISSION_SCORES_FORCE",
        "Принудительная загрузка баллов при старте",
        "false",
        "boolean",
    ),
    (
        "SEED_FACULTIES_FORCE",
        "Принудительная загрузка справочника при старте",
        "false",
        "boolean",
    ),
    ("ADMISSION_SCORES_MAX_AGE_HOURS", "Период обновления баллов, часов", "24", "number"),
    (
        "ADMISSION_SCORES_URL",
        "Источник проходных баллов",
        "https://www.nsu.ru/n/education/apply-info/itogi-priema/",
        "url",
    ),
    ("MAX_DOCUMENT_SIZE", "Максимальный размер документа, байт", "52428800", "number"),
    ("MAX_DOCUMENT_PAGES", "Максимум страниц PDF", "50", "number"),
    ("SCHEDULER_ENABLED", "Планировщик списков включён", "true", "boolean"),
    ("RATING_POLL_MIN_SECONDS", "Минимальный интервал, секунд", "300", "number"),
    ("RATING_POLL_MAX_SECONDS", "Максимальный интервал, секунд", "1800", "number"),
    ("RATING_POLL_START_SECONDS", "Начальный интервал, секунд", "1800", "number"),
    ("RATING_REQUEST_DELAY", "Пауза между запросами, секунд", "0.3", "number"),
    ("RATING_DEGREE", "Уровень обучения", "bachelor", "text"),
    ("RATING_TYPES", "Типы конкурсных списков", "30", "text"),
    ("RATING_FAILED_LISTS_THRESHOLD", "Допустимая доля ошибок", "0.2", "number"),
    ("RATING_MAX_SESSION_REFRESHES", "Повторы сессии", "3", "number"),
    ("RATING_USE_MOCK", "Тестовый источник списков", "false", "boolean"),
    ("MOCK_RATING_URL", "Адрес тестового источника", "http://localhost:8001", "url"),
    ("NOTIFY_SEND_DELAY", "Пауза отправки уведомлений, секунд", "0.035", "number"),
    ("BOT_RESTART_DELAY_SECONDS", "Пауза перезапуска канала, секунд", "5", "number"),
    ("LIGHTRAG_EMBEDDING_WORKERS", "Параллельные эмбеддинги", "2", "number"),
    ("LIGHTRAG_LLM_WORKERS", "Параллельные запросы графа", "1", "number"),
):
    define(name, label, "Обработка и источники", default, kind=kind)

define(
    "PROFILE_CHAT_MAX_TOKENS",
    "Ответы: Максимум токенов",
    "Параметры вызовов",
    "1500",
    kind="number",
)
define(
    "PROFILE_CHAT_TIMEOUT",
    "Ответы: Таймаут, секунд",
    "Параметры вызовов",
    "45",
    kind="number",
)
define(
    "PROFILE_CHAT_TEMPERATURE",
    "Ответы: Температура",
    "Параметры вызовов",
    "0.2",
    kind="number",
)
define(
    "PROFILE_GRAPH_MAX_TOKENS",
    "Граф знаний: Максимум токенов",
    "Параметры вызовов",
    "8192",
    kind="number",
)
define(
    "PROFILE_GRAPH_TIMEOUT",
    "Граф знаний: Таймаут, секунд",
    "Параметры вызовов",
    "180",
    kind="number",
)
define(
    "PROFILE_GRAPH_TEMPERATURE",
    "Граф знаний: Температура",
    "Параметры вызовов",
    "0.1",
    kind="number",
)
define(
    "PROFILE_PARSER_MAX_TOKENS",
    "Подготовка текста: Максимум токенов",
    "Параметры вызовов",
    "4000",
    kind="number",
)
define(
    "PROFILE_PARSER_TIMEOUT",
    "Подготовка текста: Таймаут, секунд",
    "Параметры вызовов",
    "90",
    kind="number",
)
define(
    "PROFILE_PARSER_TEMPERATURE",
    "Подготовка текста: Температура",
    "Параметры вызовов",
    "0.1",
    kind="number",
)
define(
    "PROFILE_VISION_MAX_TOKENS",
    "Документы: Максимум токенов",
    "Параметры вызовов",
    "8192",
    kind="number",
)
define(
    "PROFILE_VISION_TIMEOUT",
    "Документы: Таймаут, секунд",
    "Параметры вызовов",
    "240",
    kind="number",
)
define(
    "PROFILE_VISION_TEMPERATURE",
    "Документы: Температура",
    "Параметры вызовов",
    "0.1",
    kind="number",
)
define(
    "PROFILE_INTENT_MAX_TOKENS",
    "Классификация: Максимум токенов",
    "Параметры вызовов",
    "50",
    kind="number",
)
define(
    "PROFILE_INTENT_TIMEOUT",
    "Классификация: Таймаут, секунд",
    "Параметры вызовов",
    "15",
    kind="number",
)
define(
    "PROFILE_INTENT_TEMPERATURE",
    "Классификация: Температура",
    "Параметры вызовов",
    "0",
    kind="number",
)
define(
    "PROFILE_TITLE_MAX_TOKENS",
    "Заголовки: Максимум токенов",
    "Параметры вызовов",
    "100",
    kind="number",
)
define(
    "PROFILE_TITLE_TIMEOUT",
    "Заголовки: Таймаут, секунд",
    "Параметры вызовов",
    "20",
    kind="number",
)
define(
    "PROFILE_TITLE_TEMPERATURE",
    "Заголовки: Температура",
    "Параметры вызовов",
    "0.2",
    kind="number",
)
define(
    "PROFILE_EMBEDDING_TIMEOUT",
    "Эмбеддинги: Таймаут, секунд",
    "Параметры вызовов",
    "60",
    kind="number",
)

_values: dict[str, str] | None = None
_revision = 0
_secret_history: set[str] = set()


def install(values: dict[str, str], revision: int) -> None:
    global _values, _revision
    remember_secrets(values)
    _values = dict(values)
    _revision = revision


def revision() -> int:
    return _revision


@overload
def get_setting(name: str, default: str) -> str: ...
@overload
def get_setting(name: str, default: None = None) -> str | None: ...


def get_setting(name: str, default: str | None = None) -> str | None:
    if _values is None or name not in SPECS:
        return os.getenv(name, default)
    value = _values.get(name, SPECS[name].default)
    return value if value != "" else default


def enabled(name: str) -> bool:
    return get_setting(name, "false").lower() in {"true", "1", "yes", "on"}


def redact(text: str) -> str:
    from urllib.parse import quote

    secrets = _secret_history | {
        os.environ[k] for k, spec in SPECS.items() if spec.secret and os.environ.get(k)
    }
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
        text = text.replace(quote(secret, safe=""), "[REDACTED]")
    return text


def remember_secrets(values: dict[str, str]) -> None:
    _secret_history.update(
        values[k] for k, spec in SPECS.items() if spec.secret and values.get(k)
    )
