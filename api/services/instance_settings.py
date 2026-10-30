import asyncio
import math
from urllib.parse import urlsplit

import httpx
from aiogram import Bot as TelegramBot
from aiogram.client.session.aiohttp import AiohttpSession
from google import genai
from google.genai import types
from maxapi import Bot as MaxBot
from openai import AsyncOpenAI
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from config.instance import SPECS, SettingSpec, remember_secrets
from db.postgres.models import Document, FaqEntry
from db.postgres.services.instance_settings import InstanceSnapshot


class ConfigurationError(ValueError):
    pass


def merge_changes(
    current: InstanceSnapshot, changes: dict[str, str | None]
) -> dict[str, str]:
    unknown = changes.keys() - SPECS.keys()
    if unknown:
        raise ConfigurationError("Неизвестный параметр")
    values = {
        name: current.values.get(name, spec.default) for name, spec in SPECS.items()
    }
    for name, value in changes.items():
        spec = SPECS[name]
        candidate = spec.default if value is None else value.strip()
        validate_value(spec, candidate)
        values[name] = candidate
    if float(values["RATING_POLL_MIN_SECONDS"]) > float(
        values["RATING_POLL_MAX_SECONDS"]
    ):
        raise ConfigurationError("Минимальный интервал превышает максимальный")
    for channel, token in (("TELEGRAM", "BOT_TOKEN"), ("MAX", "MAX_BOT_TOKEN")):
        if values[channel + "_ENABLED"] == "true" and not values[token]:
            raise ConfigurationError(f"Для включения {channel} нужен токен")
    return values


def validate_value(spec: SettingSpec, value: str) -> None:
    if len(value) > 4000 or "\n" in value or "\r" in value:
        raise ConfigurationError("Недопустимое значение настройки")
    if spec.group == "Вуз" and (len(value) > 300 or any(c in value for c in "{}<>")):
        raise ConfigurationError("Реквизиты должны быть обычным текстом до 300 символов")
    if spec.choices and value not in spec.choices:
        raise ConfigurationError("Выберите допустимое значение")
    if spec.kind == "boolean" and value not in {"true", "false"}:
        raise ConfigurationError("Ожидается true или false")
    if spec.kind == "number":
        try:
            number = float(value)
        except ValueError:
            raise ConfigurationError("Ожидается число") from None
        if not math.isfinite(number) or number < 0:
            raise ConfigurationError("Ожидается неотрицательное конечное число")
        if (
            "температура" not in spec.label.lower()
            and "доля" not in spec.label.lower()
            and ("Пауза" not in spec.label and "Период" not in spec.label)
        ):
            try:
                integer = int(value)
            except ValueError:
                raise ConfigurationError("Ожидается положительное целое число") from None
            if integer < 1:
                raise ConfigurationError("Ожидается положительное целое число")
        if "Температура" in spec.label and number > 2:
            raise ConfigurationError("Температура должна быть от 0 до 2")
        if "доля" in spec.label and number > 1:
            raise ConfigurationError("Доля должна быть от 0 до 1")
    if value and spec.kind in {"url", "proxy"}:
        try:
            parsed = urlsplit(value)
            allowed = (
                {"http", "https"}
                if spec.kind == "url"
                else {"http", "https", "socks5", "socks5h"}
            )
            if parsed.scheme not in allowed or not parsed.hostname or parsed.fragment:
                raise ValueError
            _ = parsed.port
            if spec.kind == "url" and (parsed.username or parsed.password):
                raise ValueError
        except ValueError:
            raise ConfigurationError("Некорректный адрес подключения") from None


async def guard_embeddings(
    session: AsyncSession,
    current: InstanceSnapshot,
    values: dict[str, str],
    *,
    lock: bool = False,
) -> None:
    embedding_keys = {
        "FAQ_EMBEDDING_PROVIDER",
        "LIGHTRAG_LLM_PROVIDER",
        "LIGHTRAG_EMBEDDING_PROVIDER",
        "OPENAI_EMBEDDING_MODEL",
        "GEMINI_EMBEDDING_MODEL",
        "OPENAI_GRAPH_EMBEDDING_MODEL",
        "GEMINI_GRAPH_EMBEDDING_MODEL",
        "OPENAI_BASE_URL",
        "GEMINI_BASE_URL",
    }
    if not any(
        values[k] != current.values.get(k, SPECS[k].default) for k in embedding_keys
    ):
        return
    if lock:
        # Fence new document/FAQ inserts until the new embedding configuration
        # commits. Remote checks happen before taking these short-lived locks.
        await session.execute(text("LOCK TABLE document, faq_entry IN SHARE MODE"))
    # Also protect legacy on-disk indexes which are not registered in Document.
    import os
    from pathlib import Path

    root = Path(os.getenv("LIGHTRAG_WORKSPACE_BASE", "./data/lightrag"))
    populated = root.exists() and any(root.glob("*/kv_store_full_docs.json"))
    if (
        populated
        or await session.scalar(select(Document.id).limit(1))
        or await session.scalar(select(FaqEntry.id).limit(1))
    ):
        raise ConfigurationError(
            "Смена пространства эмбеддингов требует переиндексации; "
            "изменение заблокировано"
        )


async def check_connection(target: str, values: dict[str, str]) -> str:
    """Never include upstream exception text: URLs can contain credentials."""
    remember_secrets(values)
    try:
        async with asyncio.timeout(20):
            if target == "telegram":
                session = AiohttpSession(proxy=values["TELEGRAM_SOCKS5_PROXY"] or None)
                bot = TelegramBot(values["BOT_TOKEN"], session=session)
                try:
                    me = await bot.get_me()
                    return f"Telegram: @{me.username} (ID {me.id})"
                finally:
                    await session.close()
            if target == "max":
                bot_max = MaxBot(token=values["MAX_BOT_TOKEN"])
                try:
                    me_max = await bot_max.get_me()
                    return f"MAX: ID {me_max.user_id}"
                finally:
                    await bot_max.close_session()
            provider, purpose = target.split(":", 1)
            prefix = provider.upper()
            model_key = {
                "chat": "MODEL",
                "graph": "GRAPH_MODEL",
                "vision": "MODEL_VISION",
                "embedding": "EMBEDDING_MODEL",
                "graph_embedding": "GRAPH_EMBEDDING_MODEL",
            }[purpose]
            model = values[f"{prefix}_{model_key}"]
            key = values[f"{prefix}_API_KEY"]
            if not key:
                raise ConfigurationError("Сначала задайте API-ключ")
            base_url = values[f"{prefix}_BASE_URL"] or None
            if provider == "openai":
                async with httpx.AsyncClient(
                    proxy=values["OPENAI_SOCKS5_PROXY"] or None
                ) as transport:
                    async with AsyncOpenAI(
                        api_key=key,
                        base_url=base_url,
                        http_client=transport,
                        timeout=15,
                        max_retries=0,
                    ) as client:
                        if "embedding" in purpose:
                            await client.embeddings.create(
                                model=model, input="connection check"
                            )
                        else:
                            await client.responses.create(
                                model=model, input="Reply OK", max_output_tokens=16
                            )
            elif provider == "gemini":
                client_g = genai.Client(
                    api_key=key,
                    http_options=types.HttpOptions(base_url=base_url, timeout=15000),
                )
                try:
                    if "embedding" in purpose:
                        await client_g.aio.models.embed_content(
                            model=model, contents="connection check"
                        )
                    else:
                        await client_g.aio.models.generate_content(
                            model=model,
                            contents="Reply OK",
                            config=types.GenerateContentConfig(max_output_tokens=16),
                        )
                finally:
                    await client_g.aio.aclose()
                    client_g.close()
            else:
                raise ConfigurationError("Неизвестное подключение")
            return "Подключение и доступ к модели проверены"
    except ConfigurationError:
        raise
    except (ValueError, KeyError):
        raise ConfigurationError("Неверный тип подключения или токен") from None
    except TimeoutError:
        raise ConfigurationError("Превышено время ожидания подключения") from None
    except Exception as exc:
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if status in (401, 403) or type(exc).__name__ == "TelegramUnauthorizedError":
            message = "Ключ или токен отклонён сервисом; проверьте права доступа"
        elif status == 404:
            message = "Модель или адрес API не найден"
        elif status == 429:
            message = "Превышена квота или частота запросов провайдера"
        elif type(exc).__name__ == "TokenValidationError":
            message = "Некорректный формат токена бота"
        else:
            message = (
                "Подключение не удалось: проверьте ключ, адрес, модель и доступ к сервису"
            )
        raise ConfigurationError(message) from None


def required_checks(changes: dict[str, str | None], values: dict[str, str]) -> set[str]:
    targets = set()
    for provider in ("openai", "gemini"):
        prefix = provider.upper() + "_"
        changed = any(k.startswith(prefix) for k in changes)
        active = (
            values["LLM_PROVIDER"] == provider
            or values["FAQ_EMBEDDING_PROVIDER"] == provider
            or values["LIGHTRAG_LLM_PROVIDER"] == provider
            or values["PDF_PARSER_PROVIDER"] == provider
            or values["LIGHTRAG_EMBEDDING_PROVIDER"] == provider
        )
        if (
            changed
            or any(
                k in changes
                for k in (
                    "LLM_PROVIDER",
                    "FAQ_EMBEDDING_PROVIDER",
                    "LIGHTRAG_LLM_PROVIDER",
                    "PDF_PARSER_PROVIDER",
                    "LIGHTRAG_EMBEDDING_PROVIDER",
                )
            )
        ) and active:
            if not values[prefix + "API_KEY"]:
                raise ConfigurationError("Для выбранного провайдера нужен API-ключ")
            for purpose, selector in (
                ("chat", "LLM_PROVIDER"),
                ("graph", "LIGHTRAG_LLM_PROVIDER"),
                ("vision", "PDF_PARSER_PROVIDER"),
                ("graph_embedding", "LIGHTRAG_EMBEDDING_PROVIDER"),
                ("embedding", "FAQ_EMBEDDING_PROVIDER"),
            ):
                if values[selector] == provider:
                    targets.add(provider + ":" + purpose)
    for target, keys in (
        ("telegram", {"BOT_TOKEN", "TELEGRAM_ENABLED", "TELEGRAM_SOCKS5_PROXY"}),
        ("max", {"MAX_BOT_TOKEN", "MAX_ENABLED"}),
    ):
        if (
            keys.intersection(changes)
            and values["TELEGRAM_ENABLED" if target == "telegram" else "MAX_ENABLED"]
            == "true"
        ):
            targets.add(target)
    return targets
