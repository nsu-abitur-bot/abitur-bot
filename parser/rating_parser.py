"""Парсер рейтинговых списков абитуриентов НГУ."""

import logging
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

from parser.config.load_config import get_rating_config
from parser.utils import calculate_page_hash

logger = logging.getLogger(__name__)

BASE_URL = "https://abiturient.nsu.ru/bachelor"
API_ENDPOINT = "https://abiturient.nsu.ru/site/list-content"


@dataclass
class RatingEntry:
    identifier: str
    place: int
    status: str = ""
    competition_type: str = ""


def build_leaderboard_url(
    faculty: int,
    direction: int,
    condition: int,
    type_: int = 0,
) -> str:
    config = get_rating_config()
    base_url = config.get("url", BASE_URL)
    return (
        f"{base_url}"
        f"?faculty={faculty}"
        f"&direction={direction}"
        f"&condition={condition}"
        f"&type={type_}"
    )


def _get_csrf_token(session: requests.Session, headers: dict) -> str | None:
    """Получает CSRF-токен со страницы."""
    config = get_rating_config()
    base_url = config.get("url", BASE_URL)

    try:
        response = session.get(base_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        meta = soup.find("meta", {"name": "csrf-token"})
        if meta:
            token = meta.get("content")
            if token:
                token_str = token[0] if isinstance(token, list) else str(token)
                logger.info("CSRF-токен успешно получен")
                return token_str
        logger.error("CSRF-токен не найден на странице")
        return None
    except Exception as e:
        logger.error(f"Ошибка при получении CSRF-токена: {e}")
        return None


def parse_rating_page(url: str) -> tuple[list[RatingEntry], str]:
    """
    Получает данные рейтингового списка через Form Data POST запрос.

    Returns:
        (список записей рейтинга, хэш ответа)
    """
    config = get_rating_config()
    headers = config.get("headers", {})

    try:
        params = _extract_params_from_url(url)
    except ValueError as e:
        logger.error(f"Некорректные параметры в URL {url}: {e}")
        return [], ""

    with requests.Session() as session:
        csrf_token = _get_csrf_token(session, headers)
        if not csrf_token:
            return [], ""

        form_data = {
            "_csrf-frontend": csrf_token,
            "degree": "bachelor",
            "faculty": str(params["faculty"]),
            "direction": str(params["direction"]),
            "condition": str(params["condition"]),
            "type": str(params["type"]),
        }

        try:
            response = session.post(
                API_ENDPOINT,
                headers={**headers, "Accept": "application/json"},
                data=form_data,  # Form Data
                timeout=10,
            )
            response.raise_for_status()

            data = response.json()
            page_hash = calculate_page_hash(response.text)
            entries = _extract_entries(data)

            logger.info(f"Получено {len(entries)} записей из {url}")
            return entries, page_hash

        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при запросе {url}: {e}")
            return [], ""
        except Exception as e:
            logger.error(f"Ошибка при парсинге {url}: {e}")
            return [], ""


def _extract_params_from_url(url: str) -> dict:
    """Извлекает faculty/direction/condition/type из URL."""
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(url)
    qs = parse_qs(parsed.query)

    config = get_rating_config()
    defaults = config.get("defaults", {}) if isinstance(config, dict) else {}

    faculty_default = int(defaults.get("faculty", 8))
    direction_default = int(defaults.get("direction", 7))
    condition_default = int(defaults.get("condition", 10))
    type_default = int(defaults.get("type", 0))

    return {
        "faculty": int(qs.get("faculty", [faculty_default])[0]),
        "direction": int(qs.get("direction", [direction_default])[0]),
        "condition": int(qs.get("condition", [condition_default])[0]),
        "type": int(qs.get("type", [type_default])[0]),
    }


def _extract_entries(data: dict) -> list[RatingEntry]:
    """Извлекает записи из JSON ответа."""
    entries = []
    items = data.get("items", [])

    for item in items:
        competition_type = item.get("title", "").strip() or "без названия"
        table = item.get("table", [])

        for row in table:
            try:
                place = int(row["number"])
                identifier = str(row["name"])
                status = row.get("status", "")

                if identifier and place > 0:
                    entries.append(
                        RatingEntry(
                            identifier=identifier,
                            place=place,
                            status=status,
                            competition_type=competition_type,
                        )
                    )
            except (KeyError, ValueError):
                continue

    return entries


def find_entry_by_identifier(
    entries: list[RatingEntry], identifier: str
) -> RatingEntry | None:
    """Находит запись абитуриента по его идентификатору."""
    for entry in entries:
        if entry.identifier == identifier:
            return entry
    return None
