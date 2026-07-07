"""Парсер рейтинговых списков абитуриентов НГУ."""

import logging
import os
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup

from db.postgres.dto import RatingEntry
from parser.config.load_config import get_rating_config
from parser.utils import calculate_page_hash

logger = logging.getLogger(__name__)

BASE_URL = "https://abiturient.nsu.ru/bachelor"
API_ENDPOINT = "https://abiturient.nsu.ru/site/list-content"
SITE_ORIGIN = "https://abiturient.nsu.ru"

# Тип списка для отслеживания. 30 — «если бы зачисление состоялось сегодня»
# (показывает, прошёл бы абитуриент). Остальные типы (0 — подавшие, 20 —
# рейтинговый список, 50 — высший приоритет) не отслеживаем. Переопределяется
# через env RATING_TYPES (список через запятую).
DEFAULT_LIST_TYPES = "30"


def _resolve_type_ids() -> list[int]:
    raw = os.getenv("RATING_TYPES", DEFAULT_LIST_TYPES)
    ids = [v for part in raw.split(",") if (v := _to_int(part.strip())) is not None]
    return ids or [30]


def build_leaderboard_url(
    faculty: int,
    direction: int,
    condition: int,
    type_: int = 0,
) -> str:
    """
    Строит URL рейтингового списка с заданными параметрами.
    Args:
        faculty: Идентификатор факультета.
        direction: Идентификатор направления подготовки.
        condition: Идентификатор условия поступления (основание конкурса).
        type_: Тип списка (например, общий, целевой и т.п.); по умолчанию 0.
    Returns:
        Полный URL страницы рейтингового списка с подставленными параметрами.
    """

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


def parse_rating_page(url: str) -> tuple[list[RatingEntry], str, str]:
    """
    Получает данные рейтингового списка через Form Data POST запрос.

    Returns:
        (список записей рейтинга, хэш ответа, название направления)
    """
    config = get_rating_config()
    headers = config.get("headers", {})

    try:
        params = _extract_params_from_url(url)
    except ValueError as e:
        logger.error(f"Некорректные параметры в URL {url}: {e}")
        return [], "", ""

    with requests.Session() as session:
        csrf_token = _get_csrf_token(session, headers)
        if not csrf_token:
            return [], "", ""

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

            direction_name = ""
            if (
                data
                and data.get("items")
                and isinstance(data["items"], list)
                and len(data["items"]) > 0
            ):
                item = data["items"][0]
                if (
                    item
                    and isinstance(item, dict)
                    and item.get("info")
                    and item["info"].get("speciality")
                ):
                    direction_name = item["info"]["speciality"].get("name", "")

            logger.info(f"Получено {len(entries)} записей из {url}")
            return entries, page_hash, direction_name

        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при запросе {url}: {e}")
            return [], "", ""
        except Exception as e:
            logger.error(f"Ошибка при парсинге {url}: {e}")
            return [], "", ""


def parse_mock_rating_page(url: str) -> tuple[list[RatingEntry], str, str]:
    """
    Получает данные рейтингового списка с мок-сайта (напрямую GET-запросом).

    Args:
        url: Ссылка на эндпоинт мок-сервера.

    Returns:
        (список записей рейтинга, хэш ответа, название направления)
    """
    config = get_rating_config()
    headers = config.get("headers", {})

    try:
        response = requests.get(
            url, headers={**headers, "Accept": "application/json"}, timeout=10
        )
        response.raise_for_status()

        data = response.json()
        page_hash = calculate_page_hash(response.text)
        entries = _extract_entries(data)

        direction_name = ""
        if (
            data
            and data.get("items")
            and isinstance(data["items"], list)
            and len(data["items"]) > 0
        ):
            item = data["items"][0]
            if (
                item
                and isinstance(item, dict)
                and item.get("info")
                and item["info"].get("speciality")
            ):
                direction_name = item["info"]["speciality"].get("name", "")

        logger.info(f"Получено {len(entries)} записей из мока {url}")
        return entries, page_hash, direction_name

    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при GET запросе к моку {url}: {e}")
        return [], "", ""
    except Exception as e:
        logger.error(f"Ошибка при парсинге мока {url}: {e}")
        return [], "", ""


def _extract_direction_name(data: dict) -> str:
    """Достаёт название направления из JSON-ответа list-content."""
    items = data.get("items") if isinstance(data, dict) else None
    if not items or not isinstance(items, list):
        return ""
    item = items[0]
    if isinstance(item, dict) and isinstance(item.get("info"), dict):
        speciality = item["info"].get("speciality")
        if isinstance(speciality, dict):
            return speciality.get("name", "") or ""
    return ""


def _fetch_list_content(
    session: requests.Session,
    csrf_token: str,
    params: dict,
    headers: dict,
    url_for_log: str = "",
) -> tuple[list[RatingEntry], str, str]:
    """POST-запрос к list-content с готовой сессией и CSRF-токеном.

    Вынесено отдельно, чтобы обходить много списков, переиспользуя одну сессию
    и один CSRF-токен (не дёргать страницу за токеном на каждый список).
    """
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
            data=form_data,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        page_hash = calculate_page_hash(response.text)
        entries = _extract_entries(data)
        return entries, page_hash, _extract_direction_name(data)
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при запросе {url_for_log}: {e}")
        return [], "", ""
    except Exception as e:
        logger.error(f"Ошибка при парсинге {url_for_log}: {e}")
        return [], "", ""


def create_rating_session() -> tuple[requests.Session | None, str | None]:
    """Создаёт HTTP-сессию и получает CSRF-токен для серии запросов."""
    config = get_rating_config()
    headers = config.get("headers", {})
    session = requests.Session()
    csrf_token = _get_csrf_token(session, headers)
    if not csrf_token:
        session.close()
        return None, None
    return session, csrf_token


def parse_rating_url_with_session(
    session: requests.Session, csrf_token: str, url: str
) -> tuple[list[RatingEntry], str, str]:
    """Парсит один список, переиспользуя переданные сессию и CSRF-токен."""
    config = get_rating_config()
    headers = config.get("headers", {})
    try:
        params = _extract_params_from_url(url)
    except ValueError as e:
        logger.error(f"Некорректные параметры в URL {url}: {e}")
        return [], "", ""
    return _fetch_list_content(session, csrf_token, params, headers, url_for_log=url)


def _to_int(value: object) -> int | None:
    """Безопасно приводит значение справочника (строка/число) к int."""
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _get_taxonomy_options(
    session: requests.Session, headers: dict, path: str
) -> list[dict]:
    """GET к справочному эндпоинту (faculty/direction/condition/type) → список."""
    try:
        response = session.get(
            f"{SITE_ORIGIN}{path}",
            headers={**headers, "Accept": "application/json"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.error(f"Ошибка получения справочника {path}: {e}")
        return []


def fetch_all_leaderboard_urls(degree: str = "bachelor") -> list[str]:
    """Перебирает таксономию сайта и строит URL всех конкурсных списков.

    Комбинация: faculty × direction(факультета) × condition × type. Значения
    берутся из справочных эндпоинтов сайта (/faculty/index и т.п.), поэтому
    список автоматически подстраивается под текущий набор факультетов НГУ.
    """
    config = get_rating_config()
    headers = config.get("headers", {})
    urls: list[str] = []
    type_ids = _resolve_type_ids()

    with requests.Session() as session:
        faculties = _get_taxonomy_options(
            session, headers, f"/faculty/index?degree={degree}"
        )
        conditions = _get_taxonomy_options(
            session, headers, f"/condition/index?degree={degree}"
        )

        for faculty in faculties:
            faculty_id = _to_int(faculty.get("value"))
            if faculty_id is None:
                continue
            directions = _get_taxonomy_options(
                session,
                headers,
                f"/direction/index?degree={degree}&faculty={faculty_id}",
            )
            for direction in directions:
                direction_id = _to_int(direction.get("value"))
                if direction_id is None:
                    continue
                for condition in conditions:
                    condition_id = _to_int(condition.get("value"))
                    if condition_id is None:
                        continue
                    for type_id in type_ids:
                        urls.append(
                            build_leaderboard_url(
                                faculty=faculty_id,
                                direction=direction_id,
                                condition=condition_id,
                                type_=type_id,
                            )
                        )

    logger.info("Составлено %d URL конкурсных списков для degree=%s", len(urls), degree)
    return urls


def _extract_params_from_url(url: str) -> dict:
    """Извлекает faculty/direction/condition/type из URL."""
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
