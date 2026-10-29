"""Парсер рейтинговых списков абитуриентов НГУ."""

import logging
import os
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class ListFetchResult:
    """Результат загрузки одного конкурсного списка.

    Ошибка — отдельный признак, а не пустой список: иначе упавший запрос
    неотличим от списка, в котором действительно никого нет (#286).
    """

    entries: list[RatingEntry] = field(default_factory=list)
    page_hash: str = ""
    direction_name: str = ""
    error: str | None = None
    # Повтор с новой сессией может помочь (сеть, 5xx, протухший CSRF), а
    # битые параметры в URL — нет.
    retryable: bool = True

    @property
    def ok(self) -> bool:
        return self.error is None

    @classmethod
    def failure(cls, error: str, retryable: bool = True) -> "ListFetchResult":
        return cls(error=error, retryable=retryable)


def _parse_list_response(text: str, data: object) -> ListFetchResult:
    """Разбирает JSON list-content; неожиданный формат считается ошибкой.

    Ответ без ключа items — это не пустой список, а сменившийся протокол
    или страница-заглушка вместо данных.
    """
    if not isinstance(data, dict) or "items" not in data:
        return ListFetchResult.failure("неожиданный формат ответа: нет ключа items")
    return ListFetchResult(
        entries=extract_entries(data),
        page_hash=calculate_page_hash(text),
        direction_name=extract_direction_name(data),
    )


def _resolve_type_ids() -> list[int]:
    raw = os.getenv("RATING_TYPES", DEFAULT_LIST_TYPES)
    ids = [v for part in raw.split(",") if (v := _to_int(part.strip())) is not None]
    return ids or [30]


def build_leaderboard_url(
    faculty: int,
    direction: int,
    condition: int,
    type_: int,
) -> str:
    """
    Строит URL рейтингового списка с заданными параметрами.
    Args:
        faculty: Идентификатор факультета.
        direction: Идентификатор направления подготовки.
        condition: Идентификатор условия поступления (основание конкурса).
        type_: Тип списка (например, общий, целевой и т.п.).
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


def get_csrf_token(session: requests.Session, headers: dict) -> str | None:
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
        params = extract_params_from_url(url)
    except ValueError as e:
        logger.error(f"Некорректные параметры в URL {url}: {e}")
        return [], "", ""

    with requests.Session() as session:
        csrf_token = get_csrf_token(session, headers)
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
            entries = extract_entries(data)

            direction_name = extract_direction_name(data)

            logger.info(f"Получено {len(entries)} записей из {url}")
            return entries, page_hash, direction_name

        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при запросе {url}: {e}")
            return [], "", ""
        except Exception as e:
            logger.error(f"Ошибка при парсинге {url}: {e}")
            return [], "", ""


def parse_mock_rating_page(url: str) -> ListFetchResult:
    """Получает данные рейтингового списка с мок-сайта (напрямую GET-запросом)."""
    config = get_rating_config()
    headers = config.get("headers", {})

    try:
        response = requests.get(
            url, headers={**headers, "Accept": "application/json"}, timeout=10
        )
        response.raise_for_status()
        result = _parse_list_response(response.text, response.json())
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при GET запросе к моку {url}: {e}")
        return ListFetchResult.failure(f"запрос: {e}")
    except Exception as e:
        logger.error(f"Ошибка при парсинге мока {url}: {e}")
        return ListFetchResult.failure(f"разбор ответа: {e}")

    if result.ok:
        logger.info(f"Получено {len(result.entries)} записей из мока {url}")
    else:
        logger.error(f"Ошибка при разборе мока {url}: {result.error}")
    return result


def extract_direction_name(data: dict) -> str:
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
) -> ListFetchResult:
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
        result = _parse_list_response(response.text, response.json())
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при запросе {url_for_log}: {e}")
        return ListFetchResult.failure(f"запрос: {e}")
    except Exception as e:
        logger.error(f"Ошибка при парсинге {url_for_log}: {e}")
        return ListFetchResult.failure(f"разбор ответа: {e}")
    if not result.ok:
        logger.error(f"Ошибка при разборе {url_for_log}: {result.error}")
    return result


def create_rating_session() -> tuple[requests.Session | None, str | None]:
    """Создаёт HTTP-сессию и получает CSRF-токен для серии запросов."""
    config = get_rating_config()
    headers = config.get("headers", {})
    session = requests.Session()
    csrf_token = get_csrf_token(session, headers)
    if not csrf_token:
        session.close()
        return None, None
    return session, csrf_token


def parse_rating_url_with_session(
    session: requests.Session, csrf_token: str, url: str
) -> ListFetchResult:
    """Парсит один список, переиспользуя переданные сессию и CSRF-токен."""
    config = get_rating_config()
    headers = config.get("headers", {})
    try:
        params = extract_params_from_url(url)
    except ValueError as e:
        logger.error(f"Некорректные параметры в URL {url}: {e}")
        return ListFetchResult.failure(f"некорректный URL: {e}", retryable=False)
    return _fetch_list_content(session, csrf_token, params, headers, url_for_log=url)


class RatingListFetcher:
    """Обход многих списков в одной HTTP-сессии с одним CSRF-токеном.

    Обход идёт минутами, и токен или кука могут протухнуть на середине. Тогда
    при ошибке сессия и токен запрашиваются заново, а список — ещё раз.
    Число таких перезапусков за обход ограничено: если сайт лежит, долбить
    его новой сессией на каждый список бессмысленно.
    """

    def __init__(self, max_refreshes: int = 3) -> None:
        self._session: requests.Session | None = None
        self._csrf_token: str | None = None
        self.refreshes_left = max_refreshes

    def open(self) -> bool:
        self._session, self._csrf_token = create_rating_session()
        return self._session is not None and self._csrf_token is not None

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
        self._session = None
        self._csrf_token = None

    def _fetch_once(self, url: str) -> ListFetchResult:
        if self._session is None or self._csrf_token is None:
            return ListFetchResult.failure("нет HTTP-сессии с CSRF-токеном")
        return parse_rating_url_with_session(self._session, self._csrf_token, url)

    def fetch(self, url: str) -> ListFetchResult:
        result = self._fetch_once(url)
        if result.ok or not result.retryable or self.refreshes_left <= 0:
            return result

        self.refreshes_left -= 1
        logger.warning(
            "Ошибка списка %s (%s): перезапрашиваю сессию и CSRF-токен и повторяю",
            url,
            result.error,
        )
        self.close()
        if not self.open():
            # Токен не получить — сайт, скорее всего, лежит: дальше не перезапускаем.
            self.refreshes_left = 0
            return result
        return self._fetch_once(url)


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


def extract_params_from_url(url: str) -> dict:
    """Извлекает faculty/direction/condition/type из URL.

    Дефолты для отсутствующих в URL параметров берёт из parser/config/config.yaml
    (rating.defaults) — это единственное место, где живут эти числа (#326).
    """
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)

    config = get_rating_config()
    defaults = config.get("defaults", {}) if isinstance(config, dict) else {}

    def _default(name: str) -> int:
        value = defaults.get(name)
        if value is None:
            raise RuntimeError(
                f"В parser/config/config.yaml не задан rating.defaults.{name}"
            )
        return int(value)

    return {
        "faculty": int(qs.get("faculty", [_default("faculty")])[0]),
        "direction": int(qs.get("direction", [_default("direction")])[0]),
        "condition": int(qs.get("condition", [_default("condition")])[0]),
        "type": int(qs.get("type", [_default("type")])[0]),
    }


def extract_entries(data: dict) -> list[RatingEntry]:
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
