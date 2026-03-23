"""Общие DTO для слоя базы данных и парсера."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RatingEntry:
    """Запись рейтинга абитуриента.

    Используется как DTO между парсером (parser.rating_parser)
    и сервисом БД (db.postgres.services.rating).
    """

    identifier: str
    place: int
    status: str = ""
    competition_type: str = ""


@dataclass
class RatingChange:
    """Изменение в рейтинге абитуриента.

    Используется для определения необходимости уведомления пользователя.
    """

    user_id: int
    applicant_id: str
    leaderboard_id: str
    old_place: Optional[int]
    new_place: int
    old_status: Optional[str]
    new_status: str
    old_competition_type: Optional[str]
    new_competition_type: str
    is_new: bool = False
    direction: str = ""
    url: str = ""
