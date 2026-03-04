"""Общие DTO для слоя базы данных и парсера."""

from dataclasses import dataclass


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
