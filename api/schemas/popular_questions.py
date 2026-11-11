from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class PopularQuestion(BaseModel):
    question: str
    count: int
    variants: list[str] = Field(default_factory=list)


class PopularQuestionsResponse(BaseModel):
    questions: list[PopularQuestion]


class PopularQuestionsSnapshotResponse(PopularQuestionsResponse):
    status: Literal["pending", "ready", "error"] = "ready"
    updated_at: datetime | None = None
