from pydantic import BaseModel, Field


class PopularQuestion(BaseModel):
    question: str
    count: int
    variants: list[str] = Field(default_factory=list)


class PopularQuestionsResponse(BaseModel):
    questions: list[PopularQuestion]
