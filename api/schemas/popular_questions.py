from pydantic import BaseModel


class PopularQuestion(BaseModel):
    question: str
    count: int

class PopularQuestionsResponse(BaseModel):
    questions: list[PopularQuestion]
