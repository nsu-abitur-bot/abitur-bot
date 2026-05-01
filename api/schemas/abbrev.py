from pydantic import BaseModel, Field


class AbbrevItem(BaseModel):
    short: str = Field(..., description="Аббревиатура (например, НГУ)")
    full: str = Field(
        ...,
        description="Расшифровка (например, Новосибирский государственный университет)",
    )


class AbbrevListResponse(BaseModel):
    items: list[AbbrevItem] = Field(..., description="Список аббревиатур")
