from typing import Optional

from pydantic import BaseModel, Field


class AbbrevItem(BaseModel):
    id: Optional[str] = Field(None, description="UUID записи (заполняется при ответе)")
    short: str = Field(..., description="Аббревиатура (например, НГУ)")
    full: str = Field(
        ...,
        description="Расшифровка (например, Новосибирский государственный университет)",
    )


class AbbrevListResponse(BaseModel):
    items: list[AbbrevItem] = Field(..., description="Список аббревиатур")
