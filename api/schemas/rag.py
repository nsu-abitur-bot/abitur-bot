from pydantic import BaseModel, Field


class UploadedDocumentResult(BaseModel):
    filename: str = Field(..., description="Имя загруженного файла")
    status: str = Field(..., description="Статус обработки: indexed или skipped")
    message: str = Field(..., description="Подробности результата обработки")
    chars: int = Field(0, description="Количество символов, отправленных в RAG")


class RagUploadResponse(BaseModel):
    accepted_formats: list[str] = Field(
        ..., description="Поддерживаемые расширения файлов"
    )
    indexed_count: int = Field(
        ..., description="Количество успешно проиндексированных файлов"
    )
    skipped_count: int = Field(..., description="Количество пропущенных файлов")
    results: list[UploadedDocumentResult] = Field(
        ..., description="Результаты обработки по каждому загруженному файлу"
    )
