from pydantic import BaseModel, Field


class UploadedDocumentResult(BaseModel):
    filename: str = Field(..., description="Имя загруженного файла")
    status: str = Field(..., description="Статус обработки: indexed или skipped")
    message: str = Field(..., description="Подробности результата обработки")
    chars: int = Field(0, description="Количество символов, отправленных в RAG")


class ParsedDocument(BaseModel):
    title: str = Field(..., description="Название документа")
    url: str = Field(..., description="URL для скачивания/ссылки на документ")


class ParsedPageResult(BaseModel):
    title: str = Field(..., description="Название страницы")
    url: str = Field(..., description="URL страницы")
    text: str = Field(..., description="Сырой или предобработанный текст страницы")
    documents: list[ParsedDocument] = Field(
        ..., description="Список найденных PDF-документов"
    )


class ConfirmUploadRequest(BaseModel):
    title: str = Field(..., description="Название страницы")
    url: str = Field(..., description="URL страницы")
    text: str = Field(..., description="Отредактированный текст страницы")
    documents: list[ParsedDocument] = Field(
        ..., description="Финальный список файлов для загрузки"
    )


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


class RagDocument(BaseModel):
    id: str = Field(
        ..., description="Идентификатор документа (обычно URL или имя файла)"
    )
    status: str = Field(..., description="Статус обработки")
    content_summary: str | None = Field(None, description="Краткое содержание")
    content_length: int | None = Field(None, description="Длина контента")
    created_at: str | None = Field(None, description="Дата создания")


class RagDocumentListResponse(BaseModel):
    documents: list[RagDocument] = Field(..., description="Список документов в RAG")
