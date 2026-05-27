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


class PreprocessDocumentRequest(BaseModel):
    text: str = Field(..., description="Текст документа для предобработки")


class PreprocessDocumentResponse(BaseModel):
    text: str = Field(..., description="Текст после расширения аббревиатур и очистки")
    chars: int = Field(..., description="Количество символов в обработанном тексте")


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
    id: str = Field(..., description="Идентификатор документа (обычно URL или имя файла)")
    title: str | None = Field(None, description="Название документа")
    url: str | None = Field(None, description="Оригинальный URL или путь к файлу")
    status: str = Field(..., description="Статус обработки")
    content_hash: str | None = Field(None, description="SHA-256 сырого контента")
    content_summary: str | None = Field(None, description="Краткое содержание")
    content_length: int | None = Field(None, description="Длина контента")
    created_at: str | None = Field(None, description="Дата создания")
    updated_at: str | None = Field(None, description="Дата обновления")
    last_indexed_at: str | None = Field(None, description="Дата последней индексации")


class RagDocumentListResponse(BaseModel):
    documents: list[RagDocument] = Field(..., description="Список документов в RAG")


class CsvImportResult(BaseModel):
    title: str = Field(..., description="Название документа")
    url: str = Field(..., description="URL документа")
    success: bool = Field(..., description="Статус обработки")
    message: str | None = Field(None, description="Сообщение об ошибке или статус")


class CsvImportResponse(BaseModel):
    imported_count: int = Field(
        ..., description="Количество успешно импортированных документов"
    )
    total_found: int = Field(..., description="Всего найдено ссылок в CSV")
    results: list[CsvImportResult] = Field(
        ..., description="Результаты обработки по каждой ссылке"
    )


class RagDocumentContentResponse(BaseModel):
    id: str = Field(..., description="Идентификатор документа")
    content: str = Field(..., description="Полный текст документа")


class CsvImportPreviewResult(BaseModel):
    title: str = Field(..., description="Название документа")
    url: str = Field(..., description="URL документа")
    comment: str | None = Field(None, description="Комментарий из CSV")


class CsvImportPreviewResponse(BaseModel):
    total_found: int = Field(..., description="Всего найдено ссылок в CSV")
    results: list[CsvImportPreviewResult] = Field(
        ..., description="Список найденных документов для превью"
    )


class DocumentCheckRequest(BaseModel):
    document_ids: list[str] | None = Field(
        None, description="ID документов для проверки; если не переданы, проверяются все"
    )


class DocumentCheckResult(BaseModel):
    id: str = Field(..., description="ID документа")
    title: str = Field(..., description="Название документа")
    source_url: str | None = Field(None, description="URL источника")
    status: str = Field(..., description="changed, unchanged, skipped, failed, updated")
    content_hash: str | None = Field(None, description="Новый SHA-256")
    previous_hash: str | None = Field(None, description="Сохраненный SHA-256")
    message: str | None = Field(None, description="Ошибка или пояснение")


class DocumentCheckResponse(BaseModel):
    checked_count: int = Field(..., description="Количество проверенных документов")
    changed_count: int = Field(..., description="Количество измененных документов")
    results: list[DocumentCheckResult] = Field(..., description="Результаты проверки")


class DocumentUpdateResponse(BaseModel):
    checked_count: int = Field(..., description="Количество проверенных документов")
    updated_count: int = Field(..., description="Количество обновленных документов")
    results: list[DocumentCheckResult] = Field(..., description="Результаты обновления")
