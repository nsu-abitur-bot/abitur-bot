"""Pagination over the merged document/import list, without per-document diagnostics."""


def material_state(document: dict) -> str:
    if document.get("import_state") in {"review", "queued", "processing", "index_failed"}:
        return "attention"
    status = (document.get("status") or "").lower().replace(" ", "_")
    if status in {
        "changed",
        "изменён",
        "изменен",
        "source_unavailable",
        "источник_недоступен",
        "indexing_failed",
        "ошибка_индексации",
    }:
        return "attention"
    processing = document.get("lightrag_status")
    if processing == "processed":
        return "ready"
    if processing in {"processing", "pending", "failed"}:
        return "attention"
    return "unknown"


def page_materials(
    documents: list[dict],
    *,
    limit: int | None = None,
    offset: int = 0,
    query: str = "",
    state: str = "all",
) -> dict:
    query = query.strip().casefold()
    filtered = [
        doc
        for doc in documents
        if (
            not query
            or any(
                query in str(doc.get(key) or "").casefold()
                for key in ("title", "url", "id")
            )
        )
        and (state == "all" or material_state(doc) == state)
    ]
    filtered.sort(key=lambda doc: (doc.get("created_at") or "", doc["id"]), reverse=True)
    return {
        "documents": filtered[offset : offset + limit if limit is not None else None],
        "total": len(filtered),
        "total_all": len(documents),
        "offset": offset,
        "limit": limit,
        "has_pending": any(
            doc.get("import_state") in {"queued", "processing"}
            or doc.get("lightrag_status") in {"pending", "processing"}
            for doc in documents
        ),
    }
