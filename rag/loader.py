from chonkie import chunk_text
from rag.vectorstore import get_vectorstore


def add_texts(texts: list[str]):
    chunks = []

    for text in texts:
        # chunk_text возвращает генератор, поэтому оборачиваем в список
        text_chunks = list(chunk_text(
            text=text,
            max_chunk_size=500,
            overlap=50
        ))
        chunks.extend(text_chunks)

    # Сохраняем чанки в векторное хранилище
    db = get_vectorstore()
    db.add_texts(texts=chunks)
