from langchain_text_splitters import RecursiveCharacterTextSplitter
from chonkie import Chonkie
from rag.vectorstore import get_vectorstore


def add_texts(texts: list[str]):
    # Инициализация Chonkie
    chonkie = Chonkie(
        max_chunk_size=500,
        overlap=50,
    )

    chunks = []
    for text in texts:
        # Разбиваем текст с помощью Chonkie
        text_chunks = chonkie.split(text)
        chunks.extend(text_chunks)

    db = get_vectorstore()
    db.add_texts(chunks)
