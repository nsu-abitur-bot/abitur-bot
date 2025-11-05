from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.vectorstore import get_vectorstore


def add_texts(texts: list[str]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))

    db = get_vectorstore()
    db.add_texts(chunks)
