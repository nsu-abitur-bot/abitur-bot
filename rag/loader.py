from chonkie.chunker import RecursiveChunker

from rag.vectorstore import get_vectorstore


def add_texts(texts: list[str]):
    # Инициализация RecursiveChunker
    chunker = RecursiveChunker(chunk_size=500, rules=None)

    all_chunks: list[str] = []
    for text in texts:
        chunks = chunker.chunk(text)
        for chunk in chunks:
            all_chunks.append(chunk.text)

    db = get_vectorstore()
    db.add_texts(all_chunks)
