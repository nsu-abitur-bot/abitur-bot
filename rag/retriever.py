from rag.vectorstore import get_vectorstore


def search_similar(query: str, k: int = 3):
    db = get_vectorstore()
    return db.similarity_search(query, k=k)
