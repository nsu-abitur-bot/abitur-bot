from langchain_chroma import Chroma

from rag.embeddings import get_embedding_model

PERSIST_DIR = "./chroma_db"

# Кешируем объект базы данных
_vectorstore_instance = None


def get_vectorstore():
    global _vectorstore_instance

    if _vectorstore_instance is None:
        embedding = get_embedding_model()
        _vectorstore_instance = Chroma(
            collection_name="abitur_knowledge_base",
            embedding_function=embedding,
            persist_directory=PERSIST_DIR,
        )

    return _vectorstore_instance
