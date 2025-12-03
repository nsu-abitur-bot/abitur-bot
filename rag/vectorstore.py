import threading

from langchain_chroma import Chroma

from rag.embeddings import get_embedding_model

PERSIST_DIR = "./chroma_db"


_vectorstore_instance = None
_lock = threading.Lock()


def get_vectorstore():
    global _vectorstore_instance
    if _vectorstore_instance is None:
        with _lock:
            if _vectorstore_instance is None:
                embedding = get_embedding_model()
                _vectorstore_instance = Chroma(
                    collection_name="abitur_knowledge_base",
                    embedding_function=embedding,
                    persist_directory=PERSIST_DIR,
                )
    return _vectorstore_instance
