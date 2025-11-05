from langchain_chroma import Chroma

from rag.embeddings import get_embedding_model

PERSIST_DIR = "./chroma_db"


def get_vectorstore():
    embedding = get_embedding_model()
    db = Chroma(
        collection_name="abitur_knowledge_base",
        embedding_function=embedding,
        persist_directory=PERSIST_DIR,
    )
    return db
