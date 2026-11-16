from src.config import RETRIEVAL_K
from src.retrieval.vectorstore_manager import load_vectorstore

_retriever = None


def get_retriever():
    global _retriever
    if _retriever is None:
        vectorstore = load_vectorstore()
        _retriever = vectorstore.as_retriever(
            search_kwargs={"k": RETRIEVAL_K}
        )
    return _retriever
