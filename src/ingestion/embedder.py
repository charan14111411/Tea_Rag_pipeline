from langchain_huggingface import HuggingFaceEmbeddings

from src.config import BGE_QUERY_INSTRUCTION

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_embeddings = None


def get_embeddings():
    """
    Dense embedding model (BGE-small, 384 dimensions, runs locally on CPU).
    Questions get BGE's retrieval instruction prepended; stored chunks do not -
    this asymmetric setup is how BGE was trained for search.
    """
    global _embeddings
    if _embeddings is None:
        query_kwargs = {"prompt": BGE_QUERY_INSTRUCTION} if BGE_QUERY_INSTRUCTION else {}
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            query_encode_kwargs=query_kwargs,
        )
    return _embeddings
