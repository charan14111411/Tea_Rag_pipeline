import time

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from src.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_PATH,
    QDRANT_COLLECTION,
    RETRIEVAL_MODE,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
)
from src.ingestion.embedder import get_embeddings
from src.ingestion.sparse import BM25SparseEmbeddings

_MODES = {"hybrid": RetrievalMode.HYBRID, "dense": RetrievalMode.DENSE, "sparse": RetrievalMode.SPARSE}

_client = None
_vectorstore = None


def get_qdrant_client() -> QdrantClient:
    """
    Return a shared Qdrant client.

    Local (embedded) mode locks the storage folder, so only one client per
    process should exist - this function makes sure of that.
    """
    global _client
    if _client is None:
        if QDRANT_URL:
            _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
        else:
            _client = QdrantClient(path=QDRANT_PATH)
    return _client


def get_chunks_by_index(book: str, indexes: list, attempts: int = 3) -> dict:
    """Fetch chunks of one book by their chunk_index -> {chunk_index: page_content}."""
    client = get_qdrant_client()
    for attempt in range(1, attempts + 1):
        try:
            points, _ = client.scroll(
                QDRANT_COLLECTION,
                scroll_filter=Filter(must=[
                    FieldCondition(key="metadata.book", match=MatchValue(value=book)),
                    FieldCondition(key="metadata.chunk_index", match=MatchAny(any=list(indexes))),
                ]),
                limit=len(indexes),
                with_payload=True,
                with_vectors=False,
            )
            break
        except Exception:
            # The server connection can drop briefly - retry before giving up
            if attempt == attempts:
                raise
            time.sleep(attempt * 2)
    return {
        p.payload["metadata"]["chunk_index"]: p.payload["page_content"]
        for p in points
        if "chunk_index" in p.payload.get("metadata", {})
    }


def make_vectorstore(mode: str = RETRIEVAL_MODE, embedding=None, sparse_embedding=None) -> QdrantVectorStore:
    """
    Vector store over the tea collection in the given search mode
    ("hybrid", "dense" or "sparse"). Every chunk stores both a dense vector
    (named "dense") and a BM25 sparse vector (named "bm25"), so any mode works
    on the same collection.
    """
    if mode not in _MODES:
        raise ValueError(f"RETRIEVAL_MODE must be one of {list(_MODES)}, got '{mode}'")
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=QDRANT_COLLECTION,
        embedding=embedding or get_embeddings(),
        sparse_embedding=sparse_embedding or BM25SparseEmbeddings(),
        retrieval_mode=_MODES[mode],
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )


def load_vectorstore() -> QdrantVectorStore:
    global _vectorstore
    if _vectorstore is None:
        client = get_qdrant_client()

        if not client.collection_exists(QDRANT_COLLECTION):
            raise RuntimeError(
                f"Qdrant collection '{QDRANT_COLLECTION}' does not exist. "
                "Run ingestion first: python -m src.ingestion.ingest"
            )

        _vectorstore = make_vectorstore()
    return _vectorstore
