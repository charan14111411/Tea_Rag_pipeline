"""
Ingest all tea book PDFs into Qdrant.

Usage (from the project root):
    python -m src.ingestion.ingest

The collection is rebuilt from scratch on every run, so re-run this whenever
you add, remove or replace PDFs in the data folder.

Note: with local (embedded) Qdrant, stop the API server / CLI before running
this - only one process can open the local Qdrant folder at a time.
"""
import time
import uuid

from qdrant_client.models import (
    Distance, Filter, FilterSelector, Modifier, SparseVectorParams, VectorParams,
)

from src.config import (
    DATA_DIR, QDRANT_COLLECTION, QDRANT_URL, QDRANT_PATH, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME,
)
from src.ingestion.loader import load_pdfs
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import get_embeddings
from src.ingestion.sparse import BM25SparseEmbeddings
from src.retrieval.vectorstore_manager import get_qdrant_client, make_vectorstore


def ensure_collection_is_ours(client, vector_size):
    """
    Safety check before deleting an existing collection - important on a shared
    Qdrant server, where a wrong QDRANT_COLLECTION value could otherwise wipe
    another project's data. Only collections that look like ones this pipeline
    built (same vector size/distance, chunks carrying 'book' metadata) are deleted.
    """
    info = client.get_collection(QDRANT_COLLECTION)
    vectors = info.config.params.vectors
    # Older builds used one unnamed vector; current builds use named "dense" (+ "bm25")
    dense = vectors.get(DENSE_VECTOR_NAME) if isinstance(vectors, dict) else vectors
    same_shape = (
        isinstance(dense, VectorParams)
        and dense.size == vector_size
        and dense.distance == Distance.COSINE
    )

    points, _ = client.scroll(QDRANT_COLLECTION, limit=1, with_payload=True)
    empty = not points
    has_book = bool(points) and "book" in (points[0].payload or {}).get("metadata", {})

    if not (same_shape and (empty or has_book)):
        raise SystemExit(
            f"Refusing to delete collection '{QDRANT_COLLECTION}': it does not look like a "
            "tea-books collection built by this pipeline (different vector config or payload). "
            "Set QDRANT_COLLECTION to a different name in .env."
        )


def upload_with_retry(vectorstore, chunks, batch_size=64, max_attempts=5):
    """
    Upload chunks in batches, retrying a batch if the network drops (remote server).
    Each chunk gets a deterministic ID, so a retried batch overwrites rather than
    duplicates points that may already have reached the server.
    """
    ids = [
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c.metadata.get('source')}|{i}"))
        for i, c in enumerate(chunks)
    ]

    for start in range(0, len(chunks), batch_size):
        batch, batch_ids = chunks[start:start + batch_size], ids[start:start + batch_size]
        for attempt in range(1, max_attempts + 1):
            try:
                vectorstore.add_documents(batch, ids=batch_ids, batch_size=batch_size)
                break
            except Exception as e:
                if attempt == max_attempts:
                    raise
                wait = 2 ** attempt
                print(f"  Batch at chunk {start} failed ({type(e).__name__}), retry {attempt}/{max_attempts - 1} in {wait}s")
                time.sleep(wait)

        done = min(start + batch_size, len(chunks))
        if done % (batch_size * 10) == 0 or done == len(chunks):
            print(f"  Uploaded {done}/{len(chunks)}")


def main():
    documents = load_pdfs(DATA_DIR)

    chunks = chunk_documents(documents)

    embeddings = get_embeddings()
    vector_size = len(embeddings.embed_query("tea"))

    client = get_qdrant_client()

    # Rebuild the collection so stale chunks from removed/changed books don't linger
    if client.collection_exists(QDRANT_COLLECTION):
        ensure_collection_is_ours(client, vector_size)
        print(f"Deleting existing collection '{QDRANT_COLLECTION}'")
        client.delete_collection(QDRANT_COLLECTION)

    # Two vectors per chunk: "dense" (BGE, meaning) and "bm25" (sparse keywords).
    # Modifier.IDF makes Qdrant weight rare words higher - the IDF half of BM25.
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config={DENSE_VECTOR_NAME: VectorParams(size=vector_size, distance=Distance.COSINE)},
        sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)},
    )

    # On Windows, local Qdrant's delete_collection can silently fail to remove the
    # storage file (it is still open), and create_collection then reopens the old
    # points. Purge anything left over so chunks are never duplicated.
    leftover = client.count(QDRANT_COLLECTION, exact=True).count
    if leftover:
        print(f"Removing {leftover} leftover points from the previous build")
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=FilterSelector(filter=Filter()),
        )

    # Hybrid mode makes add_documents compute both the dense and the BM25 vector
    sparse = BM25SparseEmbeddings(
        avg_doc_length=BM25SparseEmbeddings.average_length([c.page_content for c in chunks])
    )
    vectorstore = make_vectorstore("hybrid", embedding=embeddings, sparse_embedding=sparse)

    print(f"Embedding and uploading {len(chunks)} chunks to Qdrant (dense + BM25)...")
    upload_with_retry(vectorstore, chunks)

    count = client.count(QDRANT_COLLECTION, exact=True).count
    location = QDRANT_URL or QDRANT_PATH
    print(f"Qdrant collection '{QDRANT_COLLECTION}' ready at {location} with {count} vectors")

    if QDRANT_URL:
        # Index book name and chunk position: used to filter by book and to fetch
        # a chunk's neighbours quickly
        client.create_payload_index(QDRANT_COLLECTION, "metadata.book", field_schema="keyword")
        client.create_payload_index(QDRANT_COLLECTION, "metadata.chunk_index", field_schema="integer")

    # Release the local storage lock cleanly (avoids noisy errors at interpreter shutdown)
    client.close()


if __name__ == "__main__":
    main()
