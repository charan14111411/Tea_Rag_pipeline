import os

from dotenv import load_dotenv

load_dotenv()

# Folder containing the tea book PDFs to ingest (every *.pdf in it is loaded).
DATA_DIR = os.getenv("TEA_DATA_DIR", "data/tea_books")

# OCR text for scanned pages is cached here so re-ingestion is fast.
OCR_CACHE_DIR = os.getenv("OCR_CACHE_DIR", "vectorstore/ocr_cache")

# Qdrant settings.
# - If QDRANT_URL is set, connect to a Qdrant server / Qdrant Cloud (QDRANT_API_KEY optional).
# - Otherwise, use embedded local Qdrant persisted on disk at QDRANT_PATH (no server needed).
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_PATH = os.getenv("QDRANT_PATH", "vectorstore/qdrant")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "tea_books")

# Search technique:
#   "hybrid" - dense (meaning) + BM25 (exact keywords), merged by rank in Qdrant (default)
#   "dense"  - meaning only (BGE embeddings)
#   "sparse" - keywords only (BM25)
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").lower()
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

# Optional instruction put in front of the QUESTION before dense embedding. BGE
# recommends "Represent this sentence for searching relevant passages: ", but on our
# 20-question tea test set it did not help (dense: 19/20 -> 18/20 found in top 5;
# hybrid: 20/20 -> 19/20), so it is off by default.
BGE_QUERY_INSTRUCTION = os.getenv("BGE_QUERY_INSTRUCTION", "")

# Number of chunks retrieved per search phrasing (the question is searched in
# several phrasings), and the max number of merged chunks passed on to extraction.
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "6"))

# How many recent chat messages (questions + answers) the LLM sees when understanding
# a follow-up and writing the answer. 10 = the last 5 question/answer pairs.
HISTORY_MESSAGES = int(os.getenv("HISTORY_MESSAGES", "10"))

# Characters of surrounding text (from the previous/next chunk) added to each
# retrieved chunk, so passages keep their heading and sentence endings.
NEIGHBOUR_CONTEXT_CHARS = int(os.getenv("NEIGHBOUR_CONTEXT_CHARS", "800"))
NEIGHBOUR_CHUNKS = int(os.getenv("NEIGHBOUR_CHUNKS", "2"))  # chunks fetched on each side
