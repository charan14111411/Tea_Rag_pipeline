"""
MCP server for the Tea RAG assistant.

Exposes the tea book library to MCP clients (Claude Desktop, Claude Code, IDEs,
other agents) as three tools:

  ask_tea_expert   - full LangGraph pipeline: understand -> retrieve -> extract -> answer,
                     returns a cited answer (uses Groq, like the web UI)
  search_tea_books - hybrid search only: returns the matching book passages with
                     book/page citations, so the client's own model can read them (no Groq)
  list_tea_books   - the books in the library and how many chunks each has

Run it:
  python mcp_server.py                         # stdio (Claude Desktop / Claude Code)
  python mcp_server.py --transport http        # streamable HTTP at http://127.0.0.1:8001/mcp

The FastAPI app (src/api/app.py) also mounts this server at /mcp, so
`uvicorn src.api.app:app --port 8000` serves the UI, /ask and MCP together.

Heavy imports (LangGraph, embeddings, Qdrant) happen on first use, so the MCP
handshake is fast; in standalone mode the vector store is warmed up in the background.
"""
import argparse
import asyncio
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent

logger = logging.getLogger("tea_rag.mcp")

mcp = MCPServer(
    name="tea-books",
    title="Tea Books Library",
    version="1.0.0",
    instructions=(
        "Answers questions about tea cultivation from a library of tea books (soil, drainage, "
        "pests, diseases, clones, plucking, climate, processing). Use ask_tea_expert for a "
        "finished, cited answer. Use search_tea_books to read the source passages yourself. "
        "Pass the same conversation_id to ask_tea_expert for follow-up questions."
    ),
)


# ---------------------------------------------------------------------------
# Output models (returned as structured content plus JSON text)
# ---------------------------------------------------------------------------

class Source(BaseModel):
    book: str
    page: int | str
    page_end: int | str
    start_line: int
    end_line: int


class TeaAnswer(BaseModel):
    answer: str = Field(description="Final answer, ending with a 'Sources:' line when sources were used")
    sources: list[Source] = Field(description="Book passages the answer is based on")
    conversation_id: str = Field(description="Pass this back to ask a follow-up question in the same conversation")


class Passage(BaseModel):
    book: str
    page: int | str
    page_end: int | str
    start_line: int | None = None
    end_line: int | None = None
    text: str


class SearchResult(BaseModel):
    query: str
    passages: list[Passage] = Field(description="Most relevant first")


class Book(BaseModel):
    name: str
    chunks: int


class BookList(BaseModel):
    books: list[Book]


# ---------------------------------------------------------------------------
# Vector store loading
# ---------------------------------------------------------------------------

# Loading the embedding model and Qdrant client must happen once. Local (embedded)
# Qdrant locks its folder, so two threads creating clients at the same time would
# fail - this lock serialises the background warm-up and the first tool call.
_store_lock = threading.Lock()


def _load_store():
    from src.retrieval.vectorstore_manager import load_vectorstore
    with _store_lock:
        return load_vectorstore()


async def _get_store():
    try:
        return await asyncio.to_thread(_load_store)
    except RuntimeError as e:  # collection missing
        raise ToolError(str(e)) from e
    except Exception as e:
        raise ToolError(f"Could not connect to the tea book index ({type(e).__name__}: {e})") from e


def _warm_up():
    try:
        _load_store()
        import src.graph.workflow  # noqa: F401  (LangGraph + Groq clients)
        logger.info("Tea book index and models loaded.")
    except Exception as e:
        logger.warning(f"Warm-up failed ({e}); will retry on first tool call.")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)


@mcp.tool(title="Ask the tea expert", annotations=READ_ONLY)
async def ask_tea_expert(
    question: Annotated[str, Field(description="A question about tea, in any wording; typos are fine")],
    conversation_id: Annotated[
        str | None,
        Field(description="Omit for a new question. To ask a follow-up (e.g. 'what about its control?'), "
                          "pass the conversation_id returned by the previous answer."),
    ] = None,
) -> TeaAnswer:
    """Answer a tea question from the tea book library, with citations.

    Rewrites the question, searches the books in several phrasings, extracts the
    relevant facts and writes one answer that ends with its sources (book and pages).
    If the books do not cover the question, the answer says so.
    """
    question = question.strip()
    if not question:
        raise ToolError("question cannot be empty.")

    await _get_store()
    from src.graph.workflow import app as graph
    from src.graph.nodes import RateLimitExceeded

    conversation_id = conversation_id or f"mcp-{uuid.uuid4().hex[:12]}"
    try:
        result = await graph.ainvoke(
            {"question": question},
            config={"configurable": {"thread_id": conversation_id}},
        )
    except RateLimitExceeded as e:
        raise ToolError(str(e)) from e
    except Exception as e:
        logger.error(f"ask_tea_expert failed: {e}", exc_info=True)
        raise ToolError(f"The tea assistant failed ({type(e).__name__}: {e})") from e

    return TeaAnswer(
        answer=result["answer"],
        sources=[Source(**ev) for ev in result["evidence"]],
        conversation_id=conversation_id,
    )


@mcp.tool(title="Search the tea books", annotations=READ_ONLY)
async def search_tea_books(
    query: Annotated[str, Field(description="What to look for, e.g. 'blister blight control' or 'TV24 clone yield'")],
    top_k: Annotated[int, Field(ge=1, le=20, description="Number of chunks to retrieve")] = 5,
    book: Annotated[
        str | None,
        Field(description="Only search this book (exact name from list_tea_books)"),
    ] = None,
) -> SearchResult:
    """Search the tea books and return the matching passages with book and page citations.

    Uses hybrid search (meaning + exact keywords, so clone codes, chemical and Latin
    names match). Neighbouring text is added around each hit, and hits next to each
    other are merged. No answer is written - read the passages yourself.
    """
    query = query.strip()
    if not query:
        raise ToolError("query cannot be empty.")

    store = await _get_store()
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    from src.graph.nodes import expand_with_neighbours

    search_filter = None
    if book:
        search_filter = Filter(must=[FieldCondition(key="metadata.book", match=MatchValue(value=book))])

    try:
        hits = await store.asimilarity_search(query, k=top_k, filter=search_filter)
    except Exception as e:
        raise ToolError(f"Search failed ({type(e).__name__}: {e})") from e

    if not hits and book:
        raise ToolError(f"No passages found in book '{book}'. Check the exact name with list_tea_books.")

    try:
        docs = await asyncio.to_thread(expand_with_neighbours, hits)
    except Exception as e:
        logger.warning(f"Could not add neighbouring text ({e}); returning chunks as-is.")
        docs = hits

    passages = []
    for doc in docs:
        meta = doc.metadata
        page = meta.get("page", "Unknown")
        passages.append(Passage(
            book=meta.get("book", "Unknown"),
            page=page,
            page_end=meta.get("page_end", page),
            start_line=meta.get("start_line"),
            end_line=meta.get("end_line"),
            text=doc.page_content,
        ))
    return SearchResult(query=query, passages=passages)


@mcp.tool(title="List the tea books", annotations=READ_ONLY)
async def list_tea_books() -> BookList:
    """List the books in the tea library and how many indexed chunks each has."""
    await _get_store()
    from src.config import QDRANT_COLLECTION
    from src.retrieval.vectorstore_manager import get_qdrant_client

    def count_books():
        client = get_qdrant_client()
        try:
            facets = client.facet(QDRANT_COLLECTION, key="metadata.book", limit=1000, exact=True)
            return {hit.value: hit.count for hit in facets.hits}
        except Exception:
            # Facets need a payload index on metadata.book (created for server
            # collections only) - otherwise count by scrolling the payloads.
            counts, offset = {}, None
            while True:
                points, offset = client.scroll(
                    QDRANT_COLLECTION, limit=1000, offset=offset,
                    with_payload=["metadata.book"], with_vectors=False,
                )
                for p in points:
                    name = (p.payload.get("metadata") or {}).get("book", "Unknown")
                    counts[name] = counts.get(name, 0) + 1
                if offset is None:
                    return counts

    try:
        counts = await asyncio.to_thread(count_books)
    except Exception as e:
        raise ToolError(f"Could not read the book list ({type(e).__name__}: {e})") from e
    return BookList(books=[Book(name=n, chunks=c) for n, c in sorted(counts.items())])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def http_security():
    """
    Which Host/Origin headers the HTTP endpoint accepts (protection against DNS
    rebinding). Always localhost; plus the public hostname in MCP_PUBLIC_HOST
    (e.g. your-name.ngrok-free.dev) when the server is exposed through a tunnel.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    origins = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*",
               "https://chatgpt.com", "https://chat.openai.com"]
    for public in filter(None, (h.strip() for h in os.getenv("MCP_PUBLIC_HOST", "").split(","))):
        public = public.removeprefix("https://").removeprefix("http://").rstrip("/")
        hosts.append(public)
        origins.append(f"https://{public}")
    return TransportSecuritySettings(allowed_hosts=hosts, allowed_origins=origins)


def main():
    parser = argparse.ArgumentParser(description="Tea books MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    # ngrok forwards to this port: `ngrok http 8002 ...` must use the same number.
    # 8002, not 8001, because the coffee MCP server (Desktop\Rag\Rag) uses 8001.
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    # MCP clients start this script from any folder; relative paths in the config
    # (local Qdrant folder, OCR cache) are relative to the project.
    os.chdir(PROJECT_ROOT)
    # One log line per HTTP request to Qdrant/Groq/Hugging Face is just noise here
    logging.getLogger("httpx").setLevel(logging.WARNING)

    threading.Thread(target=_warm_up, daemon=True).start()

    if args.transport == "stdio":
        mcp.run("stdio")
    else:
        # MCP_PATH: a long random path (like /mcp-8gXy1...) acts as the shared secret
        # when the endpoint is public with no authentication - keep the full URL private.
        path = "/" + os.getenv("MCP_PATH", "/mcp").strip().lstrip("/")
        security = http_security()
        public = security.allowed_hosts[3:]
        # Printed in full (secret path included) so it can be copied into ChatGPT -
        # don't share screenshots or logs of this line.
        print(f"\nLocal URL:   http://{args.host}:{args.port}{path}", flush=True)
        if public:
            print(f"Public URL:  https://{public[0]}{path}   <- paste this in ChatGPT\n", flush=True)
        else:
            print("Public URL:  (none - set MCP_PUBLIC_HOST in .env; requests through ngrok get 421)\n", flush=True)
        mcp.run("streamable-http", host=args.host, port=args.port,
                streamable_http_path=path, transport_security=security)


if __name__ == "__main__":
    main()
