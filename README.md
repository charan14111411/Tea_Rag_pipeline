# Tea RAG Assistant

LangGraph RAG assistant that answers tea questions from a library of tea book PDFs.

Pipeline: PyMuPDF loader (+ RapidOCR for scanned pages, cached in `vectorstore/ocr_cache/`) -> cross-page recursive chunking (1000 chars, 200 overlap; chunks may span a page break and are cited as "Pages 14-15") -> **hybrid search**: `BAAI/bge-small-en-v1.5` dense embeddings + BM25 sparse vectors in **Qdrant**, merged with reciprocal rank fusion -> LangGraph (understand question -> retrieve with 3 phrasings + neighbouring text -> extract relevant facts -> write the final answer with sources; Groq `openai/gpt-oss-120b`, with `openai/gpt-oss-20b` as fallback) -> FastAPI + web UI.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env`:

```
GROQ_API_KEY=your_key
# GROQ_MODEL=openai/gpt-oss-120b   (optional override)

# Optional - only if using a Qdrant server / Qdrant Cloud instead of local on-disk Qdrant
# QDRANT_URL=https://your-cluster.qdrant.io
# QDRANT_API_KEY=your_qdrant_key
```

## Add tea books and ingest

1. Copy the tea book PDFs into `data/tea_books/` (subfolders are fine).
2. Build the Qdrant collection (rebuilt from scratch on each run):

```bash
python -m src.ingestion.ingest
```

By default Qdrant runs embedded and stores data in `vectorstore/qdrant/` - no server needed.
Local mode allows only one process at a time, so stop the API/CLI before re-ingesting.

## Run

```bash
uvicorn src.api.app:app --port 8000   # web UI at http://localhost:8000
python main.py                         # CLI
```

## MCP server

The assistant is also an MCP server (`mcp_server.py`, official `mcp` Python SDK v2), so MCP clients such as Claude Desktop, Claude Code or IDE agents can use the tea library as tools:

| Tool | What it does |
|---|---|
| `ask_tea_expert(question, conversation_id?)` | Full LangGraph pipeline, returns a cited answer plus a `conversation_id` to pass back for follow-up questions (uses Groq) |
| `search_tea_books(query, top_k=5, book?)` | Hybrid search only, returns the matching passages with book/page citations so the client's own model reads them (no Groq) |
| `list_tea_books()` | Books in the library and their chunk counts (use the exact names for the `book` filter) |

Ways to run it:

```bash
uvicorn src.api.app:app --port 8000          # web UI + /ask + MCP at http://localhost:8000/mcp
python mcp_server.py                          # stdio, for Claude Desktop / Claude Code
python mcp_server.py --transport http         # standalone streamable HTTP at http://127.0.0.1:8002/mcp
```

**Claude Code** (stdio):

```bash
claude mcp add tea-books -- "C:\path\to\Tea_Rag_pipeline\venv\Scripts\python.exe" "C:\path\to\Tea_Rag_pipeline\mcp_server.py"
```

or, against the running web app: `claude mcp add --transport http tea-books http://localhost:8000/mcp`

**Claude Desktop** (`%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "tea-books": {
      "command": "C:\\path\\to\\Tea_Rag_pipeline\\venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\Tea_Rag_pipeline\\mcp_server.py"]
    }
  }
}
```

**ChatGPT via ngrok** (public URL, no auth - the random `MCP_PATH` in `.env` is the secret):

A free ngrok account has one domain, shared with the coffee MCP server (`Desktop\Rag\Rag`, port 8001). `mcp_router.py` runs both behind one tunnel, routing by secret path (routes in `mcp_router.json`; each secret is read from that project's `.env`):

```bash
python mcp_server.py --transport http        # terminal 1: tea MCP, port 8002, at MCP_PATH
# terminal 2 (coffee, in Desktop\Rag\Rag):
#   .venv\Scripts\python.exe mcp_server.py --http --allow-host <your-domain>.ngrok-free.dev
python mcp_router.py                         # terminal 3: router on port 8080
ngrok http 8080                              # terminal 4: gives https://<your-domain>.ngrok-free.dev
```

Set `MCP_PUBLIC_HOST=<your-domain>.ngrok-free.dev` in `.env` (otherwise requests via ngrok get `421 Invalid Host header`). In ChatGPT: Plugins -> + -> Add custom MCP server, URL `https://<your-domain>.ngrok-free.dev<MCP_PATH>`, authentication None. Point ngrok only at the router (8080) - never at 8000, which would publish the web UI, `/ask` and `/docs`.

The script switches to the project folder itself, so the client can start it from anywhere; `.env` is read from the project folder. The HTTP endpoints only accept `localhost`/`127.0.0.1` hosts (DNS-rebinding protection) and have no authentication - keep them on your machine.

## Configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `TEA_DATA_DIR` | `data/tea_books` | Folder of PDFs to ingest |
| `QDRANT_URL` / `QDRANT_API_KEY` | unset | Use a Qdrant server instead of local mode |
| `QDRANT_PATH` | `vectorstore/qdrant` | Local Qdrant storage folder |
| `QDRANT_COLLECTION` | `tea_books` | Collection name |
| `RETRIEVAL_MODE` | `hybrid` | `hybrid` (dense + BM25), `dense` or `sparse` |
| `RETRIEVAL_K` | `5` | Chunks retrieved per search phrasing |
