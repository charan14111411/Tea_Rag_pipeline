import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any
from src.graph.workflow import app as graph
from src.graph.nodes import RateLimitExceeded

# Set up logging for FastAPI application
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s")
logger = logging.getLogger("tea_rag.api")

# MCP server (tools: ask_tea_expert, search_tea_books, list_tea_books), served at /mcp
from mcp_server import mcp  # noqa: E402

mcp_http_app = mcp.streamable_http_app(streamable_http_path="/mcp")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # The MCP session manager must run for the app's lifetime; a mounted
    # sub-app's own lifespan is not started by FastAPI, so run it here.
    async with mcp.session_manager.run():
        yield


# Initialize FastAPI application
app = FastAPI(
    title="Tea RAG API",
    description="FastAPI interface for the LangGraph-based Tea RAG Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

# Add the MCP endpoint as a plain route (not a mount) so the URL is exactly /mcp
app.router.routes.extend(mcp_http_app.routes)

# Enable CORS for development testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic schemas for request and response validation
class AskRequest(BaseModel):
    question: str
    thread_id: str = "default_thread"

class EvidenceItem(BaseModel):
    book: str = "Unknown"
    page: Any
    page_end: Any = None  # differs from page when the chunk crosses a page break
    start_line: int
    end_line: int

class AskResponse(BaseModel):
    answer: str
    evidence: List[EvidenceItem]


@app.get("/")
def read_root():
    return FileResponse("static/index.html")


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Endpoint to ask a question to the Tea RAG assistant.
    The request goes through a LangGraph workflow that retrieves relevant chunks,
    grades their relevance, and generates the final answer using LLM.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    logger.info(f"Received question (Thread: {request.thread_id}): {question}")
    try:
        # Invoke the LangGraph workflow asynchronously with conversation memory checkpointer config
        result = await graph.ainvoke(
            {"question": question},
            config={"configurable": {"thread_id": request.thread_id}}
        )
        logger.info("Answer generated successfully.")
        return {
            "answer": result["answer"],
            "evidence": result["evidence"]
        }
    except RateLimitExceeded as e:
        logger.warning(f"Rate limited: {e}")
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error invoking graph: {str(e)}")
