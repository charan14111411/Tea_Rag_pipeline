"""
LangGraph nodes for the Tea RAG assistant.

Flow:  understand -> retrieve -> extract -> generate
  1. understand: restate the question as a clear search query and pick an answer style
  2. retrieve:   fetch the most similar chunks from Qdrant
  3. extract:    pull only the facts that answer the question out of each chunk
  4. generate:   merge the extracted facts into one final answer, plus its sources
"""
import asyncio
import json
import logging
import re
from collections import defaultdict

from langchain_core.documents import Document

from src.config import RETRIEVAL_K, MAX_CHUNKS, NEIGHBOUR_CONTEXT_CHARS, NEIGHBOUR_CHUNKS, HISTORY_MESSAGES
from src.ingestion.chunker import CHUNK_OVERLAP
from src.retrieval.vectorstore_manager import load_vectorstore, get_chunks_by_index
from src.llm.groq_client import get_llm, get_fast_llm, get_fallback_llm
from src.llm.prompts import UNDERSTAND_PROMPT, EXTRACT_PROMPT, ANSWER_PROMPT

logger = logging.getLogger("tea_rag.nodes")

NO_ANSWER = (
    "I'm sorry, but I couldn't find information about this in the tea books. "
    "Try rephrasing the question or asking about a related topic."
)

# Initialize LLM instances (Groq client initialization is fast)
llm = get_llm()                                     # final answer
# Extraction uses medium reasoning: with "low" it was inconsistent (sometimes
# dropping every relevant passage for the same input); "medium" was reliable.
extract_llm = get_llm(reasoning_effort="medium")    # extracting facts (accuracy matters)
fast_llm = get_fast_llm()                           # understanding the question (simple JSON)
fallback_llm = get_fallback_llm()                   # used when the main model's quota runs out


class RateLimitExceeded(Exception):
    """All models are rate limited; retry_after is the suggested wait in seconds."""

    def __init__(self, retry_after):
        self.retry_after = retry_after
        minutes = max(1, round(retry_after / 60))
        super().__init__(
            "The AI service's usage limit has been reached for now. "
            f"Please try again in about {minutes} minute{'s' if minutes != 1 else ''}."
        )


def page_label(metadata):
    """"Page 14", or "Pages 14-15" for a chunk that crosses a page break."""
    start = metadata.get("page", "Unknown")
    end = metadata.get("page_end", start)
    return f"Page {start}" if end == start else f"Pages {start}-{end}"


def format_history(chat_history, max_messages=HISTORY_MESSAGES):
    """Last `max_messages` messages of the conversation (10 = 5 question/answer pairs)."""
    if not chat_history:
        return "No conversation history."
    recent = chat_history[-max_messages:]
    return "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in recent)


def _page_ranges(pages):
    """[5, 6, 52, 53, 60] -> '5-6, 52-53, 60'"""
    pages = sorted(pages)
    ranges, start, prev = [], pages[0], pages[0]
    for p in pages[1:]:
        if p != prev + 1:
            ranges.append(f"{start}-{prev}" if prev != start else f"{start}")
            start = p
        prev = p
    ranges.append(f"{start}-{prev}" if prev != start else f"{start}")
    return ", ".join(ranges)


def format_sources(documents):
    """Group cited chunks by book and merge their pages: 'Drainage in tea (pp. 6, 52-53); IDM Bulletin (p. 6)'."""
    pages_by_book = {}
    for doc in documents:
        book = doc.metadata.get("book", "Unknown")
        start = doc.metadata.get("page")
        end = doc.metadata.get("page_end", start)
        pages = pages_by_book.setdefault(book, set())
        if isinstance(start, int) and isinstance(end, int):
            pages.update(range(start, end + 1))

    parts = []
    for book, pages in pages_by_book.items():
        if not pages:
            parts.append(book)
            continue
        prefix = "p." if len(pages) == 1 else "pp."
        parts.append(f"{book} ({prefix} {_page_ranges(pages)})")
    return "; ".join(parts)


MAX_RATE_LIMIT_WAIT = 30  # seconds; longer waits (e.g. daily quota) switch model instead


def _suggested_wait(error):
    """
    Groq rate-limit errors say how long to wait, e.g. 'try again in 3.7s',
    'in 450ms' or 'in 11m17.8s' (daily quota). Returns seconds, or None.
    """
    match = re.search(r"try again in ((?:\d+h)?(?:\d+m(?!s))?(?:[\d.]+(?:ms|s))?)", str(error))
    if not match or not match.group(1):
        return None
    total = 0.0
    for value, unit in re.findall(r"([\d.]+)(h|ms|m|s)", match.group(1)):
        total += float(value) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total


def _is_rate_limit(error):
    return "rate_limit" in str(error) or "429" in str(error)


async def invoke_with_retry(prompt, max_attempts=5, delay=2, model=None):
    """
    Call the LLM, retrying on errors:
    - short rate limits (per-minute): wait as long as Groq asks, then retry
    - long rate limits (e.g. daily quota used up): switch to the fallback model
      (separate Groq quota) instead of waiting minutes or hours
    - other errors: back off 2s, 4s, 8s, ...
    """
    model = model or llm
    switched = False
    for attempt in range(max_attempts):
        try:
            return await model.ainvoke(prompt)
        except Exception as e:
            suggested = _suggested_wait(e) if _is_rate_limit(e) else None

            if suggested is not None and suggested > MAX_RATE_LIMIT_WAIT:
                if not switched and model is not fallback_llm:
                    logger.warning(f"{getattr(model, 'model_name', 'model')} rate limit needs a {suggested / 60:.0f} min wait - "
                                   f"switching to fallback model {fallback_llm.model_name}")
                    model, switched = fallback_llm, True
                    continue
                raise RateLimitExceeded(suggested) from e

            if attempt == max_attempts - 1:
                if _is_rate_limit(e):
                    raise RateLimitExceeded(suggested or 60) from e
                raise e
            wait = suggested + 1 if suggested is not None else delay * (2 ** attempt)
            logger.warning(f"LLM call failed (attempt {attempt+1}/{max_attempts}): {str(e)[:120]}... retrying in {wait:.1f}s")
            await asyncio.sleep(wait)


async def understand_node(state):
    """
    Step 1 - Understand the question: rewrite it as a clear, standalone search query
    (resolving follow-ups and fixing typos) and decide whether it needs a short or a
    descriptive answer.
    """
    logger.info("--- UNDERSTAND NODE ---")
    question = state["question"]

    prompt = UNDERSTAND_PROMPT.format(
        chat_history=format_history(state.get("chat_history")),
        question=question,
    )

    search_query, search_keywords, answer_style = question, "", "descriptive"
    try:
        response = await invoke_with_retry(prompt, model=fast_llm)
        match = re.search(r"\{.*\}", response.content, re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
        search_query = (data.get("search_query") or question).strip()
        search_keywords = (data.get("search_keywords") or "").strip()
        if data.get("answer_style") in ("short", "descriptive"):
            answer_style = data["answer_style"]
    except Exception as e:
        logger.warning(f"Could not analyse question ({e}); using it as-is.")

    logger.info(f"Search query: '{search_query}' | keywords: '{search_keywords}' | answer style: {answer_style}")
    return {"search_query": search_query, "search_keywords": search_keywords, "answer_style": answer_style}


async def retrieve_node(state):
    """
    Step 2 - Retrieve: search Qdrant with several phrasings of the question (the
    clear question, a keyword version, and the user's own words), merge the
    results and keep the best chunks. Different wordings find different passages,
    so this avoids missing an answer just because of how the question was phrased.
    """
    logger.info("--- RETRIEVE NODE ---")
    queries = []
    for q in (state.get("search_query"), state.get("search_keywords"), state["question"]):
        if q and q.strip() and q.strip().lower() not in [x.lower() for x in queries]:
            queries.append(q.strip())

    async def search_all():
        # Lazily load the vector store (and embedding model) only when needed
        vectorstore = await asyncio.to_thread(load_vectorstore)
        return await asyncio.gather(*(
            vectorstore.asimilarity_search_with_score(q, k=RETRIEVAL_K) for q in queries
        ))

    # The Qdrant server connection can drop briefly - retry a few times
    for attempt in range(1, 4):
        try:
            results = await search_all()
            break
        except Exception as e:
            if attempt == 3:
                raise
            logger.warning(f"Qdrant search failed ({type(e).__name__}), retry {attempt}/2 in {attempt * 2}s")
            await asyncio.sleep(attempt * 2)

    # Merge with reciprocal rank fusion: score = sum of 1/(60 + rank) over the
    # phrasings. Raw similarity scores from different phrasings aren't comparable
    # (a short vague query scores generic chunks high), but ranks are - so a chunk
    # that ranks near the top for ANY phrasing makes it into the final set.
    fused, docs_by_key = {}, {}
    for hits in results:
        for rank, (doc, _score) in enumerate(hits, start=1):
            key = doc.metadata.get("_id") or doc.page_content
            docs_by_key[key] = doc
            fused[key] = fused.get(key, 0.0) + 1.0 / (60 + rank)

    top_keys = sorted(fused, key=fused.get, reverse=True)[:MAX_CHUNKS]
    documents = [docs_by_key[k] for k in top_keys]
    logger.info(f"Searched with {len(queries)} phrasings -> {len(fused)} unique chunks, keeping top {len(documents)}.")

    # Add the text just before/after each chunk (and merge chunks that are next to
    # each other), so a passage isn't missing a heading or the end of a sentence.
    try:
        documents = await asyncio.to_thread(expand_with_neighbours, documents)
        logger.info(f"Expanded with neighbouring text -> {len(documents)} passages.")
    except Exception as e:
        logger.warning(f"Could not fetch neighbouring chunks ({e}); using chunks as-is.")
    return {"documents": documents}


def _join_overlapping(a, b):
    """Join two consecutive chunks, dropping the text they share (the chunk overlap)."""
    for k in range(min(len(a), len(b), CHUNK_OVERLAP + 50), 15, -1):
        if a.endswith(b[:k]):
            return a + b[k:]
    return a + "\n" + b


def expand_with_neighbours(documents):
    """
    Turn retrieved chunks into passages with surrounding context:
    - chunks of the same book that are next to each other (or one apart) are merged
      into one passage, so the same text isn't sent twice;
    - each passage gets up to NEIGHBOUR_CONTEXT_CHARS of the previous and next chunk.
    Citations (page/lines) still point at the retrieved chunks themselves.
    """
    order = {id(doc): rank for rank, doc in enumerate(documents)}
    by_book, passages = defaultdict(list), []
    for doc in documents:
        if "chunk_index" in doc.metadata:
            by_book[doc.metadata.get("book")].append(doc)
        else:
            passages.append((order[id(doc)], doc))  # older data without positions

    for book, docs in by_book.items():
        docs.sort(key=lambda d: d.metadata["chunk_index"])

        # Group into runs of nearby chunks (gap of at most one chunk)
        runs = [[docs[0]]]
        for doc in docs[1:]:
            if doc.metadata["chunk_index"] - runs[-1][-1].metadata["chunk_index"] <= 2:
                runs[-1].append(doc)
            else:
                runs.append([doc])

        # Fetch up to NEIGHBOUR_CHUNKS chunks on each side: some chunks are tiny
        # (e.g. stray figure text), so the next real text can be two chunks away.
        needed = set()
        for run in runs:
            lo, hi = run[0].metadata["chunk_index"], run[-1].metadata["chunk_index"]
            needed.update(range(lo - NEIGHBOUR_CHUNKS, hi + NEIGHBOUR_CHUNKS + 1))
        texts = get_chunks_by_index(book, sorted(i for i in needed if i >= 0))

        for run in runs:
            lo, hi = run[0].metadata["chunk_index"], run[-1].metadata["chunk_index"]
            own = {d.metadata["chunk_index"]: d.page_content for d in run}

            core = ""
            for i in range(lo, hi + 1):
                part = own.get(i) or texts.get(i)
                if part:
                    core = _join_overlapping(core, part) if core else part

            before = ""
            for i in range(lo - NEIGHBOUR_CHUNKS, lo):
                part = texts.get(i)
                if part:
                    before = _join_overlapping(before, part) if before else part
            full = _join_overlapping(before, core) if before else core
            core_start = len(full) - len(core)
            for i in range(hi + 1, hi + NEIGHBOUR_CHUNKS + 1):
                part = texts.get(i)
                if part:
                    full = _join_overlapping(full, part)

            start = max(0, core_start - NEIGHBOUR_CONTEXT_CHARS)
            end = min(len(full), core_start + len(core) + NEIGHBOUR_CONTEXT_CHARS)
            # Start/end on line boundaries rather than mid-word
            if start > 0:
                nl = full.find("\n", start, core_start)
                start = nl + 1 if nl != -1 else start
            if end < len(full):
                nl = full.rfind("\n", core_start + len(core), end)
                end = nl if nl != -1 else end

            metadata = dict(run[0].metadata)
            metadata.update({
                "page_end": run[-1].metadata.get("page_end", run[-1].metadata.get("page")),
                "end_line": run[-1].metadata.get("end_line"),
            })
            rank = min(order[id(d)] for d in run)
            passages.append((rank, Document(page_content=full[start:end].strip(), metadata=metadata)))

    passages.sort(key=lambda pair: pair[0])  # most relevant first
    return [doc for _, doc in passages]


async def extract_node(state):
    """
    Step 3 - Extract: from each retrieved chunk, pull out only the facts that answer
    the question. Chunks with nothing relevant are dropped (this also acts as the
    relevance filter).
    """
    logger.info("--- EXTRACT NODE ---")
    question = state.get("search_query") or state["question"]
    documents = state["documents"]

    if not documents:
        return {"documents": [], "extracts": []}

    # One call for all passages: the instructions are sent once instead of once per
    # passage, which saves tokens (Groq free tier is 8k tokens/minute) and time.
    passages = "\n\n".join(
        f"[{i}] ({doc.metadata.get('book', 'Unknown')}, {page_label(doc.metadata)})\n\"\"\"\n{doc.page_content}\n\"\"\""
        for i, doc in enumerate(documents, start=1)
    )
    prompt = EXTRACT_PROMPT.format(question=question, passages=passages)

    try:
        response = await invoke_with_retry(prompt, model=extract_llm)
        blocks = _split_numbered_blocks(response.content, len(documents))
    except Exception as e:
        # If extraction fails (e.g. rate limit), keep the passages as-is rather than
        # losing them - the final answer step only uses what is relevant.
        logger.error(f"Extraction failed ({str(e)[:200]}); using the passages as-is.")
        return {"documents": documents, "extracts": [doc.page_content for doc in documents]}

    kept_docs, extracts = [], []
    for i, doc in enumerate(documents, start=1):
        where = f"{doc.metadata.get('book', 'Unknown')}, {page_label(doc.metadata)}"
        facts = blocks.get(i)
        if facts is None:
            # The model skipped this passage's block - keep the raw passage to be safe
            logger.warning(f"  -> Chunk {i} ({where}): no extraction block; using the passage as-is")
            kept_docs.append(doc)
            extracts.append(doc.page_content)
            continue

        if not facts or facts.upper().startswith("NONE"):
            logger.info(f"  -> Chunk {i} ({where}): nothing relevant (dropped)")
            continue

        logger.info(f"  -> Chunk {i} ({where}): extracted {len(facts)} chars")
        kept_docs.append(doc)
        extracts.append(facts)

    logger.info(f"Kept {len(kept_docs)} / {len(documents)} chunks with relevant facts.")
    return {"documents": kept_docs, "extracts": extracts}


def _split_numbered_blocks(text, count):
    """Parse '[1] ... [2] ...' extraction output into {1: '...', 2: '...'}."""
    blocks = {}
    parts = re.split(r"^\s*\[(\d+)\]\s*", text, flags=re.MULTILINE)
    # parts = [preamble, num, body, num, body, ...]
    for num, body in zip(parts[1::2], parts[2::2]):
        n = int(num)
        if 1 <= n <= count:
            blocks[n] = body.strip()
    return blocks


def clean_answer(text):
    """Safety net: strip formatting the UI shouldn't show (tables, HTML, headings)."""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    lines = []
    for line in text.splitlines():
        if re.fullmatch(r"\s*\|?[\s:\-|]+\|?\s*", line) and "-" in line:
            continue  # markdown table separator row
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|") if c.strip()]
            line = "- " + " - ".join(cells)  # flatten a stray table row into a bullet
        line = re.sub(r"^\s*#{1,6}\s*", "", line)  # headings -> plain text
        lines.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


async def generate_node(state):
    """
    Step 4 - Answer: merge the extracted facts into one final answer (short or
    descriptive), then attach the sources that were actually used.
    """
    logger.info("--- GENERATE NODE ---")
    question = state["question"]
    documents = state.get("documents") or []
    extracts = state.get("extracts") or []
    chat_history = state.get("chat_history") or []

    def finish(answer, used_docs):
        new_history = chat_history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        evidence = [{
            "book": doc.metadata.get("book", "Unknown"),
            "page": doc.metadata.get("page", "Unknown"),
            "page_end": doc.metadata.get("page_end", doc.metadata.get("page", "Unknown")),
            "start_line": doc.metadata.get("start_line", 1),
            "end_line": doc.metadata.get("end_line", 1),
        } for doc in used_docs]
        if used_docs:
            answer = f"{answer}\n\nSources: {format_sources(used_docs)}"
        return {"answer": answer, "evidence": evidence, "chat_history": new_history}

    if not extracts:
        return finish(NO_ANSWER, [])

    notes = "\n\n".join(
        f"[{i}] ({doc.metadata.get('book', 'Unknown')}, {page_label(doc.metadata)})\n{facts}"
        for i, (doc, facts) in enumerate(zip(documents, extracts), start=1)
    )

    prompt = ANSWER_PROMPT.format(
        question=question,
        search_query=state.get("search_query") or question,
        answer_style=state.get("answer_style", "descriptive"),
        notes=notes,
        chat_history=format_history(chat_history),
    )
    response = await invoke_with_retry(prompt)
    raw = response.content.strip()

    # The model ends with "USED: 1, 3" - which notes it relied on
    used_match = re.search(r"USED:\s*([\d,\s]*)\s*$", raw, re.IGNORECASE)
    answer = raw[:used_match.start()] if used_match else raw
    answer = clean_answer(answer)

    used = []
    if used_match:
        used = sorted({int(n) for n in re.findall(r"\d+", used_match.group(1)) if 1 <= int(n) <= len(documents)})
    used_docs = [documents[n - 1] for n in used] if used else documents

    if answer.strip().upper().startswith("NOT_FOUND"):
        return finish(NO_ANSWER, [])

    return finish(answer, used_docs)
