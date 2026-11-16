"""
Cross-page chunking.

All pages of a book are joined into one continuous text before splitting, so
chunks (and their 200-character overlap) continue across page breaks instead of
stopping at the end of every page. Each chunk is then mapped back to the page(s)
and line numbers it came from, using the recorded start position of every page.
"""
import re
from bisect import bisect_right
from collections import Counter, defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# A line that is only a page number, e.g. "113", "- 12 -"
_PAGE_NUMBER_LINE = re.compile(r"^\s*[-–—]?\s*(\d{1,4})\s*[-–—]?\s*$")


def _edge_line_indexes(lines):
    """Indexes of the first and last non-empty lines of a page."""
    filled = [i for i, line in enumerate(lines) if line.strip()]
    return sorted({filled[0], filled[-1]}) if filled else []


def _remove_page_number_lines(pages):
    """
    Blank out printed page numbers at the top/bottom of pages so they don't end
    up in the middle of chunks that cross a page break.

    A bare number is treated as a page number only if (printed number - PDF page
    index) matches an offset that is common in this book. Books can have several
    numbering sections, so every frequent offset is accepted. One-off numbers
    (e.g. a table value on the last line) don't match and are kept.

    The line is replaced by a single space rather than deleted, so line numbers
    in citations still match the PDF.
    """
    offsets = Counter()
    for page_no, text in pages:
        lines = text.split("\n")
        for i in _edge_line_indexes(lines):
            match = _PAGE_NUMBER_LINE.match(lines[i])
            if match:
                offsets[int(match.group(1)) - page_no] += 1

    min_count = max(3, len(pages) // 20)
    page_offsets = {offset for offset, count in offsets.items() if count >= min_count}
    if not page_offsets:
        return pages

    cleaned = []
    for page_no, text in pages:
        lines = text.split("\n")
        for i in _edge_line_indexes(lines):
            match = _PAGE_NUMBER_LINE.match(lines[i])
            if match and int(match.group(1)) - page_no in page_offsets:
                lines[i] = " "
        cleaned.append((page_no, "\n".join(lines)))
    return cleaned


def _chunk_book(pages):
    """Chunk one book (list of page Documents in page order)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,  # exact position of each chunk in the book text
    )

    page_texts = _remove_page_number_lines(
        [(doc.metadata["page"], doc.page_content.rstrip("\n")) for doc in pages]
    )

    # 1. Join pages into one text, recording where each page starts.
    #    Pages are joined with a single newline (not a blank line), so the splitter
    #    treats a page break like a normal line break and overlaps across it.
    page_starts, parts, position = [], [], 0
    for _, text in page_texts:
        page_starts.append(position)
        parts.append(text)
        position += len(text) + 1  # +1 for the joining "\n"
    book_text = "\n".join(parts)

    def locate(pos):
        """Character position in book_text -> (index into pages, line number on that page)."""
        k = bisect_right(page_starts, pos) - 1
        line = book_text.count("\n", page_starts[k], pos) + 1
        return k, line

    # 2. Split the whole book; 3. map each chunk back to its pages and lines.
    chunks = []
    for chunk_index, piece in enumerate(splitter.create_documents([book_text])):
        start = piece.metadata["start_index"]
        end = start + len(piece.page_content) - 1  # last character of the chunk

        first_k, start_line = locate(start)
        last_k, end_line = locate(end)
        covered = pages[first_k:last_k + 1]

        metadata = dict(pages[first_k].metadata)
        metadata.update({
            "page": pages[first_k].metadata["page"],  # first page (kept for compatibility)
            "page_end": pages[last_k].metadata["page"],
            "start_line": start_line,  # line on the first page
            "end_line": end_line,      # line on the last page
            "ocr": any(p.metadata.get("ocr") for p in covered),
            # Position of the chunk within its book - lets retrieval fetch the
            # neighbouring chunks (text just before/after) for extra context.
            "chunk_index": chunk_index,
        })
        chunks.append(Document(page_content=piece.page_content, metadata=metadata))

    return chunks


def chunk_documents(documents):
    """
    Chunk page Documents into overlapping chunks that may cross page breaks.
    Pages are grouped per book, so a chunk never mixes two books.
    """
    books = defaultdict(list)
    for doc in documents:
        books[doc.metadata.get("source", "unknown")].append(doc)

    chunks = []
    for pages in books.values():
        pages.sort(key=lambda d: d.metadata["page"])
        chunks.extend(_chunk_book(pages))

    cross_page = sum(1 for c in chunks if c.metadata["page_end"] != c.metadata["page"])
    print(
        f"Created {len(chunks)} chunks from {len(books)} book(s) "
        f"({cross_page} cross a page break)"
    )

    return chunks
