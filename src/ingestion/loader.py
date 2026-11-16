import json
import re
from pathlib import Path

import fitz  # PyMuPDF
from langchain_community.document_loaders import PyMuPDFLoader

from src.config import OCR_CACHE_DIR

# A page with fewer readable characters than this is treated as "no text"
# (blank, scanned image, or garbage from a broken font encoding) and is OCR'd.
MIN_PAGE_CHARS = 20

# Pages whose text layer scores below this are suspected garbage (e.g. bad OCR of a
# rotated table) and are re-OCR'd; the OCR text is kept only if it scores at least
# OCR_MIN_GAIN higher. The margin keeps legitimate code-heavy pages ("TV24",
# "Stock No.503/9/18") on their original text layer.
QUALITY_THRESHOLD = 0.65
OCR_MIN_GAIN = 0.2

_CLEAN_TOKEN = re.compile(
    r"^[(\[\"']?([A-Za-z]+(['’-][A-Za-z]+)*|[\d.,:/%°+\-–]+)[)\]\"'.,:;!?%]*$"
)

_ocr_engine = None


def book_title_from_path(pdf_path: Path) -> str:
    """Readable book name from a file name, e.g. 'The_Tea-Book_ocr.pdf' -> 'The Tea Book'."""
    title = " ".join(pdf_path.stem.replace("_", " ").replace("-", " ").split())
    return re.sub(r"\s+ocr$", "", title, flags=re.IGNORECASE)


def readable_chars(text: str) -> int:
    """Count characters that are real text (ignores control-char garbage)."""
    return sum(1 for ch in text if ch.isprintable() and not ch.isspace())


def text_quality(text: str) -> float:
    """
    Fraction of tokens that look like clean words or numbers. OCR garbage such as
    '9LS8' or 'V6-101Pr>' (letters and digits jumbled together) scores low.
    """
    tokens = text.split()
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if _CLEAN_TOKEN.match(t)) / len(tokens)


def get_ocr_engine():
    """Lazily create the RapidOCR engine. Returns None if OCR isn't installed."""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr import RapidOCR
        except ImportError:
            return None
        _ocr_engine = RapidOCR()
    return _ocr_engine


def _cache_path(pdf_path: Path) -> Path:
    return Path(OCR_CACHE_DIR) / f"{pdf_path.stem}.json"


def _load_ocr_cache(pdf_path: Path, file_key: str) -> dict:
    path = _cache_path(pdf_path)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("file_key") == file_key:
            return data.get("pages", {})
    return {}


def _save_ocr_cache(pdf_path: Path, file_key: str, pages: dict):
    path = _cache_path(pdf_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"file_key": file_key, "pages": pages}, ensure_ascii=False),
        encoding="utf-8",
    )


def needs_ocr(text: str) -> bool:
    """No usable text layer (scanned page), or a text layer that looks like garbage."""
    return readable_chars(text) < MIN_PAGE_CHARS or text_quality(text) < QUALITY_THRESHOLD


def ocr_pages(path: Path, documents):
    """
    OCR pages that have no usable text layer (scanned pages) or a garbage one,
    and use the OCR text where it is clearly better.
    Results are cached so re-ingestion doesn't redo the slow OCR step.
    """
    pages_to_ocr = [doc for doc in documents if needs_ocr(doc.page_content)]
    if not pages_to_ocr:
        return

    engine = get_ocr_engine()
    if engine is None:
        print(
            f"  WARNING: {len(pages_to_ocr)} page(s) in '{path.name}' have missing or garbled text and "
            "OCR is not installed (pip install rapidocr onnxruntime) - they are used as-is."
        )
        return

    stat = path.stat()
    file_key = f"{stat.st_size}-{int(stat.st_mtime)}"
    cache = _load_ocr_cache(path, file_key)

    pdf = fitz.open(str(path))
    todo = [doc for doc in pages_to_ocr if str(doc.metadata["page"]) not in cache]
    if todo:
        print(f"  Running OCR on {len(todo)} scanned/garbled page(s) of '{path.name}' (cached afterwards)...")

    for n, doc in enumerate(todo, start=1):
        page_no = doc.metadata["page"]
        pixmap = pdf[page_no].get_pixmap(dpi=200)
        result = engine(pixmap.tobytes("png"))
        cache[str(page_no)] = "\n".join(result.txts or [])
        if n % 10 == 0 or n == len(todo):
            print(f"    OCR {n}/{len(todo)} pages done")
            _save_ocr_cache(path, file_key, cache)  # save progress incrementally

    pdf.close()

    for doc in pages_to_ocr:
        ocr_text = cache.get(str(doc.metadata["page"]), "")
        original = doc.page_content

        if readable_chars(original) < MIN_PAGE_CHARS:
            use_ocr = True  # nothing usable in the text layer
        else:
            use_ocr = (
                readable_chars(ocr_text) >= MIN_PAGE_CHARS
                and text_quality(ocr_text) >= text_quality(original) + OCR_MIN_GAIN
            )

        if use_ocr:
            doc.page_content = ocr_text
            doc.metadata["ocr"] = True


def load_pdf(pdf_path: str):
    """
    Load a single PDF and return LangChain Documents (one per page),
    tagged with the book name. Scanned pages are OCR'd.
    """
    path = Path(pdf_path)
    documents = PyMuPDFLoader(str(path)).load()

    book = book_title_from_path(path)
    for doc in documents:
        doc.metadata["book"] = book
        doc.metadata["ocr"] = False

    ocr_pages(path, documents)

    # Collapse runs of spaces (PDF tables are often padded with dozens of spaces).
    # Padding wastes chunk space, blurs embeddings and can stop the chunk overlap from
    # reaching the previous line. Only spaces change - line breaks (and so line
    # numbers in citations) are kept.
    for doc in documents:
        doc.page_content = "\n".join(
            re.sub(r"[ \t ]{2,}", " ", line).strip()
            for line in doc.page_content.split("\n")
        )

    # PyMuPDF numbers pages from 0. From here on use the page number a PDF viewer
    # shows (first page = 1), so citations match what users see when they open the PDF.
    # (OCR above still needs the 0-based index, so this happens after it.)
    for doc in documents:
        doc.metadata["page"] = doc.metadata["page"] + 1

    # Drop pages that still have no usable text (blank pages, failed OCR)
    text_pages = [doc for doc in documents if readable_chars(doc.page_content) >= MIN_PAGE_CHARS]
    ocr_count = sum(1 for doc in text_pages if doc.metadata["ocr"])

    print(
        f"Loaded {len(text_pages)}/{len(documents)} text pages from '{path.name}'"
        + (f" ({ocr_count} via OCR)" if ocr_count else "")
    )

    return text_pages


def load_pdfs(data_dir: str):
    """
    Load every PDF in data_dir (recursively) and return all pages as Documents.
    """
    pdf_paths = sorted(Path(data_dir).rglob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in '{data_dir}'. Put the tea book PDFs there and re-run ingestion."
        )

    documents = []
    for pdf_path in pdf_paths:
        documents.extend(load_pdf(str(pdf_path)))

    print(f"Loaded {len(documents)} pages in total from {len(pdf_paths)} PDF(s)")

    return documents
