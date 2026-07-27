"""PDF text extraction using PyMuPDF (fitz).

Extracts text page-by-page and preserves the source filename and 1-based
page number in each page's metadata. Corrupted files and pages without
extractable text are handled gracefully so one bad PDF never aborts a run.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class PageText:
    """One extracted page."""

    source: str  # PDF filename (basename only)
    page: int  # 1-based page number
    text: str


@dataclass
class LoadResult:
    """Outcome of loading a single PDF file."""

    source: str
    pages: list[PageText] = field(default_factory=list)
    error: str | None = None  # set if the file could not be opened at all
    empty: bool = False  # True if the file opened but yielded no text


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes (for change detection)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def list_pdf_files(documents_dir: Path) -> list[Path]:
    """Return all *.pdf files in the documents directory, sorted by name."""
    if not documents_dir.exists():
        return []
    return sorted(
        p for p in documents_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf"
    )


@dataclass
class TopicPDF:
    """A PDF located inside a topic subfolder of the data directory."""

    path: Path
    topic: str  # immediate subfolder name under data/
    rel: str  # path relative to data/ (unique key, e.g. "01 - .../file.pdf")
    source: str  # filename only (for citations)


def list_topic_pdfs(data_dir: Path) -> list[TopicPDF]:
    """Return every PDF under data/, tagged with its topic (subfolder name).

    Each top-level subfolder of ``data_dir`` is a topic. PDFs placed directly in
    data/ (no subfolder) are grouped under the "General" topic. Search is
    recursive so PDFs in deeper subfolders keep their top-level topic.
    """
    if not data_dir.exists():
        return []
    out: list[TopicPDF] = []
    for p in sorted(data_dir.rglob("*")):
        if not (p.is_file() and p.suffix.lower() == ".pdf"):
            continue
        rel_parts = p.relative_to(data_dir).parts
        topic = rel_parts[0] if len(rel_parts) > 1 else "General"
        out.append(
            TopicPDF(
                path=p,
                topic=topic,
                rel=p.relative_to(data_dir).as_posix(),
                source=p.name,
            )
        )
    return out


def _extract_tables(page, max_tables: int = 4) -> str:
    """Render a page's tables as pipe-separated rows.

    These specifications carry their real values in "Guarantee Tables"
    (``Rated voltage ....... kV``). Plain text extraction flattens those into
    an unreadable stream and the row/value pairing is lost, so tables are also
    emitted in a structured form that retrieval and the model can follow.
    """
    try:
        finder = page.find_tables()
    except Exception:
        return ""

    blocks: list[str] = []
    for table in list(getattr(finder, "tables", []) or [])[:max_tables]:
        try:
            rows = table.extract()
        except Exception:
            continue
        lines = []
        for row in rows:
            cells = [
                " ".join(str(c).split()) if c is not None else "" for c in row
            ]
            if not any(cells):
                continue
            lines.append(" | ".join(cells))
        if len(lines) >= 2:  # a header plus at least one data row
            blocks.append("[TABLE]\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def load_pdf(path: Path) -> LoadResult:
    """Extract text from a single PDF, page by page.

    Never raises for a corrupt/unreadable file; instead returns a LoadResult
    with ``error`` set. Pages that have no extractable text are skipped, and
    if the whole document yields nothing, ``empty`` is set to True.
    """
    source = path.name
    result = LoadResult(source=source)

    try:
        doc = fitz.open(path)
    except Exception as exc:  # corrupted / not a real PDF / permission issue
        result.error = f"Could not open '{source}': {exc}"
        return result

    try:
        for index in range(doc.page_count):
            try:
                page = doc.load_page(index)
                text = page.get_text("text").strip()
                tables = _extract_tables(page)
                if tables:
                    text = f"{text}\n\n{tables}" if text else tables
            except Exception as exc:  # a single bad page
                result.error = (
                    (result.error or "")
                    + f"[page {index + 1}: {exc}] "
                )
                continue
            if text:
                result.pages.append(
                    PageText(source=source, page=index + 1, text=text)
                )
    finally:
        doc.close()

    if not result.pages:
        result.empty = True

    return result
