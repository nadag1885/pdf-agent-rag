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
