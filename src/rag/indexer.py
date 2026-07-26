"""Index building and incremental updates.

Reads PDFs from ``documents/``, extracts text page-by-page, splits into
overlapping chunks (each tagged with its source filename and page number),
embeds them locally, and stores them in a persistent Chroma collection.

Change detection uses a SHA-256 manifest so re-running only processes files
that were added or changed, and purges chunks for files that were removed —
never creating duplicate chunks.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .pdf_loader import file_sha256, list_pdf_files, load_pdf
from .store import get_embeddings, get_vectorstore


# ---------------------------------------------------------------------------
# Manifest (filename -> content hash) for change detection
# ---------------------------------------------------------------------------
def load_manifest() -> dict:
    if config.MANIFEST_PATH.exists():
        try:
            return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # Corrupt manifest -> treat as empty so a re-index rebuilds it.
            return {"files": {}}
    return {"files": {}}


def save_manifest(manifest: dict) -> None:
    config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    config.MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Result reporting
# ---------------------------------------------------------------------------
@dataclass
class IndexReport:
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    corrupted: list[str] = field(default_factory=list)  # could not open
    no_text: list[str] = field(default_factory=list)  # opened, no extractable text
    chunks_added: int = 0
    chunks_removed: int = 0

    def summary(self) -> str:
        lines = [
            "Indexing report",
            "----------------",
            f"  Added     : {len(self.added)}  {self.added or ''}",
            f"  Changed   : {len(self.changed)}  {self.changed or ''}",
            f"  Removed   : {len(self.removed)}  {self.removed or ''}",
            f"  Unchanged : {len(self.unchanged)}",
            f"  Chunks +  : {self.chunks_added}",
            f"  Chunks -  : {self.chunks_removed}",
        ]
        if self.corrupted:
            lines.append(f"  SKIPPED (corrupted / unreadable): {self.corrupted}")
        if self.no_text:
            lines.append(
                f"  SKIPPED (no extractable text - scanned image PDF?): {self.no_text}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _boilerplate_lines(pages, min_fraction: float = 0.4, min_len: int = 4) -> set[str]:
    """Identify header/footer lines that repeat across most pages.

    This spec repeats a ministry/company header on every page. Left in every
    chunk, that identical text blurs retrieval (especially cross-lingual: the
    Arabic header matches Arabic queries and drowns out the real content).
    """
    from collections import Counter

    counts: Counter[str] = Counter()
    for page in pages:
        for line in {ln.strip() for ln in page.text.splitlines() if len(ln.strip()) >= min_len}:
            counts[line] += 1
    threshold = max(2, int(len(pages) * min_fraction))
    return {line for line, c in counts.items() if c >= threshold}


# Numbered section headings like "6-CIRCUIT BREAKERS", "4-GENERAL DATA".
_NUM_HEAD = re.compile(r"^\s*(\d{1,2})\s*[-.)]\s*([A-Z][A-Z0-9 &/,.\-]{2,})\s*$")
# Panel subsections within "9-TECHNICAL SPECIFICATIONS".
_PANEL_HEAD = re.compile(
    r"(incoming feeder panel|outgoing feeder panel|bus[\s-]?coupler panel|"
    r"bus[\s-]?riser|busbar panel)",
    re.I,
)


def _heading_label(line: str) -> str | None:
    """Return a normalized section label if ``line`` is a section heading."""
    s = line.strip()
    m = _NUM_HEAD.match(s)
    if m:
        return m.group(2).strip().title()
    # Panel subheadings are short title lines; ignore matches inside prose
    # (e.g. the Scope paragraph that lists "incoming ... feeder panel").
    if len(s) <= 40:
        pm = _PANEL_HEAD.search(s)
        if pm:
            return pm.group(1).strip().title()
    return None


def _chunk_documents(path: Path) -> tuple[list[Document], list[str], str, bool, bool]:
    """Return (documents, ids, error, corrupted, no_text) for one PDF.

    Each chunk keeps metadata:
      - ``source`` (filename) and ``page`` (1-based) for citation,
      - ``seq``: global reading-order index, used for neighbor expansion,
      - ``section``: the current section heading (e.g. "Circuit Breakers"),
        used for section-aware reranking.

    Repeated boilerplate (page headers) is kept only on its FIRST occurrence so
    document identity stays searchable while later chunks are de-noised. Text is
    split at section headings so a chunk belongs to a single section.
    """
    result = load_pdf(path)
    if result.error and not result.pages:
        return [], [], result.error, True, False
    if result.empty:
        return [], [], "", False, True

    boiler = _boilerplate_lines(result.pages)
    seen_boiler: set[str] = set()
    splitter = _splitter()
    documents: list[Document] = []
    ids: list[str] = []
    seq = 0
    current_section = ""

    for page in result.pages:
        # 1) drop repeated header/footer (keep first occurrence).
        lines = []
        for ln in page.text.splitlines():
            s = ln.strip()
            if s in boiler:
                if s in seen_boiler:
                    continue
                seen_boiler.add(s)
            lines.append(ln)

        # 2) split the page into blocks that each belong to one section.
        blocks: list[tuple[str, str]] = []
        buf: list[str] = []
        for ln in lines:
            label = _heading_label(ln)
            if label is not None:
                if buf:
                    blocks.append((current_section, "\n".join(buf)))
                    buf = []
                current_section = label
            buf.append(ln)
        if buf:
            blocks.append((current_section, "\n".join(buf)))

        # 3) chunk each block, carrying its section + a global seq.
        for section, text in blocks:
            for chunk in splitter.split_text(text):
                chunk = chunk.strip()
                if not chunk:
                    continue
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": page.source,
                            "page": page.page,
                            "seq": seq,
                            "section": section,
                        },
                    )
                )
                ids.append(f"{page.source}::{seq:05d}")
                seq += 1

    if not documents:
        return [], [], "", False, True
    return documents, ids, "", False, False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def reindex(rebuild: bool = False) -> IndexReport:
    """Build or incrementally update the vector index.

    Args:
        rebuild: If True, wipe the existing collection and manifest and index
            everything from scratch.
    """
    report = IndexReport()

    if not config.DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"documents/ folder not found at {config.DOCUMENTS_DIR}. "
            "Create it and add approved PDF files."
        )

    pdf_paths = list_pdf_files(config.DOCUMENTS_DIR)
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in {config.DOCUMENTS_DIR}. "
            "Add approved PDFs there, then run the indexer again."
        )

    # Fresh rebuild: drop the persisted store + manifest.
    if rebuild:
        import shutil

        if config.VECTORSTORE_DIR.exists():
            shutil.rmtree(config.VECTORSTORE_DIR)

    manifest = {"files": {}} if rebuild else load_manifest()
    manifest_files: dict = manifest.get("files", {})

    # Warm the embedding model once (clear error if the download fails).
    get_embeddings()
    vs = get_vectorstore()

    current: dict[str, str] = {p.name: file_sha256(p) for p in pdf_paths}
    by_name = {p.name: p for p in pdf_paths}

    added = [n for n in current if n not in manifest_files]
    changed = [
        n for n in current
        if n in manifest_files and manifest_files[n].get("hash") != current[n]
    ]
    removed = [n for n in manifest_files if n not in current]
    unchanged = [
        n for n in current
        if n in manifest_files and manifest_files[n].get("hash") == current[n]
    ]

    report.removed = removed
    report.unchanged = unchanged

    # 1) Purge chunks for removed + changed files (prevents duplicates/staleness).
    for name in removed + changed:
        try:
            existing = vs.get(where={"source": name})
            n = len(existing.get("ids", []))
            if n:
                vs.delete(ids=existing["ids"])
                report.chunks_removed += n
        except Exception as exc:  # never abort the whole run on one delete
            print(f"  ! Could not purge old chunks for {name}: {exc}")
        manifest_files.pop(name, None)

    # 2) (Re)index added + changed files.
    for name in added + changed:
        path = by_name[name]
        docs, ids, error, corrupted, no_text = _chunk_documents(path)
        if corrupted:
            report.corrupted.append(name)
            print(f"  ! Skipping corrupted/unreadable PDF: {name} ({error})")
            continue
        if no_text:
            report.no_text.append(name)
            print(f"  ! Skipping (no extractable text): {name}")
            continue

        vs.add_documents(documents=docs, ids=ids)
        report.chunks_added += len(docs)
        manifest_files[name] = {
            "hash": current[name],
            "chunks": len(docs),
        }
        (report.changed if name in changed else report.added).append(name)
        print(f"  + Indexed {name}: {len(docs)} chunks")

    manifest["files"] = manifest_files
    save_manifest(manifest)
    return report


def index_stats() -> dict:
    """Return basic stats about the current index (for the app's health check)."""
    manifest = load_manifest()
    files = manifest.get("files", {})
    total_chunks = sum(f.get("chunks", 0) for f in files.values())
    return {
        "exists": config.VECTORSTORE_DIR.exists() and bool(files),
        "num_files": len(files),
        "num_chunks": total_chunks,
        "files": sorted(files.keys()),
    }
