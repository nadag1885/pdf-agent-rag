"""Central configuration for the PDF knowledge-base RAG app.

All paths, model names, and tunable constants live here so the indexing
script, the QA pipeline, and the Streamlit UI share one source of truth.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths (all resolved relative to the project root, so it works on Windows
# regardless of the current working directory).
# ---------------------------------------------------------------------------
# config.py -> src/rag -> src -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DOCUMENTS_DIR: Path = PROJECT_ROOT / "documents"
VECTORSTORE_DIR: Path = PROJECT_ROOT / "vectorstore"
MANIFEST_PATH: Path = VECTORSTORE_DIR / "manifest.json"
ENV_PATH: Path = PROJECT_ROOT / ".env"

# Load environment variables from the project's .env file.
load_dotenv(ENV_PATH)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Free, local embedding model (downloaded once, then cached).
# Multilingual model so both English and Arabic questions retrieve well
# (the documents are bilingual Arabic/English). Overridable via EMBEDDING_MODEL.
EMBEDDING_MODEL_NAME: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Groq chat model used for answer generation.
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ChromaDB collection name.
COLLECTION_NAME: str = "pdf_knowledge_base"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 150

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
# Number of chunks to retrieve for each question. Set higher to improve recall,
# especially for cross-lingual queries (e.g. an Arabic question against English
# text), where the most on-point passage can rank outside the top few.
RETRIEVAL_K: int = 12

# When listing sources, only show chunks whose distance is within this margin
# of the best match, so weakly-related passages aren't cited as sources.
SOURCE_DISPLAY_MARGIN: float = 0.20

# ---------------------------------------------------------------------------
# Context assembly: neighbor expansion + section-aware reranking
# (still the same architecture — Chroma retrieval + local embeddings + Groq).
# ---------------------------------------------------------------------------
# For every retrieved chunk, also pull this many chunks before/after it (by the
# document reading-order `seq`), so specs split across nearby chunks/pages are
# kept together. 1 = previous + next.
NEIGHBOR_RADIUS: int = 1

# Neighbors rank just after their anchor chunk (small distance penalty).
NEIGHBOR_PENALTY: float = 0.05

# Chunks in these sections are preferred (their effective distance is reduced by
# SECTION_BOOST) so equipment specs win over tables of contents / boilerplate.
PREFERRED_SECTIONS: tuple = (
    "Circuit Breakers",
    "Incoming Feeder Panel",
    "General Data",
    "Technical Specifications",
)
SECTION_BOOST: float = 0.15

# Max chunks sent to the LLM after expansion + reranking.
CONTEXT_MAX: int = 12

# Chroma returns a cosine *distance* (0 = identical, 2 = opposite). Chunks
# whose distance exceeds this are treated as irrelevant and dropped. Calibrated
# for the MiniLM model: genuinely relevant chunks score well below ~0.8, while
# unrelated text clusters around ~1.0. Tunable; higher = more permissive. This
# is a cheap first-pass filter — the strict system prompt in qa.py is the
# primary guard against answering from outside the documents.
MAX_RELEVANCE_DISTANCE: float = 0.9

# ---------------------------------------------------------------------------
# Behaviour constants
# ---------------------------------------------------------------------------
# The exact sentence shown when the documents don't contain the answer.
NOT_FOUND_MESSAGE: str = "This information was not found in the available documents."


def get_groq_api_key() -> str:
    """Return the Groq API key or raise a clear, actionable error.

    The key is read from the environment (.env). It is never logged or
    printed anywhere in the codebase.
    """
    key = os.getenv("GROQ_API_KEY", "").strip()
    placeholder = key.upper().startswith("PASTE_") or key == "your_groq_api_key_here"
    if not key or placeholder:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Create a .env file in the project root "
            "containing:\n\n    GROQ_API_KEY=your_real_key\n\n"
            "Get a key at https://console.groq.com (API Keys)."
        )
    return key
