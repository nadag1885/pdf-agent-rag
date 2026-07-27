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

# Source PDFs live in data/, organized into one subfolder per topic.
DATA_DIR: Path = PROJECT_ROOT / "data"
# Kept for backwards compatibility; not the primary source anymore.
DOCUMENTS_DIR: Path = PROJECT_ROOT / "documents"
VECTORSTORE_DIR: Path = PROJECT_ROOT / "vectorstore"
MANIFEST_PATH: Path = VECTORSTORE_DIR / "manifest.json"
# Shared "learned answers" store — question->answer pairs learned from ALL
# users' interactions and reused for everyone. Kept in a separate directory so
# a document --rebuild does not erase it.
LEARNED_DIR: Path = PROJECT_ROOT / "learned_store"
ENV_PATH: Path = PROJECT_ROOT / ".env"

# Load environment variables from the project's .env file.
load_dotenv(ENV_PATH)


def _from_streamlit_secrets(var: str) -> str:
    """Read a value from Streamlit secrets when deployed.

    A hosted app has no .env file, so settings and keys come from the app's
    secrets instead. Defined before first use and imported lazily, so the CLI
    scripts never require Streamlit.
    """
    try:
        import streamlit as st

        return str(st.secrets.get(var, "")).strip()
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Where embeddings come from:
#   "local"  - sentence-transformers on this machine (free, needs torch, ~2 GB)
#   "google" - Gemini embedding API (no torch, tiny memory: use this for hosting)
EMBEDDING_PROVIDER: str = (
    os.getenv("EMBEDDING_PROVIDER") or _from_streamlit_secrets("EMBEDDING_PROVIDER")
    or "local"
).strip().lower()

# Local embedding model. Multilingual so both English and Arabic questions
# retrieve well (the documents are bilingual).
EMBEDDING_MODEL_NAME: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Gemini embedding model. 3072 dimensions natively; truncated to keep the
# vector store small enough to publish while retaining quality.
GOOGLE_EMBEDDING_MODEL: str = os.getenv(
    "GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001"
)
GOOGLE_EMBEDDING_DIMS: int = int(os.getenv("GOOGLE_EMBEDDING_DIMS", "768"))

# Which service generates the answers: "groq" or "google" (Gemini).
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").strip().lower()

# Groq chat model used for answer generation.
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Gemini model. Use the rolling "-latest" aliases: specific versions such as
# gemini-2.5-flash are no longer served to newly-created API keys.
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

# ChromaDB collection name.
COLLECTION_NAME: str = "pdf_knowledge_base"

# Shared learned-answers collection + how similar a new question must be to an
# already-answered one to reuse the stored answer (cosine distance; smaller =
# stricter/more-identical). Kept tight so only near-duplicate questions reuse.
LEARNED_COLLECTION: str = "learned_qa"
LEARNED_MATCH_DISTANCE: float = 0.20

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

# ---------------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------------
# Weight of the keyword (BM25) ranker relative to the semantic one in the
# rank fusion. 1.0 = equal say.
BM25_WEIGHT: float = 1.0

# Optional cross-encoder reranker (empty string disables it). Downloads once.
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "")

# Chroma returns a cosine *distance* (0 = identical, 2 = opposite). Chunks
# whose distance exceeds this are treated as irrelevant and dropped. Calibrated
# for the MiniLM model: genuinely relevant chunks score well below ~0.8, while
# unrelated text clusters around ~1.0. Tunable; higher = more permissive. This
# is a cheap first-pass filter — the strict system prompt in qa.py is the
# primary guard against answering from outside the documents.
# Calibrated on the full 5,829-chunk corpus: real questions score up to ~0.52,
# unrelated questions bottom out at ~0.56, so 0.58 keeps every genuine question
# while rejecting off-topic ones. (Re-measure if the embedding model changes.)
MAX_RELEVANCE_DISTANCE: float = 0.58

# ---------------------------------------------------------------------------
# Behaviour constants
# ---------------------------------------------------------------------------
# The exact sentence shown when the documents don't contain the answer.
NOT_FOUND_MESSAGE: str = "This information was not found in the available documents."


def _read_key(var: str, where: str) -> str:
    """Read an API key from the environment or raise an actionable error.

    Locally the key comes from .env; on Streamlit Cloud there is no .env, so
    the app's secrets are used instead. Keys are never logged or printed.
    """
    key = os.getenv(var, "").strip() or _from_streamlit_secrets(var)
    placeholder = (
        key.upper().startswith("PASTE_")
        or key in {"your_groq_api_key_here", "your_google_api_key_here"}
    )
    if not key or placeholder:
        raise RuntimeError(
            f"{var} is missing. Add it to the .env file in the project root:\n\n"
            f"    {var}=your_real_key\n\nGet a key at {where}."
        )
    return key


def get_groq_api_key() -> str:
    return _read_key("GROQ_API_KEY", "https://console.groq.com (API Keys)")


def get_google_api_key() -> str:
    return _read_key("GOOGLE_API_KEY", "https://aistudio.google.com/apikey")


def get_llm_api_key() -> str:
    """Return the key for whichever provider is configured."""
    if LLM_PROVIDER == "google":
        return get_google_api_key()
    return get_groq_api_key()


def active_model() -> str:
    """Human-readable name of the model answering questions."""
    return GEMINI_MODEL if LLM_PROVIDER == "google" else GROQ_MODEL
