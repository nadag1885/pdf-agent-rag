"""Question-answering pipeline: retrieval + Groq generation.

Guarantees the app only answers from the indexed PDFs:
  1. Relevance guard - if no retrieved chunk is close enough, we return the
     standard "not found" message WITHOUT calling the LLM.
  2. Prompt guard - the system prompt forbids outside knowledge and instructs
     the model to reply with the exact "not found" sentence when the context
     does not contain the answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from . import config
from .indexer import index_stats
from .store import get_vectorstore


class NotIndexedError(RuntimeError):
    """Raised when there is no usable vector index yet."""


@dataclass
class Answer:
    text: str
    sources: list[dict] = field(default_factory=list)  # [{"source":..., "page":...}]
    found: bool = True


SYSTEM_PROMPT = (
    "You are a careful assistant answering questions about a fixed set of "
    "internal PDF documents. The documents are bilingual (English and Arabic). "
    "You must follow these rules strictly:\n"
    "1. Use ONLY the information in the CONTEXT below. Never use outside or "
    "general knowledge.\n"
    "2. The facts needed to answer are often spread across several excerpts and "
    "pages. Read ALL excerpts and combine every relevant detail into one answer.\n"
    "3. When the question asks for ratings, specifications, requirements, "
    "characteristics, or operating characteristics, compile ALL supported "
    "details from the CONTEXT into a complete, structured checklist (one item "
    "per line, grouped logically). Preserve specifics exactly: the type and "
    "operating mechanism of each component (e.g. motor-operated vs magnetic "
    "actuator), and every numeric value together with its unit and condition. "
    "Do not omit any relevant detail that appears in the CONTEXT.\n"
    "4. Answer the parts of the question that the CONTEXT supports, even if only "
    "partially. Never invent facts, values, or page numbers, and never fill "
    "gaps with assumptions.\n"
    "5. Only if the CONTEXT contains NOTHING relevant to the question, reply "
    "with EXACTLY this sentence and nothing else:\n"
    f'   "{config.NOT_FOUND_MESSAGE}"\n'
    "6. Be concise and factual, and answer in the same language as the question "
    "when possible."
)


@lru_cache(maxsize=1)
def get_llm():
    """Return a configured Groq chat model. Raises if the key is missing."""
    # Import here so a missing key gives a clean message, not an import error.
    from langchain_groq import ChatGroq

    api_key = config.get_groq_api_key()  # raises RuntimeError with guidance
    return ChatGroq(
        model=config.GROQ_MODEL,
        api_key=api_key,
        temperature=0,
        max_retries=2,
    )


def _retrieve(question: str) -> list[tuple]:
    """Return [(Document, distance)] filtered by the relevance threshold."""
    stats = index_stats()
    if not stats["exists"] or stats["num_chunks"] == 0:
        raise NotIndexedError(
            "The document index is empty or missing. An administrator must run "
            "`python scripts/index_documents.py` after adding PDFs to documents/."
        )

    vs = get_vectorstore()
    results = vs.similarity_search_with_score(question, k=config.RETRIEVAL_K)
    # Chroma cosine distance: smaller = more similar. Keep only close matches.
    return [(doc, dist) for doc, dist in results if dist <= config.MAX_RELEVANCE_DISTANCE]


def _expand_and_rerank(hits: list[tuple]) -> list[Document]:
    """Expand retrieved chunks with neighbors and rerank by section.

    1. Neighbor expansion: for each hit, also include the chunks immediately
       before/after it in reading order (by ``seq``), spanning page boundaries.
       This pulls in specs that were split across adjacent chunks/pages.
    2. Section-aware reranking: chunks in preferred equipment sections (Circuit
       Breakers, Incoming Feeder Panel, General Data, Technical Specifications)
       get a distance bonus so they win over tables-of-contents / boilerplate.

    Returns up to CONTEXT_MAX documents ordered by document reading order.
    """
    vs = get_vectorstore()
    # key = (source, seq) -> [doc, best_distance]
    cand: dict[tuple, list] = {}

    def add(doc: Document, dist: float) -> None:
        key = (doc.metadata.get("source"), doc.metadata.get("seq"))
        if key in cand:
            if dist < cand[key][1]:
                cand[key][1] = dist
        else:
            cand[key] = [doc, dist]

    for doc, dist in hits:
        add(doc, dist)

    # Neighbor expansion via the reading-order `seq`.
    for doc, dist in hits:
        src = doc.metadata.get("source")
        seq = doc.metadata.get("seq")
        if seq is None:
            continue
        wanted = [seq + d for d in range(-config.NEIGHBOR_RADIUS, config.NEIGHBOR_RADIUS + 1) if d != 0]
        try:
            got = vs.get(where={"$and": [{"source": src}, {"seq": {"$in": wanted}}]})
        except Exception:
            continue
        for content, meta in zip(got.get("documents", []), got.get("metadatas", [])):
            add(Document(page_content=content, metadata=meta), dist + config.NEIGHBOR_PENALTY)

    # Rerank: effective distance minus a bonus for preferred sections.
    preferred = tuple(p.lower() for p in config.PREFERRED_SECTIONS)
    scored = []
    for doc, dist in cand.values():
        section = (doc.metadata.get("section") or "").lower()
        boost = config.SECTION_BOOST if any(p in section for p in preferred) else 0.0
        scored.append((dist - boost, doc))
    scored.sort(key=lambda x: x[0])

    top = [doc for _s, doc in scored[: config.CONTEXT_MAX]]
    # Present in reading order so the LLM sees a coherent, ordered context.
    top.sort(key=lambda d: (d.metadata.get("source", ""), d.metadata.get("seq", 0)))
    return top


def _format_context(docs: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        section = doc.metadata.get("section") or "-"
        blocks.append(
            f"[Excerpt {i} | {src} | page {page} | section: {section}]\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


def _sources_from_docs(docs: list[Document], limit: int = 8) -> list[dict]:
    """Unique (source, page) pairs from the final context, ordered by page."""
    seen = set()
    out = []
    for doc in docs:
        key = (doc.metadata.get("source"), doc.metadata.get("page"))
        if key not in seen:
            seen.add(key)
            out.append({"source": key[0], "page": key[1]})
    out.sort(key=lambda s: (s["source"], s["page"] if isinstance(s["page"], int) else 0))
    return out[:limit]


def _friendly_llm_error(exc: Exception) -> str:
    """Map Groq/LLM exceptions to a clear, user-safe message (no key leakage)."""
    raw = str(exc)
    name = type(exc).__name__.lower()
    text = raw.lower()
    status = getattr(exc, "status_code", None)
    if "ratelimit" in name or status == 429 or "rate limit" in text or "rate_limit" in text:
        # Extract the server's suggested wait, e.g. "try again in 3m56.7s".
        m = re.search(r"try again in ([0-9hms.\s]+)", raw, re.I)
        wait = m.group(1).strip().rstrip(".") if m else None
        is_daily = ("per day" in text) or ("tpd" in text) or ("tokens per day" in text)
        if is_daily:
            base = (
                "The Groq free-tier daily usage limit has been reached. It "
                "refreshes gradually through the day"
            )
            if wait:
                return f"{base} — please try again in about {wait}."
            return f"{base}. Please try again later, or upgrade the Groq plan for higher limits."
        if wait:
            return f"The answer service is briefly rate-limited (Groq). Please try again in about {wait}."
        return (
            "The answer service is temporarily rate-limited (Groq). "
            "Please wait a few seconds and try again."
        )
    if "authentication" in name or status in (401, 403) or "api key" in text:
        return (
            "The Groq API key was rejected. An administrator should check "
            "GROQ_API_KEY in the .env file."
        )
    if status and 500 <= int(status) < 600 or "internal" in text:
        return "The answer service (Groq) had a server error. Please try again shortly."
    return f"The answer service is unavailable right now ({type(exc).__name__})."


def answer_question(question: str) -> Answer:
    """Answer a question strictly from the indexed PDFs."""
    question = (question or "").strip()
    if not question:
        return Answer(text="Please enter a question.", found=False)

    hits = _retrieve(question)  # may raise NotIndexedError (handled by caller)
    if not hits:
        # Nothing relevant in the documents -> do not call the LLM at all.
        return Answer(text=config.NOT_FOUND_MESSAGE, sources=[], found=False)

    context_docs = _expand_and_rerank(hits)
    context = _format_context(context_docs)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION: {question}"),
    ]

    try:
        llm = get_llm()
        response = llm.invoke(messages)
    except RuntimeError as exc:  # missing API key (from config.get_groq_api_key)
        return Answer(text=str(exc), sources=[], found=False)
    except Exception as exc:  # rate limit / API / network errors
        return Answer(text=_friendly_llm_error(exc), sources=[], found=False)

    answer_text = (response.content or "").strip()

    # Prompt-guard: model judged the context insufficient.
    if config.NOT_FOUND_MESSAGE.rstrip(".").lower() in answer_text.lower():
        return Answer(text=config.NOT_FOUND_MESSAGE, sources=[], found=False)

    return Answer(text=answer_text, sources=_sources_from_docs(context_docs), found=True)
