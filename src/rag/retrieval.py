"""Hybrid retrieval: keyword (BM25) fused with semantic (vector) search.

Dense embeddings are strong on meaning but weak on exact tokens — and these
specifications are full of exact tokens (``IEC 60076-11``, ``EDMS 08-200-5``,
``25 kA``, ``Dyn11``). BM25 nails those; the vector index handles paraphrase and
cross-language. Results are fused with Reciprocal Rank Fusion so a chunk that
either method ranks highly survives.

An optional cross-encoder reranker re-scores the fused shortlist by true
relevance to the question.
"""
from __future__ import annotations

import re
from functools import lru_cache

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from . import config
from .store import get_vectorstore

# Latin + digits + Arabic letters. Codes like "60076-11" split into parts, and
# the parts are what make them findable.
_TOKEN = re.compile(r"[a-z0-9؀-ۿ]+")

RRF_K = 60  # standard reciprocal-rank-fusion damping


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


@lru_cache(maxsize=1)
def _corpus() -> dict:
    """Load every chunk once, grouped by topic, with a BM25 index per topic."""
    data = get_vectorstore().get()
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []

    by_topic: dict[str, dict] = {}
    for text, meta in zip(docs, metas):
        topic = (meta or {}).get("topic", "")
        slot = by_topic.setdefault(topic, {"texts": [], "metas": []})
        slot["texts"].append(text)
        slot["metas"].append(meta or {})

    for slot in by_topic.values():
        slot["bm25"] = BM25Okapi([_tokenize(t) for t in slot["texts"]])
    return by_topic


def reset_corpus() -> None:
    """Drop the cached BM25 corpus (call after re-indexing)."""
    _corpus.cache_clear()


# A term this rare (roughly <5% of chunks) is distinctive enough that matching
# it is real evidence of relevance; common words are not.
_DISTINCTIVE_IDF = 2.0


def _bm25_hits(
    query: str, topic: str, documents: list[str] | None, n: int
) -> tuple[list[Document], bool]:
    """Return (hits, matched_something_distinctive).

    The flag matters: matching only stopwords ("is", "the") is not evidence that
    the documents contain the answer, but matching "60076" or "dyn11" is.
    """
    slot = _corpus().get(topic)
    if not slot:
        return [], False

    tokens = _tokenize(query)
    bm25 = slot["bm25"]
    # Rare *and* technical-looking: contains a digit (60076, dyn11, 25) or is a
    # long word. This excludes rare-but-empty words like "best" or "who".
    distinctive = {
        t
        for t in tokens
        if bm25.idf.get(t, 0.0) >= _DISTINCTIVE_IDF
        and (any(ch.isdigit() for ch in t) or len(t) >= 6)
    }

    scores = bm25.get_scores(tokens)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out: list[Document] = []
    solid = False
    for i in order:
        if scores[i] <= 0:
            break
        meta = slot["metas"][i]
        if documents and meta.get("source") not in documents:
            continue
        text = slot["texts"][i]
        if distinctive and not solid:
            if distinctive & set(_tokenize(text)):
                solid = True
        out.append(Document(page_content=text, metadata=meta))
        if len(out) >= n:
            break
    return out, solid


def _vector_hits(
    query: str, topic: str | None, documents: list[str] | None, n: int
) -> list[tuple[Document, float]]:
    clauses = []
    if topic:
        clauses.append({"topic": {"$eq": topic}})
    if documents:
        clauses.append({"source": {"$in": list(documents)}})
    where = {"$and": clauses} if len(clauses) > 1 else (clauses[0] if clauses else None)
    return get_vectorstore().similarity_search_with_score(query, k=n, filter=where)


def _key(doc: Document) -> tuple:
    return (doc.metadata.get("rel"), doc.metadata.get("seq"))


def hybrid_search_scored(
    query: str,
    topic: str | None = None,
    documents: list[str] | None = None,
    k: int | None = None,
) -> tuple[list[tuple[Document, float]], float]:
    """Return ([(Document, pseudo_distance)], best_vector_distance).

    ``pseudo_distance`` is a cosine-distance-like value derived from the fused
    rank, so downstream reranking/threshold logic keeps working unchanged.
    ``best_vector_distance`` is the true semantic distance of the closest chunk
    and is what the "is anything relevant at all?" guard uses.
    """
    k = k or config.RETRIEVAL_K
    pool = max(k * 3, 24)

    vec = _vector_hits(query, topic, documents, pool)
    best_dist = min((d for _doc, d in vec), default=99.0)
    lex, lex_solid = (
        _bm25_hits(query, topic, documents, pool) if topic else ([], False)
    )

    # Reciprocal rank fusion.
    scores: dict[tuple, float] = {}
    keep: dict[tuple, Document] = {}
    for rank, (doc, _d) in enumerate(vec):
        kk = _key(doc)
        scores[kk] = scores.get(kk, 0.0) + 1.0 / (RRF_K + rank + 1)
        keep.setdefault(kk, doc)
    for rank, doc in enumerate(lex):
        kk = _key(doc)
        scores[kk] = scores.get(kk, 0.0) + config.BM25_WEIGHT / (RRF_K + rank + 1)
        keep.setdefault(kk, doc)

    ranked = sorted(scores, key=lambda kk: scores[kk], reverse=True)[:k]
    ranked = rerank(query, [keep[kk] for kk in ranked])

    # Map rank -> a distance-like value in the same range as cosine distances.
    n = max(len(ranked), 1)
    out = [(doc, 0.20 + 0.55 * (i / n)) for i, doc in enumerate(ranked)]
    # A *distinctive* keyword match counts as relevance even when the vector
    # distance is loose (exact codes are the classic case). Matching only
    # common words does not, so unrelated questions still return nothing.
    if lex_solid:
        best_dist = min(best_dist, config.MAX_RELEVANCE_DISTANCE)
    return out, best_dist


def hybrid_search(
    query: str,
    topic: str | None = None,
    documents: list[str] | None = None,
    k: int | None = None,
) -> list[Document]:
    docs, _best = hybrid_search_scored(query, topic, documents, k)
    return [d for d, _s in docs]


# ---------------------------------------------------------------------------
# Optional cross-encoder reranking
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _reranker():
    name = config.RERANKER_MODEL
    if not name:
        return None
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(name)
    except Exception:
        return None


def rerank(query: str, docs: list[Document]) -> list[Document]:
    """Re-score a shortlist with a cross-encoder (no-op if disabled)."""
    if len(docs) < 2:
        return docs
    model = _reranker()
    if model is None:
        return docs
    try:
        pairs = [(query, d.page_content[:1200]) for d in docs]
        scores = model.predict(pairs)
    except Exception:
        return docs
    order = sorted(range(len(docs)), key=lambda i: float(scores[i]), reverse=True)
    return [docs[i] for i in order]
