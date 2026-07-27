"""Shared learned-answers store.

Every successful answer is saved here (question -> answer + sources, scoped by
topic). When a new question closely matches one that was already answered, the
stored answer is reused for ALL users — giving faster, consistent replies and
reducing Groq calls. This is an automatic loop (no approval gate).
"""
from __future__ import annotations

import hashlib
import json

from . import config
from .store import get_learned_store


def _qa_id(topic: str, question: str) -> str:
    key = f"{topic or ''}::{question.strip().lower()}"
    return "qa::" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def find_learned(question: str, topic: str | None) -> dict | None:
    """Return a stored answer for a near-duplicate question, or None.

    Only reuses when the closest previously-answered question is within
    ``LEARNED_MATCH_DISTANCE`` (i.e. essentially the same question) and in the
    same topic.
    """
    question = (question or "").strip()
    if not question:
        return None
    try:
        vs = get_learned_store()
        where = {"topic": topic} if topic else None
        results = vs.similarity_search_with_score(question, k=1, filter=where)
    except Exception:
        return None
    if not results:
        return None

    doc, dist = results[0]
    if dist > config.LEARNED_MATCH_DISTANCE:
        return None

    meta = doc.metadata or {}

    def _load(key: str) -> list:
        try:
            return json.loads(meta.get(key) or "[]")
        except (json.JSONDecodeError, TypeError):
            return []

    return {
        "answer": meta.get("answer", ""),
        "sources": _load("sources_json"),
        "excerpts": _load("excerpts_json"),
        "matched_question": doc.page_content,
        "distance": dist,
    }


def save_learned(
    question: str,
    answer: str,
    topic: str | None,
    sources: list[dict],
    excerpts: list[dict] | None = None,
) -> None:
    """Store (or update) a learned question->answer pair for all users."""
    question = (question or "").strip()
    answer = (answer or "").strip()
    if not question or not answer:
        return
    try:
        vs = get_learned_store()
        vs.add_texts(
            texts=[question],
            metadatas=[
                {
                    "topic": topic or "",
                    "answer": answer,
                    "sources_json": json.dumps(sources or [], ensure_ascii=False),
                    "excerpts_json": json.dumps(excerpts or [], ensure_ascii=False),
                }
            ],
            ids=[_qa_id(topic or "", question)],  # deterministic -> upsert
        )
    except Exception:
        # Learning must never break answering.
        pass


def clear_learned() -> None:
    """Wipe the shared learned-answers store (admin action)."""
    import shutil

    if config.LEARNED_DIR.exists():
        shutil.rmtree(config.LEARNED_DIR)


def learned_count() -> int:
    """Number of learned Q&A pairs stored."""
    try:
        return get_learned_store()._collection.count()
    except Exception:
        return 0
