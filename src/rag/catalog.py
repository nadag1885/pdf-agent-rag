"""Product catalog: what variants exist inside each topic.

The catalog (``catalog.json``) is built from the documents themselves and tells
the assistant, for every topic, which distinct products it covers (e.g.
transformers -> oil-immersed / dry-type / coupling), how to recognise which one
the user means, and what to ask when it is ambiguous.

It powers three things:
  1. Deterministic clarifying questions with clickable options (no LLM call).
  2. Variant-scoped retrieval - once the product is known, only that product's
     documents are searched, which makes answers far more precise.
  3. Product knowledge injected into the prompt so the model never confuses
     one variant's specs for another's.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from . import config

CATALOG_PATH = config.PROJECT_ROOT / "catalog.json"


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    """Load catalog.json (empty structure if it does not exist yet)."""
    if not CATALOG_PATH.exists():
        return {"topics": []}
    try:
        data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"topics": []}
    except (json.JSONDecodeError, OSError):
        return {"topics": []}


@lru_cache(maxsize=64)
def get_topic(topic: str) -> dict | None:
    """Return the catalog entry for a topic folder name."""
    for entry in load_catalog().get("topics", []):
        if entry.get("topic") == topic:
            return entry
    return None


def topic_label(topic: str) -> str:
    entry = get_topic(topic)
    return (entry or {}).get("label") or topic


def example_questions(topic: str) -> list[str]:
    entry = get_topic(topic)
    return (entry or {}).get("example_questions", [])[:3]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9؀-ۿ ]+", " ", (text or "").lower())


def _contains(hay: str, term: str) -> bool:
    """Whole-word-ish containment of a normalized term in a normalized haystack."""
    term = term.strip()
    if len(term) < 2:
        return False
    return f" {term} " in hay


def _option_terms(opt: dict) -> list[str]:
    terms = {_norm(opt.get("label", "")).strip()}
    for kw in opt.get("match_keywords", []):
        terms.add(_norm(kw).strip())
    return [t for t in terms if len(t) >= 2]


def _discriminating_aliases(entry: dict) -> dict:
    """alias -> variants, keeping only aliases that don't match most variants."""
    variants = entry.get("variants", [])
    counts: dict[str, list] = {}
    for v in variants:
        for alias in v.get("aliases", []):
            a = _norm(alias).strip()
            if len(a) < 3:
                continue
            counts.setdefault(a, []).append(v)
    limit = max(1, len(variants) // 2)
    return {a: vs for a, vs in counts.items() if len(vs) <= limit}


def _attr_tokens(variant: dict) -> list[str]:
    """Distinguishing tokens from a variant's attributes + name."""
    toks = set()
    for a in variant.get("key_attributes", []):
        val = _norm(str(a.get("value", "")))
        for piece in re.split(r"[/,;()]| or | and ", val):
            piece = piece.strip()
            if 2 <= len(piece) <= 40:
                toks.add(piece)
            for num in re.findall(r"\d+(?:\s\d+)?", piece):
                if len(num) >= 2:
                    toks.add(num.strip())
    return [t for t in toks if t]


def match_variants(text: str, topic: str) -> list[dict]:
    """Return the catalog variants the text refers to.

    Two levels: first the product FAMILY (using the curated, mutually-exclusive
    clarify-option keywords), then narrowing inside that family by any
    distinguishing attribute the user mentioned (voltage class, rating, ...).
    """
    entry = get_topic(topic)
    if not entry:
        return []
    hay = " " + _norm(text) + " "
    variants = entry.get("variants", [])
    if not variants:
        return []

    # --- level 1: family, via clarify options -------------------------------
    selected: list[dict] = []
    for opt in entry.get("clarify_options", []):
        terms = _option_terms(opt)
        if not any(_contains(hay, t) for t in terms):
            continue
        for v in variants:
            vname = " " + _norm(v.get("name", "")) + " "
            if any(_contains(vname, t) for t in terms) and v not in selected:
                selected.append(v)

    # --- fallback: distinguishing aliases -----------------------------------
    if not selected:
        for alias, vs in _discriminating_aliases(entry).items():
            if _contains(hay, alias):
                for v in vs:
                    if v not in selected:
                        selected.append(v)

    if not selected:
        return []

    # --- level 2: narrow within the family by mentioned attributes ----------
    if len(selected) > 1:
        scored = []
        for v in selected:
            score = sum(1 for t in _attr_tokens(v) if _contains(hay, t))
            scored.append((score, v))
        best = max(s for s, _ in scored)
        if best > 0:
            selected = [v for s, v in scored if s == best]

    return selected


def match_option(text: str, topic: str) -> dict | None:
    """Return the clarify option whose keywords/label the text matches."""
    entry = get_topic(topic)
    if not entry:
        return None
    hay = " " + _norm(text) + " "
    for opt in entry.get("clarify_options", []):
        cands = [opt.get("label", "")] + list(opt.get("match_keywords", []))
        for c in cands:
            c = _norm(c).strip()
            if c and (f" {c} " in hay or c in hay):
                return opt
    return None


def documents_for_variants(variants: list[dict]) -> list[str]:
    """Filenames covered by the given variants."""
    docs: list[str] = []
    for v in variants:
        for d in v.get("documents", []):
            if d not in docs:
                docs.append(d)
    return docs


def topic_overview(topic: str, variants: list[dict] | None = None) -> str:
    """Compact product knowledge for the prompt (whole topic or one variant)."""
    entry = get_topic(topic)
    if not entry:
        return ""
    chosen = variants if variants else entry.get("variants", [])
    lines = [f"TOPIC: {entry.get('label', topic)} — {entry.get('summary', '')}".strip()]
    if chosen:
        lines.append("Products covered:")
        for v in chosen[:8]:
            attrs = ", ".join(
                f"{a.get('name')}: {a.get('value')}"
                for a in v.get("key_attributes", [])[:4]
                if a.get("name") and a.get("value")
            )
            bit = f"- {v.get('name')}"
            if attrs:
                bit += f" ({attrs})"
            lines.append(bit)
    return "\n".join(lines)


def clarification_for(topic: str) -> dict | None:
    """Return {question, options[]} for a topic, or None if not needed."""
    entry = get_topic(topic)
    if not entry or not entry.get("needs_clarification"):
        return None
    question = entry.get("clarify_question") or ""
    options = entry.get("clarify_options") or []
    if not question or len(options) < 2:
        return None
    return {"question": question, "options": options}


def extra_dimensions(topic: str) -> list[dict]:
    entry = get_topic(topic)
    return (entry or {}).get("extra_dimensions", [])


def catalog_stats() -> dict:
    topics = load_catalog().get("topics", [])
    return {
        "topics": len(topics),
        "variants": sum(len(t.get("variants", [])) for t in topics),
        "documents": sum(
            len({d for v in t.get("variants", []) for d in v.get("documents", [])})
            for t in topics
        ),
    }
