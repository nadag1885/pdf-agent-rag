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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from . import catalog, config
from .indexer import index_stats
from .learned import find_learned, save_learned
from .retrieval import hybrid_search_scored
from .store import get_vectorstore


class NotIndexedError(RuntimeError):
    """Raised when there is no usable vector index yet."""


CLARIFY_TAG = "[CLARIFY]"


@dataclass
class Answer:
    text: str
    sources: list[dict] = field(default_factory=list)  # [{"source":..., "page":...}]
    found: bool = True
    clarify: bool = False  # True when the reply is a clarifying question
    learned: bool = False  # True when reused from the shared learned store
    options: list[dict] = field(default_factory=list)  # clickable clarify options
    variant: str = ""  # product variant the answer was scoped to
    excerpts: list[dict] = field(default_factory=list)  # verbatim supporting text
    unverified: list[str] = field(default_factory=list)  # numbers not found in sources
    follow_ups: list[str] = field(default_factory=list)  # suggested next questions


CHAT_PROMPT = (
    "You are a knowledgeable engineering assistant for the EEHC / EDMS "
    "electrical specification library. You are talking with an engineer.\n"
    "\n"
    "This particular message is CONVERSATIONAL — a greeting, a thank-you, a "
    "question about you, or an unfinished thought. It is NOT a request to look "
    "up a specification.\n"
    "\n"
    "Reply like a helpful colleague, not a search engine:\n"
    "- Be warm, natural and brief (1-3 sentences). No bullet lists unless useful.\n"
    "- If they greeted you, greet back and say what you can help with here.\n"
    "- If the message is incomplete (e.g. just \"how\" or \"and?\"), don't guess "
    "and don't search — ask what they'd like to know, and suggest one concrete "
    "example from the topic below.\n"
    "- If they ask what you can do, explain that you answer questions from this "
    "topic's specification documents and always cite the file and page.\n"
    "- NEVER state specification values, ratings or standards from memory. If "
    "they want facts, invite the real question instead of inventing numbers.\n"
    "- Reply in the same language the user used (English or Arabic).\n"
)

SYSTEM_PROMPT = (
    "You are a knowledgeable engineering assistant answering questions from a "
    "fixed set of internal specification PDFs (English and Arabic). Write like a "
    "helpful expert colleague — natural, direct and confident — never like a "
    "search engine dumping text. Lead with the answer, then the detail.\n"
    "Follow these rules strictly.\n"
    "\n"
    "GROUNDING\n"
    "1. Use ONLY the information in the CONTEXT below. Never use outside or "
    "general knowledge, and never invent facts, values, product names, options, "
    "or page numbers.\n"
    "\n"
    "CLARIFY BEFORE ANSWERING (only when truly ambiguous)\n"
    "2. The CONTEXT may cover several DISTINCT products, types, models, or "
    "documents. Ask for clarification ONLY when the question could match TWO OR "
    "MORE of them roughly equally and you genuinely cannot tell which the user "
    "means. Then do NOT guess: ask ONE short clarifying question and list the "
    "distinct options as SHORT, human-readable labels (a few words each) "
    "describing the product/type, derived from the document titles — do NOT "
    "paste raw filenames or codes, and never invent options.\n"
    f"3. Any clarifying reply MUST begin with the exact tag {CLARIFY_TAG} on the "
    "first line, followed by the question and the option labels (one per line).\n"
    "4. PREFER ANSWERING over asking. If the question — together with anything "
    "the user already said earlier in the conversation — points to ONE product "
    "or document, answer directly from it, even if other loosely-related "
    "documents also appear in the CONTEXT. Ask about the same subject at most "
    "once; if you already asked and the user has now specified, answer.\n"
    "\n"
    "ANSWERING\n"
    "5. Read ALL excerpts and combine every relevant detail into one answer. "
    "For ratings, specifications, requirements, characteristics, or operating "
    "characteristics, compile ALL supported details from the CONTEXT into a "
    "complete, structured checklist (one item per line, grouped logically). "
    "Preserve specifics exactly: the type and operating mechanism of each "
    "component (e.g. motor-operated vs magnetic actuator) and every numeric "
    "value with its unit and condition. Do not omit any relevant detail.\n"
    "6. Answer the parts the CONTEXT supports, even if only partial. If the "
    "CONTEXT contains NOTHING relevant to the question, reply with EXACTLY this "
    "sentence and nothing else:\n"
    f'   "{config.NOT_FOUND_MESSAGE}"\n'
    "7. Be concise and factual, and answer in the same language as the question "
    "when possible."
)


@lru_cache(maxsize=1)
def get_llm():
    """Return the configured chat model (Groq or Gemini).

    Raises RuntimeError with guidance if the provider's key is missing.
    """
    if config.LLM_PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.get_google_api_key(),
            temperature=0,
            max_retries=2,
        )

    # Import here so a missing key gives a clean message, not an import error.
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=config.GROQ_MODEL,
        api_key=config.get_groq_api_key(),
        temperature=0,
        max_retries=2,
    )


def _retrieve(
    question: str, topic: str | None = None, documents: list[str] | None = None
) -> list[tuple]:
    """Return [(Document, distance)] filtered by the relevance threshold.

    ``topic`` restricts retrieval to that topic; ``documents`` further restricts
    it to a specific product variant's files.
    """
    stats = index_stats()
    if not stats["exists"] or stats["num_chunks"] == 0:
        raise NotIndexedError(
            "The document index is empty or missing. An administrator must run "
            "`python scripts/index_documents.py` after adding PDFs to data/."
        )

    # Hybrid: keyword (BM25) fused with semantic search, optionally reranked.
    results, best_dist = hybrid_search_scored(
        question, topic=topic, documents=documents, k=config.RETRIEVAL_K
    )
    # Relevance gate: if nothing is semantically close AND nothing matched by
    # keyword, treat it as "not in the documents" rather than answering noise.
    hits = results if best_dist <= config.MAX_RELEVANCE_DISTANCE else []
    # If a variant filter was too narrow to find anything, fall back to the
    # whole topic rather than wrongly reporting "not found".
    if not hits and documents:
        return _retrieve(question, topic=topic, documents=None)
    return hits


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
    # key = (rel, seq) -> [doc, best_distance]  (rel is unique per document)
    cand: dict[tuple, list] = {}

    def add(doc: Document, dist: float) -> None:
        key = (doc.metadata.get("rel"), doc.metadata.get("seq"))
        if key in cand:
            if dist < cand[key][1]:
                cand[key][1] = dist
        else:
            cand[key] = [doc, dist]

    for doc, dist in hits:
        add(doc, dist)

    # Neighbor expansion via the reading-order `seq`, within the same document.
    for doc, dist in hits:
        rel = doc.metadata.get("rel")
        seq = doc.metadata.get("seq")
        if seq is None or rel is None:
            continue
        wanted = [seq + d for d in range(-config.NEIGHBOR_RADIUS, config.NEIGHBOR_RADIUS + 1) if d != 0]
        try:
            got = vs.get(where={"$and": [{"rel": rel}, {"seq": {"$in": wanted}}]})
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


_NUM = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def _verify_numbers(answer: str, docs: list[Document]) -> list[str]:
    """Return numbers stated in the answer that don't appear in the sources.

    A cheap, deterministic hallucination check: every figure in a specification
    answer should be traceable to the retrieved text.
    """
    haystack = " ".join(d.page_content for d in docs)
    hay_nums = set(_NUM.findall(haystack.replace(",", "")))
    missing = []
    for raw in _NUM.findall(answer.replace(",", "")):
        # ignore list numbering and trivial values
        if raw in {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0"}:
            continue
        if raw not in hay_nums and raw not in missing:
            missing.append(raw)
    return missing[:6]


def _excerpts_from_docs(docs: list[Document], limit: int = 5) -> list[dict]:
    """Verbatim snippets so the user can check the answer against the source."""
    out = []
    for d in docs[:limit]:
        text = re.sub(r"\s+", " ", d.page_content).strip()
        out.append(
            {
                "source": d.metadata.get("source", "?"),
                "page": d.metadata.get("page", "?"),
                "text": text[:600] + ("…" if len(text) > 600 else ""),
            }
        )
    return out


def _suggest_follow_ups(topic: str | None, variant: str) -> list[str]:
    """Next-question chips, taken from the catalog (no LLM call)."""
    if not topic:
        return []
    entry = catalog.get_topic(topic) or {}
    out: list[str] = []
    for dim in entry.get("extra_dimensions", [])[:1]:
        q = dim.get("question")
        if q:
            out.append(q)
    for q in catalog.example_questions(topic):
        if q not in out:
            out.append(q)
    if variant:
        out.append(f"What tests are required for the {variant.split(',')[0]}?")
    return out[:3]


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
    service = "Gemini" if config.LLM_PROVIDER == "google" else "Groq"

    # Gemini: a model that is no longer served to new API keys.
    if "not_found" in text or status == 404:
        return (
            f"The configured model '{config.active_model()}' is not available "
            "for this API key. An administrator should set a current model in "
            ".env (e.g. GEMINI_MODEL=gemini-flash-latest)."
        )
    if "resource_exhausted" in text or "quota" in text:
        return (
            f"The {service} usage quota for this key has been reached. "
            "Please try again later, or an administrator can raise the quota "
            "in the provider console."
        )
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


def _message_text(response) -> str:
    """Return a message's text, whatever shape the provider used.

    Groq returns a plain string; Gemini returns a list of content blocks.
    """
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
        return "".join(parts).strip()
    return str(content or "").strip()


def _clean_title(source: str) -> str:
    """Turn a source filename into a readable title (drop code prefix + .pdf)."""
    name = re.sub(r"\.pdf$", "", source, flags=re.I)
    name = re.sub(r"^\s*EDMS\s*[0-9\-–\s]+-\s*", "", name)  # strip "EDMS 08-100-5 - "
    return name.strip() or source


def _search_query(question: str, history: list[dict] | None) -> str:
    """Build the retrieval query, carrying the previous user turn for context.

    A terse follow-up (e.g. answering a clarifying question with "Diesel") only
    retrieves well when combined with what it is answering.
    """
    if history:
        prev_user = next(
            (m["content"] for m in reversed(history) if m.get("role") == "user"),
            None,
        )
        if prev_user:
            return f"{prev_user} {question}".strip()
    return question


def _history_messages(history: list[dict] | None, max_msgs: int = 4) -> list:
    """Convert recent chat turns into LLM messages (bounded, truncated)."""
    if not history:
        return []
    msgs = []
    for m in history[-max_msgs:]:
        content = (m.get("content") or "")[:600]
        if not content:
            continue
        if m.get("role") == "user":
            msgs.append(HumanMessage(content=content))
        else:
            msgs.append(AIMessage(content=content))
    return msgs


# ---------------------------------------------------------------------------
# Intent routing: talk when the user is talking, search when they need a fact.
# ---------------------------------------------------------------------------
_GREETING = re.compile(
    r"^(hi|hello+|hey+|yo|hallo|good\s*(morning|afternoon|evening|day)|"
    r"how are you|salam|as-?salam[ou]?\s*alaik?um|السلام|مرحب|اهلا|أهلا|صباح|مساء)\b",
    re.I,
)
_ACK = re.compile(
    r"^(thanks?|thank you|thx|ty|ok(ay)?|k|great|nice|cool|perfect|good|got it|"
    r"understood|fine|alright|bye|goodbye|شكرا|شكرًا|تمام|حسنا|طيب|ماشي|مع السلامة)"
    r"[\s!.،]*$",
    re.I,
)
_META = re.compile(
    r"(what can you do|who are you|what are you|how (do|does) (you|this|it) work|"
    r"how to use|help me|^help$|what do you know|your capabilit|what is this|"
    r"how can you help|ماذا تستطيع|من انت|من أنت|كيف تعمل|مساعدة)",
    re.I,
)
# Words that mean the user really is asking about the documents.
_DOMAIN_HINT = re.compile(
    r"(spec|standard|rating|rated|voltage|current|kv|kva|amp|power|test|type|"
    r"requirement|dimension|material|insulat|temperature|class|table|clause|"
    r"page|document|capacity|protection|cable|panel|switch|transformer|fuse|"
    r"meter|relay|breaker|earth|مواصف|جهد|تيار|قدرة|اختبار|كابل|محول)",
    re.I,
)
# A bare value like "25 kA" or "630 A" is an answer to a clarifying question.
_VALUE_HINT = re.compile(r"\d+\s*(kv|ka|kva|mva|mw|kw|hz|mm|sq|a|v)\b", re.I)


def classify_intent(question: str, history: list[dict] | None) -> str:
    """Return "chat" or "docs" for a user message."""
    q = (question or "").strip()
    if not q:
        return "chat"
    if _GREETING.search(q) or _ACK.match(q) or _META.search(q):
        return "chat"
    if _VALUE_HINT.search(q):
        return "docs"

    # A short reply right after the assistant asked something is the user
    # answering that question (e.g. "oil-immersed"), not small talk.
    last_assistant = next(
        (m.get("content", "") for m in reversed(history or [])
         if m.get("role") == "assistant"),
        "",
    )
    if last_assistant.rstrip().endswith("?") or "\n-" in last_assistant:
        return "docs"

    words = [w for w in re.split(r"\s+", q) if w]
    # Very short message with no domain word: usually an unfinished thought
    # ("how", "and?", "ok so"). Talk to the user instead of searching.
    if len(words) <= 3 and not _DOMAIN_HINT.search(q) and "?" not in q:
        return "chat"
    if len(words) <= 2 and not _DOMAIN_HINT.search(q):
        return "chat"
    return "docs"


def _chat_reply(question: str, topic: str | None, history: list[dict] | None) -> Answer:
    """Conversational turn: no retrieval, no citations, just a normal reply."""
    context_bits = []
    if topic:
        entry = catalog.get_topic(topic) or {}
        context_bits.append(f'Current topic: "{catalog.topic_label(topic)}"')
        if entry.get("summary"):
            context_bits.append(f"It covers: {entry['summary']}")
        variants = [v.get("name", "") for v in entry.get("variants", [])][:6]
        if variants:
            context_bits.append("Product types here: " + "; ".join(variants))
        examples = catalog.example_questions(topic)
        if examples:
            context_bits.append("Example questions users ask: " + " | ".join(examples))
    context = "\n".join(context_bits)

    messages = [SystemMessage(content=CHAT_PROMPT + ("\n" + context if context else ""))]
    messages += _history_messages(history, max_msgs=6)
    messages.append(HumanMessage(content=question))

    try:
        response = get_llm().invoke(messages)
    except RuntimeError as exc:
        return Answer(text=str(exc), found=False)
    except Exception as exc:
        return Answer(text=_friendly_llm_error(exc), found=False)

    return Answer(text=_message_text(response), sources=[], found=True)


def _active_variants(question: str, topic: str, history: list[dict] | None) -> list[dict]:
    """Which product variant(s) the conversation is about.

    The family usually comes from an earlier turn ("Oil-immersed") while the
    narrowing detail comes from the current one ("the 22/0.4 kV one"), so the
    recent user turns and the current question are matched together.
    """
    if not topic:
        return []
    # Current question alone: if it already pins one product, use it.
    found = catalog.match_variants(question, topic)
    if len(found) == 1:
        return found

    # Otherwise combine the recent user turns with the current question, so a
    # family chosen earlier can be narrowed by a detail given now.
    recent = [
        m.get("content", "")
        for m in (history or [])
        if m.get("role") == "user"
    ][-3:]
    if recent:
        combined = " ".join(recent) + " " + question
        merged = catalog.match_variants(combined, topic)
        if merged:
            return merged
    return found


def _is_general_question(question: str) -> bool:
    """True for topic-wide questions that don't need a specific product."""
    q = question.lower()
    general = (
        "what documents", "which documents", "list the documents", "what topics",
        "what is covered", "what does this cover", "overview", "list all",
        "ما هي المستندات", "نظرة عامة",
    )
    return any(g in q for g in general)


def answer_question(
    question: str,
    topic: str | None = None,
    history: list[dict] | None = None,
    skip_clarify: bool = False,
) -> Answer:
    """Answer a question strictly from the indexed PDFs.

    ``topic`` scopes the search. ``history`` enables conversational
    clarification. When the topic covers several distinct products and the user
    has not said which one, a clarifying question with clickable options is
    returned instead of a guessed answer (unless ``skip_clarify``).
    """
    question = (question or "").strip()
    if not question:
        return Answer(text="Please enter a question.", found=False)

    # --- Talking, not searching? -------------------------------------------
    # Greetings, thanks, "what can you do", or an unfinished thought get a
    # normal conversational reply — the documents are a skill, not a reflex.
    if classify_intent(question, history) == "chat":
        return _chat_reply(question, topic, history)

    # --- Which product is this about? ---------------------------------------
    variants = _active_variants(question, topic or "", history) if topic else []
    variant_name = variants[0]["name"] if len(variants) == 1 else ""

    # Ambiguous -> ask, using the catalog's own options. Deterministic, instant,
    # and it never invents an option that does not exist.
    if topic and not variants and not skip_clarify and not _is_general_question(question):
        clar = catalog.clarification_for(topic)
        if clar:
            return Answer(
                text=clar["question"],
                sources=[],
                found=False,
                clarify=True,
                options=clar["options"],
            )

    # --- Shared learning (scoped by product so variants never mix) ----------
    learn_key = f"{variant_name} :: {question}" if variant_name else question
    reused = find_learned(learn_key, topic)
    if reused:
        return Answer(
            text=reused["answer"], sources=reused["sources"], found=True,
            learned=True, variant=variant_name,
            excerpts=reused.get("excerpts", []),
            follow_ups=_suggest_follow_ups(topic, variant_name),
        )

    variant_docs = catalog.documents_for_variants(variants) if variants else []
    # Search with the user's own words. The variant is already enforced by the
    # document filter; prefixing its long name only dilutes the query and pushes
    # the specific answer out of the retrieved context.
    search_text = _search_query(question, history)
    hits = _retrieve(search_text, topic=topic, documents=variant_docs)  # may raise
    if not hits:
        return Answer(text=config.NOT_FOUND_MESSAGE, sources=[], found=False)

    context_docs = _expand_and_rerank(hits)
    context = _format_context(context_docs)
    available = sorted({_clean_title(d.metadata.get("source", "?")) for d in context_docs})
    doc_list = "\n".join(f"- {name}" for name in available)

    # Product knowledge from the catalog, so the model knows the landscape and
    # does not blend one variant's specifications into another's.
    overview = catalog.topic_overview(topic, variants) if topic else ""
    focus = (
        f"\nThe user is asking specifically about: {variant_name}. Answer only "
        "about that product.\n" if variant_name else ""
    )

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages += _history_messages(history)
    messages.append(
        HumanMessage(
            content=(
                (f"PRODUCT CATALOG:\n{overview}\n{focus}\n" if overview else "")
                + f"CONTEXT:\n{context}\n\n"
                f"DOCUMENTS PRESENT IN THE CONTEXT (use only these names if you "
                f"must clarify):\n{doc_list}\n\n"
                f"QUESTION: {question}"
            )
        )
    )

    try:
        llm = get_llm()
        response = llm.invoke(messages)
    except RuntimeError as exc:  # missing API key (from config.get_groq_api_key)
        return Answer(text=str(exc), sources=[], found=False)
    except Exception as exc:  # rate limit / API / network errors
        return Answer(text=_friendly_llm_error(exc), sources=[], found=False)

    answer_text = _message_text(response)

    # Clarification the model asked for itself (catalog didn't cover it).
    if answer_text.startswith(CLARIFY_TAG):
        clarified = answer_text[len(CLARIFY_TAG):].strip()
        return Answer(text=clarified, sources=[], found=False, clarify=True)

    # Prompt-guard: the model judged the context insufficient. Only treat this
    # as a real "not found" when the reply is essentially just that sentence —
    # if it answered some parts and flagged one as missing, keep the answer.
    target = config.NOT_FOUND_MESSAGE.rstrip(".").lower()
    stripped = answer_text.strip().strip('"').rstrip(".").lower()
    if stripped == target or (
        target in stripped and len(answer_text) <= len(config.NOT_FOUND_MESSAGE) + 60
    ):
        return Answer(text=config.NOT_FOUND_MESSAGE, sources=[], found=False)

    # Learn this answer so all users benefit from it next time.
    answer_sources = _sources_from_docs(context_docs)
    answer_excerpts = _excerpts_from_docs(context_docs)
    save_learned(learn_key, answer_text, topic, answer_sources, answer_excerpts)
    return Answer(
        text=answer_text,
        sources=answer_sources,
        found=True,
        variant=variant_name,
        excerpts=answer_excerpts,
        unverified=_verify_numbers(answer_text, context_docs),
        follow_ups=_suggest_follow_ups(topic, variant_name),
    )
