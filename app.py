"""User-facing chat application (Streamlit).

Flow: pick a topic -> ask a question -> if the topic covers several distinct
products the assistant asks which one (with clickable options from the product
catalog) -> the answer is drawn ONLY from that product's documents and cites
filename + page. Every answer is added to a shared learned store so it is reused
for all users. There is no upload component: documents are added by an
administrator into data/<topic>/ and indexed with
`python scripts/index_documents.py`.
"""
from __future__ import annotations

import sys

# ChromaDB needs sqlite3 >= 3.35, but some hosts (Streamlit Community Cloud)
# ship an older system sqlite3. Swap in the bundled modern build before
# anything imports chromadb. No-op where pysqlite3 isn't installed.
try:  # pragma: no cover - environment specific
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ModuleNotFoundError:
    pass

from pathlib import Path  # noqa: E402

import streamlit as st  # noqa: E402

# Make ``src`` importable regardless of launch directory.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import catalog, config  # noqa: E402
from src.rag.indexer import index_stats, list_topics, topic_files  # noqa: E402
from src.rag.qa import (  # noqa: E402
    NotIndexedError,
    answer_question,
    classify_intent,
)

st.set_page_config(page_title="Document Q&A", page_icon="📚", layout="centered")

ALL_TYPES = "🔎 Search all types"


def _key_configured() -> bool:
    try:
        config.get_llm_api_key()
        return True
    except RuntimeError:
        return False


def _render_sources(sources: list[dict]) -> None:
    """Compact source summary: one line per document with its page numbers."""
    by_doc: dict[str, set] = {}
    for s in sources:
        by_doc.setdefault(s["source"], set()).add(s["page"])

    lines = []
    for src in sorted(by_doc):
        pages = sorted(p for p in by_doc[src] if isinstance(p, (int, float)))
        pg = ", ".join(str(int(p)) for p in pages)
        label = "p." if len(pages) == 1 else "pp."
        lines.append(f"• {src} — {label} {pg}")
    st.caption("📎 **Sources**  \n" + "  \n".join(lines))


def _render_options(msg: dict, key_prefix: str) -> None:
    """Clickable clarification options (buttons)."""
    options = msg.get("options") or []
    if not options:
        return
    cols = st.columns(min(len(options), 3))
    for i, opt in enumerate(options):
        col = cols[i % len(cols)]
        label = opt.get("label", f"Option {i+1}")
        hint = opt.get("hint", "")
        if col.button(label, key=f"{key_prefix}_opt{i}", use_container_width=True,
                      help=hint or None):
            st.session_state.pending_input = label
            st.rerun()
    if st.button(ALL_TYPES, key=f"{key_prefix}_all", use_container_width=True,
                 help="Search every product in this topic"):
        st.session_state.pending_input = msg.get("for_question", "")
        st.session_state.pending_skip_clarify = True
        st.rerun()


def _welcome_text(topic: str) -> str:
    """Friendly opening message so the chat starts like a conversation."""
    entry = catalog.get_topic(topic) or {}
    label = catalog.topic_label(topic)
    n_docs = len(topic_files(topic))
    lines = [
        f"👋 Hi! I'm your assistant for **{label}** — I've read the "
        f"{n_docs} specification document{'s' if n_docs != 1 else ''} in this topic."
    ]
    variants = [v.get("name", "") for v in entry.get("variants", [])]
    if len(variants) > 1:
        shown = ", ".join(variants[:3])
        more = f" (+{len(variants) - 3} more)" if len(variants) > 3 else ""
        lines.append(f"It covers **{len(variants)} product types** — {shown}{more}.")
        lines.append(
            "Just tell me what you need. If your question could apply to more "
            "than one of them, I'll ask which one you mean."
        )
    else:
        lines.append("Ask me anything about it — I'll cite the file and page.")
    return "\n\n".join(lines)


def render_topic_picker(topics: list[dict]) -> None:
    """First screen: ask the user which topic they want to ask about."""
    st.title("📚 Document Q&A")
    st.subheader("👋 Which topic would you like to ask about?")
    st.caption(
        "Pick the topic your question is about. Answers come only from that "
        "topic's approved documents, with the source file and page shown."
    )

    labeled = {}
    for t in topics:
        entry = catalog.get_topic(t["topic"]) or {}
        n_var = len(entry.get("variants", []))
        extra = f" · {n_var} product types" if n_var > 1 else ""
        labeled[f"{t['topic']}   ({t['num_files']} docs{extra})"] = t["topic"]

    placeholder = "— Select a topic —"
    choice = st.selectbox("Topic", [placeholder] + list(labeled.keys()), index=0)

    if choice != placeholder:
        entry = catalog.get_topic(labeled[choice])
        if entry:
            if entry.get("summary"):
                st.info(entry["summary"])
            variants = entry.get("variants", [])
            if len(variants) > 1:
                with st.expander(f"📋 Product types in this topic ({len(variants)})"):
                    for v in variants:
                        attrs = " · ".join(
                            f"{a['name']}: {a['value']}"
                            for a in v.get("key_attributes", [])[:3]
                            if a.get("name") and a.get("value")
                        )
                        st.markdown(f"**{v['name']}**")
                        if attrs:
                            st.caption(attrs)

    if st.button("Start chatting →", type="primary",
                 disabled=(choice == placeholder)):
        chosen = labeled[choice]
        st.session_state.topic = chosen
        st.session_state.messages = [
            {"role": "assistant", "content": _welcome_text(chosen)}
        ]
        st.rerun()

    st.stop()


def render_sidebar(stats: dict, topic: str) -> None:
    with st.sidebar:
        st.header("📚 Knowledge base")
        st.metric("Topics", stats.get("num_topics", 0))
        st.metric("Documents (total)", stats["num_files"])

        st.divider()
        st.markdown(f"**Current topic**\n\n🗂️ {catalog.topic_label(topic)}")
        if st.button("🔄 Change topic", use_container_width=True):
            for k in ("topic", "messages", "pending_input", "pending_skip_clarify"):
                st.session_state.pop(k, None)
            st.rerun()
        if st.button("🧹 New conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        entry = catalog.get_topic(topic) or {}
        variants = entry.get("variants", [])
        if variants:
            with st.expander(f"📋 Product types ({len(variants)})", expanded=False):
                for v in variants:
                    st.markdown(f"**{v['name']}**")
                    if v.get("summary"):
                        st.caption(v["summary"])
                    for a in v.get("key_attributes", [])[:4]:
                        if a.get("name") and a.get("value"):
                            st.caption(f"• {a['name']}: {a['value']}")

        files = topic_files(topic)
        with st.expander(f"📄 Documents ({len(files)})", expanded=False):
            for name in files:
                st.write(f"• {name}")

        st.divider()
        st.caption(
            "Administrators add PDFs to `data/<topic>/` and run "
            "`python scripts/index_documents.py` to update this index."
        )


def _render_excerpts(excerpts: list[dict], key_prefix: str) -> None:
    """Verbatim source text so the answer can be checked against the PDF."""
    with st.expander(f"🔍 Show the exact text ({len(excerpts)} excerpts)"):
        for i, ex in enumerate(excerpts):
            st.markdown(f"**{ex['source']}** — page {ex['page']}")
            st.markdown(
                f"<div style='border-left:3px solid #888;padding-left:10px;"
                f"color:#888;font-size:0.87em'>{ex['text']}</div>",
                unsafe_allow_html=True,
            )
            if i < len(excerpts) - 1:
                st.write("")


def _render_follow_ups(msg: dict, key_prefix: str) -> None:
    ups = msg.get("follow_ups") or []
    if not ups:
        return
    st.caption("Ask next:")
    cols = st.columns(len(ups))
    for i, q in enumerate(ups):
        if cols[i].button(q, key=f"{key_prefix}_fu{i}", use_container_width=True):
            st.session_state.pending_input = q
            st.rerun()


def _render_assistant(msg: dict, key_prefix: str, is_last: bool) -> None:
    st.markdown(msg["content"])
    if msg.get("variant"):
        st.caption(f"🎯 Answered for: **{msg['variant']}**")
    if msg.get("learned"):
        st.caption("↩︎ Reused from a previously learned answer.")
    if msg.get("unverified"):
        st.warning(
            "Please double-check these figures against the source — I couldn't "
            "match them to the retrieved text: "
            + ", ".join(msg["unverified"])
        )
    if msg.get("sources"):
        _render_sources(msg["sources"])
    if msg.get("excerpts"):
        _render_excerpts(msg["excerpts"], key_prefix)
    if msg.get("options") and is_last:
        _render_options(msg, key_prefix)
    elif is_last and msg.get("follow_ups"):
        _render_follow_ups(msg, key_prefix)


def main() -> None:
    stats = index_stats()

    if not stats["exists"] or stats["num_chunks"] == 0:
        st.title("📚 Document Q&A")
        st.warning(
            "No documents are indexed yet. An administrator needs to add PDFs "
            "to `data/<topic>/` and run `python scripts/index_documents.py`."
        )
        st.stop()

    if not _key_configured():
        st.title("📚 Document Q&A")
        st.error(
            "The answer service is not configured (missing GROQ_API_KEY). "
            "An administrator must set it in the `.env` file."
        )
        st.stop()

    topics = list_topics()
    if not topics:
        st.title("📚 Document Q&A")
        st.warning("No topics found in the index.")
        st.stop()

    if "topic" not in st.session_state:
        render_topic_picker(topics)  # calls st.stop()

    topic = st.session_state.topic
    render_sidebar(stats, topic)

    st.title("📚 Document Q&A")
    st.caption(
        f"Topic: **{catalog.topic_label(topic)}** — answers come only from this "
        "topic's documents."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Starter suggestions on an empty conversation.
    if not st.session_state.messages:
        examples = catalog.example_questions(topic)
        if examples:
            st.caption("💡 Try one of these:")
            cols = st.columns(len(examples))
            for i, ex in enumerate(examples):
                if cols[i].button(ex, key=f"ex{i}", use_container_width=True):
                    st.session_state.pending_input = ex
                    st.rerun()

    last_i = len(st.session_state.messages) - 1
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_assistant(msg, f"m{i}", i == last_i)
            else:
                st.markdown(msg["content"])

    # --- Input (typed, or a clicked option/example) -------------------------
    typed = st.chat_input(f"Ask a question about “{catalog.topic_label(topic)}”…")
    pending = st.session_state.pop("pending_input", None)
    skip_clarify = bool(st.session_state.pop("pending_skip_clarify", False))
    prompt = typed or pending
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    thinking = (
        "Thinking…"
        if classify_intent(prompt, history) == "chat"
        else "Checking the documents…"
    )
    with st.chat_message("assistant"):
        with st.spinner(thinking):
            try:
                answer = answer_question(
                    prompt, topic=topic, history=history, skip_clarify=skip_clarify
                )
            except NotIndexedError as exc:
                st.warning(str(exc))
                st.session_state.messages.append(
                    {"role": "assistant", "content": str(exc)}
                )
                return
            except Exception as exc:  # last-resort safety net
                msg = f"Something went wrong while answering: {type(exc).__name__}."
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                return

        entry = {
            "role": "assistant",
            "content": answer.text,
            "sources": answer.sources if answer.found else [],
            "learned": answer.learned,
            "variant": answer.variant,
            "options": answer.options,
            "excerpts": answer.excerpts if answer.found else [],
            "unverified": answer.unverified,
            "follow_ups": answer.follow_ups if answer.found else [],
            "for_question": prompt,
        }
        _render_assistant(entry, f"m{len(st.session_state.messages)}", True)

    st.session_state.messages.append(entry)


if __name__ == "__main__":
    main()
