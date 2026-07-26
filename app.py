"""User-facing chat application (Streamlit).

Users can ONLY ask questions about the pre-indexed PDF documents. There is no
upload component: documents are added by an administrator and indexed with
`python scripts/index_documents.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make ``src`` importable regardless of launch directory.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import config  # noqa: E402
from src.rag.indexer import index_stats  # noqa: E402
from src.rag.qa import NotIndexedError, answer_question  # noqa: E402

st.set_page_config(page_title="Document Q&A", page_icon="📚", layout="centered")


def _key_configured() -> bool:
    try:
        config.get_groq_api_key()
        return True
    except RuntimeError:
        return False


def render_sidebar(stats: dict) -> None:
    with st.sidebar:
        st.header("📚 Knowledge base")
        st.caption(
            "Ask questions about the documents below. Answers come only from "
            "these files — nothing is uploaded by users."
        )
        st.metric("Documents indexed", stats["num_files"])
        st.metric("Text chunks", stats["num_chunks"])
        if stats["files"]:
            with st.expander("Available documents", expanded=False):
                for name in stats["files"]:
                    st.write(f"• {name}")
        st.divider()
        st.caption(
            "Administrators add PDFs to the `documents/` folder and run "
            "`python scripts/index_documents.py` to update this index."
        )


def main() -> None:
    st.title("📚 Document Q&A")
    st.caption("Answers are grounded only in the approved PDF documents.")

    stats = index_stats()
    render_sidebar(stats)

    # --- Health checks ------------------------------------------------------
    if not stats["exists"] or stats["num_chunks"] == 0:
        st.warning(
            "No documents are indexed yet. An administrator needs to add PDFs "
            "to the `documents/` folder and run "
            "`python scripts/index_documents.py`."
        )
        st.stop()

    if not _key_configured():
        st.error(
            "The answer service is not configured (missing GROQ_API_KEY). "
            "An administrator must set it in the `.env` file."
        )
        st.stop()

    # --- Chat history -------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"])

    # --- Chat input ---------------------------------------------------------
    prompt = st.chat_input("Ask a question about the documents…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents…"):
            try:
                answer = answer_question(prompt)
            except NotIndexedError as exc:
                st.warning(str(exc))
                st.session_state.messages.append(
                    {"role": "assistant", "content": str(exc)}
                )
                return
            except Exception as exc:  # last-resort safety net
                msg = f"Something went wrong while answering: {type(exc).__name__}."
                st.error(msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": msg}
                )
                return

        st.markdown(answer.text)
        if answer.found and answer.sources:
            _render_sources(answer.sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer.text,
            "sources": answer.sources if answer.found else [],
        }
    )


def _render_sources(sources: list[dict]) -> None:
    with st.expander(f"📎 Sources ({len(sources)})", expanded=True):
        for s in sources:
            st.markdown(f"- **{s['source']}** — page {s['page']}")


if __name__ == "__main__":
    main()
