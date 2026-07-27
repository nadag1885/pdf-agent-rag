"""Shared factories for the embedding model and the Chroma vector store.

Both the indexing script and the QA pipeline import from here so they use
an identical embedding model and collection configuration.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from . import config


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the local sentence-transformers embedding model (cached).

    The model is downloaded once to the local Hugging Face cache and then
    reused. Runs fully offline after the first download; no API key needed.
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        # Normalizing gives cleaner cosine distances for the relevance guard.
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(embeddings: HuggingFaceEmbeddings | None = None) -> Chroma:
    """Return a persistent Chroma vector store handle.

    Uses cosine distance so the relevance threshold in config is meaningful.
    """
    if embeddings is None:
        embeddings = get_embeddings()

    config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.VECTORSTORE_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def get_learned_store(embeddings: HuggingFaceEmbeddings | None = None) -> Chroma:
    """Return the persistent Chroma store of learned question->answer pairs.

    Questions are embedded (same model as the documents) so a new question can
    be matched against previously answered ones. Stored in its own directory so
    a document rebuild does not wipe learned answers.
    """
    if embeddings is None:
        embeddings = get_embeddings()

    config.LEARNED_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=config.LEARNED_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(config.LEARNED_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
