"""Shared factories for the embedding model and the Chroma vector store.

Both the indexing script and the QA pipeline import from here so they use
an identical embedding model and collection configuration.
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from . import config

_log = logging.getLogger("pdf_agent.store")


class GeminiEmbeddings(Embeddings):
    """Gemini embeddings with real batching and free-tier rate limiting.

    The LangChain wrapper issues one HTTP request per text, which exhausts the
    free tier's 100 requests/minute almost immediately (indexing this library
    would take hours). The native SDK accepts a whole list per request, so a
    few hundred chunks cost a single request. Requests are additionally paced
    under the limit and retried when the API reports exhaustion.

    Vectors are truncated to ``dims`` (Matryoshka) and re-normalised, which
    keeps cosine distances meaningful and the published index small.

    Note the free tier's real constraint is 1000 embed *requests per day*, not
    per token, so the batch size should stay large: at 100 texts per request a
    full 7,000-chunk index costs about 70 requests, while a batch of 20 would
    cost 350. Shrinking batches to dodge rate limits is counter-productive.
    """

    def __init__(
        self,
        model: str,
        dims: int,
        api_key: str,
        batch_size: int = 100,
        rpm: int = 45,
    ):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._dims = dims
        self._batch = batch_size
        self._rpm = rpm
        self._calls: deque = deque()

    def _pace(self) -> None:
        now = time.monotonic()
        while self._calls and now - self._calls[0] > 60:
            self._calls.popleft()
        if len(self._calls) >= self._rpm:
            wait = 60 - (now - self._calls[0]) + 0.5
            if wait > 0:
                time.sleep(wait)
            now = time.monotonic()
            while self._calls and now - self._calls[0] > 60:
                self._calls.popleft()
        self._calls.append(time.monotonic())

    def _call(self, fn, tries: int = 6):
        delay = 3.0
        for attempt in range(tries):
            try:
                self._pace()
                return fn()
            except Exception as exc:
                text = str(exc)
                exhausted = "RESOURCE_EXHAUSTED" in text or "429" in text
                if not exhausted or attempt == tries - 1:
                    raise
                # Back off past the current rate-limit window, and slow the
                # pace so the next requests don't hit the same wall.
                _log.warning("embedding rate-limited, waiting %.0fs", delay)
                time.sleep(delay)
                delay = min(delay * 2, 90)
                self._rpm = max(15, self._rpm - 5)
        raise RuntimeError("unreachable")

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types

        resp = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=self._dims),
        )
        out = []
        for emb in resp.embeddings:
            vec = list(emb.values)
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            group = texts[i: i + self._batch]
            out.extend(self._call(lambda g=group: self._embed_batch(g)))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._call(lambda: self._embed_batch([text]))[0]


@lru_cache(maxsize=1)
def get_embeddings():
    """Return the configured embedding model (cached).

    "local"  - sentence-transformers, free and offline, but needs torch and
               roughly 2 GB of RAM, so it is unsuitable for small hosts.
    "google" - Gemini embedding API: no heavy dependencies, which is what makes
               hosted deployment possible.

    Imports are done lazily so the unused provider's dependencies are never
    required at start-up.
    """
    if config.EMBEDDING_PROVIDER == "google":
        return GeminiEmbeddings(
            model=config.GOOGLE_EMBEDDING_MODEL,
            dims=config.GOOGLE_EMBEDDING_DIMS,
            api_key=config.get_google_api_key(),
        )

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        # Normalizing gives cleaner cosine distances for the relevance guard.
        encode_kwargs={"normalize_embeddings": True},
        # Keep the encoding progress bar off: it writes to stderr, which on a
        # hosted runtime can be a closed pipe, and that write raises
        # BrokenPipeError in the middle of answering. (Passing
        # show_progress_bar via encode_kwargs instead would collide with the
        # value LangChain already supplies.)
        show_progress=False,
    )


def get_vectorstore(embeddings=None) -> Chroma:
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


def get_learned_store(embeddings=None) -> Chroma:
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
