"""Administrator indexing command.

Usage (from the project root, inside the WSL venv):

    # Incremental update (default): index new/changed PDFs, purge removed ones
    python scripts/index_documents.py

    # Full rebuild: wipe the vector store and re-index everything
    python scripts/index_documents.py --rebuild

    # Show what is currently indexed, without changing anything
    python scripts/index_documents.py --status

Only the administrator/developer runs this. End users never do.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src`` importable when run as a plain script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import config  # noqa: E402
from src.rag.indexer import index_stats, reindex  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or update the PDF knowledge-base vector index."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe the existing index and re-index all PDFs from scratch.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current index stats and exit (no changes made).",
    )
    parser.add_argument(
        "--clear-learned",
        action="store_true",
        help="Wipe the shared learned-answers store (does not touch documents).",
    )
    args = parser.parse_args()

    if args.clear_learned:
        from src.rag.learned import clear_learned, learned_count  # noqa: E402

        n = learned_count()
        clear_learned()
        print(f"Cleared the learned-answers store ({n} pairs removed).")
        return 0

    if args.status:
        from src.rag.indexer import list_topics  # noqa: E402

        stats = index_stats()
        print("Index status")
        print("------------")
        print(f"  Store exists : {stats['exists']}")
        print(f"  Topics       : {stats.get('num_topics', 0)}")
        print(f"  Files indexed: {stats['num_files']}")
        print(f"  Total chunks : {stats['num_chunks']}")
        from src.rag.catalog import catalog_stats  # noqa: E402
        from src.rag.learned import learned_count  # noqa: E402

        print(f"  Learned Q&A  : {learned_count()}")
        cat = catalog_stats()
        print(
            f"  Catalog      : {cat['topics']} topics, {cat['variants']} product "
            f"types, {cat['documents']} docs"
        )
        if cat["documents"] != stats["num_files"]:
            print(
                "    ! catalog is out of date "
                f"({cat['documents']} catalogued vs {stats['num_files']} indexed). "
                "Ask Claude to rebuild catalog.json so new documents get product types."
            )
        for t in list_topics():
            print(f"    - {t['topic']}  ({t['num_files']} docs, {t['num_chunks']} chunks)")
        return 0

    print(f"Data folder     : {config.DATA_DIR}")
    print(f"Vector store    : {config.VECTORSTORE_DIR}")
    embed_name = (
        f"{config.GOOGLE_EMBEDDING_MODEL} ({config.GOOGLE_EMBEDDING_DIMS}d, API)"
        if config.EMBEDDING_PROVIDER == "google"
        else f"{config.EMBEDDING_MODEL_NAME} (local)"
    )
    print(f"Embedding model : {embed_name}")
    print(f"Mode            : {'REBUILD' if args.rebuild else 'incremental update'}")
    print("-" * 60)

    try:
        report = reindex(rebuild=args.rebuild)
    except FileNotFoundError as exc:
        # Empty documents/ folder or missing folder.
        print(f"\nERROR: {exc}")
        return 2
    except Exception as exc:  # embedding download failure, disk error, etc.
        print(f"\nERROR while indexing: {exc}")
        return 1

    print("-" * 60)
    print(report.summary())
    print("\nDone. The application will use the updated index on its next query.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
