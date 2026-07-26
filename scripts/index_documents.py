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
    args = parser.parse_args()

    if args.status:
        stats = index_stats()
        print("Index status")
        print("------------")
        print(f"  Store exists : {stats['exists']}")
        print(f"  Files indexed: {stats['num_files']}")
        print(f"  Total chunks : {stats['num_chunks']}")
        for name in stats["files"]:
            print(f"    - {name}")
        return 0

    print(f"Documents folder: {config.DOCUMENTS_DIR}")
    print(f"Vector store    : {config.VECTORSTORE_DIR}")
    print(f"Embedding model : {config.EMBEDDING_MODEL_NAME}")
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
