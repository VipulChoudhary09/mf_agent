#!/usr/bin/env python3
"""Force a full rebuild of the local ChromaDB knowledge-base index.

The LiveKit worker can auto-index during process setup when ``AUTO_INDEX_KB``
is true. Run this command after KB changes or when preparing an index artifact.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(".env.local", override=False)
load_dotenv(".env", override=False)

from rag.chunker import chunk_all_kb_files
from rag.embedder import reset_collection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb-dir", default="KB/pdftomds")
    parser.add_argument("--db-path", default="./chroma_db")
    args = parser.parse_args()

    kb_dir = Path(args.kb_dir)
    if not kb_dir.exists():
        raise SystemExit(f"KB directory not found: {kb_dir}")

    chunks = chunk_all_kb_files(kb_dir)
    if not chunks:
        raise SystemExit(f"No chunks generated from: {kb_dir}")

    print(f"Indexing {len(chunks)} chunks into {args.db_path}")
    collection = reset_collection(args.db_path)

    batch_size = 5
    max_retries = 3
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                collection.add(
                    ids=[chunk["id"] for chunk in batch],
                    documents=[chunk["text"] for chunk in batch],
                    metadatas=[chunk["metadata"] for chunk in batch],
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                message = str(exc)
                if "429" not in message and "RESOURCE_EXHAUSTED" not in message:
                    raise
                delay = 5 * (2**attempt)
                print(f"Rate limited; retrying in {delay}s")
                time.sleep(delay)

        if last_error is not None:
            raise RuntimeError("Indexing failed after retries") from last_error

        completed = min(start + batch_size, len(chunks))
        print(f"Embedded {completed}/{len(chunks)}")
        if completed < len(chunks):
            time.sleep(3)

    print(f"Index ready with {collection.count()} chunks")


if __name__ == "__main__":
    main()
