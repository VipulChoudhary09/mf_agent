#!/usr/bin/env python3
"""Index all KB markdown files into ChromaDB.

Usage:
    python scripts/index_kb.py                     # default paths
    python scripts/index_kb.py --kb-dir KB/pdftomds --db-path ./chroma_db

This is idempotent: re-running deletes and recreates the collection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from rag.chunker import chunk_all_kb_files
from rag.embedder import reset_collection, get_collection


def main() -> None:
    parser = argparse.ArgumentParser(description="Index KB docs into ChromaDB")
    parser.add_argument(
        "--kb-dir",
        default="KB/pdftomds",
        help="Path to markdown KB directory (default: KB/pdftomds)",
    )
    parser.add_argument(
        "--db-path",
        default="./chroma_db",
        help="ChromaDB persistent storage path (default: ./chroma_db)",
    )
    args = parser.parse_args()

    kb_dir = Path(args.kb_dir)
    if not kb_dir.exists():
        print(f"❌ KB directory not found: {kb_dir}")
        sys.exit(1)

    # Step 1: Chunk all documents
    print(f"📂 Chunking documents from {kb_dir}...")
    chunks = chunk_all_kb_files(kb_dir)
    print(f"   → {len(chunks)} chunks created")

    # Print per-file breakdown
    file_counts: dict[str, int] = {}
    for chunk in chunks:
        fname = chunk["metadata"]["source_file"]
        file_counts[fname] = file_counts.get(fname, 0) + 1

    for fname, count in sorted(file_counts.items()):
        print(f"   {fname:20s} → {count} chunks")

    # Step 2: Reset and populate collection
    print(f"\n🗄️  Indexing into ChromaDB at {args.db_path}...")
    collection = reset_collection(args.db_path)

    # ChromaDB batch add — small batches + retry for Gemini API rate limits
    import time

    BATCH_SIZE = 5
    MAX_RETRIES = 3
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_ids = [c["id"] for c in batch]
        batch_docs = [c["text"] for c in batch]
        batch_metas = [c["metadata"] for c in batch]

        for attempt in range(MAX_RETRIES):
            try:
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas,  # type: ignore[arg-type]
                )
                break
            except (ValueError, Exception) as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 5 * (2 ** attempt)  # 5s, 10s, 20s
                    print(f"   ⏳ Rate limited, waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(wait)
                else:
                    raise

        embedded = min(i + BATCH_SIZE, len(chunks))
        print(f"   Embedded {embedded}/{len(chunks)} chunks...")
        if embedded < len(chunks):
            time.sleep(3)  # rate limit cooldown between batches

    print(f"   ✅ {collection.count()} chunks embedded and stored")

    # Step 3: Quick sanity check
    print("\n🔍 Sanity check — querying 'exit load' for each scheme...")
    for scheme_id in ["Dynamic_Asset_Allocation", "Large_Cap", "Flexi_Cap"]:
        results = collection.query(
            query_texts=["What is the exit load?"],
            n_results=1,
            where={"scheme_id": scheme_id},
        )
        if results and results["documents"] and results["documents"][0]:
            doc = results["documents"][0][0]
            dist = results["distances"][0][0]  # type: ignore[index]
            preview = doc[:120].replace("\n", " ")
            print(f"   {scheme_id:30s} (dist={dist:.3f}): {preview}...")
        else:
            print(f"   {scheme_id:30s}: ❌ no results")

    print("\n✅ Indexing complete!")


if __name__ == "__main__":
    main()
