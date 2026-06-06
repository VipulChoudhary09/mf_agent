"""Retrieval interface for the PPFAS KB vector store.

Provides filtered, confidence-gated retrieval of KB chunks.
All queries are hard-filtered by ``scheme_id`` to prevent
cross-fund contamination — critical for financial accuracy.
"""

from __future__ import annotations

import logging
from typing import Any

from rag.embedder import get_collection

logger = logging.getLogger("rag.retriever")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_N_RESULTS = 3
CONFIDENCE_THRESHOLD = 0.6  # cosine distance; lower = more similar


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def retrieve(
    question: str,
    scheme_id: str,
    n_results: int = DEFAULT_N_RESULTS,
    db_path: str = "./chroma_db",
) -> list[str]:
    """Retrieve relevant KB chunks for a user question.

    Args:
        question: The user's natural language question (from STT).
        scheme_id: Fund identifier to filter by (e.g. ``Flexi_Cap``).
            Ensures we never return chunks from a different fund.
        n_results: Maximum number of chunks to return.
        db_path: Path to ChromaDB persistent storage.

    Returns:
        List of chunk text strings, ordered by relevance.
        Empty list if no chunks pass the confidence threshold.
    """
    collection = get_collection(db_path)

    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        where={"scheme_id": scheme_id},
    )

    if not results or not results.get("documents"):
        logger.warning(f"No results for query='{question}' scheme={scheme_id}")
        return []

    documents: list[str] = results["documents"][0]
    distances: list[float] = results["distances"][0]  # type: ignore[index]
    metadatas: list[dict[str, Any]] = results["metadatas"][0]  # type: ignore[index]

    # Filter by confidence threshold
    filtered: list[str] = []
    for doc, dist, meta in zip(documents, distances, metadatas):
        if dist > CONFIDENCE_THRESHOLD:
            logger.debug(
                f"Rejected chunk (dist={dist:.3f} > {CONFIDENCE_THRESHOLD}): "
                f"{meta.get('section_title', '?')}"
            )
            continue

        # Double-check scheme_id (belt-and-suspenders for financial data)
        if meta.get("scheme_id") != scheme_id:
            logger.error(
                f"Cross-fund contamination detected! "
                f"Expected {scheme_id}, got {meta.get('scheme_id')}"
            )
            continue

        filtered.append(doc)
        logger.info(
            f"Retrieved chunk (dist={dist:.3f}): "
            f"{meta.get('section_title', '?')} from {meta.get('source_file', '?')}"
        )

    return filtered


def retrieve_formatted(
    question: str,
    scheme_id: str,
    n_results: int = DEFAULT_N_RESULTS,
    db_path: str = "./chroma_db",
) -> str:
    """Retrieve and format chunks into a prompt-ready context block.

    Args:
        question: The user's question.
        scheme_id: Fund identifier to filter by.
        n_results: Maximum chunks.
        db_path: ChromaDB path.

    Returns:
        Formatted context string for injection into the system prompt.
        Returns empty string if no chunks retrieved.
    """
    chunks = retrieve(question, scheme_id, n_results, db_path)
    if not chunks:
        return ""
    return "\n\n---\n\n".join(chunks)


def get_collection_stats(db_path: str = "./chroma_db") -> dict[str, Any]:
    """Get stats about the current collection for debugging.

    Returns:
        Dict with total count and per-scheme breakdown.
    """
    collection = get_collection(db_path)
    total = collection.count()

    # Get per-scheme counts
    scheme_counts = {}
    for scheme_id in ["Dynamic_Asset_Allocation", "Large_Cap", "Flexi_Cap"]:
        results = collection.get(where={"scheme_id": scheme_id})
        scheme_counts[scheme_id] = len(results["ids"])

    return {
        "total_chunks": total,
        "per_scheme": scheme_counts,
        "collection_name": collection.name,
    }
