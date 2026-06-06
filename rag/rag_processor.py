"""RAG-powered context injection for the RECOMMENDATION state.

Instead of injecting the entire ~47K char knowledge base into the system prompt,
this module retrieves only the 3-5 most relevant chunks per user question via
ChromaDB vector search with hard scheme_id filtering.

Architecture:
  - RAGContextInjector: Called from bot.py's recommendation flow to augment
    user messages with relevant KB context before they reach the LLM.
  - Falls back to full context injection if ChromaDB is unavailable or
    retrieval fails.

Usage in bot.py:
  from rag.rag_processor import RAGContextInjector
  rag_injector = RAGContextInjector(db_path="./chroma_db")
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from loguru import logger

from rag.retriever import retrieve_formatted


class RAGContextInjector:
    """Retrieves relevant KB chunks for a user question in RECOMMENDATION state.

    This is NOT a Pipecat FrameProcessor — it's a stateless helper called
    directly from the function handler to inject context into the system prompt.
    This design avoids adding latency to the frame pipeline for non-RAG states.
    """

    def __init__(
        self,
        db_path: str = "./chroma_db",
        n_results: int = 4,
        enabled: bool = True,
    ):
        self.db_path = db_path
        self.n_results = n_results
        self._enabled = enabled and Path(db_path).exists()

        if not self._enabled:
            logger.warning(
                f"[RAG] Disabled — ChromaDB not found at {db_path}. "
                f"Run 'python scripts/index_kb.py' to populate."
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def retrieve_context(self, query: str, scheme_id: str) -> str | None:
        """Retrieve relevant KB chunks for a user question.

        Args:
            query: The user's question text.
            scheme_id: The fund scheme ID (e.g. 'Flexi_Cap').

        Returns:
            Formatted context string with relevant chunks, or None if
            retrieval fails or returns no results.
        """
        if not self._enabled:
            return None

        try:
            start = time.monotonic()
            context = retrieve_formatted(
                query=query,
                scheme_id=scheme_id,
                n_results=self.n_results,
                db_path=self.db_path,
            )
            elapsed_ms = (time.monotonic() - start) * 1000

            if context:
                logger.info(
                    f"[RAG] Retrieved {self.n_results} chunks for "
                    f"scheme={scheme_id} in {elapsed_ms:.0f}ms "
                    f"({len(context)} chars)"
                )
                return context
            else:
                logger.warning(
                    f"[RAG] No results for query='{query[:50]}' "
                    f"scheme={scheme_id}"
                )
                return None

        except Exception as e:
            logger.error(f"[RAG] Retrieval failed: {e}")
            return None
