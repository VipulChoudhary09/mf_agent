"""RAG-powered context retrieval for recommendation prompts.

The helper is intentionally transport-neutral. ``advisor_core.py`` uses it for
the initial recommendation, and LiveKit's ``on_user_turn_completed`` hook uses
it indirectly for follow-up questions.
"""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger

from rag.retriever import retrieve_formatted


class RAGContextInjector:
    """Retrieve fund-filtered KB chunks for one user question."""

    def __init__(
        self,
        db_path: str = "./chroma_db",
        n_results: int = 4,
        enabled: bool = True,
    ) -> None:
        self.db_path = db_path
        self.n_results = n_results
        self._enabled = enabled and Path(db_path).exists()

        if enabled and not self._enabled:
            logger.warning(
                "[RAG] ChromaDB not found at {}. Full-KB fallback will be used.",
                db_path,
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def retrieve_context(self, query: str, scheme_id: str) -> str | None:
        """Return formatted chunks or ``None`` when retrieval cannot ground a turn."""

        if not self._enabled:
            return None

        try:
            started = time.monotonic()
            context = retrieve_formatted(
                question=query,
                scheme_id=scheme_id,
                n_results=self.n_results,
                db_path=self.db_path,
            )
            elapsed_ms = (time.monotonic() - started) * 1000

            if context:
                logger.info(
                    "[RAG] Retrieved {} chunks for scheme={} in {:.0f}ms ({} chars)",
                    self.n_results,
                    scheme_id,
                    elapsed_ms,
                    len(context),
                )
                return context

            logger.warning(
                '[RAG] No results for query="{}" scheme={}',
                query[:50],
                scheme_id,
            )
            return None
        except Exception as exc:
            logger.error("[RAG] Retrieval failed: {}", type(exc).__name__)
            return None
