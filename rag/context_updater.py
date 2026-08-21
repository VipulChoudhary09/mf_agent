"""Framework-neutral per-turn RAG prompt refresh.

LiveKit calls this helper from ``Agent.on_user_turn_completed``. It contains no
LiveKit or Pipecat imports, which keeps retrieval logic independently testable
and reusable by other transports.
"""

from __future__ import annotations

from typing import Any, Protocol

from loguru import logger

from prompts import build_recommendation_prompt_rag


class RAGInjector(Protocol):
    def retrieve_context(self, query: str, scheme_id: str) -> str | None: ...


class RAGContextUpdater:
    """Refresh the recommendation prompt with fund-filtered RAG context."""

    def __init__(self, state: Any, rag_injector: RAGInjector) -> None:
        self._state = state
        self._rag = rag_injector
        self._comparison_table = ""
        self._fallback_prompt: str | None = None
        self._last_query: str | None = None

    def configure(self, *, comparison_table: str, fallback_prompt: str) -> None:
        """Set recommendation data after the winning fund is known."""

        self._comparison_table = comparison_table
        self._fallback_prompt = fallback_prompt
        self._last_query = None

    def build_prompt(self, user_text: str) -> str | None:
        """Retrieve fresh context and return the next system prompt.

        ``None`` means no prompt change is required. If retrieval misses for a
        new question, the full-KB fallback is returned instead of leaving stale
        chunks from the previous turn active.
        """

        current = getattr(self._state.current, "value", self._state.current)
        query = user_text.strip()

        if (
            current != "recommendation"
            or not self._state.using_rag
            or not self._state.recommended_fund
            or not query
        ):
            return None

        if query == self._last_query:
            return None
        self._last_query = query

        scheme_id = self._state.recommended_fund
        logger.info(
            '[RAG-LIVEKIT] Retrieving for: "{}" scheme={}',
            query[:80],
            scheme_id,
        )
        rag_context = self._rag.retrieve_context(query=query, scheme_id=scheme_id)

        if not rag_context:
            logger.warning(
                '[RAG-LIVEKIT] No chunks for: "{}"; using full-KB fallback',
                query[:80],
            )
            return self._fallback_prompt

        prompt = build_recommendation_prompt_rag(
            fund_id=scheme_id,
            profile_summary=self._state.profile_summary or "",
            rag_context=rag_context,
            comparison_table=self._comparison_table,
        )
        logger.info(
            '[RAG-LIVEKIT] Refreshed prompt with {} chars for "{}"',
            len(rag_context),
            query[:60],
        )
        return prompt
