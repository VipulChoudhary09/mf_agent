"""Per-turn RAG context updater for the RECOMMENDATION state.

This FrameProcessor sits between the user aggregator and the LLM in the
pipeline.  On every turn in the RECOMMENDATION state, it:

  1. Intercepts the ``LLMContextFrame``
  2. Extracts the latest user message text
  3. Queries ChromaDB with that user question + scheme_id filter
  4. Rebuilds the system prompt with fresh RAG chunks
  5. Passes the updated context downstream to the LLM

This solves the "one-shot RAG" problem where follow-up questions like
"Who manages this fund?" would get "I don't have that detail" because
the initial RAG retrieval used a generic query that didn't cover those topics.

Architecture note:
  - Runs synchronously on the pipeline thread (ChromaDB queries are ~50ms
    locally; embedding API is ~200-500ms).  For production with cloud vector
    DBs, consider moving to an async wrapper.
  - Only activates when ``state.current == "recommendation"`` and
    ``state.using_rag is True``.
"""

from __future__ import annotations

from loguru import logger

from pipecat.frames.frames import Frame, LLMContextFrame, LLMUpdateSettingsFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from rag.rag_processor import RAGContextInjector
from prompts import build_recommendation_prompt_rag, FUND_NAMES, FUND_INCEPTION_DATES


class RAGContextUpdater(FrameProcessor):
    """Intercepts LLMContextFrame and enriches with per-turn RAG chunks."""

    def __init__(
        self,
        state,
        rag_injector: RAGContextInjector,
        comparison_table: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._state = state
        self._rag = rag_injector
        self._comparison_table = comparison_table
        # Cache the last query to avoid redundant retrieval
        self._last_query: str | None = None

    def set_comparison_table(self, table: str):
        """Set the comparison table after it's loaded (called from bot.py)."""
        self._comparison_table = table

    def _extract_latest_user_text(self, context) -> str | None:
        """Walk context messages backwards to find the last user utterance."""
        messages = context.messages
        for msg in reversed(messages):
            role = msg.get("role", "")
            if role != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                # Skip tool/function responses that also have role=user
                if content.startswith("{") and "function_response" not in content:
                    return content
                if not content.startswith("{"):
                    return content
            # Handle list-style content (multimodal)
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part["text"]
        return None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Intercept LLMContextFrame for RAG enrichment, pass everything else."""
        await super().process_frame(frame, direction)

        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        # Only activate during RAG-enabled recommendation state
        if (
            self._state.current != "recommendation"
            or not self._state.using_rag
            or not self._state.recommended_fund
        ):
            await self.push_frame(frame, direction)
            return

        # Extract the latest user question
        user_text = self._extract_latest_user_text(frame.context)
        if not user_text or user_text == self._last_query:
            await self.push_frame(frame, direction)
            return

        self._last_query = user_text
        scheme_id = self._state.recommended_fund

        # Retrieve fresh RAG chunks for this specific question
        logger.info(f"[RAG-LIVE] Retrieving for: \"{user_text[:60]}\" scheme={scheme_id}")
        rag_context = self._rag.retrieve_context(
            query=user_text,
            scheme_id=scheme_id,
        )

        if not rag_context:
            logger.warning(f"[RAG-LIVE] No chunks for: \"{user_text[:60]}\"")
            await self.push_frame(frame, direction)
            return

        # Rebuild system prompt with fresh chunks
        new_prompt = build_recommendation_prompt_rag(
            fund_id=scheme_id,
            profile_summary=self._state.profile_summary or "",
            rag_context=rag_context,
            comparison_table=self._comparison_table,
        )

        logger.info(
            f"[RAG-LIVE] Updated system prompt with {len(rag_context)} chars "
            f"for \"{user_text[:40]}\" → scheme={scheme_id}"
        )

        # Update the LLM settings with the new system prompt
        await self.push_frame(
            LLMUpdateSettingsFrame(settings={"system_instruction": new_prompt}),
            direction,
        )

        # Then pass the original context frame through
        await self.push_frame(frame, direction)
