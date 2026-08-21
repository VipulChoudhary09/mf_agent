"""Framework-neutral conversation state and recommendation orchestration.

The LiveKit agent and the terminal text harness both use this module. Keeping
fund selection and prompt transitions outside the transport layer makes the
voice migration testable without opening a WebRTC room or spending provider
credits.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol

from knowledge_base import get_comparison_table, get_scheme_context
from prompts import (
    FUND_NAMES,
    build_recommendation_prompt,
    build_recommendation_prompt_rag,
    build_tie_breaker_prompt,
)
from rag.context_updater import RAGContextUpdater
from scoring import recommend_mutual_fund


class ConversationState(str, Enum):
    """Conversation states shared by all agent front ends."""

    PROFILING = "profiling"
    TIE_BREAKER = "tie_breaker"
    RECOMMENDATION = "recommendation"


@dataclass
class BotState:
    """Mutable, per-session state for the advisor flow."""

    current: ConversationState = ConversationState.PROFILING
    recommended_fund: str | None = None
    profile_summary: str | None = None
    using_rag: bool = False


@dataclass(frozen=True)
class FlowTransition:
    """Result of processing a profile tool call."""

    state: ConversationState
    prompt: str
    tool_result: dict[str, Any]


class RAGInjector(Protocol):
    """Small protocol implemented by ``RAGContextInjector`` and test fakes."""

    @property
    def enabled(self) -> bool: ...

    def retrieve_context(self, query: str, scheme_id: str) -> str | None: ...


class AdvisorFlow:
    """Own the deterministic state machine independently of LiveKit.

    The LLM extracts profile fields, but it never chooses the fund. This class
    runs the deterministic scorer, builds the next system prompt, and manages
    per-turn RAG refreshes.
    """

    INITIAL_RAG_QUERY = (
        "fund overview key features exit load performance investment details"
    )

    def __init__(
        self,
        *,
        rag_injector: RAGInjector | None = None,
        use_rag: bool = True,
        scheme_context_getter: Callable[[str], str] = get_scheme_context,
        comparison_table_getter: Callable[[], str] = get_comparison_table,
    ) -> None:
        self.state = BotState()
        self._rag = rag_injector
        self._use_rag = use_rag and bool(rag_injector and rag_injector.enabled)
        self._get_scheme_context = scheme_context_getter
        self._get_comparison_table = comparison_table_getter
        self._rag_updater = (
            RAGContextUpdater(state=self.state, rag_injector=rag_injector)
            if self._use_rag and rag_injector is not None
            else None
        )

    def handle_profile(self, profile: dict[str, Any]) -> FlowTransition:
        """Score a profile and transition to tie-breaker or recommendation."""

        result = recommend_mutual_fund(profile)

        if result["is_tie"] and self.state.current == ConversationState.PROFILING:
            self.state.current = ConversationState.TIE_BREAKER
            prompt = build_tie_breaker_prompt(
                result["winner"], result["runner_up"]
            )
            tool_result = {
                "status": "tie_breaker_needed",
                "fund1": FUND_NAMES[result["winner"]],
                "fund2": FUND_NAMES[result["runner_up"]],
                "message": (
                    "The profile fits both funds. Ask one clarifying question "
                    "that distinguishes the two choices."
                ),
            }
            return FlowTransition(self.state.current, prompt, tool_result)

        winner_id = result["winner"]
        profile_summary = result["profile_summary"]
        comparison_table = self._get_comparison_table()
        scheme_context = self._get_scheme_context(winner_id)
        fallback_prompt = build_recommendation_prompt(
            winner_id,
            profile_summary,
            scheme_context,
            comparison_table,
        )

        self.state.current = ConversationState.RECOMMENDATION
        self.state.recommended_fund = winner_id
        self.state.profile_summary = profile_summary

        prompt = fallback_prompt
        context_mode = "full_kb"

        if self._use_rag and self._rag is not None:
            rag_context = self._rag.retrieve_context(
                query=self.INITIAL_RAG_QUERY,
                scheme_id=winner_id,
            )
            if rag_context:
                self.state.using_rag = True
                context_mode = "rag"
                prompt = build_recommendation_prompt_rag(
                    winner_id,
                    profile_summary,
                    rag_context,
                    comparison_table,
                )
            else:
                self.state.using_rag = False

        if self._rag_updater is not None:
            self._rag_updater.configure(
                comparison_table=comparison_table,
                fallback_prompt=fallback_prompt,
            )

        tool_result = {
            "status": "recommendation_ready",
            "recommended_fund": FUND_NAMES[winner_id],
            "profile_summary": profile_summary,
            "scores": result["scores"],
            "context_mode": context_mode,
        }
        return FlowTransition(self.state.current, prompt, tool_result)

    def refresh_rag_prompt(self, user_text: str) -> str | None:
        """Return a refreshed recommendation prompt for a follow-up turn."""

        if self._rag_updater is None:
            return None
        return self._rag_updater.build_prompt(user_text)
