"""Network-free regression tests for the LiveKit migration seam."""

from __future__ import annotations

from advisor_core import AdvisorFlow, ConversationState


class FakeRAG:
    enabled = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def retrieve_context(self, query: str, scheme_id: str) -> str | None:
        self.calls.append((query, scheme_id))
        if "missing" in query:
            return None
        return f"retrieved context for {scheme_id}: {query}"


def make_flow(rag: FakeRAG | None = None) -> AdvisorFlow:
    return AdvisorFlow(
        rag_injector=rag,
        use_rag=rag is not None,
        scheme_context_getter=lambda fund_id: f"FULL KB FOR {fund_id}",
        comparison_table_getter=lambda: "COMPARISON TABLE",
    )


def test_clear_winner_uses_rag_and_updates_state() -> None:
    rag = FakeRAG()
    flow = make_flow(rag)

    transition = flow.handle_profile(
        {
            "horizon_years": 10,
            "risk_tolerance": "high",
            "growth_style": "diversified_global",
            "initial_investment": 100_000,
            "investment_mode": "lumpsum",
        }
    )

    assert transition.state == ConversationState.RECOMMENDATION
    assert flow.state.recommended_fund == "Flexi_Cap"
    assert flow.state.using_rag is True
    assert transition.tool_result["context_mode"] == "rag"
    assert "retrieved context for Flexi_Cap" in transition.prompt
    assert len(rag.calls) == 1


def test_near_tie_then_second_profile_resolves() -> None:
    flow = make_flow()

    first = flow.handle_profile(
        {
            "horizon_years": 4,
            "risk_tolerance": "moderate",
            "growth_style": "unspecified",
            "initial_investment": 20_000,
            "investment_mode": "lumpsum",
        }
    )
    assert first.state == ConversationState.TIE_BREAKER
    assert first.tool_result["status"] == "tie_breaker_needed"

    second = flow.handle_profile(
        {
            "horizon_years": 4,
            "risk_tolerance": "moderate",
            "growth_style": "downside_protection",
            "initial_investment": 20_000,
            "investment_mode": "lumpsum",
        }
    )
    assert second.state == ConversationState.RECOMMENDATION
    assert flow.state.recommended_fund == "Dynamic_Asset_Allocation"
    assert second.tool_result["context_mode"] == "full_kb"


def test_follow_up_refreshes_and_miss_uses_full_kb_fallback() -> None:
    rag = FakeRAG()
    flow = make_flow(rag)
    flow.handle_profile(
        {
            "horizon_years": 10,
            "risk_tolerance": "high",
            "growth_style": "diversified_global",
            "initial_investment": 100_000,
            "investment_mode": "lumpsum",
        }
    )

    refreshed = flow.refresh_rag_prompt("Who manages this fund?")
    assert refreshed is not None
    assert "Who manages this fund?" in refreshed

    # Repeating the exact query should keep the current prompt and skip a call.
    call_count = len(rag.calls)
    assert flow.refresh_rag_prompt("Who manages this fund?") is None
    assert len(rag.calls) == call_count

    fallback = flow.refresh_rag_prompt("missing detail")
    assert fallback is not None
    assert "FULL KB FOR Flexi_Cap" in fallback
