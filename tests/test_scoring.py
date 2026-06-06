"""
Unit tests for the deterministic scoring engine.

Tests cover:
  - Clear winner scenarios (each fund wins for its ideal profile)
  - Hard constraints (minimum investment disqualification)
  - Near-tie detection
  - Edge cases (zero horizon, missing optional fields)
  - Score sanity checks
"""

import pytest
from scoring import recommend_mutual_fund


class TestClearWinners:
    """Each fund should win decisively for its ideal user profile."""

    def test_short_horizon_low_risk_recommends_dynamic(self):
        """2-year horizon + low risk → Dynamic Asset Allocation (clear winner)."""
        result = recommend_mutual_fund({
            "horizon_years": 2,
            "risk_tolerance": "low",
            "initial_investment": 10000,
        })
        assert result["winner"] == "Dynamic_Asset_Allocation"
        assert not result["is_tie"]

    def test_long_horizon_high_risk_global_recommends_flexi_cap(self):
        """10-year horizon + high risk + diversified_global → Flexi Cap."""
        result = recommend_mutual_fund({
            "horizon_years": 10,
            "risk_tolerance": "high",
            "growth_style": "diversified_global",
            "initial_investment": 100000,
        })
        assert result["winner"] == "Flexi_Cap"
        assert not result["is_tie"]

    def test_long_horizon_moderate_risk_bluechip_recommends_large_cap(self):
        """7-year horizon + moderate risk + stable_bluechip → Large Cap."""
        result = recommend_mutual_fund({
            "horizon_years": 7,
            "risk_tolerance": "moderate",
            "growth_style": "stable_bluechip",
            "initial_investment": 50000,
        })
        assert result["winner"] == "Large_Cap"
        assert not result["is_tie"]


class TestHardConstraints:
    """Minimum investment disqualifies Dynamic Asset Allocation."""

    def test_investment_below_5000_disqualifies_dynamic(self):
        """Investment < ₹5,000 → Dynamic gets score -999."""
        result = recommend_mutual_fund({
            "horizon_years": 2,
            "risk_tolerance": "low",
            "initial_investment": 3000,
        })
        assert result["scores"]["Dynamic_Asset_Allocation"] == -999
        assert result["winner"] != "Dynamic_Asset_Allocation"

    def test_investment_exactly_5000_does_not_disqualify_dynamic(self):
        """Investment = ₹5,000 → Dynamic is NOT disqualified."""
        result = recommend_mutual_fund({
            "horizon_years": 2,
            "risk_tolerance": "low",
            "initial_investment": 5000,
        })
        assert result["scores"]["Dynamic_Asset_Allocation"] != -999

    def test_investment_zero_does_not_disqualify_dynamic(self):
        """Investment = 0 → no hard constraint triggered (0 < inv is False)."""
        result = recommend_mutual_fund({
            "horizon_years": 2,
            "risk_tolerance": "low",
            "initial_investment": 0,
        })
        assert result["scores"]["Dynamic_Asset_Allocation"] != -999


class TestNearTie:
    """Near-tie detection: top two scores within 1 point → is_tie=True."""

    def test_tie_scenario(self):
        """Medium horizon + moderate risk + unspecified style → near tie possible."""
        # horizon=4 → Dynamic +3, Large_Cap +2, Flexi_Cap +1
        # risk=moderate → Dynamic +2, Large_Cap +3
        # style=unspecified → no bonus
        # Totals: Dynamic=5, Large_Cap=5, Flexi_Cap=1
        result = recommend_mutual_fund({
            "horizon_years": 4,
            "risk_tolerance": "moderate",
            "initial_investment": 10000,
        })
        # Dynamic and Large_Cap are tied at 5, difference = 0 ≤ 1
        assert result["is_tie"] is True

    def test_clear_winner_is_not_tie(self):
        """Strong profile → difference > 1 → is_tie=False."""
        result = recommend_mutual_fund({
            "horizon_years": 1,
            "risk_tolerance": "low",
            "growth_style": "downside_protection",
            "initial_investment": 50000,
        })
        # Dynamic should dominate: 5 + 5 + 4 = 14
        assert result["is_tie"] is False


class TestEdgeCases:
    """Edge cases that should never crash."""

    def test_zero_horizon_no_crash(self):
        """horizon=0 → treated as < 3 (short term), should not crash."""
        result = recommend_mutual_fund({
            "horizon_years": 0,
            "risk_tolerance": "moderate",
            "initial_investment": 10000,
        })
        assert "winner" in result
        assert "scores" in result

    def test_missing_optional_fields_no_crash(self):
        """Missing growth_style and investment_mode → should not crash."""
        result = recommend_mutual_fund({
            "horizon_years": 5,
            "risk_tolerance": "high",
            "initial_investment": 25000,
        })
        assert "winner" in result
        assert result["profile_summary"]  # non-empty

    def test_empty_profile_no_crash(self):
        """Completely empty profile → uses all defaults, should not crash."""
        result = recommend_mutual_fund({})
        assert "winner" in result

    def test_negative_horizon_no_crash(self):
        """Negative horizon → treated as < 3, should not crash."""
        result = recommend_mutual_fund({
            "horizon_years": -1,
            "risk_tolerance": "low",
            "initial_investment": 10000,
        })
        assert "winner" in result


class TestReturnStructure:
    """Verify the return dict has all required keys with correct types."""

    def test_return_keys(self):
        result = recommend_mutual_fund({
            "horizon_years": 5,
            "risk_tolerance": "moderate",
            "initial_investment": 50000,
        })
        assert set(result.keys()) == {
            "winner", "winner_name", "scores", "is_tie",
            "runner_up", "runner_up_name", "profile_summary",
        }

    def test_return_types(self):
        result = recommend_mutual_fund({
            "horizon_years": 5,
            "risk_tolerance": "moderate",
            "initial_investment": 50000,
        })
        assert isinstance(result["winner"], str)
        assert isinstance(result["winner_name"], str)
        assert isinstance(result["scores"], dict)
        assert isinstance(result["is_tie"], bool)
        assert isinstance(result["runner_up"], str)
        assert isinstance(result["runner_up_name"], str)
        assert isinstance(result["profile_summary"], str)

    def test_winner_is_valid_scheme_id(self):
        result = recommend_mutual_fund({
            "horizon_years": 5,
            "risk_tolerance": "high",
            "initial_investment": 100000,
        })
        valid_ids = {"Dynamic_Asset_Allocation", "Large_Cap", "Flexi_Cap"}
        assert result["winner"] in valid_ids
        assert result["runner_up"] in valid_ids
        assert result["winner"] != result["runner_up"]

    def test_profile_summary_contains_profile_info(self):
        result = recommend_mutual_fund({
            "horizon_years": 5,
            "risk_tolerance": "high",
            "growth_style": "diversified_global",
            "initial_investment": 100000,
        })
        assert "5-year" in result["profile_summary"]
        assert "high" in result["profile_summary"]
        assert "diversified global" in result["profile_summary"]
