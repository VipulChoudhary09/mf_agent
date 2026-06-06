"""
Deterministic Scoring Engine for PPFAS Mutual Fund Recommendation.

This module implements a soft-scoring recommendation engine that takes a
user's investment profile (extracted by the LLM via function calling) and
returns a deterministic fund recommendation. The LLM does NOT choose the
fund — this pure Python function does.

Scoring rules are based on:
- Time horizon (exit load penalties, track record bonuses)
- Risk tolerance (Dynamic's debt cushion vs equity-heavy options)
- Growth style preference (downside protection / bluechip / global)
- Hard constraints (minimum investment amounts)

Near-tie detection (within 1 point) triggers a tie-breaker flow.
"""


def recommend_mutual_fund(profile: dict) -> dict:
    """Deterministic soft-scoring recommendation engine.

    Args:
        profile: Dict with keys from the LLM's extract_user_profile function call:
            - horizon_years (int): Years the user plans to stay invested
            - risk_tolerance (str): "low" | "moderate" | "high"
            - growth_style (str, optional): "downside_protection" | "stable_bluechip" |
              "diversified_global" | "unspecified"
            - initial_investment (int): Amount in INR
            - investment_mode (str, optional): "lumpsum" | "sip" | "both" | "unspecified"

    Returns:
        dict with keys:
            - winner (str): Scheme ID of the recommended fund
            - winner_name (str): Human-readable fund name
            - scores (dict): All 3 fund scores
            - is_tie (bool): True if top two are within 1 point
            - runner_up (str): Scheme ID of second-place fund
            - runner_up_name (str): Human-readable runner-up name
            - profile_summary (str): Natural language summary for the LLM
    """
    scores = {
        "Dynamic_Asset_Allocation": 0,
        "Large_Cap": 0,
        "Flexi_Cap": 0,
    }

    horizon = profile.get("horizon_years", 0)
    risk = profile.get("risk_tolerance", "moderate")
    style = profile.get("growth_style", "unspecified")
    investment = profile.get("initial_investment", 0)

    # --- Rule 1: Time Horizon ---
    if horizon < 3:
        scores["Dynamic_Asset_Allocation"] += 5
        scores["Flexi_Cap"] -= 2  # Heavy exit load penalty (2% < 1yr)
    elif 3 <= horizon < 5:
        scores["Dynamic_Asset_Allocation"] += 3
        scores["Large_Cap"] += 2  # NIL exit load attractive here
        scores["Flexi_Cap"] += 1
    elif horizon >= 5:
        scores["Large_Cap"] += 3
        scores["Flexi_Cap"] += 4  # 13-year proven track record
        scores["Dynamic_Asset_Allocation"] += 1

    # --- Rule 2: Risk Tolerance ---
    if risk == "low":
        scores["Dynamic_Asset_Allocation"] += 5  # Debt cushion, moderately high risk
        scores["Large_Cap"] += 1
    elif risk == "moderate":
        scores["Dynamic_Asset_Allocation"] += 2
        scores["Large_Cap"] += 3  # Index-like, low active share
    elif risk == "high":
        scores["Flexi_Cap"] += 4  # Very High risk, global exposure
        scores["Large_Cap"] += 2

    # --- Rule 3: Growth Style ---
    if style == "downside_protection":
        scores["Dynamic_Asset_Allocation"] += 4
    elif style == "stable_bluechip":
        scores["Large_Cap"] += 4
    elif style == "diversified_global":
        scores["Flexi_Cap"] += 4
    # "unspecified" adds no score — relies on horizon + risk

    # --- Rule 4: HARD CONSTRAINTS ---
    if 0 < investment < 5000:
        scores["Dynamic_Asset_Allocation"] = -999  # Requires ₹5,000 min lumpsum

    # --- Determine Winner ---
    sorted_funds = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    first, second = sorted_funds[0], sorted_funds[1]

    is_tie = (first[1] - second[1]) <= 1

    # Build profile summary for the LLM explanation
    horizon_desc = f"{horizon}-year horizon" if horizon else "unspecified horizon"
    risk_desc = f"{risk} risk tolerance"
    style_desc = style.replace("_", " ") if style != "unspecified" else ""
    summary_parts = [horizon_desc, risk_desc]
    if style_desc:
        summary_parts.append(f"prefers {style_desc}")

    profile_summary = ", ".join(summary_parts)

    return {
        "winner": first[0],
        "winner_name": first[0].replace("_", " "),
        "scores": dict(scores),
        "is_tie": is_tie,
        "runner_up": second[0],
        "runner_up_name": second[0].replace("_", " "),
        "profile_summary": profile_summary,
    }
