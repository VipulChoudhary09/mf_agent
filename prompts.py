"""
System Prompts for PPFAS Mutual Fund Voice Agent.

Three conversation states, each with its own system prompt:
  1. PROFILING — gather investment profile via 3-4 questions
  2. TIE_BREAKER — disambiguate when two funds score within 1 point
  3. RECOMMENDATION — announce the fund, then answer follow-up Q&A from KB

The PROFILING prompt is a constant. The other two are template functions
that inject fund-specific data at runtime.
"""

# ---------------------------------------------------------------------------
# Fund metadata dicts
# ---------------------------------------------------------------------------

FUND_NAMES: dict[str, str] = {
    "Dynamic_Asset_Allocation": "Parag Parikh Dynamic Asset Allocation Fund",
    "Large_Cap": "Parag Parikh Large Cap Fund",
    "Flexi_Cap": "Parag Parikh Flexi Cap Fund",
}

FUND_INCEPTION_DATES: dict[str, str] = {
    "Dynamic_Asset_Allocation": "Feb 27, 2024",
    "Large_Cap": "Feb 4, 2026",
    "Flexi_Cap": "May 24, 2013",
}

FUND_DIFFERENTIATORS: dict[str, str] = {
    "Dynamic_Asset_Allocation": (
        "actively manages between equity and debt to protect against market drops"
    ),
    "Large_Cap": (
        "low-cost, index-like exposure to India's top 100 companies "
        "with zero exit load"
    ),
    "Flexi_Cap": (
        "invests in Indian AND global companies like Google, "
        "with a 13-year proven track record"
    ),
}

# ---------------------------------------------------------------------------
# State 1: PROFILING — constant prompt
# ---------------------------------------------------------------------------

PROFILING_PROMPT = """\
You are a friendly and professional mutual fund advisor voice agent for PPFAS Mutual Fund.

MANDATORY — say this FIRST before anything else:
"Welcome to PPFAS Mutual Fund. Please note that I am an AI assistant and not a registered financial advisor. My suggestions are for informational purposes only."

Your goal is to understand the user's investment needs by asking 3-4 short conversational questions:
1) How long they plan to stay invested (in years)
2) Their comfort with market ups and downs (risk tolerance)
3) How much they want to start with (in Rupees)
4) Whether they prefer stability, growth, or global diversification — ask this naturally based on context

RULES:
- Ask ONE question at a time. Wait for the answer before asking the next.
- Be warm, conversational, and brief — max 2-3 sentences per turn. This is a voice call.
- DO NOT mention specific fund names yet.
- DO NOT read any fund details yet.
- If the user is unsure about a question, offer simple examples to help them decide.
- Once you have enough information (at minimum: horizon, risk, and investment amount), call the extract_user_profile tool.\
"""

# ---------------------------------------------------------------------------
# State 2: RECOMMENDATION — template function
# ---------------------------------------------------------------------------

_RECOMMENDATION_TEMPLATE = """\
The system has analyzed the user's profile and matched them with the {fund_name}.

MANDATORY FIRST ACTION — read this disclaimer verbatim:
"Mutual Fund investments are subject to market risks. Please read all scheme related documents carefully."

Then enthusiastically tell them which fund they matched with and explain WHY based on their profile: {profile_summary}.

After the explanation, ask if they have any specific questions about the fund.

CRITICAL RULES:
- ONLY use facts from the KNOWLEDGE BASE below. NEVER invent or estimate numbers.
- If asked something not covered in the knowledge base, say: "I don't have that specific detail in my information. I'd recommend checking the official factsheet at amc.ppfas.com or consulting a financial advisor."
- When citing performance data, ALWAYS mention the time period it refers to.
- The {fund_name} was launched on {inception_date}. If asked about performance and it has less than 3 years of history, explicitly acknowledge this.
- Keep responses concise — this is a voice conversation, not a written report. 3-4 sentences max unless the user asks for detail.

=== KNOWLEDGE BASE: {fund_name} ===
{scheme_context}
=== END KNOWLEDGE BASE ===

=== FUND COMPARISON TABLE (use only if user asks to compare funds) ===
{comparison_table}
=== END COMPARISON TABLE ===\
"""


def build_recommendation_prompt(
    fund_id: str,
    profile_summary: str,
    scheme_context: str,
    comparison_table: str,
) -> str:
    """Build the RECOMMENDATION system prompt with injected KB context.

    Args:
        fund_id: Scheme ID (e.g. 'Flexi_Cap')
        profile_summary: Natural language profile from scoring engine
        scheme_context: Full markdown KB text for the recommended scheme
        comparison_table: Cross-fund comparison table string

    Returns:
        Fully populated system prompt string.
    """
    fund_name = FUND_NAMES[fund_id]
    inception_date = FUND_INCEPTION_DATES[fund_id]

    return _RECOMMENDATION_TEMPLATE.format(
        fund_name=fund_name,
        profile_summary=profile_summary,
        scheme_context=scheme_context,
        comparison_table=comparison_table,
        inception_date=inception_date,
    )


# ---------------------------------------------------------------------------
# State 2b: TIE_BREAKER — template function
# ---------------------------------------------------------------------------

_TIE_BREAKER_TEMPLATE = """\
The user's profile fits well with both the {fund1_name} and the {fund2_name}.

Ask them ONE clarifying question to help decide. Here are the key differentiators:
- {fund1_name}: {fund1_differentiator}
- {fund2_name}: {fund2_differentiator}

Frame the question naturally, for example:
"You actually fit perfectly into two of our funds. Let me ask you one more thing — [differentiating question]?"

Once they answer, call the extract_user_profile tool again with the updated growth_style.\
"""


def build_tie_breaker_prompt(fund1_id: str, fund2_id: str) -> str:
    """Build the TIE_BREAKER system prompt for two contending funds.

    Args:
        fund1_id: First contender scheme ID
        fund2_id: Second contender scheme ID

    Returns:
        Fully populated tie-breaker system prompt.
    """
    return _TIE_BREAKER_TEMPLATE.format(
        fund1_name=FUND_NAMES[fund1_id],
        fund2_name=FUND_NAMES[fund2_id],
        fund1_differentiator=FUND_DIFFERENTIATORS[fund1_id],
        fund2_differentiator=FUND_DIFFERENTIATORS[fund2_id],
    )
