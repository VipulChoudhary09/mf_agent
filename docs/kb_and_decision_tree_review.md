# Review: Knowledge Base & Decision Tree Assessment

After reading all 6 extracted MDs and [gemini_suggestion_for_decision_tree.md](file:///home/op/projects/mf_agent/docs/gemini_suggestion_for_decision_tree.md), here's my assessment.

---

## ✅ What's Solid

1. **The 3-step hybrid pipeline** (LLM Function Calling → Python Scoring → In-Context KB) is the correct architecture. Gemini's suggestion nails this.
2. **The function calling schema** (`extract_user_profile`) is well-designed. Using the LLM to parse natural voice into structured fields is exactly the Gap 1 fix we identified.
3. **The state machine** (PROFILING → RECOMMENDATION_AND_QA → TIE_BREAKER) is clean and implementable in Pipecat.
4. **The extracted MDs are high quality** — tables, performance data, holdings, and scheme facts all came through cleanly.

---

## ⚠️ Issues Found After Cross-Referencing with Actual Fund Data

### Issue 1: Large Cap Fund Is Brand New — No Track Record

From [LCpres.md](file:///home/op/projects/mf_agent/LCpres.md):
> NFO OPENS: JANUARY 19, 2026. NFO CLOSES: JANUARY 30, 2026. Scheme Reopens on February 06, 2026.

**The Large Cap Fund is only ~4 months old** (inception: Feb 4, 2026). It has **zero** published performance data — no 1-year, 3-year, or 5-year returns. Compare this to Flexi Cap which has a **13-year track record** with 17.80% CAGR since inception.

> [!IMPORTANT]
> The scoring engine and the agent's explanation should acknowledge this. If a user asks "How has the Large Cap Fund performed?", the agent must say it's too new to have meaningful performance data, rather than fabricating or dodging.

### Issue 2: Exit Load Is a Major Differentiator — Not in the Scoring Engine

The 3 funds have dramatically different exit loads:

| Fund | Exit Load |
|:-----|:----------|
| **Large Cap** | **NIL** (zero exit load!) |
| **Dynamic Asset Allocation** | 1% if redeemed within 1 year |
| **Flexi Cap** | **2%** within 1 year, **1%** within 2 years |

This is a critical factor for investors, especially those with shorter horizons or uncertain timelines. Gemini's scoring engine completely ignores this. A user who says *"I might need this money back in 18 months"* should be steered away from Flexi Cap (which would charge 1% exit load at 18 months).

### Issue 3: Minimum Investment Amounts Differ

| Fund | Lumpsum Minimum | SIP Minimum |
|:-----|:----------------|:------------|
| **Dynamic Asset Allocation** | **₹5,000** | ₹1,000/month |
| **Flexi Cap** | ₹1,000 | ₹1,000/month |
| **Large Cap** | ₹1,000 | ₹1,000/month |

Gemini's scoring engine has the ₹5,000 hard constraint for Dynamic — **this is correct** ✅. But it should also consider SIP vs lumpsum preference since the minimums differ.

### Issue 4: Cost (TER) Is a Big Selling Point for Large Cap

From [LCpres.md](file:///home/op/projects/mf_agent/LCpres.md):
> Direct Plan: 0.15%, Regular Plan: 0.55%

This is **extremely cheap** compared to typical active funds (which charge 1-2%). The Flexi Cap and Dynamic funds don't publish exact TER in these documents but refer to a website link. For cost-conscious investors, Large Cap is the clear winner.

### Issue 5: Flexi Cap Has Global Exposure — Decision Tree's `aggressive_global` Is Slightly Misleading

The Gemini suggestion uses `"aggressive_global"` as a growth style for Flexi Cap. While Flexi Cap does invest globally (Alphabet Inc is in the top 10 holdings), it's not purely "aggressive." It's a value-oriented fund with:
- "Margin of Safety" philosophy
- Cash-generating, low-debt businesses
- "Minority-shareholder friendly managements"

The global exposure is more about diversification and reduced country risk than aggressive growth. A better label might be `"diversified_global"` or the profiling question should frame it as:
> *"Do you want your investments to include global companies like Google and Microsoft, or prefer staying focused on Indian companies?"*

### Issue 6: Dynamic Asset Allocation Is Not Just "Short-Term" — It's for Risk-Averse at Any Horizon

The current scoring engine heavily weights Dynamic for short horizons (< 3 years, +5 points). But looking at the actual fund documents:

From [Dpres.md](file:///home/op/projects/mf_agent/Dpres.md):
> "Typically suitable for **medium to long-term** investors looking for balanced risk and return"

Dynamic is also suitable for **long-term investors who are risk-averse** (e.g., retirees). The scoring engine should give Dynamic points for low risk tolerance **regardless of horizon**, not just for short horizons.

---

## 🔧 Proposed Refinements to the Scoring Engine

```python
def recommend_mutual_fund(profile: dict) -> str:
    scores = {
        "Dynamic_Asset_Allocation": 0,
        "Large_Cap": 0,
        "Flexi_Cap": 0
    }
    
    horizon = profile.get("horizon_years", 0)
    risk = profile.get("risk_tolerance", "moderate")
    style = profile.get("growth_style", "unspecified")
    investment = profile.get("initial_investment", 0)

    # --- Rule 1: Time Horizon ---
    if horizon < 3:
        scores["Dynamic_Asset_Allocation"] += 5
        # Short horizon penalizes Flexi Cap due to heavy exit load (2% < 1yr)
        scores["Flexi_Cap"] -= 2
    elif 3 <= horizon < 5:
        # Medium term: Dynamic is still valid, equity starts becoming viable
        scores["Dynamic_Asset_Allocation"] += 3
        scores["Large_Cap"] += 2  # NIL exit load is attractive
        scores["Flexi_Cap"] += 1  # Still has 1% exit load at 18 months
    elif horizon >= 5:
        scores["Large_Cap"] += 3
        scores["Flexi_Cap"] += 4  # Long term is where Flexi Cap shines (13yr track record)
        scores["Dynamic_Asset_Allocation"] += 1

    # --- Rule 2: Risk Tolerance ---
    if risk == "low":
        scores["Dynamic_Asset_Allocation"] += 5  # Moderately High risk, but has debt cushion
        scores["Large_Cap"] += 1
    elif risk == "moderate":
        scores["Dynamic_Asset_Allocation"] += 2
        scores["Large_Cap"] += 3  # Index-like, low active share = lower risk within equity
    elif risk == "high":
        scores["Flexi_Cap"] += 4  # Very High risk, global exposure
        scores["Large_Cap"] += 2

    # --- Rule 3: Growth Style ---
    if style == "downside_protection":
        scores["Dynamic_Asset_Allocation"] += 4
    elif style == "stable_bluechip":
        scores["Large_Cap"] += 4
    elif style == "diversified_global":  # Renamed from aggressive_global
        scores["Flexi_Cap"] += 4
    # "unspecified" adds no score — relies on horizon + risk

    # --- Rule 4: HARD CONSTRAINTS ---
    # Dynamic requires ₹5,000 minimum lumpsum
    if 0 < investment < 5000:
        scores["Dynamic_Asset_Allocation"] = -999

    # --- Determine Winner ---
    sorted_funds = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    first, second = sorted_funds[0], sorted_funds[1]

    # Tie or near-tie (within 1 point): ask clarifying question
    if first[1] - second[1] <= 1:
        return f"TIE:{first[0]}:{second[0]}"
    
    return first[0]
```

### Key Changes from Gemini's Version:
1. **Exit load penalty** for Flexi Cap at short horizons
2. **Medium horizon (3-5 years)** is no longer a dead zone — Large Cap gets points for NIL exit load
3. **Low risk always boosts Dynamic** (not just short term), reflecting the fund's suitability for risk-averse long-term investors
4. **Flexi Cap gets extra points at 5+ years** because it has a proven 13-year track record
5. **Renamed `aggressive_global` → `diversified_global`** to better reflect Flexi Cap's value-oriented global strategy
6. **Near-tie detection** (within 1 point) returns both contenders for a clarification question, not just exact ties

---

## 🔧 Proposed Refinements to the Function Calling Schema

```json
{
  "name": "extract_user_profile",
  "description": "Call this tool once the user has provided enough information about their investment timeframe, risk appetite, and starting amount.",
  "parameters": {
    "type": "OBJECT",
    "properties": {
      "horizon_years": {
        "type": "INTEGER",
        "description": "The number of years the user plans to stay invested."
      },
      "risk_tolerance": {
        "type": "STRING",
        "enum": ["low", "moderate", "high"],
        "description": "Low = fears losing money / wants capital protection. Moderate = accepts some drops for growth. High = comfortable with significant volatility."
      },
      "growth_style": {
        "type": "STRING",
        "enum": ["downside_protection", "stable_bluechip", "diversified_global", "unspecified"],
        "description": "downside_protection = wants active debt/equity management to reduce losses. stable_bluechip = wants low-cost exposure to India's top 100 companies. diversified_global = wants exposure to Indian AND international stocks like Google/Alphabet."
      },
      "initial_investment": {
        "type": "INTEGER",
        "description": "The initial amount in INR the user wants to invest."
      },
      "investment_mode": {
        "type": "STRING",
        "enum": ["lumpsum", "sip", "both", "unspecified"],
        "description": "Whether the user wants to invest a one-time lump sum, set up a monthly SIP, or both."
      }
    },
    "required": ["horizon_years", "risk_tolerance", "initial_investment"]
  }
}
```

### Changes:
- Added `investment_mode` (lumpsum/SIP) — relevant for minimum amounts
- Renamed `aggressive_global` → `diversified_global` with better description
- Improved `risk_tolerance` descriptions for clearer LLM parsing

---

## 📋 Quick Data Summary for the Knowledge Base

Here's a comparison table the LLM should have access to (or you can hardcode in the recommendation explanation):

| Feature | Dynamic Asset Allocation | Large Cap | Flexi Cap |
|:--------|:------------------------|:----------|:----------|
| **Inception** | Feb 27, 2024 (~2 yrs) | Feb 4, 2026 (~4 mo) | May 24, 2013 (~13 yrs) |
| **Category** | Dynamic Asset Allocation | Large Cap (quasi-index) | Flexi Cap (active) |
| **Risk Level** | Moderately High | Very High | Very High |
| **Benchmark** | CRISIL Hybrid 50+50 Moderate | Nifty 100 TRI | Nifty 500 TRI |
| **Min Lumpsum** | ₹5,000 | ₹1,000 | ₹1,000 |
| **Min SIP** | ₹1,000/month | ₹1,000/month | ₹1,000/month |
| **Exit Load** | 1% within 1 yr | **NIL** | 2% ≤1yr, 1% ≤2yr |
| **TER (Direct)** | Not disclosed | **0.15%** | Not disclosed |
| **Global Stocks** | No | No | **Yes** (e.g. Alphabet) |
| **Since Inception CAGR** | 7.49% (Direct) | N/A (too new) | **18.63%** (Direct) |
| **Equity Allocation** | 0-100% (dynamic) | 80-100% (Nifty 100) | 65-100% (flexible) |
| **Current Equity %** | ~28% | ~100% (index) | ~85%+ (active) |

---

## 📝 Final Verdict on the Plan

The overall architecture is **ready to build**. The adjustments above are refinements, not redesigns. Here's the priority order:

1. ✅ **Incorporate exit load + fund age into scoring** (the biggest data-driven fix)
2. ✅ **Add the `investment_mode` field** to the function calling schema
3. ✅ **Rename `aggressive_global` → `diversified_global`** to accurately reflect Flexi Cap
4. ✅ **Add the comparison table** to the system prompt so the agent can answer cross-fund questions
5. ✅ **Hard-code the "too new for performance data" disclaimer** for Large Cap
