

# Technical Implementation Spec: Mutual Fund Voice Agent Gemini's input on the plan.

**Context for AI Coding Agent:**
We are building a voice-based mutual fund recommending agent using **Pipecat**, **Deepgram (STT)**, **Gemini 1.5/2.0 Flash (LLM)**, and **Cartesia (TTS)**.

Do NOT use RAG or vector databases. We have exactly 3 funds (Parag Parikh Dynamic Asset Allocation, Large Cap, and Flexi Cap). We are using a 3-Step Hybrid Recommendation Pipeline:

1. **LLM Function Calling** to extract a structured profile from natural conversation.
2. **Deterministic Python Soft-Scoring** to select the appropriate fund.
3. **In-Context Context Injection** where we feed the chosen fund's markdown text into the LLM to explain the recommendation and answer follow-ups.

Please implement the following architecture.

## 1. The Gemini Tool (Function Calling Schema)

Register this tool with the Gemini model in Pipecat. The LLM's goal during the "Profiling State" is to fill this out based on the user's answers.

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
        "description": "Low = fears losing money. Moderate = accepts some drops. High = comfortable with volatility."
      },
      "growth_style": {
        "type": "STRING",
        "enum": ["downside_protection", "stable_bluechip", "aggressive_global", "unspecified"],
        "description": "downside_protection = wants debt/equity mix. stable_bluechip = wants top 100 Indian companies. aggressive_global = wants global exposure and smaller caps."
      },
      "initial_investment": {
        "type": "INTEGER",
        "description": "The initial amount in INR the user wants to invest."
      }
    },
    "required": ["horizon_years", "risk_tolerance", "initial_investment"]
  }
}

```

## 2. The Python Scoring Engine (The Logic)

When Gemini calls `extract_user_profile`, intercept it in Pipecat, parse the JSON, and pass it through this exact scoring function.

```python
def recommend_mutual_fund(profile: dict) -> str:
    # Initialize score board
    scores = {
        "Dynamic_Asset_Allocation": 0,
        "Large_Cap": 0,
        "Flexi_Cap": 0
    }
    
    horizon = profile.get("horizon_years", 0)
    risk = profile.get("risk_tolerance", "moderate")
    style = profile.get("growth_style", "unspecified")
    investment = profile.get("initial_investment", 0)

    # Rule 1: Time Horizon
    if horizon < 3:
        scores["Dynamic_Asset_Allocation"] += 5
    elif horizon >= 5:
        scores["Large_Cap"] += 3
        scores["Flexi_Cap"] += 3
        scores["Dynamic_Asset_Allocation"] += 1

    # Rule 2: Risk Tolerance
    if risk == "low":
        scores["Dynamic_Asset_Allocation"] += 4
    elif risk == "moderate":
        scores["Dynamic_Asset_Allocation"] += 2
        scores["Large_Cap"] += 1
    elif risk == "high":
        scores["Flexi_Cap"] += 3
        scores["Large_Cap"] += 2

    # Rule 3: Growth Style
    if style == "downside_protection":
        scores["Dynamic_Asset_Allocation"] += 3
    elif style == "stable_bluechip":
        scores["Large_Cap"] += 3
    elif style == "aggressive_global":
        scores["Flexi_Cap"] += 3

    # Rule 4: HARD CONSTRAINT - Minimum Investment
    # Dynamic Asset Allocation requires minimum Rs 5000
    if investment > 0 and investment < 5000:
        scores["Dynamic_Asset_Allocation"] = -999  # Disqualify

    # Sort and determine winner
    sorted_funds = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    first_place = sorted_funds[0]
    second_place = sorted_funds[1]

    # Handle Tie-Breaker Scenario
    if first_place[1] == second_place[1]:
        return "TIE_BREAKER_REQUIRED"
    
    return first_place[0] # Returns the winning fund string

```

## 3. Pipecat State Management & Prompts

Implement a basic state machine in your Pipecat script.

### State 1: `PROFILING`

**System Prompt:** > "You are a helpful mutual fund voice agent for PPFAS Mutual Fund. Your goal is to gather 3 pieces of information to recommend a fund: 1) Their investment timeframe in years, 2) Their comfort with market volatility (risk tolerance), and 3) Their starting investment amount in Rupees. Ask these conversationally, one by one. Do NOT read out the actual funds yet. Once you have all the information, quietly call the `extract_user_profile` tool."

### State 2: `RECOMMENDATION_AND_QA`

*Triggered after the Python function runs and returns a winning fund.*
*Action:* Load the corresponding Markdown file texts (e.g., `FCfly.md` and `FCpres.md`) and append them to the system prompt.
**System Prompt Update:**

> "The system has matched the user with the {Fund Name}.
> **MANDATORY FIRST ACTION:** You MUST read this disclaimer verbatim: 'Mutual Fund investments are subject to market risks, read all scheme related documents carefully.'
> Next, enthusiastically tell them which fund they matched with and why, based on the profile they gave you. Finally, ask if they have any specific questions about the fund.
> USE ONLY THE FOLLOWING KNOWLEDGE BASE TO ANSWER QUESTIONS. DO NOT MAKE UP NUMBERS:
> [Insert Parsed Markdown Content Here]"

### State 3: `TIE_BREAKER` (Edge Case)

*Triggered if Python returns "TIE_BREAKER_REQUIRED"*
**System Prompt Update:**

> "The user qualifies for both the Large Cap and Flexi Cap fund. Ask them: 'You actually fit perfectly into two of our equity funds. Do you prefer investing strictly in India's top 100 stable companies, or do you want a fund that also looks for aggressive growth in international markets?'"

## 4. Instructions for the Developer

1. Set up the Pipecat pipeline and load the `google` (Gemini), `deepgram`, and `cartesia` plugins.
2. Load the 6 provided Markdown files into memory variables on startup so they are ready to be injected into the prompt instantly without file I/O latency.
3. Please write the complete Python script to initialize this pipeline. Ensure you use Pipecat's `FunctionCall` capabilities to handle the Gemini tool execution gracefully.

---

