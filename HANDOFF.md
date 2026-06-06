# Project Handoff: Mutual Fund Voice Agent

> **For:** Claude (or any AI coding agent) on Antigravity IDE
> **From:** Previous brainstorming session (June 6, 2026)
> **Project:** `/home/op/projects/mf_agent`
> **Venv:** `/home/op/projects/mf_agent/venv` (Python, only `python-dotenv` installed)

---

## 1. What We're Building

A **Pipecat voice agent** that:
1. Greets the user and asks 3-4 conversational questions to understand their investment profile
2. Runs a **deterministic Python scoring engine** (not LLM) to recommend one of 3 PPFAS mutual fund schemes
3. Announces the recommendation with a disclaimer
4. Answers follow-up questions about the recommended fund using **in-context knowledge** (all fund docs loaded into the system prompt)

### Tech Stack
| Component | Service | Free Tier |
|:----------|:--------|:----------|
| **Voice Pipeline** | Pipecat (framework) | Open source |
| **STT** | Deepgram (Nova-3) | ~45 min/month |
| **LLM** | Google Gemini 2.0 Flash | 15 RPM, 1M TPM |
| **TTS** | Cartesia | Limited free tier |
| **Transport** | Daily WebRTC | Free for dev |
| **VAD** | Silero (local, no API) | Free |

### The 3 Funds
| ID | Fund Name | Risk | Inception |
|:---|:----------|:-----|:----------|
| `Dynamic_Asset_Allocation` | Parag Parikh Dynamic Asset Allocation Fund | Moderately High | Feb 27, 2024 |
| `Large_Cap` | Parag Parikh Large Cap Fund | Very High | Feb 4, 2026 (~4 months old, NO performance data) |
| `Flexi_Cap` | Parag Parikh Flexi Cap Fund | Very High | May 24, 2013 (flagship, 13-year track record) |

---

## 2. Existing Files in the Project

### Knowledge Base (already extracted from PDFs by the user)
All extracted MDs are in `KB/pdftomds/`:
- `KB/pdftomds/Dfly.md` (6KB) — Dynamic Asset Allocation flyer
- `KB/pdftomds/Dpres.md` (19KB) — Dynamic Asset Allocation presentation
- `KB/pdftomds/FCfly.md` (17KB) — Flexi Cap flyer
- `KB/pdftomds/FCpres.md` (30KB) — Flexi Cap presentation
- `KB/pdftomds/LCfly.md` (15KB) — Large Cap flyer
- `KB/pdftomds/LCpres.md` (17KB) — Large Cap presentation

Original PDFs are in `KB/pdf/`.

**Total knowledge base: ~104KB of markdown text.** This fits easily in Gemini's context window. No vector DB / RAG needed.

### Reference Files
- `docs/gemini_suggestion_for_decision_tree.md` — Earlier brainstorming input (useful reference but superseded by this handoff)
- `docs/` — Also contains `implementation_plan.md`, `mutual_fund_agent_brainstorm.md`, `plan_critical_review.md`, `kb_and_decision_tree_review.md` from the brainstorming session
- `demo_pipeline/` — May contain reference Pipecat pipeline files from another project. Use as architectural reference but DO NOT copy wholesale — build minimal.

### Environment
- `venv/` — Python virtual environment (only `python-dotenv` installed)
- `.antigravitycli/` — IDE configuration (do not modify)
- `.env` — Needs to be created with API keys

---

## 3. Architecture Decisions (Already Made)

### 3.1 Recommendation: Deterministic Scoring, NOT LLM
The LLM does NOT choose the fund. A local Python function scores each fund based on the user's profile and picks the winner deterministically. The LLM only explains the recommendation and answers follow-ups.

**Why:** Financial recommendations must be consistent and auditable. An LLM might recommend different funds for identical profiles on different runs.

### 3.2 Knowledge Base: In-Context Injection, NOT RAG
All 6 markdown files are loaded into memory at startup (~104KB). When a fund is recommended, its flyer + presentation text is injected directly into the system prompt. No vector database, no embeddings, no chunking.

**Why:** Only 3 schemes, ~104KB total. Well within Gemini's context window. In-context gives 100% retrieval accuracy with zero infrastructure.

### 3.3 Voice-to-Structured-Data: LLM Function Calling
The user speaks naturally ("maybe a couple of years, I don't wanna lose money"). The LLM extracts structured fields via a **function call** (`extract_user_profile`), which triggers the Python scoring engine.

### 3.4 State Machine: 3 States with Dynamic System Prompt Swaps
States: `PROFILING` → `RECOMMENDATION` (or `TIE_BREAKER` → `RECOMMENDATION`)
On state transition, the system prompt is updated via `LLMUpdateSettingsFrame`.

### 3.5 Gemini-Specific: System Prompt via Settings, NOT Context
Gemini does NOT support `{"role": "system"}` messages in the conversation context. System instructions MUST be set via `GoogleLLMService.Settings(system_instruction=...)` and updated via `LLMUpdateSettingsFrame`.

---

## 4. Pipecat + Gemini Integration Details

### Installation
```bash
pip install "pipecat-ai[google,deepgram,cartesia,silero,daily]" python-dotenv loguru
```

### Key Imports
```python
# LLM
from pipecat.services.google.llm import GoogleLLMService

# STT & TTS
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService

# Function Calling (same for all LLM providers)
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

# Pipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

# Context
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)

# VAD
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

# Transport
from pipecat.transports.daily.transport import DailyParams, DailyTransport

# Frames
from pipecat.frames.frames import LLMRunFrame, LLMUpdateSettingsFrame

# Observers (minimal logging)
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
```

### LLM Setup
```python
llm = GoogleLLMService(
    api_key=os.environ["GOOGLE_API_KEY"],
    settings=GoogleLLMService.Settings(
        model="gemini-2.0-flash",
        system_instruction=PROFILING_PROMPT,  # Initial state prompt
    ),
)
```

### Dynamic System Prompt Update (State Transition)
```python
await task.queue_frame(
    LLMUpdateSettingsFrame(
        settings={"system_instruction": new_system_prompt}
    )
)
```

### Function Calling (identical to OpenAI pattern)
```python
# Define schema
tools = ToolsSchema(standard_tools=[
    FunctionSchema(
        name="extract_user_profile",
        description="...",
        properties={...},
        required=[...],
    ),
])

# Register handler
llm.register_function("extract_user_profile", handle_extract_user_profile)

# Pass tools to context
context = LLMContext(tools=tools)
```

### Pipeline Chain
```python
pipeline = Pipeline([
    transport.input(),
    stt,
    user_aggregator,
    llm,
    tts,
    transport.output(),
    assistant_aggregator,
])
```

---

## 5. Function Calling Schema

```python
FunctionSchema(
    name="extract_user_profile",
    description="Call this tool once the user has provided enough information about their investment timeframe, risk appetite, and starting amount.",
    properties={
        "horizon_years": {
            "type": "integer",
            "description": "The number of years the user plans to stay invested.",
        },
        "risk_tolerance": {
            "type": "string",
            "enum": ["low", "moderate", "high"],
            "description": "Low = fears losing money / wants capital protection. Moderate = accepts some drops for growth. High = comfortable with significant volatility.",
        },
        "growth_style": {
            "type": "string",
            "enum": ["downside_protection", "stable_bluechip", "diversified_global", "unspecified"],
            "description": "downside_protection = wants active debt/equity management to reduce losses. stable_bluechip = wants low-cost exposure to India's top 100 companies. diversified_global = wants exposure to Indian AND international stocks like Google/Alphabet.",
        },
        "initial_investment": {
            "type": "integer",
            "description": "The initial amount in INR the user wants to invest.",
        },
        "investment_mode": {
            "type": "string",
            "enum": ["lumpsum", "sip", "both", "unspecified"],
            "description": "Whether the user wants to invest a one-time lump sum, set up a monthly SIP, or both.",
        },
    },
    required=["horizon_years", "risk_tolerance", "initial_investment"],
)
```

---

## 6. Scoring Engine (Complete Implementation)

```python
def recommend_mutual_fund(profile: dict) -> dict:
    """
    Deterministic soft-scoring recommendation engine.
    Returns the winning fund, scores, and whether a tie-breaker is needed.
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
        scores["Dynamic_Asset_Allocation"] += 5
        scores["Large_Cap"] += 1
    elif risk == "moderate":
        scores["Dynamic_Asset_Allocation"] += 2
        scores["Large_Cap"] += 3  # Index-like, low active share
    elif risk == "high":
        scores["Flexi_Cap"] += 4
        scores["Large_Cap"] += 2

    # --- Rule 3: Growth Style ---
    if style == "downside_protection":
        scores["Dynamic_Asset_Allocation"] += 4
    elif style == "stable_bluechip":
        scores["Large_Cap"] += 4
    elif style == "diversified_global":
        scores["Flexi_Cap"] += 4
    # "unspecified" adds no score

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
```

---

## 7. System Prompts

### PROFILING_PROMPT
```
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
- Once you have enough information (at minimum: horizon, risk, and investment amount), call the extract_user_profile tool.
```

### RECOMMENDATION_PROMPT (template — fill {variables} at runtime)
```
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
=== END COMPARISON TABLE ===
```

### TIE_BREAKER_PROMPT (template)
```
The user's profile fits well with both the {fund1_name} and the {fund2_name}.

Ask them ONE clarifying question to help decide. Here are the key differentiators:
- {fund1_name}: {fund1_differentiator}
- {fund2_name}: {fund2_differentiator}

Frame the question naturally, for example:
"You actually fit perfectly into two of our funds. Let me ask you one more thing — [differentiating question]?"

Once they answer, call the extract_user_profile tool again with the updated growth_style.
```

### COMPARISON TABLE (hardcoded string)
```
| Feature | Dynamic Asset Allocation | Large Cap | Flexi Cap |
|:--------|:------------------------|:----------|:----------|
| Inception | Feb 27, 2024 (~2 yrs) | Feb 4, 2026 (~4 mo) | May 24, 2013 (~13 yrs) |
| Risk Level | Moderately High | Very High | Very High |
| Benchmark | CRISIL Hybrid 50+50 Moderate | Nifty 100 TRI | Nifty 500 TRI |
| Min Lumpsum | ₹5,000 | ₹1,000 | ₹1,000 |
| Min SIP | ₹1,000/month | ₹1,000/month | ₹1,000/month |
| Exit Load | 1% within 1 yr | NIL | 2% ≤1yr, 1% ≤2yr |
| TER (Direct) | Not disclosed | 0.15% | Not disclosed |
| Global Stocks | No | No | Yes (e.g. Alphabet) |
| Since Inception CAGR (Direct) | 7.49% | N/A (too new) | 18.63% |
| Equity Range | 0-100% (dynamic) | 80-100% (Nifty 100) | 65-100% (flexible) |
```

---

## 8. File Structure to Create

```
mf_agent/
├── .env                    # API keys
├── requirements.txt        # Dependencies
├── bot.py                  # Main Pipecat voice pipeline
├── scoring.py              # Deterministic scoring engine
├── prompts.py              # System prompts for each state
├── knowledge_base.py       # Loads 6 MD files, provides per-scheme context
├── text_bot.py             # Text-mode testing harness (no STT/TTS)
├── tests/
│   └── test_scoring.py     # Unit tests for the scoring engine
├── tasks.md                # Build tracker
├── docs/                   # Reference docs (already exists)
├── *.md                    # Fund knowledge base files (already exist)
└── venv/                   # Python venv (already exists)
```

---

## 9. Critical Gotchas

1. **Gemini system prompt**: MUST use `system_instruction` in Settings. `{"role": "system"}` in context causes API errors.
2. **Dynamic prompt updates**: Use `LLMUpdateSettingsFrame(settings={"system_instruction": new_prompt})`.
3. **Large Cap has NO performance data** — it's only ~4 months old. The agent must handle "how has it performed?" gracefully.
4. **Exit loads differ dramatically**: Flexi Cap charges 2% within 1 year. Large Cap charges ZERO. This is factored into the scoring engine.
5. **Dynamic Asset Allocation requires ₹5,000 minimum lumpsum** (vs ₹1,000 for the other two). Hard constraint in scoring.
6. **Deepgram free tier is tight** (~45 min/month). Build and test in text mode (`text_bot.py`) first. Only use voice for final demos.
7. **`pipecat-ai[google]`** uses the `google-genai` SDK, not httpx. The `TimedOpenAILLMService` pattern from the reference code won't work — use Pipecat's built-in `MetricsLogObserver` instead.
