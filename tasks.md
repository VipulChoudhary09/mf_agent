# Tasks: Mutual Fund Voice Agent

> **Read `HANDOFF.md` first** — it contains all architectural decisions, schemas, code snippets, and integration details needed to build this project.

---

## Stage 1: Foundation (scoring + knowledge base + prompts)
**Goal:** Build the non-voice components that can be tested independently.

- [ ] **1.1 Set up dependencies**
  - Create `requirements.txt` with: `pipecat-ai[google,deepgram,cartesia,silero,daily]`, `python-dotenv`, `loguru`, `pytest`
  - Install into the existing venv at `/home/op/projects/mf_agent/venv`
  - Create `.env` file with placeholder keys (`GOOGLE_API_KEY`, `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY`, `DAILY_API_KEY`)
  - **Acceptance:** `source venv/bin/activate && python -c "import pipecat; print(pipecat.__version__)"` succeeds

- [ ] **1.2 Create `knowledge_base.py`**
  - Load all 6 MD files from `KB/pdftomds/` (Dfly.md, Dpres.md, FCfly.md, FCpres.md, LCfly.md, LCpres.md)
  - `SCHEME_FILES` dict mapping scheme IDs to their file pairs
  - `get_scheme_context(scheme_id: str) -> str` — returns combined flyer + presentation text
  - `get_comparison_table() -> str` — returns the hardcoded comparison table from HANDOFF.md §7
  - `load_all()` — reads all files at startup, stores in module-level dict
  - **Acceptance:** Running `python -c "from knowledge_base import *; load_all(); print(len(get_scheme_context('Flexi_Cap')))"` prints a number > 40000

- [ ] **1.3 Create `scoring.py`**
  - Implement `recommend_mutual_fund(profile: dict) -> dict` exactly as specified in HANDOFF.md §6
  - Return dict with keys: `winner`, `winner_name`, `scores`, `is_tie`, `runner_up`, `runner_up_name`, `profile_summary`
  - **Acceptance:** All tests in Stage 1.4 pass

- [ ] **1.4 Create `tests/test_scoring.py`**
  - Test cases (minimum):
    - Short horizon (2yr) + low risk → Dynamic_Asset_Allocation
    - Long horizon (10yr) + high risk + diversified_global → Flexi_Cap
    - Long horizon (7yr) + moderate risk + stable_bluechip → Large_Cap
    - Investment < ₹5,000 → Dynamic disqualified (score = -999)
    - Near-tie scenario → `is_tie` is True
    - Edge case: horizon=0, risk="moderate", investment=10000 → should not crash
    - Edge case: missing optional fields (`growth_style`, `investment_mode` absent) → should not crash
  - **Acceptance:** `pytest tests/test_scoring.py -v` — all tests pass

- [ ] **1.5 Create `prompts.py`**
  - Define `PROFILING_PROMPT` (constant string) — see HANDOFF.md §7
  - Define `build_recommendation_prompt(fund_id, profile_summary, scheme_context, comparison_table) -> str` — fills template
  - Define `build_tie_breaker_prompt(fund1_id, fund2_id) -> str` — fills template with differentiators
  - Differentiators dict:
    - Dynamic: "actively manages between equity and debt to protect against market drops"
    - Large_Cap: "low-cost, index-like exposure to India's top 100 companies with zero exit load"
    - Flexi_Cap: "invests in Indian AND global companies like Google, with a 13-year proven track record"
  - Inception dates dict: Dynamic="Feb 27, 2024", Large_Cap="Feb 4, 2026", Flexi_Cap="May 24, 2013"
  - Human-readable names dict: Dynamic="Parag Parikh Dynamic Asset Allocation Fund", etc.
  - **Acceptance:** `python -c "from prompts import *; print(build_recommendation_prompt('Flexi_Cap', 'test', 'context', 'table')[:100])"` prints a filled prompt

---

## Stage 2: Text-Mode Bot (test the full flow without voice)
**Goal:** End-to-end conversation in terminal. Uses Gemini for LLM but no STT/TTS.

- [ ] **2.1 Create `text_bot.py`**
  - Uses `google-genai` SDK directly (NOT Pipecat) for simplicity
  - Implements the 3-state machine: PROFILING → (TIE_BREAKER?) → RECOMMENDATION
  - On startup: loads knowledge base, sets initial system prompt to PROFILING_PROMPT
  - Chat loop: reads user input from stdin, sends to Gemini with function calling enabled
  - When Gemini calls `extract_user_profile`: runs scoring engine, transitions state, swaps system prompt
  - Prints bot responses to stdout
  - Requires only `GOOGLE_API_KEY` in `.env`
  - **Acceptance:** Can complete a full conversation:
    1. Agent greets with disclaimer ✓
    2. Agent asks 3-4 questions one at a time ✓
    3. Agent calls extract_user_profile (visible in logs) ✓
    4. Scoring engine runs and picks a fund ✓
    5. Agent announces recommendation with disclaimer ✓
    6. User asks a follow-up question → agent answers from KB ✓
    7. User asks something NOT in KB → agent says "I don't have that detail" ✓

- [ ] **2.2 Test edge cases in text mode**
  - Simulate: user says "I don't know" → agent handles gracefully
  - Simulate: user gives contradictory info ("I want safe but maximum returns") → agent asks clarifier
  - Simulate: tie scenario → agent asks one more question, then resolves
  - Simulate: investment < ₹5,000 → Dynamic gets disqualified
  - Simulate: user asks about Large Cap performance → agent says "too new for performance data"
  - **Acceptance:** All 5 edge cases handled correctly (manual testing log documented)

---

## Stage 3: Voice Pipeline (Pipecat + Deepgram + Cartesia + Daily)
**Goal:** Full voice agent running over Daily WebRTC.

- [ ] **3.1 Create `bot.py`** — the Pipecat voice pipeline
  - Pipeline: `transport.input() → stt → user_aggregator → llm → tts → transport.output() → assistant_aggregator`
  - GoogleLLMService with system_instruction=PROFILING_PROMPT
  - DeepgramSTTService with model="nova-3"
  - CartesiaTTSService with any generic English voice
  - DailyTransport with audio_in/out enabled
  - SileroVADAnalyzer for voice activity detection
  - LLMContext with tools (extract_user_profile schema)
  - Register `handle_extract_user_profile` function:
    - Calls `scoring.recommend_mutual_fund()`
    - If tie: swap system prompt to TIE_BREAKER, return tie info to LLM
    - If winner: swap system prompt to RECOMMENDATION (with KB injected), return result to LLM
  - Use `LLMUpdateSettingsFrame` for system prompt swaps (see HANDOFF.md §4)
  - on_first_participant_joined → queue LLMRunFrame() to make bot speak first
  - on_participant_left → cancel pipeline
  - MetricsLogObserver for basic TTFB logging
  - **Acceptance:** 
    1. `python bot.py` starts without errors
    2. Joining the Daily room → bot greets with disclaimer
    3. Full voice conversation completes the profiling → recommendation flow
    4. Bot speaks the recommendation and answers a follow-up question

- [ ] **3.2 Create a simple launcher**
  - Either a `run.sh` script or a minimal `server.py` that:
    - Creates a Daily room via the REST API
    - Prints the room URL for the user to join
    - Starts the bot
  - **Acceptance:** Single command to start the demo: `./run.sh` or `python server.py`

---

## Stage 4: Polish & Testing
**Goal:** Robustness, edge cases, and demo readiness.

- [ ] **4.1 Handle interruptions**
  - When user speaks while bot is talking, Pipecat auto-cancels TTS
  - Verify the bot recovers gracefully and doesn't lose conversation context
  - **Acceptance:** Interrupt bot mid-sentence → bot stops, processes new input, continues normally

- [ ] **4.2 Add end_call tool**
  - Register an `end_call` function → when user says goodbye, bot says farewell and cancels pipeline
  - **Acceptance:** User says "bye" → bot says farewell → pipeline shuts down

- [ ] **4.3 Validate all scoring paths with voice**
  - Test 3 distinct profiles via voice that each recommend a different fund
  - Test 1 tie-breaker scenario via voice
  - **Acceptance:** All 4 voice tests produce correct recommendations

- [ ] **4.4 Add README.md**
  - Setup instructions (API keys, venv, dependencies)
  - How to run text mode vs voice mode
  - Architecture diagram (text-based)
  - **Acceptance:** A new developer can set up and run the project from the README alone

---

## Build Order & Dependencies

```
Stage 1 (no API keys needed for 1.2-1.5, only for pip install)
  └─ 1.1 → 1.2 → 1.3 → 1.4 → 1.5

Stage 2 (needs GOOGLE_API_KEY only)
  └─ 2.1 → 2.2

Stage 3 (needs all 4 API keys)
  └─ 3.1 → 3.2

Stage 4 (needs all 4 API keys)
  └─ 4.1, 4.2, 4.3 (parallel) → 4.4
```

> [!TIP]
> **Start with Stage 1 and 2.** These can be fully tested with just a Gemini API key and no voice infrastructure. Stage 3 requires Deepgram, Cartesia, and Daily keys.
