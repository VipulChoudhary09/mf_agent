# PPFAS Mutual Fund Voice Agent

A deterministic, Pipecat-based voice AI agent that profiles users through natural conversation and recommends the most suitable PPFAS mutual fund scheme.

## Architecture

```
User ↔ Daily WebRTC ↔ Deepgram STT → Gemini 2.5 Flash → Cartesia TTS → Daily WebRTC
                                           ↓ (function call)
                                    Scoring Engine (Python)
                                           ↓
                                    Knowledge Base (6 MD files)
```

**Hybrid approach**: The LLM handles natural language parsing and explanation, while a deterministic Python scoring engine selects the fund — ensuring consistency, auditability, and compliance.

## Supported Funds

| Fund | Key Characteristic |
|---|---|
| **Flexi Cap** | Global diversification (India + international stocks) |
| **Dynamic Asset Allocation** | Active debt/equity rebalancing for downside protection |
| **Large Cap** | Low-cost passive exposure to India's top 100 companies |

## State Machine

```
PROFILING → extract_user_profile() → RECOMMENDATION
                  ↓ (if tie)
            TIE_BREAKER → extract_user_profile() → RECOMMENDATION
```

1. **PROFILING**: Bot asks about investment horizon, risk tolerance, amount, and growth preferences
2. **TIE_BREAKER**: If two funds score within 1 point, bot asks one clarifying question
3. **RECOMMENDATION**: Bot explains the selected fund using scheme-specific knowledge

## Quick Start

### Prerequisites

- Python 3.12+
- API keys for: [Google AI](https://aistudio.google.com/), [Deepgram](https://deepgram.com/), [Cartesia](https://cartesia.ai/), [Daily](https://daily.co/)

### Setup

```bash
# Clone and enter the project
cd mf_agent

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your keys
```

### Run

```bash
# Launch the voice agent
python run_bot.py
```

This will:
1. Validate all 4 API keys
2. Create a temporary Daily.co room
3. Print a join URL — open it in your browser
4. Start the voice pipeline

### Text Mode (for testing)

```bash
python text_bot.py
```

Terminal-based harness with the same state machine — useful for testing without voice services.

### Tests

```bash
# Unit tests (scoring engine)
pytest tests/test_scoring.py -v

# E2E tests (requires GOOGLE_API_KEY)
pytest tests/test_e2e.py -v
```

## Project Structure

```
mf_agent/
├── bot.py              # Pipecat voice pipeline (Stage 3)
├── run_bot.py           # Launcher — creates Daily room
├── text_bot.py          # Terminal-based text harness (Stage 2)
├── scoring.py           # Deterministic fund selection engine
├── knowledge_base.py    # Loads & caches scheme markdown files
├── prompts.py           # System prompts for each state
├── requirements.txt     # Python dependencies
├── .env                 # API keys (not committed)
├── KB/                  # Knowledge base markdown files
│   ├── Flexi_Cap.md
│   ├── Dynamic_Asset_Allocation.md
│   ├── Large_Cap.md
│   ├── comparison.md
│   ├── general_faq.md
│   └── why_ppfas.md
└── tests/
    ├── test_scoring.py  # 16 unit tests
    └── test_e2e.py      # 4 automated e2e scenarios
```

## Key Design Decisions

- **In-context knowledge injection** (~104KB) instead of RAG — simpler, deterministic, no retrieval latency
- **System prompt swapping** via `LLMUpdateSettingsFrame` for state transitions (Gemini doesn't support mutable system messages in chat)
- **Hard constraints**: Dynamic Asset Allocation requires ₹5,000 minimum investment; exit load penalties for Flexi Cap
- **Near-tie detection**: When top-2 scores are within 1 point, triggers tie-breaker flow instead of potentially wrong recommendation

## Services

| Service | Purpose | Model |
|---|---|---|
| Deepgram | Speech-to-Text | nova-3-general |
| Google Gemini | LLM (conversation + function calling) | gemini-2.5-flash |
| Cartesia | Text-to-Speech | sonic-2 |
| Daily | WebRTC transport | — |
| Silero | Voice Activity Detection (local) | v5 |
