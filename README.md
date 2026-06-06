# PPFAS Mutual Fund Voice Agent

A production-grade voice AI agent that profiles investors through natural conversation and recommends the most suitable PPFAS mutual fund scheme, powered by RAG-based knowledge retrieval for accurate follow-up Q&A.

## Architecture

```
                                ┌─────────────────────────────┐
User ↔ Daily WebRTC ↔ Deepgram STT → RAGContextUpdater → Gemini 2.5 Flash → Cartesia TTS → Daily WebRTC
                                │         ↓                        ↓ (function call)        │
                                │   ChromaDB (per-turn)    Scoring Engine (Python)           │
                                │   gemini-embedding-001   ├── deterministic scoring         │
                                │   66 chunks, 3 schemes   └── tie-breaker detection         │
                                │                                                            │
                                │   SmartTurn v3 (local)   ← turn-end detection              │
                                │   Silero VAD v5          ← voice activity detection         │
                                └─────────────────────────────────────────────────────────────┘
```

**Hybrid approach**: The LLM handles natural language understanding and explanation, while a deterministic Python scoring engine selects the fund — ensuring consistency, auditability, and compliance. RAG retrieves relevant KB chunks per question for accurate follow-up answers.

## Supported Funds

| Fund | Key Characteristic | Risk Level |
|---|---|---|
| **Parag Parikh Flexi Cap Fund** | Global diversification (India + international stocks like Google) | Very High |
| **Parag Parikh Dynamic Asset Allocation Fund** | Active debt/equity rebalancing for downside protection | Moderately High |
| **Parag Parikh Large Cap Fund** | Low-cost passive exposure to India's top 100 companies | Very High |

## Conversation State Machine

```
PROFILING → extract_user_profile() → RECOMMENDATION ← (follow-up Q&A with per-turn RAG)
                  ↓ (if tie)
            TIE_BREAKER → extract_user_profile() → RECOMMENDATION
```

1. **PROFILING**: Bot asks about investment horizon, risk tolerance, SIP/lumpsum preference, amount, and growth style
2. **TIE_BREAKER**: If two funds score within 1 point, bot asks one clarifying question
3. **RECOMMENDATION**: Bot explains the selected fund; each follow-up question triggers dynamic RAG retrieval for accurate answers

## Quick Start

### Prerequisites

- Python 3.12+
- API keys for: [Google AI](https://aistudio.google.com/), [Deepgram](https://deepgram.com/), [Cartesia](https://cartesia.ai/), [Daily](https://daily.co/)

### Setup

```bash
cd mf_agent

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your 4 API keys:
#   GOOGLE_API_KEY, DEEPGRAM_API_KEY, CARTESIA_API_KEY, DAILY_API_KEY
```

### Run

```bash
python server.py
```

On startup, the server will:
1. Validate all 4 API keys
2. Auto-index the KB into ChromaDB if needed (first run or empty store)
3. Pre-load the knowledge base into memory
4. Start the FastAPI server on port 7860

Then in another terminal:

```bash
curl -s -X POST http://localhost:7860/start | jq .
```

Returns a Daily room URL — open it in your browser to start the conversation.

#### Alternative modes

| Command | Use case |
|---|---|
| `python text_bot.py` | Terminal text harness — same state machine, no voice/RAG, only needs GOOGLE_API_KEY |
| `python scripts/index_kb.py` | Force re-index KB into ChromaDB (server does this automatically) |

### Measure Latency

```bash
python measure_latency.py
```

Parses `pipecat.log` and displays:
- Per-turn latency breakdown (perceived, VAD, user→bot, LLM TTFB, TTS TTFB)
- River-flow timeline visualization
- Token counts, streaming rates, and cost breakdown (Gemini 2.5 Flash + Cartesia + Deepgram pricing)
- Perceived Latency (VAD breakdown) — shows what % of wait is VAD vs pipeline
- Conversation transcript with timestamps
- Auto-saves transcripts to `transcripts.log`

```bash
# Options
python measure_latency.py -n 1     # previous conversation
python measure_latency.py --list   # list all conversations in log
python measure_latency.py --table  # tabular format
python measure_latency.py --html   # export to HTML
```

### Tests

```bash
# All tests (57)
pytest tests/ -v

# By module
pytest tests/test_scoring.py -v    # 16 scoring engine tests
pytest tests/test_e2e.py -v        # 4 automated e2e scenarios
pytest tests/test_chunker.py -v    # 25 markdown chunker tests
pytest tests/test_retriever.py -v  # 16 retrieval + 10 golden question tests
```

## Project Structure

```
mf_agent/
├── bot.py                  # Pipecat voice pipeline + state machine
├── server.py               # ← single entry point (auto-indexes, loads KB, serves API)
├── text_bot.py             # Terminal text harness (no voice/RAG, testing only)
├── scoring.py              # Deterministic fund selection engine
├── knowledge_base.py       # Loads & caches scheme markdown files
├── prompts.py              # System prompts for each state (profiling/rec/tie-breaker)
├── measure_latency.py      # Latency analyzer and cost calculator
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed)
│
├── rag/                    # RAG pipeline
│   ├── chunker.py          # Section-aware markdown splitter (H2/H3, table integrity)
│   ├── embedder.py         # ChromaDB + gemini-embedding-001 (768d cosine)
│   ├── retriever.py        # Filtered retrieval with 0.6 confidence threshold
│   ├── rag_processor.py    # Stateless RAG helper for initial recommendation
│   └── context_updater.py  # Per-turn FrameProcessor for follow-up Q&A
│
├── scripts/
│   └── index_kb.py         # Batch embed KB → ChromaDB (rate-limit aware)
│
├── KB/                     # Knowledge base source documents
│   └── pdftomds/           # 6 markdown files (flyer + presentation per fund)
│       ├── FCfly.md / FCpres.md    # Flexi Cap
│       ├── Dfly.md / Dpres.md      # Dynamic Asset Allocation
│       └── LCfly.md / LCpres.md    # Large Cap
│
├── tests/                  # 57 tests total
│   ├── test_scoring.py     # Scoring engine unit tests
│   ├── test_e2e.py         # Automated e2e conversation scenarios
│   ├── test_chunker.py     # Markdown chunker tests
│   └── test_retriever.py   # RAG retrieval + golden question tests
│
└── docs/
    └── rag_strategy.md     # RAG architecture planning doc
```

## RAG Pipeline

The agent uses a two-stage RAG approach:

1. **Initial Retrieval** (at recommendation time): A broad query fetches general fund context for the system prompt
2. **Per-Turn Retrieval** (each follow-up question): `RAGContextUpdater` intercepts `LLMContextFrame` in the pipeline, extracts the user's question, and queries ChromaDB with that specific question + scheme_id filter

This means each follow-up gets tailored context:
- *"Who manages this fund?"* → retrieves fund manager chunk
- *"What are the top holdings?"* → retrieves TOP 10 STOCKS chunk
- *"3-year performance?"* → retrieves LUMPSUM & SIP PERFORMANCE chunk

**Storage**: ChromaDB persistent local store (`chroma_db/`), 66 chunks across 3 schemes, embedded with `gemini-embedding-001` (768-dim cosine similarity).

## Key Design Decisions

- **Per-turn RAG** over full-context injection: Reduces system prompt from ~47K chars to ~2-4K chars per turn while maintaining accuracy across diverse follow-up questions
- **Deterministic scoring engine**: The LLM never picks the fund — a Python function with auditable rules does, ensuring compliance and reproducibility
- **System prompt swapping** via `LLMUpdateSettingsFrame` for state transitions (Gemini doesn't support mutable system messages in chat history)
- **SmartTurn v3** for turn-end detection: Uses a local ML model instead of fixed silence timers, reducing false turn-ends mid-sentence
- **Hard constraints**: Dynamic Asset Allocation requires ₹5,000 minimum for lumpsum (SIP exempt); exit load penalties for Flexi Cap under 2 years
- **Near-tie detection**: When top-2 scores are within 1 point, triggers tie-breaker flow instead of a potentially wrong recommendation

## Services & Pricing

| Service | Purpose | Model | Cost |
|---|---|---|---|
| Google Gemini | LLM (conversation + function calling) | gemini-2.5-flash | $0.30/$2.50 per M input/output tokens |
| Deepgram | Speech-to-Text (streaming) | nova-3-general | $0.0077/min |
| Cartesia | Text-to-Speech | sonic-2 | ~$0.03/min (~1 credit/char) |
| Daily | WebRTC transport | — | Free tier available |
| Silero | Voice Activity Detection (local) | v5 | Free (local inference) |
| ChromaDB | Vector store (local) | — | Free (local) |

**Typical session cost**: ~$0.04–0.10 per 3-5 minute conversation.

## Environment Variables

```bash
GOOGLE_API_KEY=       # Google AI Studio API key (Gemini + embeddings)
DEEPGRAM_API_KEY=     # Deepgram STT API key
CARTESIA_API_KEY=     # Cartesia TTS API key
DAILY_API_KEY=        # Daily.co API key (WebRTC rooms)
```
