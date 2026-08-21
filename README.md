# PPFAS Mutual Fund LiveKit Agent

A voice advisor that profiles an investor, selects a PPFAS fund through a deterministic scoring engine, and answers follow-up questions using fund-filtered RAG.

The voice stack is now LiveKit-native:

```text
User
  <-> LiveKit WebRTC room
  -> Deepgram Nova-3 multilingual STT
  -> deterministic profile scoring
  -> per-turn ChromaDB RAG filtered by scheme_id
  -> Gemini 2.5 Flash-Lite
  -> Cartesia Sonic 3 TTS
  <-> LiveKit WebRTC room
```

## Architecture

The LLM extracts structured profile fields and explains results. It does **not** choose the fund. `scoring.py` makes that decision from explicit rules, including horizon, risk tolerance, growth style, investment mode, and minimum-investment constraints.

The conversation states are:

```text
PROFILING -> extract_user_profile -> RECOMMENDATION
                    |
                    +-> near tie -> TIE_BREAKER -> extract_user_profile -> RECOMMENDATION
```

In recommendation mode, LiveKit's `on_user_turn_completed` hook retrieves evidence for the latest question before the LLM generates. Preemptive generation is intentionally disabled so a response cannot begin with stale RAG context.

## Supported funds

| Fund | Primary characteristic | Risk level |
|---|---|---|
| Parag Parikh Flexi Cap Fund | Indian and international equity diversification | Very High |
| Parag Parikh Dynamic Asset Allocation Fund | Active equity/debt allocation for downside protection | Moderately High |
| Parag Parikh Large Cap Fund | Low-cost exposure to India's top 100 companies | Very High |

## Setup

Requirements:

- Python 3.10 or newer. Python 3.12 is recommended.
- A LiveKit Cloud project or compatible self-hosted LiveKit deployment.
- API keys for Google Gemini, Deepgram, and Cartesia.

```bash
git checkout feat/livekit-agents-port
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env.local
# Fill .env.local with your LiveKit and provider credentials.
```

Real credentials belong in `.env.local` or the deployment secret store. `.env`, `.env.local`, logs, metrics, and the local vector database are ignored by Git.

## Run locally

LiveKit's conventional entrypoint is `agent.py`.

```bash
# Local terminal session. Useful for a quick microphone test.
python agent.py console

# Connect the worker to the configured LiveKit project.
python agent.py dev
```

`lk agent dev` also works because the LiveKit CLI discovers `agent.py` automatically.

The worker registers under the dispatch name `mf-advisor`. Open the Agents Console for the LiveKit project, select that agent, and start a session. The repository does not include a custom browser frontend; the Agents Console is the test client for this branch.

Legacy commands remain available:

```bash
python bot.py dev
python server.py dev
```

`server.py` is now a compatibility shim. FastAPI, Daily room creation, and the old `/start` endpoint have been removed because LiveKit owns dispatch, rooms, media, and worker health.

## Knowledge-base indexing

With `AUTO_INDEX_KB=true`, the worker checks the local Chroma database during process setup and creates it when missing. A cross-process file lock prevents multiple LiveKit prewarm processes from rebuilding the same index.

To force a rebuild after editing source documents:

```bash
python scripts/index_kb.py
```

Set `AUTO_INDEX_KB=false` to skip startup indexing. When no usable index exists, the advisor falls back to the full winning-fund knowledge base rather than failing the call.

## Tests

Network-free regression tests cover scoring, prompt transitions, tie resolution, RAG refresh, duplicate-query suppression, and full-KB fallback:

```bash
pytest tests/ -v
```

The direct Gemini text harness exercises the shared state machine without LiveKit, STT, or TTS:

```bash
python text_bot.py
```

A provider-level voice smoke test uses:

```bash
python agent.py console
```

That final command consumes provider API usage because it exercises Deepgram, Gemini, and Cartesia.

## Latency and observability

LiveKit Cloud supplies session traces and transcripts through its Agents observability tools. For a small local summary, the worker writes privacy-minimized metrics without transcript text to `livekit_metrics.jsonl`.

```bash
python measure_latency.py
python measure_latency.py --list
python measure_latency.py --room <room-name>
```

The report includes assistant turn count, mean, median, p95, and maximum end-to-end latency.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LIVEKIT_URL` | none | LiveKit WebSocket endpoint |
| `LIVEKIT_API_KEY` | none | LiveKit API key |
| `LIVEKIT_API_SECRET` | none | LiveKit API secret |
| `LIVEKIT_AGENT_NAME` | `mf-advisor` | Explicit dispatch name |
| `DEEPGRAM_API_KEY` | none | Direct Deepgram plugin authentication |
| `GOOGLE_API_KEY` | none | Gemini LLM and embedding authentication |
| `CARTESIA_API_KEY` | none | Direct Cartesia plugin authentication |
| `STT_MODEL` | `nova-3` | Deepgram speech model |
| `STT_LANGUAGE` | `multi` | Multilingual and Hinglish-friendly recognition |
| `LLM_MODEL` | `gemini-2.5-flash-lite` | Google text model |
| `TTS_MODEL` | `sonic-3` | Cartesia speech model |
| `TTS_VOICE` | repository default | Cartesia voice ID |
| `USE_RAG` | `true` | Enable Chroma retrieval |
| `AUTO_INDEX_KB` | `true` | Build missing local index during process setup |
| `DELETE_ROOM_ON_CLOSE` | `true` | Delete the room after the advisor completes the `end_call` flow |
| `LIVEKIT_METRICS_FILE` | `livekit_metrics.jsonl` | Local metrics destination |

## Project structure

```text
agent.py                  LiveKit CLI entrypoint
bot.py                    LiveKit AgentServer and voice session
advisor_core.py           framework-neutral state machine and prompt transitions
startup.py                provider validation, KB loading, safe auto-indexing
server.py                 compatibility entrypoint
text_bot.py               direct Gemini terminal harness
scoring.py                deterministic recommendation engine
prompts.py                state-specific voice prompts
knowledge_base.py         source-document cache
rag/
  chunker.py              structure-aware Markdown splitting
  embedder.py             ChromaDB and Gemini embeddings
  retriever.py            scheme-filtered retrieval
  rag_processor.py        transport-neutral retrieval helper
  context_updater.py      per-turn prompt refresh and stale-context fallback
scripts/index_kb.py       manual index rebuild
tests/                    scoring, retrieval, chunking, and migration regressions
docs/livekit_migration.md architecture decision record
```

## Deployment

The included Dockerfile starts `python agent.py start` as a non-root user and exposes the AgentServer health port. Supply all credentials through the deployment's secret manager. Do not add them to the image, Docker build arguments, or repository variables that are visible in logs.
