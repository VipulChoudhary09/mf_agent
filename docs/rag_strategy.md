# RAG strategy: PPFAS Mutual Fund LiveKit agent

> **Status:** Implemented  
> **Transport:** LiveKit Agents  
> **Scope:** Fund-filtered retrieval for recommendation and follow-up questions

## 1. Why RAG is separate from recommendation

The deterministic scorer chooses one fund from the user's horizon, risk tolerance, investment mode, amount, and growth preference. Retrieval never chooses the fund. It only supplies facts for explaining the selected fund and answering follow-up questions.

This division is intentional:

- `scoring.py` is deterministic, auditable, and covered by unit tests.
- ChromaDB retrieves evidence only after a winner exists.
- Gemini handles conversation and explanation, not portfolio selection.

## 2. Runtime path

```text
LiveKit audio
  -> Deepgram Nova-3 transcript
  -> LiveKit end-of-turn detector
  -> Agent.on_user_turn_completed
  -> Gemini embedding + Chroma query filtered by scheme_id
  -> refreshed grounded instructions
  -> Gemini 2.5 Flash-Lite
  -> Cartesia Sonic 3
  -> LiveKit audio
```

Retrieval is skipped during `PROFILING` and `TIE_BREAKER`. It activates only in `RECOMMENDATION`.

## 3. Two-stage retrieval

### Initial recommendation

After `extract_user_profile` runs, the agent issues a broad query for overview, key features, exit load, performance, and investment details. The result grounds the first recommendation explanation.

### Follow-up questions

For every new recommendation-state question, `on_user_turn_completed` passes the final transcript to the framework-neutral `RAGContextUpdater`. The updater:

1. Rejects empty, duplicate, or out-of-state queries.
2. Queries ChromaDB with the winning `scheme_id` as a hard metadata filter.
3. Rebuilds the recommendation prompt with the fresh chunks.
4. Returns the full-KB prompt if retrieval misses, preventing stale chunks from the previous question from remaining active.

## 4. Accuracy safeguards

- Every chunk carries a `scheme_id`; cross-fund retrieval is rejected by the query filter.
- Markdown tables stay atomic during chunking so labels and financial values remain together.
- The prompt prohibits inventing or estimating figures.
- Performance answers must name the relevant time period.
- If neither retrieved nor full KB context contains the requested fact, the agent directs the user to the official factsheet or a qualified financial advisor.

## 5. Latency policy

The RAG call is synchronous internally, so LiveKit runs it through `asyncio.to_thread` to avoid blocking media handling. Preemptive LLM generation is disabled because generation must not begin before current-turn evidence is available.

Per-turn `e2e_latency` is recorded without transcripts in `livekit_metrics.jsonl`. Run `python measure_latency.py` for mean, median, p95, and maximum response latency by room.

## 6. Storage and scaling

The current implementation uses a local persistent Chroma database for a small corpus. First-run indexing is guarded by a cross-process file lock because LiveKit prewarms isolated job processes.

Recommended scaling path:

- Small demo: local ChromaDB and runtime indexing.
- Repeated container deployment: build or restore an index artifact during deployment.
- Multiple replicas or a much larger fund universe: shared managed vector storage with versioned documents and freshness metadata.
