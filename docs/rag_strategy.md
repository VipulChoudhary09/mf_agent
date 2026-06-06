# RAG Strategy: PPFAS Mutual Fund Voice Agent

> **Status:** Planning — to be implemented on a feature branch  
> **Last updated:** 2026-06-06  
> **Scope:** MVP demo of RAG for knowledge retrieval, designed to scale to 50+ fund documents

---

## 1. Problem Statement

Currently the bot injects the **entire** flyer + presentation markdown (~7K–12K tokens) for the winning fund into the system prompt. This works for 3 funds × 2 docs, but:

- At **50+ funds × 2-4 docs each**, the KB grows to 300K–600K tokens — can't stuff all possible winner docs
- Even for a single fund, injecting 12K tokens when the user asks "what's the exit load?" wastes ~11K tokens of irrelevant context
- Financial accuracy depends on retrieving the **exact paragraph** with the number, not hoping the LLM picks it from a wall of text

## 2. Architecture: Hybrid Deterministic + RAG

```
┌──────────────────────────────────────────────────────────────┐
│                    Voice Pipeline (Pipecat)                   │
│                                                              │
│  PROFILING state → No KB needed (3-4 questions only)         │
│          ↓                                                   │
│  scoring.py → deterministic winner selection                 │
│          ↓                                                   │
│  RECOMMENDATION state                                        │
│          ↓                                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ RAG Layer (replaces full context injection)             │ │
│  │                                                         │ │
│  │  User question → embed → ChromaDB query (filtered by   │ │
│  │  scheme_id) → top-K chunks → inject into system prompt │ │
│  └─────────────────────────────────────────────────────────┘ │
│          ↓                                                   │
│  LLM responds with grounded answer                           │
└──────────────────────────────────────────────────────────────┘
```

**Key insight:** The scoring engine already selects the fund. RAG doesn't replace fund selection — it replaces "dump the entire fund document" with "retrieve only the paragraphs relevant to the user's question."

### 2.1 Why NOT a generic RAG-over-all-funds approach

In a typical RAG system you'd search across all documents. That's **dangerous** for financial data:

| Risk | Example |
|------|---------|
| **Cross-fund contamination** | User matched to Flexi Cap, asks about exit load. RAG retrieves the Dynamic Fund's "1% within 1 yr" instead of Flexi Cap's "2% ≤1yr, 1% ≤2yr" |
| **Stale data mixing** | Old factsheet chunk mixes with new one, giving contradictory returns |
| **Hallucination amplification** | LLM gets 3 chunks from 3 different funds, synthesizes a wrong answer |

**Solution:** Metadata-filtered retrieval. Every chunk is tagged with `scheme_id`. At query time, we filter to **only** the winner's chunks. The vector search happens within a single fund's document space.

```python
collection.query(
    query_texts=["what is the exit load?"],
    n_results=3,
    where={"scheme_id": "Flexi_Cap"}  # ← hard filter
)
```

---

## 3. Chunking Strategy

Financial documents have **structured sections** (markdown headings). Generic recursive text splitting would break tables and cut numbers in half. Instead:

### 3.1 Section-Aware Chunking

```
Document: FCpres.md
  ├── chunk: "WHY INVEST IN A FLEXI CAP FUND?" (section)
  ├── chunk: "KEY FEATURES" (section)
  ├── chunk: "ASSET ALLOCATION" (section + table)
  ├── chunk: "LUMPSUM & SIP PERFORMANCE" (section + table)
  ├── chunk: "TOP 10 STOCKS" (section + table)
  ├── chunk: "SCHEME FACTS" (section — exit load, min investment, etc.)
  └── chunk: "PRODUCT LABELLING & RISKOMETER" (section)
```

**Rules:**
1. Split on `## ` (H2 headings) — each section becomes a chunk
2. If a section > 1000 tokens, split on `### ` (H3) sub-sections
3. **Never split within a markdown table** — tables are atomic units
4. Prepend each chunk with metadata header: `[Fund: Parag Parikh Flexi Cap Fund | Source: Presentation | Section: SCHEME FACTS]`
5. If a section < 100 tokens, merge with the next section

### 3.2 Metadata per Chunk

```python
{
    "scheme_id": "Flexi_Cap",          # filter key
    "fund_name": "Parag Parikh Flexi Cap Fund",
    "source_type": "presentation",      # "flyer" or "presentation"
    "source_file": "FCpres.md",
    "section_title": "SCHEME FACTS",
    "has_table": True,                  # flag for structured data
    "has_numbers": True,                # flag for financial figures
}
```

### 3.3 Estimated Chunk Counts (current 6 docs)

| Document | Sections | Est. Chunks |
|----------|----------|-------------|
| Dfly.md  | 8 | 8 |
| Dpres.md | ~12 | 14 |
| LCfly.md | ~8 | 9 |
| LCpres.md | ~10 | 12 |
| FCfly.md | ~8 | 9 |
| FCpres.md | ~14 | 16 |
| **Total** | | **~68 chunks** |

At 50 funds × 3 docs × ~12 chunks = **~1,800 chunks** — well within ChromaDB's in-memory capacity.

---

## 4. Technology Choices

### 4.1 Vector Store: ChromaDB (in-memory, persistent)

| Why ChromaDB | Why not alternatives |
|-------------|---------------------|
| Zero infrastructure — runs in-process | Pinecone/Weaviate = external service = latency + cost |
| `PersistentClient` survives restarts | FAISS = no metadata filtering (need scheme_id filter) |
| Native metadata filtering (`where=`) | Vertex AI Vector Search = overkill for <2K chunks |
| First-class Gemini embedding function | |
| Embeds at add-time, no separate step | |

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="ppfas_kb",
    embedding_function=google_ef,
    metadata={"hnsw:space": "cosine"}
)
```

### 4.2 Embedding Model: `gemini-embedding-001`

- **768 dimensions** (MRL — can reduce to save memory)
- **$0.15 / 1M tokens** (free tier for dev)
- Same API key as our LLM (`GOOGLE_API_KEY`)
- `task_type="RETRIEVAL_DOCUMENT"` for indexing, `"RETRIEVAL_QUERY"` for queries

### 4.3 Why NOT a local embedding model

| Consideration | Gemini API | Local (e.g. all-MiniLM) |
|---------------|-----------|------------------------|
| Latency | ~80-120ms API call | ~20ms local inference |
| Quality | State-of-the-art multilingual | Good for English, weak on ₹/Hindi |
| Dependencies | None (API) | PyTorch + model download (~100MB) |
| Voice latency impact | Adds ~100ms to first Q&A turn | Minimal |

**Recommendation:** Use `gemini-embedding-001` for accuracy on financial text (₹ amounts, CAGR percentages, SEBI terminology). The ~100ms embedding call happens once per user question and is parallelizable with other processing.

---

## 5. Retrieval Strategy for Voice Agent

### 5.1 The Voice Latency Problem

In a text chatbot, you retrieve → generate → display. In a voice agent:
```
User stops speaking → VAD triggers → STT finalizes → [RAG HERE] → LLM starts → TTS starts → user hears
```

RAG adds latency **between STT and LLM**. The budget is tight:
- STT TTFB: ~200ms
- **RAG (embed + search): ~100-150ms** ← new
- LLM TTFB: ~400ms
- TTS TTFB: ~100ms

150ms is acceptable. But we must avoid:
- Retrieving on every turn (PROFILING turns don't need KB)
- Multiple retrieval round-trips
- Re-embedding the same question

### 5.2 When to Retrieve

| State | Action |
|-------|--------|
| `PROFILING` | No retrieval — questions only |
| `TIE_BREAKER` | No retrieval — one clarifying question |
| `RECOMMENDATION` — first turn | Inject **fund summary chunk** (always) + **key facts chunk** (always) = 2 pre-fetched chunks. No query embedding needed. |
| `RECOMMENDATION` — follow-up Q&A | Embed user's STT text → query ChromaDB → top-3 chunks → inject |

### 5.3 Retrieval Integration Point

The retrieval happens in a **custom FrameProcessor** in the Pipecat pipeline, sitting between STT and the user_aggregator:

```
transport.input() → stt → [RAGProcessor] → user_aggregator → llm → tts → transport.output()
```

The `RAGProcessor`:
1. Receives `TranscriptionFrame` from STT
2. If state == `RECOMMENDATION`: embeds the transcription, queries ChromaDB
3. Updates the system prompt with retrieved chunks (via `LLMUpdateSettingsFrame`)
4. Passes the frame through

### 5.4 Prompt Template with RAG

```
The system has analyzed the user's profile and matched them with the {fund_name}.

[MANDATORY DISCLAIMER]

RETRIEVED CONTEXT (use ONLY these facts to answer):
---
[Chunk 1: SCHEME FACTS — exit load, min investment...]
[Chunk 2: PERFORMANCE — CAGR, returns table...]
[Chunk 3: ASSET ALLOCATION — equity/debt split...]
---

RULES:
- Answer ONLY from the RETRIEVED CONTEXT above.
- If the information is not in the retrieved context, say: "I don't have that specific detail..."
- NEVER combine information from different chunks to fabricate a number.
```

---

## 6. Accuracy Safeguards for Financial Data

### 6.1 Chunk Integrity

- Tables are **never split** — a performance table is one atomic chunk
- Numbers are always in context with their labels (e.g., "Since Inception CAGR: 18.63%" stays together)
- Each chunk includes its source file and section for traceability

### 6.2 Retrieval Validation

```python
# After retrieval, verify all chunks belong to the correct fund
results = collection.query(
    query_texts=[user_question],
    n_results=5,
    where={"scheme_id": winner_id}
)

# Double-check: reject any chunk that doesn't match
chunks = [
    doc for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    if meta["scheme_id"] == winner_id
]
```

### 6.3 Confidence Threshold

If the best retrieval distance is > 0.6 (low similarity), don't inject the chunk — let the LLM use the base facts (fund name, key differentiators from `prompts.py`) and acknowledge the gap.

### 6.4 Always-Injected Base Facts

Even with RAG, always inject a minimal fact card (hardcoded, ~200 tokens):

```
Fund: Parag Parikh Flexi Cap Fund
Inception: May 24, 2013
Risk: Very High
Benchmark: Nifty 500 TRI
Min Investment: ₹1,000
Exit Load: 2% ≤1yr, 1% ≤2yr, nil after 2yr
```

This ensures the bot can always answer basic questions even if retrieval fails.

---

## 7. File Structure

```
mf_agent/
├── rag/
│   ├── __init__.py
│   ├── chunker.py          # Section-aware markdown chunking
│   ├── embedder.py         # ChromaDB + Gemini embedding setup
│   ├── retriever.py        # Query interface with metadata filtering
│   └── rag_processor.py    # Pipecat FrameProcessor for pipeline integration
├── chroma_db/              # Persistent vector store (gitignored)
├── KB/
│   └── pdftomds/           # Source markdown files
└── scripts/
    └── index_kb.py         # One-time: chunk + embed + store all docs
```

---

## 8. Implementation Stages

### Stage R1: Chunker + Indexer
- [ ] `rag/chunker.py` — section-aware markdown splitter
- [ ] `scripts/index_kb.py` — processes all 6 docs, stores in ChromaDB
- [ ] Unit tests: chunk count, no split tables, metadata correctness

### Stage R2: Retriever
- [ ] `rag/retriever.py` — filtered query interface
- [ ] `rag/embedder.py` — ChromaDB client + Gemini embedding function
- [ ] Unit tests: retrieval accuracy (exit load → SCHEME FACTS section)

### Stage R3: Pipeline Integration
- [ ] `rag/rag_processor.py` — Pipecat FrameProcessor
- [ ] Modify `bot.py` RECOMMENDATION state to use RAG instead of full injection
- [ ] Modify `prompts.py` to use retrieved-context template
- [ ] E2E test: bot correctly answers "what's the exit load?" using retrieved chunk

### Stage R4: Validation + Comparison
- [ ] Run 10 test questions, compare RAG answers vs. full-context answers
- [ ] Measure latency impact (before/after via `measure_latency.py`)
- [ ] Document accuracy/latency tradeoffs

---

## 9. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Retrieval misses relevant chunk | Medium | High — wrong answer | Always-inject base facts + confidence threshold |
| Cross-fund chunk contamination | Low | Critical — wrong fund data | Hard `scheme_id` filter on every query |
| Latency spike on embedding API | Low | Medium — slow first response | Cache embeddings, timeout + fallback to full injection |
| Table split during chunking | Low | High — broken financial data | Atomic table rule in chunker |
| ChromaDB persistence corruption | Low | Medium — need re-index | `index_kb.py` is idempotent, re-run rebuilds |

---

## 10. Future Scaling Path

| Scale Point | Approach |
|-------------|----------|
| **50 funds** | ChromaDB PersistentClient, scheme_id filtering, ~1.8K chunks |
| **200+ funds** | Migrate to Vertex AI Vector Search (managed, auto-scaling) |
| **Multi-language** | `gemini-embedding-001` handles Hindi/English natively |
| **Real-time NAV updates** | Separate "live data" chunks with TTL, re-index nightly |
| **Multi-modal** (fund charts)** | `gemini-embedding-2` supports image embeddings |
