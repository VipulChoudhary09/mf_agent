# Deployment Plan: PPFAS Mutual Fund Voice Agent

> **Status:** Planning — not yet executed.
> **Last updated:** 2026-06-06
> **Author:** Auto-generated, to be reviewed before execution.

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        User (Browser)                        │
│              daily.co room URL → WebRTC client               │
└─────────────────────────┬────────────────────────────────────┘
                          │ WebRTC (audio)
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                      Daily.co SFU                            │
│   (Selective Forwarding Unit — managed by Daily)             │
│   - ICE/STUN/TURN negotiation                                │
│   - Audio routing between participants                       │
└─────────────────────────┬────────────────────────────────────┘
                          │ WebRTC (audio)
                          ▼
┌──────────────────────────────────────────────────────────────┐
│              Bot Process (our deployment target)             │
│                                                              │
│  run_bot.py → bot.py (Pipecat pipeline)                     │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌─────────┐      │
│  │Deepgram │→ │ Gemini  │→ │ Cartesia │→ │  Daily  │      │
│  │  STT    │  │   LLM   │  │   TTS    │  │Transport│      │
│  └─────────┘  └─────────┘  └──────────┘  └─────────┘      │
│                                                              │
│  + SileroVAD (local CPU)                                     │
│  + SmartTurn v3 (local CPU, ~50ms model)                     │
│  + scoring.py (deterministic, <1ms)                          │
│  + knowledge_base.py (6 cached MD files)                     │
└──────────────────────────────────────────────────────────────┘
```

### Key Insight: WebRTC is NOT self-hosted

Daily.co handles all WebRTC complexity (ICE, STUN/TURN, NAT traversal, codec negotiation). Our bot process is a **Daily participant** — it joins the room as a peer via Daily's SDK, not by hosting a WebRTC endpoint. This means:

- **No TURN server needed** on our infra
- **No public UDP ports needed** on our side (Daily SDK uses outbound connections)
- **No STUN/ICE server** configuration
- The bot can run behind a NAT/firewall (including Cloud Run's network)

---

## 2. Deployment Strategy: Containerized vs. Bare Process

### Option A: Docker Container on Cloud Run

| Pros | Cons |
|------|------|
| Immutable deployments | Cold start latency (container spin-up + SmartTurn model load ~3-5s) |
| Secret Manager integration (native) | Requires min-instances=1 to avoid cold starts |
| Auto-scaling to zero when idle | Cloud Run has a 60-min max request timeout |
| No VM management | Stateless — each call gets a fresh container |
| Built-in logging to Cloud Logging | |

### Option B: Docker Container on Compute Engine (VM)

| Pros | Cons |
|------|------|
| Always-on — no cold start | Manual scaling, patching, monitoring |
| No request timeout limits | Paying for idle compute |
| Full OS control | Need to manage systemd, restarts, log rotation |
| Can run SmartTurn model warm | Single point of failure without MIG |

### Option C: Cloud Run with Always-On (min-instances=1)

| Pros | Cons |
|------|------|
| Best of both: no cold starts + managed infra | Higher base cost (~$15-25/mo for 2 vCPU) |
| Secret Manager, Cloud Logging, IAM — native | Still limited to 60-min request timeout |
| Container auto-heals | |
| Can scale up for concurrent calls | |

### ✅ Recommended: Option C (Cloud Run, min-instances=1)

**Rationale:**
- The bot process lifecycle is bounded by call duration (typically 3-10 minutes, max ~30 minutes). Cloud Run's 60-minute timeout is sufficient.
- SmartTurn v3 model loading takes ~2-3s. With `min-instances=1`, the model stays warm.
- Daily.co SDK makes outbound WebRTC connections — no inbound ports needed. Cloud Run's egress-only network model is compatible.
- Secret Manager provides native `.env` variable injection — no file management.
- Auto-scaling handles concurrent calls without infrastructure provisioning.

---

## 3. WebRTC-Specific Deployment Concerns

### 3.1 Daily Transport Network Requirements

Daily's Python SDK (`daily-python`) uses **libdaily** under the hood, which:
- Makes outbound WebSocket connections to Daily's signaling server
- Establishes outbound WebRTC media connections (UDP preferred, TCP fallback)
- Does NOT require inbound port listening

**Cloud Run compatibility:** ✅ Cloud Run allows outbound UDP and TCP. Daily's SDK has been tested in containerized environments and works behind NATs.

### 3.2 Potential Issues & Mitigations

| Issue | Impact | Mitigation |
|-------|--------|------------|
| **Cloud Run CPU throttling** between requests | Audio stuttering if CPU drops to fractional | Set `--cpu-always-allocated` flag on Cloud Run |
| **Cold start + model load** | 3-5s delay before bot can join call | `min-instances=1` + health check that pre-loads model |
| **Container ephemeral filesystem** | `pipecat.log` lost on restart | Stream logs to Cloud Logging; use loguru sink or standard stdout |
| **Concurrent call limit** | Default Cloud Run concurrency=1 is correct for us (1 bot = 1 call) | Explicit `--concurrency=1` per container, scale via instances |
| **UDP restrictions** | Some corporate networks block UDP | Daily handles TCP fallback automatically |
| **Daily room lifecycle** | Rooms expire after `exp` time | `run_bot.py` already sets 1-hour expiry; health check cleans up |

### 3.3 Audio Latency Considerations

End-to-end voice latency chain:
```
User speaks → [~50ms WebRTC] → Daily SFU → [~50ms WebRTC] → Bot receives audio
→ [~100ms VAD] → [~200ms STT TTFB] → [~400ms LLM TTFB] → [~100ms TTS TTFB]
→ [~50ms WebRTC] → Daily SFU → [~50ms WebRTC] → User hears response
```

Total perceived latency: **~900-1200ms** (measured via `measure_latency.py`)

**Deployment should NOT add significant latency because:**
- Daily SFU handles media routing (our container is just another peer)
- GCP `us-central1` co-locates with Daily's US infrastructure
- CPU-always-allocated prevents throttling-induced audio drops

---

## 4. Container Design

```dockerfile
# Dockerfile (planned, not yet created)
FROM python:3.14-slim

WORKDIR /app

# System deps for daily-python (libdaily requires specific libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopus0 libvpx-dev libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects PORT env var
ENV PORT=8080

# Health check endpoint (FastAPI or simple HTTP)
# The bot needs an HTTP trigger to start a session
EXPOSE 8080

CMD ["python", "server.py"]
```

### 4.1 `server.py` (HTTP Wrapper — to be created)

Cloud Run requires an HTTP trigger. The flow:
1. Client hits `POST /start` → `server.py` creates Daily room + spawns bot process
2. Returns `{ room_url, token }` to client
3. Client joins the Daily room in browser
4. Bot process runs until call ends, then exits

This is essentially what `run_bot.py` does, wrapped in FastAPI.

---

## 5. Secrets Management

| Secret | Source | Cloud Run Config |
|--------|--------|-----------------|
| `GOOGLE_API_KEY` | Google AI Studio | Secret Manager → env var |
| `DEEPGRAM_API_KEY` | Deepgram dashboard | Secret Manager → env var |
| `CARTESIA_API_KEY` | Cartesia dashboard | Secret Manager → env var |
| `DAILY_API_KEY` | Daily.co dashboard | Secret Manager → env var |

All secrets injected as env vars via `--set-secrets` flag on `gcloud run deploy`.

---

## 6. Logging & Observability

### Production Logging Strategy

| Layer | Sink | Purpose |
|-------|------|---------|
| Loguru → stdout/stderr | Cloud Logging (automatic) | Centralized log search |
| `pipecat.log` (file) | Local dev only | `measure_latency.py` analysis |
| Cloud Run metrics | Cloud Monitoring | Request count, latency, errors |

In production:
- All loguru output goes to **stdout** (Cloud Run auto-captures to Cloud Logging)
- `pipecat.log` file sink is **disabled** (ephemeral filesystem)
- `measure_latency.py` can be run against Cloud Logging exports if needed

---

## 7. Cost Estimate (per month, low traffic)

| Service | Estimate |
|---------|----------|
| Cloud Run (2 vCPU, 2GB RAM, min-instances=1) | ~$25/mo |
| Secret Manager (4 secrets) | ~$0.02/mo |
| Cloud Logging (ingestion) | ~$0.50/mo |
| **Total GCP** | **~$26/mo** |
| Daily.co (API calls) | Free tier covers ~1000 min/mo |
| Deepgram (STT) | Pay-per-use |
| Google AI (Gemini) | Pay-per-use |
| Cartesia (TTS) | Plan-based |

---

## 8. Staged Execution Plan

> These stages will be executed on a `deployment` branch.

### Stage D1: Containerization
- [ ] Create `Dockerfile`
- [ ] Create `server.py` (FastAPI wrapper)
- [ ] Create `requirements.txt` (pinned versions)
- [ ] Local `docker build` + `docker run` test
- [ ] Verify bot joins Daily room from container

### Stage D2: GCP Foundation
- [ ] Store 4 API keys in Secret Manager
- [ ] Create Artifact Registry repository
- [ ] Push container image to Artifact Registry
- [ ] Initial `gcloud run deploy` with `--set-secrets`

### Stage D3: Production Hardening
- [ ] Enable `--cpu-always-allocated`
- [ ] Set `--min-instances=1`
- [ ] Set `--concurrency=1`
- [ ] Health check endpoint (`GET /health`)
- [ ] Graceful shutdown handling (SIGTERM)

### Stage D4: Observability
- [ ] Cloud Logging filter for `[LATENCY_BREAKDOWN]` and `[TURN]` events
- [ ] Uptime check on `/health`
- [ ] Alert policy for error rate

---

## 9. Open Questions

1. **Custom domain?** — Do we want a domain like `voice.ppfas.com` or is the Cloud Run URL sufficient?
2. **Authentication** — Should the `/start` endpoint require auth (API key, Firebase Auth)?
3. **Region** — Default `us-central1`; should it be `asia-south1` (Mumbai) for Indian users?
4. **Concurrent calls** — How many simultaneous calls do we need to support?
5. **Call recording** — Does Daily need to record calls for compliance?
