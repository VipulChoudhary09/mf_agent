# Pipecat to LiveKit migration

## Decision summary

The transport and orchestration layer has moved from Pipecat plus Daily to the current LiveKit Agents server model. The financial-domain logic remains framework-neutral.

| Concern | Previous design | LiveKit design |
|---|---|---|
| Media transport | Daily WebRTC room created by FastAPI | LiveKit room and RTC session dispatch |
| Worker lifecycle | One in-process task per `/start` call | One isolated LiveKit job process per room |
| STT | Pipecat Deepgram service | LiveKit Deepgram plugin, Nova-3 multilingual |
| LLM | Pipecat Google service | LiveKit Google plugin, Gemini 2.5 Flash-Lite |
| TTS | Pipecat Cartesia service | LiveKit Cartesia plugin, Sonic 3 |
| Turn completion | Pipecat SmartTurn plus Silero | LiveKit audio turn detector and adaptive interruption |
| RAG integration | Pipecat `FrameProcessor` | `Agent.on_user_turn_completed` hook |
| Health endpoint | Custom FastAPI `/health` | AgentServer health endpoint |
| Call creation | Custom Daily `/start` endpoint | LiveKit dispatch and Agents Console |

## Important implementation choices

### RAG blocks generation

Preemptive generation is disabled. In recommendation mode, the current user question is embedded and filtered by `scheme_id` before the LLM starts. This avoids a response beginning with stale evidence from the previous turn.

### Framework-neutral state machine

`advisor_core.py` owns scoring, state transitions, prompt selection, and the full-KB fallback. LiveKit tools delegate to it. The terminal harness uses the same state machine, so transport changes cannot silently fork recommendation behavior.

### Safe process initialization

LiveKit runs sessions in isolated job processes. `startup.py` loads the KB during process setup and protects first-run Chroma indexing with a cross-process file lock. Only one process pays the embedding/indexing cost; others wait for the completed index.

For a larger deployment, move the vector store to a managed shared database rather than relying on a container-local Chroma directory.

### Named agent dispatch

The worker registers as `mf-advisor` by default. A named agent uses explicit dispatch, including selection from LiveKit's Agents Console. Override the name with `LIVEKIT_AGENT_NAME` when multiple deployments share a project.

### Secrets

No credential is present in source, examples, Docker layers, logs, or metrics. Runtime credentials come from `.env.local` for development or the deployment's secret store.

## Compatibility entrypoints

- `agent.py` is the conventional LiveKit CLI entrypoint.
- `bot.py` contains the implementation and remains import-compatible with the old repository.
- `server.py` remains as a shim, but no longer starts FastAPI or creates Daily rooms.
