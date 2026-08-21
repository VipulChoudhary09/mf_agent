"""LiveKit voice agent for the PPFAS mutual-fund advisor.

Pipeline:
    LiveKit WebRTC -> Deepgram STT -> Gemini 2.5 Flash-Lite
    -> Cartesia TTS -> LiveKit WebRTC

The deterministic scorer, state machine, and per-turn RAG logic live in
framework-neutral modules so they can be tested without a room or paid calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Keep process/container environment authoritative. Local files fill only gaps.
load_dotenv(".env.local", override=False)
load_dotenv(".env", override=False)

from livekit.agents import (  # noqa: E402
    Agent,
    AgentServer,
    AgentSession,
    ChatContext,
    ChatMessage,
    ConversationItemAddedEvent,
    JobContext,
    JobProcess,
    RunContext,
    TurnHandlingOptions,
    cli,
    function_tool,
    inference,
)
from livekit.agents.beta.tools import EndCallTool  # noqa: E402
from livekit.plugins import cartesia, deepgram, google  # noqa: E402

from advisor_core import AdvisorFlow, ConversationState  # noqa: E402
from prompts import PROFILING_PROMPT  # noqa: E402
from rag.rag_processor import RAGContextInjector  # noqa: E402
from startup import (  # noqa: E402
    ensure_kb_indexed,
    env_flag,
    load_knowledge_base,
    validate_environment,
)

logger = logging.getLogger("mf-agent")

AGENT_NAME = os.getenv("LIVEKIT_AGENT_NAME", "mf-advisor")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
STT_MODEL = os.getenv("STT_MODEL", "nova-3")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "multi")
STT_ENDPOINTING_MS = int(os.getenv("STT_ENDPOINTING_MS", "150"))
TTS_MODEL = os.getenv("TTS_MODEL", "sonic-3")
TTS_VOICE = os.getenv(
    "TTS_VOICE",
    "79a125e8-cd45-4c13-8a67-188112f4dd22",
)
USE_RAG = env_flag("USE_RAG", True)
AUTO_INDEX_KB = env_flag("AUTO_INDEX_KB", True)
DELETE_ROOM_ON_CLOSE = env_flag("DELETE_ROOM_ON_CLOSE", True)
METRICS_FILE = Path(os.getenv("LIVEKIT_METRICS_FILE", "livekit_metrics.jsonl"))

PROFILE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": "extract_user_profile",
    "description": (
        "Call once the user has provided their investment horizon, risk "
        "tolerance, investment mode, and starting amount."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "horizon_years": {
                "type": "integer",
                "description": "How many years the user plans to stay invested.",
                "minimum": 1,
            },
            "risk_tolerance": {
                "type": "string",
                "enum": ["low", "moderate", "high"],
                "description": "The user's comfort with investment volatility.",
            },
            "growth_style": {
                "type": "string",
                "enum": [
                    "downside_protection",
                    "stable_bluechip",
                    "diversified_global",
                    "unspecified",
                ],
                "description": "The user's preferred portfolio style.",
            },
            "initial_investment": {
                "type": "integer",
                "description": "Starting investment in Indian Rupees.",
                "minimum": 1,
            },
            "investment_mode": {
                "type": "string",
                "enum": ["lumpsum", "sip", "both", "unspecified"],
                "description": "One-time lumpsum, monthly SIP, both, or unknown.",
            },
        },
        "required": [
            "horizon_years",
            "risk_tolerance",
            "initial_investment",
            "investment_mode",
        ],
        "additionalProperties": False,
    },
}


def _json_safe(value: Any) -> Any:
    """Convert SDK metric objects into JSON-safe values without transcripts."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "__dict__"):
        return {
            str(key): _json_safe(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def _append_metric(record: dict[str, Any]) -> None:
    """Append one privacy-minimized metric record for local analysis."""

    try:
        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with METRICS_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        logger.exception("Unable to write local LiveKit metrics")


def _prewarm(proc: JobProcess) -> None:
    """Initialize KB assets once in each isolated LiveKit job process."""

    validate_environment(
        ["GOOGLE_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY"]
    )
    if USE_RAG and AUTO_INDEX_KB:
        ensure_kb_indexed()
    load_knowledge_base()
    proc.userdata["rag_injector"] = RAGContextInjector(
        db_path="./chroma_db",
        n_results=4,
        enabled=USE_RAG,
    )


class MutualFundAdvisor(Agent):
    """LiveKit agent that delegates decisions to the deterministic core."""

    def __init__(self, rag_injector: RAGContextInjector) -> None:
        end_call_tool = EndCallTool(
            extra_description=(
                "Only end the call after the user clearly says they are done "
                "or has no more questions."
            ),
            delete_room=DELETE_ROOM_ON_CLOSE,
            end_instructions=(
                "Briefly thank the user for their time and remind them that "
                "mutual-fund investments are subject to market risks."
            ),
            ignore_on_enter=True,
        )
        super().__init__(
            instructions=PROFILING_PROMPT,
            tools=end_call_tool.tools,
        )
        self._flow = AdvisorFlow(
            rag_injector=rag_injector,
            use_rag=USE_RAG,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                "Greet the user briefly as the PPFAS mutual fund advisor and "
                "ask the first profiling question."
            )
        )

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        del turn_ctx  # The stateful system prompt is refreshed instead.
        if self._flow.state.current != ConversationState.RECOMMENDATION:
            return

        user_text = new_message.text_content.strip()
        if not user_text:
            return

        # Chroma + Gemini embeddings are synchronous. Keep them off LiveKit's
        # event loop so media and interruption handling remain responsive.
        prompt = await asyncio.to_thread(
            self._flow.refresh_rag_prompt,
            user_text,
        )
        if prompt:
            await self.update_instructions(prompt)

    @function_tool(raw_schema=PROFILE_TOOL_SCHEMA)
    async def extract_user_profile(
        self,
        raw_arguments: dict[str, object],
        context: RunContext,
    ) -> dict[str, Any]:
        """Run deterministic scoring after the required profile is collected."""

        del context
        profile = dict(raw_arguments)
        logger.info("Scoring extracted profile fields: %s", sorted(profile))

        transition = await asyncio.to_thread(self._flow.handle_profile, profile)
        await self.update_instructions(transition.prompt)

        logger.info(
            "Advisor state changed to %s; context=%s",
            transition.state.value,
            transition.tool_result.get("context_mode", "n/a"),
        )
        return transition.tool_result


server = AgentServer(initialize_process_timeout=300)
server.setup_fnc = _prewarm


@server.rtc_session(agent_name=AGENT_NAME)
async def mf_advisor_session(ctx: JobContext) -> None:
    """Create one isolated voice session for a dispatched LiveKit room."""

    ctx.log_context_fields = {"room": ctx.room.name, "agent": AGENT_NAME}
    rag_injector: RAGContextInjector = ctx.proc.userdata["rag_injector"]

    session = AgentSession(
        stt=deepgram.STT(
            model=STT_MODEL,
            language=STT_LANGUAGE,
            endpointing_ms=STT_ENDPOINTING_MS,
            smart_format=True,
            filler_words=True,
            keyterm=[
                "PPFAS",
                "Parag Parikh",
                "Flexi Cap",
                "Large Cap",
                "Dynamic Asset Allocation",
                "SIP",
                "lumpsum",
            ],
        ),
        llm=google.LLM(
            model=LLM_MODEL,
            temperature=0.3,
            max_output_tokens=512,
        ),
        tts=cartesia.TTS(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            language="en",
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            interruption={"mode": "adaptive"},
            # RAG must complete before generation; speculative generation can
            # otherwise start with stale evidence from the previous question.
            preemptive_generation={"enabled": False},
        ),
    )

    @session.on("conversation_item_added")
    def on_conversation_item_added(event: ConversationItemAddedEvent) -> None:
        item = event.item
        if not isinstance(item, ChatMessage) or not item.metrics:
            return

        metrics = _json_safe(item.metrics)
        record = {
            "type": "turn_metrics",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "room": ctx.room.name,
            "role": item.role,
            "metrics": metrics,
        }
        _append_metric(record)

        if item.role == "assistant" and isinstance(metrics, Mapping):
            e2e = metrics.get("e2e_latency")
            if isinstance(e2e, (int, float)):
                logger.info("End-to-end response latency: %.3fs", e2e)

    async def on_shutdown() -> None:
        _append_metric(
            {
                "type": "session_usage",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "room": ctx.room.name,
                "usage": _json_safe(session.usage),
            }
        )

    ctx.add_shutdown_callback(on_shutdown)

    await session.start(
        agent=MutualFundAdvisor(rag_injector),
        room=ctx.room,
    )
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
