"""
Pipecat Voice Pipeline — PPFAS Mutual Fund Advisor Bot

Pipeline: Daily WebRTC → Deepgram STT → Gemini LLM → Cartesia TTS → Daily WebRTC

Run with: python run_bot.py (which creates a Daily room and launches this)
Requires all 4 API keys in .env:
  GOOGLE_API_KEY, DEEPGRAM_API_KEY, CARTESIA_API_KEY, DAILY_API_KEY
"""

import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from loguru import logger

# ── Pipecat: function calling schemas ───────────────────────
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

# ── Pipecat: VAD (Voice Activity Detection) ─────────────────
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

# ── Pipecat: frame types ─────────────────────────────────────
from pipecat.frames.frames import LLMRunFrame, LLMUpdateSettingsFrame

# ── Pipecat: pipeline + runner ───────────────────────────────
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

# ── Pipecat: observers ────────────────────────────────────────
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver

# ── Pipecat: context management ─────────────────────────────
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)

# ── Pipecat: AI services ─────────────────────────────────────
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.llm_service import FunctionCallParams

# ── Pipecat: Daily transport ─────────────────────────────────
from pipecat.transports.daily.transport import DailyParams, DailyTransport

# ── Local modules ────────────────────────────────────────────
from knowledge_base import load_all, get_scheme_context, get_comparison_table
from scoring import recommend_mutual_fund
from prompts import (
    PROFILING_PROMPT,
    FUND_NAMES,
    build_recommendation_prompt,
    build_tie_breaker_prompt,
)

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_MODEL = "gemini-2.5-flash"
STT_MODEL = "nova-3-general"
STT_ENDPOINTING = 150
TTS_MODEL = "sonic-2"
TTS_VOICE = "79a125e8-cd45-4c13-8a67-188112f4dd22"  # British Lady

VAD_PARAMS = VADParams(
    start_secs=0.2,
    stop_secs=0.3,
    confidence=0.8,
    min_volume=0.3,
)

# ---------------------------------------------------------------------------
# Function Calling Schema (Pipecat format — matches HANDOFF.md §5)
# ---------------------------------------------------------------------------

tools = ToolsSchema(
    standard_tools=[
        FunctionSchema(
            name="extract_user_profile",
            description=(
                "Call this tool once the user has provided enough information about "
                "their investment timeframe, risk appetite, and starting amount."
            ),
            properties={
                "horizon_years": {
                    "type": "integer",
                    "description": "The number of years the user plans to stay invested.",
                },
                "risk_tolerance": {
                    "type": "string",
                    "enum": ["low", "moderate", "high"],
                    "description": (
                        "Low = fears losing money / wants capital protection. "
                        "Moderate = accepts some drops for growth. "
                        "High = comfortable with significant volatility."
                    ),
                },
                "growth_style": {
                    "type": "string",
                    "enum": [
                        "downside_protection",
                        "stable_bluechip",
                        "diversified_global",
                        "unspecified",
                    ],
                    "description": (
                        "downside_protection = wants active debt/equity management to "
                        "reduce losses. stable_bluechip = wants low-cost exposure to "
                        "India's top 100 companies. diversified_global = wants exposure "
                        "to Indian AND international stocks like Google/Alphabet."
                    ),
                },
                "initial_investment": {
                    "type": "integer",
                    "description": "The initial amount in INR the user wants to invest.",
                },
                "investment_mode": {
                    "type": "string",
                    "enum": ["lumpsum", "sip", "both", "unspecified"],
                    "description": (
                        "Whether the user wants to invest a one-time lump sum, "
                        "set up a monthly SIP, or both."
                    ),
                },
            },
            required=["horizon_years", "risk_tolerance", "initial_investment"],
        ),
    ]
)


# ---------------------------------------------------------------------------
# State Machine
# ---------------------------------------------------------------------------

class BotState:
    """Mutable state container shared by the function handler and event handlers."""

    def __init__(self):
        self.current = "profiling"  # profiling → tie_breaker → recommendation


# ---------------------------------------------------------------------------
# Main Bot Function
# ---------------------------------------------------------------------------

async def run_bot(room_url: str, token: str):
    """Build and run the Pipecat voice pipeline."""

    logger.info(f"Bot starting — joining room: {room_url}")

    # Load knowledge base at startup
    logger.info("Loading knowledge base...")
    load_all()
    logger.info("Knowledge base loaded ✓")

    state = BotState()

    # ─────────────────────────────────────────
    # STEP 1: TRANSPORT
    # ─────────────────────────────────────────
    transport = DailyTransport(
        room_url,
        token,
        "MF Advisor",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    # ─────────────────────────────────────────
    # STEP 2: STT — Speech-to-Text (Deepgram)
    # ─────────────────────────────────────────
    stt = DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        model=STT_MODEL,
        endpointing=STT_ENDPOINTING,
    )

    # ─────────────────────────────────────────
    # STEP 3: LLM — Google Gemini
    # ─────────────────────────────────────────
    llm = GoogleLLMService(
        api_key=os.environ["GOOGLE_API_KEY"],
        settings=GoogleLLMService.Settings(
            model=LLM_MODEL,
            system_instruction=PROFILING_PROMPT,
        ),
    )

    # ─────────────────────────────────────────
    # STEP 4: TTS — Cartesia
    # ─────────────────────────────────────────
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice=TTS_VOICE,
            model=TTS_MODEL,
        ),
    )

    # ─────────────────────────────────────────
    # STEP 5: FUNCTION CALLING — extract_user_profile
    # ─────────────────────────────────────────

    async def handle_extract_user_profile(params: FunctionCallParams):
        """Pipecat function handler for extract_user_profile.

        Runs the scoring engine, transitions state, and updates the
        system prompt via LLMUpdateSettingsFrame.
        """
        args = params.arguments
        logger.info(f"🔧 Function call: extract_user_profile({json.dumps(args)})")

        # Run deterministic scoring engine
        result = recommend_mutual_fund(args)
        logger.info(
            f"📊 Scores: {result['scores']} | "
            f"Winner: {result['winner']} | Tie: {result['is_tie']}"
        )

        if result["is_tie"] and state.current == "profiling":
            # --- Transition to TIE_BREAKER ---
            state.current = "tie_breaker"
            new_prompt = build_tie_breaker_prompt(
                result["winner"], result["runner_up"]
            )
            response_data = {
                "status": "tie_breaker_needed",
                "fund1": FUND_NAMES[result["winner"]],
                "fund2": FUND_NAMES[result["runner_up"]],
                "message": (
                    f"The user's profile fits well with both "
                    f"{FUND_NAMES[result['winner']]} and "
                    f"{FUND_NAMES[result['runner_up']]}. "
                    f"Ask ONE clarifying question to help decide."
                ),
            }
            logger.info(
                f"🔀 State → TIE_BREAKER "
                f"({result['winner']} vs {result['runner_up']})"
            )
        else:
            # --- Transition to RECOMMENDATION ---
            winner_id = result["winner"]
            state.current = "recommendation"
            scheme_context = get_scheme_context(winner_id)
            comparison_table = get_comparison_table()
            new_prompt = build_recommendation_prompt(
                winner_id,
                result["profile_summary"],
                scheme_context,
                comparison_table,
            )
            response_data = {
                "status": "recommendation_ready",
                "recommended_fund": FUND_NAMES[winner_id],
                "profile_summary": result["profile_summary"],
                "scores": result["scores"],
            }
            logger.info(f"✅ State → RECOMMENDATION ({winner_id})")

        # Update system prompt dynamically
        await task.queue_frame(
            LLMUpdateSettingsFrame(
                settings={"system_instruction": new_prompt}
            )
        )

        # Return function result to the LLM
        await params.result_callback(response_data)

    llm.register_function("extract_user_profile", handle_extract_user_profile)

    # ─────────────────────────────────────────
    # STEP 6: CONTEXT + AGGREGATORS
    # ─────────────────────────────────────────
    context = LLMContext(tools=tools)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VAD_PARAMS),
        ),
    )

    # ─────────────────────────────────────────
    # STEP 7: PIPELINE
    # ─────────────────────────────────────────
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    # ─────────────────────────────────────────
    # STEP 8: PIPELINE TASK
    # ─────────────────────────────────────────
    task = PipelineTask(
        pipeline,
        observers=[MetricsLogObserver()],
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # ─────────────────────────────────────────
    # STEP 9: EVENT HANDLERS
    # ─────────────────────────────────────────
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info(f"Participant joined: {participant.get('user_name', 'unknown')}")
        # Small delay for WebRTC audio track to stabilise
        await asyncio.sleep(0.5)
        # Bot speaks first — trigger the greeting
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info("Participant left. Shutting down bot.")
        await task.cancel()

    # ─────────────────────────────────────────
    # STEP 10: RUN
    # ─────────────────────────────────────────
    runner = PipelineRunner()
    await runner.run(task)

    logger.info("Bot session ended.")
