#!/usr/bin/env python3
"""
Parses pipecat.log and reports per-turn latency metrics for each conversation.

Default mode: river-flow timeline showing USER and BOT tracks on a shared time axis.
Use --table for the original tabular format.

Usage:
  python3 measure_latency.py           # river flow (latest conversation)
  python3 measure_latency.py --table   # original tabular format
  python3 measure_latency.py -n 1      # one conversation back
  python3 measure_latency.py --list    # list all rooms in log
"""

import ast
import os
import re
import shutil
import sys
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

DEFAULT_LOG = "pipecat.log"

# ── Pricing constants ─────────────────────────────────────────────────────────
# Verify these at provider dashboards before using for billing decisions.
# OpenAI gpt-4.1 (as of June 2026): https://openai.com/api/pricing
#   Input:        $2.00 / 1M tokens
#   Cached input: $0.50 / 1M tokens  (automatic prompt caching, 128-token chunks)
#   Output:       $8.00 / 1M tokens
# Cartesia: charges per second of audio generated (subscription-based credits)
#   Pro $4/mo → 133 min  → ~$0.000501/sec
#   Scale $239/mo → 10,667 min → ~$0.000373/sec
#   Update this constant to match your plan.
_OAI_GPT41_IN_USD_PER_TOK      = 2.00 / 1_000_000
_OAI_GPT41_CACHED_USD_PER_TOK  = 0.50 / 1_000_000
_OAI_GPT41_OUT_USD_PER_TOK     = 8.00 / 1_000_000
_CARTESIA_USD_PER_AUDIO_SEC    = 0.000430   # mid-point estimate; adjust to your plan
# Deepgram nova-3 streaming (verify at https://deepgram.com/pricing). Billed per
# second of audio streamed over the connection — i.e. roughly the whole call, not
# per utterance — so STT is reported only as a session total from call duration.
_DEEPGRAM_USD_PER_AUDIO_SEC    = 0.0077 / 60   # ≈ $0.000128/sec; adjust to your plan

# ── ANSI color codes ─────────────────────────────────────────────────────────
_RESET     = "\033[0m"
_DIM       = "\033[2m"
_GREEN     = "\033[32m"   # user speaking
_BLUE      = "\033[34m"   # bot speaking
_YELLOW    = "\033[33m"   # LLM phase
_CYAN      = "\033[36m"   # Agg phase
_MAGENTA   = "\033[35m"   # TTS phase
_DIM_YEL   = "\033[2;33m" # VAD+TurnAnalyzer silence window
_RED       = "\033[31m"   # errors

# ── Timeline characters ───────────────────────────────────────────────────────
_USER_SPK  = "▓"   # user speaking
_VAD_CH    = "·"   # VAD silence + TurnAnalyzer window
_BOT_SPK   = "█"   # bot speaking
_LLM_CH    = "░"   # LLM TTFB phase
_AGG_CH    = "▒"   # TxtAgg phase
_TTS_CH    = "▓"   # TTS TTFB phase
_IDLE      = "─"   # silence / idle


def _use_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


# ── HTML color map (white-background palette) ─────────────────────────────────
_HTML: dict[str, str] = {
    _GREEN:   "#15803d",
    _BLUE:    "#1d4ed8",
    _YELLOW:  "#b45309",
    _CYAN:    "#0e7490",
    _MAGENTA: "#7c3aed",
    _DIM_YEL: "#a16207",
    _DIM:     "#9ca3af",
    _RED:     "#dc2626",
}


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _hspan(text: str, ansi: str) -> str:
    css = _HTML.get(ansi, "")
    return f'<span style="color:{css}">{_esc(text)}</span>' if css else _esc(text)


def _colorize_row_html(row: list, color_spans: list) -> str:
    pos_color: dict[int, str] = {}
    for x0, x1, code in color_spans:
        css = _HTML.get(code)
        if css:
            for x in range(max(0, x0), min(len(row), x1 + 1)):
                pos_color[x] = css
    parts: list[str] = []
    cur: Optional[str] = None
    for x, ch in enumerate(row):
        c = pos_color.get(x)
        if c != cur:
            if cur is not None:
                parts.append("</span>")
            if c is not None:
                parts.append(f'<span style="color:{c}">')
            cur = c
        parts.append(_esc(ch))
    if cur is not None:
        parts.append("</span>")
    return "".join(parts)


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class Turn:
    num: int
    user_stop_time: datetime
    user_start_time:      Optional[datetime] = None
    bot_start_time:       Optional[datetime] = None
    bot_stop_time:        Optional[datetime] = None
    llm_ttfb:             Optional[float] = None
    llm_http_ttfb:        Optional[float] = None   # httpx hook: HTTP request sent → response headers
    tts_ttfb:             Optional[float] = None
    stt_ttfb:             Optional[float] = None
    user_turn_secs:       Optional[float] = None   # VAD stop → turn release (observer)
    llm_prompt_tokens:    Optional[int]   = None
    llm_completion_tokens: Optional[int]  = None
    tts_chars:            int = 0                  # total chars sent to Cartesia
    llm_cache_read_tokens: Optional[int]  = None   # OpenAI prompt cache hits
    llm_error:            Optional[str]   = None   # LLM exception message if any
    user_text:            Optional[str]   = None   # STT transcript of user utterance
    user_transcript_time: Optional[datetime] = None  # wall-clock when Deepgram returned final transcript
    bot_text:             Optional[str]   = None   # bot response text
    llm_called:           bool = False
    interrupted:          bool = False
    turn_complete:        Optional[bool]  = None  # TurnAnalyzer final decision
    turn_probability:     Optional[float] = None  # confidence score (0–1)
    turn_analyzer_ms:     Optional[float] = None  # e2e processing time
    llm_processing_secs:  Optional[float] = None  # full LLM streaming duration (TTFB → last token)
    txt_agg_direct:       Optional[float] = None  # from CartesiaTTSService text aggregation time log
    has_breakdown:        bool = False            # a [LATENCY_BREAKDOWN] line was seen for this turn
    # Each (start, stop) is one contiguous span of bot audio. Tool-call turns
    # produce several spans (pre-tool reply, post-tool reply) separated by silent
    # gaps; summing the spans excludes those gaps from "bot speaking time".
    bot_speech_segments:  list = field(default_factory=list)
    # Per-token streaming stats reconstructed from the "LLM GENERATING: '<tok>' at <t>s"
    # log lines (the `at` value is pipecat's own monotonic clock). Aggregated across
    # invocations within the turn; only intra-invocation gaps are counted so the silent
    # tool-call gap never inflates the rate.
    llm_stream_ttfb:      Optional[float] = None  # first GENERATING token − START RESPONSE (first invocation)
    llm_stream_span:      float = 0.0             # Σ (last_token − first_token) per invocation
    llm_stream_gaps:      int   = 0               # Σ (token_count − 1) per invocation
    # One dict per function call: {name, call_id, call_ts, result_ts}. call→result
    # gives tool-execution duration (result frame present in the richer recent logs).
    tool_calls:           list = field(default_factory=list)
    # One entry per LLM→TTS cycle in the turn (tool-call turns have 2+), anchored on
    # real log wall-clock times so each cycle draws at its true position. Keys:
    # first_token, llm_ttfb, end (🧠 logs only), tts_frame, tts_ttfb.
    llm_invocations:      list = field(default_factory=list)

    @property
    def tool_names(self) -> list:
        return [c["name"] for c in self.tool_calls]

    @property
    def user_bot_latency(self) -> Optional[float]:
        if self.bot_start_time is not None:
            return (self.bot_start_time - self.user_stop_time).total_seconds()
        return None

    @property
    def perceived(self) -> Optional[float]:
        """Physical voice stop → bot start.

        = user_turn_secs (silence→turn-release, measured in-process by the
        UserBotLatencyObserver) + user_bot_latency (turn-release→bot start).
        Returns None when no observer breakdown exists for the turn — we never
        substitute the VAD stop_secs config constant, which is a threshold, not
        a measurement.
        """
        ub = self.user_bot_latency
        if self.user_turn_secs is not None and ub is not None:
            return self.user_turn_secs + ub
        return None

    @property
    def txt_agg(self) -> Optional[float]:
        if self.txt_agg_direct is not None:
            return self.txt_agg_direct
        ub = self.user_bot_latency
        if ub is not None and self.llm_ttfb is not None and self.tts_ttfb is not None:
            return max(0.0, ub - self.llm_ttfb - self.tts_ttfb)
        return None

    @property
    def stt_call_secs(self) -> Optional[float]:
        """Total STT API time per turn: first speech frame → final transcript.

        Unlike stt_ttfb (first word latency), this captures the full round-trip
        across the whole utterance. Available when both user_start_time and
        user_transcript_time were captured from the log.
        """
        if self.user_start_time and self.user_transcript_time:
            return (self.user_transcript_time - self.user_start_time).total_seconds()
        return None

    @property
    def llm_call_secs(self) -> Optional[float]:
        """Total LLM API call duration: request sent → last token received (= llm_processing_secs)."""
        return self.llm_processing_secs

    @property
    def user_speech_secs(self) -> Optional[float]:
        if self.user_start_time and self.user_stop_time:
            return (self.user_stop_time - self.user_start_time).total_seconds()
        return None

    @property
    def bot_speech_secs(self) -> Optional[float]:
        # Sum of contiguous speech spans — excludes silent gaps during tool calls.
        if self.bot_speech_segments:
            return sum((b - a).total_seconds() for a, b in self.bot_speech_segments)
        if self.bot_start_time and self.bot_stop_time:
            return (self.bot_stop_time - self.bot_start_time).total_seconds()
        return None

    @property
    def llm_stream_tok_per_sec(self) -> Optional[float]:
        """Pure inter-token streaming throughput (excludes TTFB and TTS/agg).

        Measured straight from the gaps between GENERATING tokens, so it is a
        cleaner generation rate than completion_tokens / (user→bot − TTFB).
        """
        if self.llm_stream_span > 0 and self.llm_stream_gaps > 0:
            return self.llm_stream_gaps / self.llm_stream_span
        return None

    @property
    def tts_chars_per_sec(self) -> Optional[float]:
        bspk = self.bot_speech_secs
        if self.tts_chars > 0 and bspk and bspk > 0:
            return self.tts_chars / bspk
        return None

    @property
    def llm_input_cost_usd(self) -> Optional[float]:
        """Uncached prompt tokens (full input rate)."""
        if not self.llm_prompt_tokens:
            return None
        uncached = max(0, self.llm_prompt_tokens - (self.llm_cache_read_tokens or 0))
        return uncached * _OAI_GPT41_IN_USD_PER_TOK

    @property
    def llm_cache_cost_usd(self) -> Optional[float]:
        """Cached prompt tokens (discounted input rate)."""
        if not self.llm_cache_read_tokens:
            return None
        return self.llm_cache_read_tokens * _OAI_GPT41_CACHED_USD_PER_TOK

    @property
    def llm_output_cost_usd(self) -> Optional[float]:
        """Completion tokens."""
        if not self.llm_completion_tokens:
            return None
        return self.llm_completion_tokens * _OAI_GPT41_OUT_USD_PER_TOK

    @property
    def llm_cost_usd(self) -> Optional[float]:
        if self.llm_prompt_tokens is None and self.llm_completion_tokens is None:
            return None
        return ((self.llm_input_cost_usd or 0.0)
                + (self.llm_cache_cost_usd or 0.0)
                + (self.llm_output_cost_usd or 0.0))

    @property
    def tts_cost_usd(self) -> Optional[float]:
        secs = self.bot_speech_secs
        return (secs * _CARTESIA_USD_PER_AUDIO_SEC) if secs else None

    @property
    def total_cost_usd(self) -> Optional[float]:
        llm = self.llm_cost_usd
        tts = self.tts_cost_usd
        if llm is None and tts is None:
            return None
        return (llm or 0.0) + (tts or 0.0)

    @property
    def note(self) -> str:
        if self.interrupted:
            return "interrupted"
        if self.llm_error:
            return "error"
        if not self.llm_called:
            return "no inference"
        if self.bot_start_time is None:
            return "no response"
        return ""


@dataclass
class Conversation:
    room_id: str
    start_time: Optional[datetime]
    vad_stop_secs: float = 0.0
    turns: list = field(default_factory=list)
    llm_services: set = field(default_factory=set)  # e.g. {"OpenAILLMService", "GroqLLMService"}
    stt_ws_connect_secs: Optional[float] = None   # Deepgram WebSocket handshake time
    tts_ws_connect_secs: Optional[float] = None   # Cartesia WebSocket handshake time

    @property
    def llm_provider_label(self) -> str:
        if not self.llm_services:
            return "unknown"
        return "+".join(sorted(self.llm_services))

    @property
    def mixed_providers(self) -> bool:
        return len(self.llm_services) > 1


# ── Log parser ───────────────────────────────────────────────────────────────

def parse_ts(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return None


def _parse_context(msg: str) -> Optional[list]:
    """Extract message list from 'Generating chat from context [...]' log line."""
    m = re.search(r'Generating chat from context (\[.+)', msg)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(1))
    except Exception:
        return None


def parse_log(log_file: str) -> list[Conversation]:
    conversations: list[Conversation] = []
    conv: Optional[Conversation] = None
    active_turn: Optional[Turn] = None
    active_bot_turn: Optional[Turn] = None
    prev_turn: Optional[Turn] = None    # for back-filling bot_text
    user_speaking = False
    pending_stt: Optional[float] = None
    pending_user_start: Optional[datetime] = None
    pending_turn_metrics: Optional[dict] = None
    pending_inference: bool = False     # "inference triggered" fires before "stopped speaking"
    pending_user_text: Optional[str] = None
    pending_user_text_time: Optional[datetime] = None
    active_llm_turn: Optional[Turn] = None   # turn currently being generated
    llm_seg_buf: list = []               # tokens for the current LLM invocation
    llm_time_buf: list = []              # per-token `at <t>s` stamps for the current invocation
    llm_start_at: Optional[float] = None # `at <t>s` of the current LLM START RESPONSE
    bot_seg_start: Optional[datetime] = None  # open bot-speech span (start with no stop yet)

    with open(log_file) as f:
        lines = f.readlines()

    for line in lines:
        parts = line.split(" | ", 2)
        if len(parts) < 3:
            continue
        ts = parse_ts(parts[0])
        if ts is None:
            continue
        level = parts[1].strip()
        msg   = parts[2].strip()

        # ── New conversation boundary ──────────────────────────────────────
        m = re.search(r'Bot starting — joining room: .+/([^\s/]+)\s*$', msg)
        if m:
            conv = Conversation(room_id=m.group(1), start_time=ts)
            conversations.append(conv)
            active_turn = active_bot_turn = prev_turn = None
            user_speaking = False
            pending_stt = pending_user_start = pending_turn_metrics = None
            pending_user_text = pending_user_text_time = None
            pending_inference = False
            active_llm_turn = None
            llm_seg_buf = []
            llm_time_buf = []
            llm_start_at = None
            bot_seg_start = None
            continue

        if conv is None:
            continue

        # ── Track which LLM provider(s) this conversation used ─────────────
        # Surfaced in the header and used to flag config-blended sessions.
        mp = re.search(r'([A-Za-z]+LLMService)#\d', msg)
        if mp:
            conv.llm_services.add(mp.group(1))

        # ── VAD params ────────────────────────────────────────────────────
        # Matches both pipecat native "Setting VAD params" and our [SESSION] vad line
        if conv.vad_stop_secs == 0.0 and ("Setting VAD params" in msg or "[SESSION] vad" in msg):
            m2 = re.search(r'stop_secs=([\d.]+)', msg)
            if m2:
                conv.vad_stop_secs = float(m2.group(1))
            continue

        # ── [HTTP_CALL] connection / request timing ───────────────────────
        # Emitted by TimedOpenAILLMService (per turn) and TimedCartesiaTTSService /
        # TimedDeepgramSTTService (once per session on WebSocket connect).
        #   [HTTP_CALL] service=llm http_ttfb=1.234s status=200   ← per turn
        #   [HTTP_CALL] service=tts ws_connect=0.123s             ← session
        #   [HTTP_CALL] service=stt ws_connect=0.456s             ← session
        if "[HTTP_CALL]" in msg:
            svc_m = re.search(r'service=(\w+)', msg)
            svc = svc_m.group(1) if svc_m else ""
            if svc == "llm":
                m2 = re.search(r'http_ttfb=([\d.]+)s', msg)
                if m2:
                    target = active_turn or active_bot_turn
                    if target is not None and target.llm_http_ttfb is None:
                        target.llm_http_ttfb = float(m2.group(1))
            elif svc == "tts":
                m2 = re.search(r'ws_connect=([\d.]+)s', msg)
                if m2:
                    conv.tts_ws_connect_secs = float(m2.group(1))
            elif svc == "stt":
                m2 = re.search(r'ws_connect=([\d.]+)s', msg)
                if m2:
                    conv.stt_ws_connect_secs = float(m2.group(1))
            continue

        # ── User started speaking ─────────────────────────────────────────
        if "LLMUserAggregator" in msg and "User started speaking" in msg:
            user_speaking = True
            pending_stt = None
            pending_user_start = ts
            continue

        # ── Deepgram final transcript (TranscriptionLogObserver) ──────────
        # Fires before "User stopped speaking"; buffer and apply at Turn creation.
        # "INTERIM TRANSCRIPTION" lines are skipped — only final results.
        if "💬" in msg and "→ TRANSCRIPTION:" in msg and "INTERIM" not in msg:
            m2 = re.search(r"TRANSCRIPTION: '(.+?)' from", msg)
            if m2:
                pending_user_text = m2.group(1)
                pending_user_text_time = ts
            continue

        # ── Inference triggered ───────────────────────────────────────────
        # Fires BEFORE "User stopped speaking" so active_turn doesn't exist yet.
        # Buffer in pending_inference and apply it when the Turn is created.
        if "LLMUserAggregator" in msg and "User turn inference triggered" in msg:
            pending_inference = True
            continue

        # ── STT TTFB during speech window ─────────────────────────────────
        if user_speaking and "DeepgramSTTService" in msg and "TTFB:" in msg:
            m2 = re.search(r'TTFB: ([-\d.]+)s', msg)
            if m2:
                pending_stt = float(m2.group(1))
            continue

        # ── User stopped speaking → start latency clock ───────────────────
        if "LLMUserAggregator" in msg and "User stopped speaking" in msg:
            user_speaking = False
            if active_turn is not None and active_turn.bot_start_time is None:
                active_turn.interrupted = active_turn.llm_called
            # prev_turn = the last turn before this new one, regardless of whether
            # active_turn or active_bot_turn holds it right now
            prev_turn   = conv.turns[-1] if conv.turns else None
            tm = pending_turn_metrics
            active_turn = Turn(
                num=len(conv.turns) + 1,
                user_stop_time=ts,
                user_start_time=pending_user_start,
                stt_ttfb=pending_stt,
                llm_called=pending_inference,
                user_text=pending_user_text,
                user_transcript_time=pending_user_text_time,
                turn_complete=tm['complete'] if tm else None,
                turn_probability=tm['prob'] if tm else None,
                turn_analyzer_ms=tm['e2e_ms'] if tm else None,
            )
            conv.turns.append(active_turn)
            pending_stt = pending_user_start = pending_turn_metrics = None
            pending_user_text = pending_user_text_time = None
            pending_inference = False
            continue

        # ── LLM context dump → user_text + prev turn bot_text ─────────────
        if "Generating chat from context" in msg:
            target = active_turn or active_bot_turn
            messages = _parse_context(msg)
            if messages and target is not None:
                # Fallback user_text from context dump if TranscriptionLogObserver didn't fire.
                if messages[-1].get("role") == "user" and target.user_text is None:
                    target.user_text = messages[-1].get("content", "")
            continue

        # ── SmartTurn/TurnAnalyzer decision (rolling — keep last before turn) ──
        # Format 1 (custom observer, legacy): [TURN_METRICS] COMPLETE prob=0.85 e2e=12.3ms
        # Format 2 (MetricsLogObserver):      📊 [TurnAnalyzer] TURN...: COMPLETE (probability: 85.00%, e2e: 12.3ms)
        if "[TURN_METRICS]" in msg:
            m2 = re.search(r'(COMPLETE|INCOMPLETE)\s+prob=([\d.]+)\s+e2e=([\d.]+)ms', msg)
            if m2:
                pending_turn_metrics = {
                    'complete': m2.group(1) == 'COMPLETE',
                    'prob': float(m2.group(2)),
                    'e2e_ms': float(m2.group(3)),
                }
            continue
        if "TURN" in msg and ("COMPLETE" in msg or "INCOMPLETE" in msg) and "probability:" in msg:
            m2 = re.search(r'(COMPLETE|INCOMPLETE).*?probability: ([\d.]+)%.*?e2e: ([\d.]+)ms', msg)
            if m2:
                pending_turn_metrics = {
                    'complete': m2.group(1) == 'COMPLETE',
                    'prob': float(m2.group(2)) / 100.0,
                    'e2e_ms': float(m2.group(3)),
                }
            continue
        # Format 3 (pipecat TurnAnalyzer native): "End of Turn result: EndOfTurnState.COMPLETE"
        if "EndOfTurnState." in msg:
            m2 = re.search(r'EndOfTurnState\.(\w+)', msg)
            if m2:
                is_complete = m2.group(1) == 'COMPLETE'
                if pending_turn_metrics is None:
                    pending_turn_metrics = {'complete': is_complete, 'prob': None, 'e2e_ms': None}
                else:
                    pending_turn_metrics['complete'] = is_complete
            continue

        # ── [LATENCY_BREAKDOWN] from UserBotLatencyObserver ───────────────
        # This is the bot's OWN in-process breakdown — the closest analog to the
        # OTEL `turn.user_bot_latency` span attribute. Prefer every field here
        # over the values recomputed from individual service log lines, because
        # these are measured inside the pipeline rather than from log wall-clock.
        # Format: [LATENCY_BREAKDOWN] user_turn_secs=0.610 ttfb_OpenAILLMService#2=1.972
        #         ttfb_DeepgramSTTService#2=0.604 ttfb_CartesiaTTSService#2=0.436 text_agg=0.000
        # user_turn_secs may be "n/a" (e.g. greeting before any user turn) — left None.
        if "[LATENCY_BREAKDOWN]" in msg:
            target = active_turn or active_bot_turn
            if target is not None:
                target.has_breakdown = True
                m2 = re.search(r'user_turn_secs=([\d.]+)', msg)
                if m2:
                    target.user_turn_secs = float(m2.group(1))
                for svc, val in re.findall(r'ttfb_([A-Za-z]+)(?:#\d+)?=([\d.]+)', msg):
                    v = float(val)
                    # Order matters: "DeepgramSTTService" also contains "TTS" as a
                    # substring, so STT must be tested before TTS.
                    if "LLM" in svc:
                        target.llm_ttfb = v
                    elif "Deepgram" in svc or "STT" in svc:
                        target.stt_ttfb = v
                    elif "Cartesia" in svc or "TTS" in svc:
                        target.tts_ttfb = v
                m3 = re.search(r'text_agg=([\d.]+)', msg)
                if m3:
                    target.txt_agg_direct = float(m3.group(1))
            continue

        # ── Function / tool call ──────────────────────────────────────────
        # pipecat: "...Calling function [book_appointment:call_abc] with arguments {...}"
        # Provider-independent (OpenAI/Groq); fires once per call.
        if "Calling function [" in msg:
            target = active_turn or active_bot_turn
            if target is not None:
                m2 = re.search(r'Calling function \[([A-Za-z_]+):([^\]]+)\]', msg)
                if m2:
                    target.tool_calls.append({
                        "name": m2.group(1), "call_id": m2.group(2),
                        "call_ts": ts, "result_ts": None,
                    })
            continue

        # ── Function result (richer recent logs) → tool-execution duration ─
        # "...FunctionCallResultFrame: [book_appointment:call_abc]" — match by call_id.
        if "FunctionCallResultFrame" in msg:
            target = active_turn or active_bot_turn
            m2 = re.search(r'FunctionCallResultFrame: \[[A-Za-z_]+:([^\]]+)\]', msg)
            if target is not None and m2:
                cid = m2.group(1)
                for c in target.tool_calls:
                    if c["call_id"] == cid and c["result_ts"] is None:
                        c["result_ts"] = ts
                        break
            continue

        # ── LLM error ────────────────────────────────────────────────────
        if level == "ERROR" and "LLMService" in msg and "exception" in msg:
            target = active_turn or active_bot_turn
            if target is not None and target.llm_error is None:
                # Extract the exception message after the last ': '
                m2 = re.search(r'exception[^:]*: (.+)$', msg)
                if m2:
                    err = m2.group(1).strip()
                    # Trim to first sentence to keep it short
                    target.llm_error = err.split(".")[0].strip()
            continue

        if active_turn is None and active_bot_turn is None:
            continue

        # ── STT TTFB (late) ───────────────────────────────────────────────
        if (
            active_turn is not None
            and active_turn.stt_ttfb is None
            and "DeepgramSTTService" in msg
            and "TTFB:" in msg
        ):
            m2 = re.search(r'TTFB: ([-\d.]+)s', msg)
            if m2:
                active_turn.stt_ttfb = float(m2.group(1))
            continue

        # ── LLM TTFB ─────────────────────────────────────────────────────
        # Each LLMService TTFB log marks one invocation's first token. Record it
        # for EVERY cycle (tool-call turns have 2+), not just the first, so each
        # cycle can be drawn at its real position. The log fires at first-token, so
        # the request started ttfb seconds earlier.
        if "LLMService" in msg and "TTFB:" in msg:
            target = active_turn or active_bot_turn
            m2 = re.search(r'TTFB: ([-\d.]+)s', msg)
            if target is not None and m2:
                val = float(m2.group(1))
                target.llm_invocations.append({"first_token": ts, "llm_ttfb": val})
                if active_turn is not None and active_turn.llm_ttfb is None:
                    active_turn.llm_ttfb = val
            continue

        # ── TTS TTFB ──────────────────────────────────────────────────────
        # Attach to the most recent invocation still missing its TTS frame.
        if "CartesiaTTSService" in msg and "TTFB:" in msg:
            target = active_turn or active_bot_turn
            m2 = re.search(r'TTFB: ([-\d.]+)s', msg)
            if target is not None and m2:
                val = float(m2.group(1))
                for iv in reversed(target.llm_invocations):
                    if iv.get("tts_frame") is None:
                        iv["tts_frame"] = ts
                        iv["tts_ttfb"] = val
                        break
                if active_turn is not None and active_turn.tts_ttfb is None:
                    active_turn.tts_ttfb = val
            continue

        # ── TTS text aggregation time (direct measurement — prefer over derived) ─
        if (
            active_turn is not None
            and active_turn.txt_agg_direct is None
            and "CartesiaTTSService" in msg
            and "text aggregation time:" in msg
        ):
            m2 = re.search(r'text aggregation time: ([\d.]+)', msg)
            if m2:
                active_turn.txt_agg_direct = float(m2.group(1))
            continue

        # ── LLM token usage ───────────────────────────────────────────────
        # A tool-call turn fires the LLM 2+ times (decide tool, then narrate the
        # result), so ACCUMULATE across calls rather than keeping only the first.
        # Each call logs usage twice — the native "prompt tokens:" line and the
        # observer "LLM TOKEN USAGE" line — so take prompt/completion from the
        # native line only (always emitted) and cache_read from the observer line
        # only (the native line omits it); otherwise the two would double-count.
        if "prompt tokens:" in msg and "completion tokens:" in msg:
            target = active_turn or active_bot_turn
            if target is not None:
                m2 = re.search(r'prompt tokens: (\d+), completion tokens: (\d+)', msg)
                if m2:
                    target.llm_prompt_tokens     = (target.llm_prompt_tokens or 0) + int(m2.group(1))
                    target.llm_completion_tokens = (target.llm_completion_tokens or 0) + int(m2.group(2))
            continue
        if "LLM TOKEN USAGE" in msg and "prompt:" in msg:
            target = active_turn or active_bot_turn
            if target is not None:
                m3 = re.search(r'cache_read: (\d+)', msg)
                if m3:
                    target.llm_cache_read_tokens = (target.llm_cache_read_tokens or 0) + int(m3.group(1))
            continue

        # ── TTS character usage ───────────────────────────────────────────
        # Use only the native pipecat log (fires per TTS call).
        # MetricsLogObserver emits the same values — skip it to avoid 2× count.
        if "usage characters:" in msg and "CartesiaTTSService" in msg:
            target = active_turn or active_bot_turn
            if target is not None:
                m2 = re.search(r'usage characters: (\d+)', msg)
                if m2:
                    target.tts_chars += int(m2.group(1))
            continue

        # ── LLM full processing duration (TTFB → last token) ─────────────
        if "LLMService" in msg and "processing time:" in msg:
            target = active_turn or active_bot_turn
            if target is not None and target.llm_processing_secs is None:
                m2 = re.search(r'processing time: ([\d.]+)s', msg)
                if m2:
                    target.llm_processing_secs = float(m2.group(1))
            continue

        # ── LLM generation (LLMLogObserver) ──────────────────────────────
        # START RESPONSE: new invocation begins — attach to current active turn.
        if "🧠" in msg and "LLM START RESPONSE" in msg:
            active_llm_turn = active_turn or active_bot_turn
            llm_seg_buf = []
            llm_time_buf = []
            st_m = re.search(r' at ([\d.]+)s', msg)
            llm_start_at = float(st_m.group(1)) if st_m else None
            continue

        # GENERATING: accumulate tokens (repr-formatted by pipecat — use ast.literal_eval)
        # and the per-token `at <t>s` stamp (pipecat's own monotonic clock).
        if "🧠" in msg and "LLM GENERATING:" in msg and active_llm_turn is not None:
            idx = msg.find("LLM GENERATING: ")
            if idx != -1:
                rest = msg[idx + len("LLM GENERATING: "):]
                ts_m = re.search(r' at ([\d.]+)s$', rest)
                if ts_m:
                    token_repr = rest[:ts_m.start()]
                    try:
                        llm_seg_buf.append(ast.literal_eval(token_repr))
                        llm_time_buf.append(float(ts_m.group(1)))
                    except Exception:
                        pass
            continue

        # END RESPONSE: finalize this invocation's segment.
        # For tool-call turns there are two invocations — concatenate both so
        # the transcript shows the full spoken response (pre-tool + post-tool).
        if "🧠" in msg and "LLM END RESPONSE" in msg and active_llm_turn is not None:
            if llm_seg_buf:
                segment = "".join(llm_seg_buf)
                if active_llm_turn.bot_text is None:
                    active_llm_turn.bot_text = segment
                else:
                    active_llm_turn.bot_text += " " + segment
            # Fold this invocation's token timing into the turn's streaming stats.
            if llm_time_buf:
                if active_llm_turn.llm_stream_ttfb is None and llm_start_at is not None:
                    active_llm_turn.llm_stream_ttfb = max(0.0, llm_time_buf[0] - llm_start_at)
                if len(llm_time_buf) >= 2:
                    active_llm_turn.llm_stream_span += llm_time_buf[-1] - llm_time_buf[0]
                    active_llm_turn.llm_stream_gaps += len(llm_time_buf) - 1
            # Mark the generation end on the matching invocation (enables the gen
            # phase to be drawn precisely rather than estimated from the TTS handoff).
            for iv in reversed(active_llm_turn.llm_invocations):
                if iv.get("first_token") is not None and iv.get("end") is None:
                    iv["end"] = ts
                    break
            llm_seg_buf = []
            llm_time_buf = []
            llm_start_at = None
            continue

        # ── Bot started speaking → stop latency clock ─────────────────────
        if "Bot started speaking" in msg:
            if active_turn is not None and active_turn.bot_start_time is None:
                # First speech span of a new turn.
                active_turn.bot_start_time = ts
                active_bot_turn = active_turn
                active_turn = None
                bot_seg_start = ts
            elif active_bot_turn is not None:
                # A later span of the SAME turn (e.g. post-tool-call reply).
                bot_seg_start = ts
            continue

        # ── Bot stopped speaking ──────────────────────────────────────────
        # Tool calls produce multiple start/stop pairs per turn (pre-tool reply +
        # post-tool reply). bot_stop_time tracks the LAST stop (for timeline span),
        # while each (start, stop) span is recorded so bot_speech_secs can sum only
        # the spoken spans and exclude the silent function-execution gaps between them.
        if "Bot stopped speaking" in msg and active_bot_turn is not None:
            active_bot_turn.bot_stop_time = ts
            if bot_seg_start is not None:
                active_bot_turn.bot_speech_segments.append((bot_seg_start, ts))
                bot_seg_start = None
            continue

    return conversations


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fmt(val: Optional[float], width: int = 9) -> str:
    if val is None:
        return "   n/a  ".rjust(width)
    return f"{val:7.3f}s".rjust(width)


def _pipe_phases(turn, px):
    """PIPE-row segments for a turn: list of (x0, x1, char, color).

    One LLM→TTS cycle per invocation (tool-call turns have 2+), each anchored on
    real log wall-clock times so the post-tool cycle lands in the gap between bot
    segments instead of leaving the row empty:
      ░ LLM TTFB  = request → first token   (request = first_token − ttfb)
      ▒ txt agg   = first token → TTS request  (the breakdown's txt_agg slice;
                    LLM generation overlaps this window — see metrics row LLM gen)
      ▓ TTS TTFB  = TTS request → first audio frame
    These three tile the user→bot window exactly (LLM TTFB + txt agg + TTS TTFB).
    Falls back to laying the turn's metric values end-to-end from turn release when
    no per-invocation data exists (older logs without service TTFB lines).
    """
    out = []

    def seg(t0, t1, ch, col):
        x0 = px(t0)
        x1 = max(x0, px(t1) - 1)
        out.append((x0, x1, ch, col))

    invs = [iv for iv in turn.llm_invocations
            if iv.get("first_token") is not None and iv.get("llm_ttfb") is not None]
    if invs:
        for iv in invs:
            ft = iv["first_token"]
            seg(ft - timedelta(seconds=max(0.0, iv["llm_ttfb"])), ft, _LLM_CH, _YELLOW)
            tts_frame, tts_ttfb = iv.get("tts_frame"), iv.get("tts_ttfb")
            tts_req = (tts_frame - timedelta(seconds=max(0.0, tts_ttfb))
                       if tts_frame is not None and tts_ttfb is not None else None)
            # Middle = the txt_agg slice that tiles to the TTS request; fall back to
            # the LLM END time only when there is no TTS cycle (e.g. end_call).
            agg_end = tts_req or iv.get("end")
            if agg_end is not None and agg_end > ft:
                seg(ft, agg_end, _AGG_CH, _CYAN)
            if tts_frame is not None and tts_req is not None:
                seg(tts_req, tts_frame, _TTS_CH, _MAGENTA)
        return out

    # Fallback: no per-invocation data — lay metric values end-to-end from release.
    if turn.user_stop_time is None:
        return out
    t_cur = turn.user_stop_time
    if turn.llm_ttfb and turn.llm_ttfb > 0:
        nxt = t_cur + timedelta(seconds=turn.llm_ttfb); seg(t_cur, nxt, _LLM_CH, _YELLOW); t_cur = nxt
    if turn.txt_agg and turn.txt_agg > 0.005:
        nxt = t_cur + timedelta(seconds=turn.txt_agg); seg(t_cur, nxt, _AGG_CH, _CYAN); t_cur = nxt
    if turn.tts_ttfb and turn.tts_ttfb > 0:
        nxt = t_cur + timedelta(seconds=turn.tts_ttfb); seg(t_cur, nxt, _TTS_CH, _MAGENTA)
    return out


def _avg(vals: list[float]) -> Optional[float]:
    return sum(vals) / len(vals) if vals else None


def _fmt_tool_calls(tool_calls: list) -> str:
    """'book_appointment (4.4s) → end_call' — appends exec time when a result frame
    was logged (richer recent logs); name only otherwise."""
    parts = []
    for c in tool_calls:
        if c.get("result_ts") is not None and c.get("call_ts") is not None:
            parts.append(f"{c['name']} ({(c['result_ts'] - c['call_ts']).total_seconds():.3f}s)")
        else:
            parts.append(c["name"])
    return " → ".join(parts)


def _turn_label(n: int) -> str:
    if n <= 20:
        return chr(0x2460 + n - 1)
    return f"[{n}]"


def _trunc(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else s[:max_len - 1] + "…"


# ── Tabular display ───────────────────────────────────────────────────────────

def display(conv: Conversation, label: str) -> None:
    stop = conv.vad_stop_secs
    stop_label = f"{stop:.2f}s" if stop else "unknown"

    print(f"\n{'═' * 72}")
    print(f"  Room       : {conv.room_id}  ({label})")
    if conv.start_time:
        print(f"  Started    : {conv.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  VAD stop   : {stop_label}  (silence wait before turn fires)")
    print(f"  Turns      : {len(conv.turns)}")
    print(f"{'═' * 72}")

    if not conv.turns:
        print("  No user turns detected.\n")
        return

    hdr = (
        f"  {'#':>4}  {'Perceived':>10}  {'User→Bot':>9}"
        f"  {'LLM TTFB':>9}  {'TxtAgg':>8}  {'TTS TTFB':>9}  {'STT TTFB':>9}  {'Note'}"
    )
    sep = (
        f"  {'─'*4}  {'─'*10}  {'─'*9}"
        f"  {'─'*9}  {'─'*8}  {'─'*9}  {'─'*9}  {'─'*12}"
    )
    print(f"\n{hdr}")
    print(sep)

    perc_vals, ub_vals, llm_vals, agg_vals, tts_vals, stt_vals = [], [], [], [], [], []

    n_missing = 0
    for t in conv.turns:
        ub  = t.user_bot_latency
        agg = t.txt_agg
        perceived = t.perceived
        if perceived is None and ub is not None:
            n_missing += 1

        print(
            f"  {t.num:>4}  {_fmt(perceived, 10):>10}  {_fmt(ub):>9}"
            f"  {_fmt(t.llm_ttfb):>9}  {_fmt(agg, 8):>8}  {_fmt(t.tts_ttfb):>9}  {_fmt(t.stt_ttfb):>9}"
            + (f"  {t.note}" if t.note else "")
        )
        if perceived is not None: perc_vals.append(perceived)
        if ub  is not None: ub_vals.append(ub)
        if t.llm_ttfb is not None: llm_vals.append(t.llm_ttfb)
        if agg is not None: agg_vals.append(agg)
        if t.tts_ttfb is not None: tts_vals.append(t.tts_ttfb)
        if t.stt_ttfb is not None: stt_vals.append(t.stt_ttfb)

    print(sep)
    print(
        f"  {'avg':>4}  {_fmt(_avg(perc_vals), 10):>10}  {_fmt(_avg(ub_vals)):>9}"
        f"  {_fmt(_avg(llm_vals)):>9}  {_fmt(_avg(agg_vals), 8):>8}"
        f"  {_fmt(_avg(tts_vals)):>9}  {_fmt(_avg(stt_vals)):>9}"
    )
    print(f"\n  Perceived = user_turn_secs (observer) + User→Bot   [physical voice stop → bot start]")
    print(f"  User→Bot  = turn release → bot start  (= LLM TTFB + TxtAgg + TTS TTFB)")
    print(f"  STT TTFB  = diagnostic only (transcript ready before latency clock starts)")
    if n_missing:
        print(f"  note: {n_missing} turn(s) show perceived n/a — no observer breakdown logged "
              f"(VAD stop {stop_label} is NOT substituted)")
    print()


# ── River-flow timeline display ───────────────────────────────────────────────

def _build_char_row(width: int, segments: list[tuple[int, int, str]]) -> list[str]:
    row = [_IDLE] * width
    for x0, x1, ch in segments:
        for x in range(max(0, x0), min(width, x1 + 1)):
            row[x] = ch
    return row


def _colorize_row(
    row: list[str],
    color_spans: list[tuple[int, int, str]],
    use_color: bool,
) -> str:
    if not use_color:
        return "".join(row)

    pos_color: dict[int, str] = {}
    for x0, x1, code in color_spans:
        for x in range(max(0, x0), min(len(row), x1 + 1)):
            pos_color[x] = code

    parts: list[str] = []
    cur: Optional[str] = None
    for x, ch in enumerate(row):
        c = pos_color.get(x)
        if c != cur:
            if cur is not None:
                parts.append(_RESET)
            if c is not None:
                parts.append(c)
            cur = c
        parts.append(ch)
    if cur is not None:
        parts.append(_RESET)
    return "".join(parts)


def _tick_interval(total_secs: float, plot_w: int) -> int:
    for iv in (1, 2, 5, 10, 15, 30, 60, 120):
        if total_secs / iv <= plot_w / 7:
            return iv
    return 120


def display_timeline(conv: Conversation, label: str, width: int = None) -> None:
    color = _use_color()
    if width is None:
        width = min(shutil.get_terminal_size().columns, 160)

    LABEL_W = 6
    plot_w = width - LABEL_W

    # ── Time range ───────────────────────────────────────────────────────
    all_times = []
    for t in conv.turns:
        for dt in (t.user_start_time, t.user_stop_time, t.bot_start_time, t.bot_stop_time):
            if dt is not None:
                all_times.append(dt)

    if len(all_times) < 2:
        display(conv, label)
        return

    t_start = min(all_times)
    t_end   = max(all_times)
    total_secs = (t_end - t_start).total_seconds()
    if total_secs < 0.1:
        display(conv, label)
        return

    cps = plot_w / total_secs

    def px(dt: datetime) -> int:
        return max(0, min(plot_w - 1, int((dt - t_start).total_seconds() * cps)))

    # ── USER row ─────────────────────────────────────────────────────────
    user_segs:   list[tuple[int, int, str]] = []
    user_colors: list[tuple[int, int, str]] = []

    for turn in conv.turns:
        if not (turn.user_start_time and turn.user_stop_time):
            continue
        if turn.user_turn_secs is not None:
            physical_stop = turn.user_stop_time - timedelta(seconds=turn.user_turn_secs)
            x0, x1, x2 = px(turn.user_start_time), px(physical_stop), px(turn.user_stop_time)
            if x1 >= x0:
                user_segs.append((x0, x1, _USER_SPK))
                user_colors.append((x0, x1, _GREEN))
            if x2 > x1:
                user_segs.append((x1 + 1, x2, _VAD_CH))
                user_colors.append((x1 + 1, x2, _DIM_YEL))
        else:
            x0, x1 = px(turn.user_start_time), px(turn.user_stop_time)
            user_segs.append((x0, x1, _USER_SPK))
            user_colors.append((x0, x1, _GREEN))

    # ── PIPE row (LLM→Agg→TTS = user→bot gap) + BOT row (speech only) ─────
    pipe_segs:  list[tuple[int, int, str]] = []
    pipe_colors: list[tuple[int, int, str]] = []
    bot_segs:   list[tuple[int, int, str]] = []
    bot_colors: list[tuple[int, int, str]] = []

    for turn in conv.turns:
        if turn.user_stop_time is None:
            continue

        for x0, x1, ch, col in _pipe_phases(turn, px):
            pipe_segs.append((x0, x1, ch))
            pipe_colors.append((x0, x1, col))

        # Draw each contiguous speech span separately so silent tool-call gaps
        # render as idle (─) rather than solid speech — consistent with the
        # gap-excluding "bot speaking time" metric.
        if turn.bot_speech_segments:
            for seg_start, seg_stop in turn.bot_speech_segments:
                bot_segs.append((px(seg_start), px(seg_stop), _BOT_SPK))
                bot_colors.append((px(seg_start), px(seg_stop), _BLUE))
        elif turn.bot_start_time and turn.bot_stop_time:
            x0 = px(turn.bot_start_time)
            x1 = px(turn.bot_stop_time)
            bot_segs.append((x0, x1, _BOT_SPK))
            bot_colors.append((x0, x1, _BLUE))

    user_row = _build_char_row(plot_w, user_segs)
    pipe_row = _build_char_row(plot_w, pipe_segs)
    bot_row  = _build_char_row(plot_w, bot_segs)

    # ── Turn annotation row ───────────────────────────────────────────────
    ann_row = [" "] * plot_w
    for turn in conv.turns:
        if turn.user_start_time and turn.user_stop_time:
            mid = (px(turn.user_start_time) + px(turn.user_stop_time)) // 2
        elif turn.user_stop_time:
            mid = px(turn.user_stop_time)
        else:
            continue
        for i, ch in enumerate(_turn_label(turn.num)):
            if 0 <= mid + i < plot_w:
                ann_row[mid + i] = ch

    # ── Time tick lines ───────────────────────────────────────────────────
    iv = _tick_interval(total_secs, plot_w)
    tick_row = [" "] * plot_w
    tick_lbl = [" "] * plot_w
    major_xs = set()
    n = 0
    while True:
        offset = n * iv
        if offset > total_secs + 0.5:
            break
        x = px(t_start + timedelta(seconds=offset))
        if 0 <= x < plot_w:
            tick_row[x] = "┼"
            major_xs.add(x)
            for i, ch in enumerate(f"{offset:.0f}s"):
                if x + i < plot_w:
                    tick_lbl[x + i] = ch
        n += 1

    # Minor ticks — finer grid at iv//5 intervals, skipping major positions.
    _minor_map = {1: 1, 2: 1, 5: 1, 10: 2, 15: 3, 30: 5, 60: 10, 120: 30}
    minor_iv = _minor_map.get(iv, max(1, iv // 5))
    minor_row = ["─"] * plot_w
    if minor_iv < iv:
        m = 0
        while True:
            offset = m * minor_iv
            if offset > total_secs + 0.5:
                break
            x = px(t_start + timedelta(seconds=offset))
            if 0 <= x < plot_w:
                minor_row[x] = "┼" if x in major_xs else "·"
            m += 1

    # ── Header ────────────────────────────────────────────────────────────
    stop = conv.vad_stop_secs
    stop_lbl = f"{stop:.2f}s" if stop else "n/a"
    started = conv.start_time.strftime("%Y-%m-%d %H:%M") if conv.start_time else "unknown"
    pad = " " * LABEL_W

    print(f"\n{'═' * width}")
    print(f"  {conv.room_id}  │  {started}  │  VAD stop: {stop_lbl}  │  "
          f"LLM: {conv.llm_provider_label}  │  {len(conv.turns)} turns  ({label})")
    if conv.mixed_providers:
        warn = f"  ⚠ this session mixes LLM providers ({conv.llm_provider_label}) — TTFB values are not comparable across turns"
        print(f"{_RED}{warn}{_RESET}" if color else warn)
    if conv.start_time is not None:
        prelude = (t_start - conv.start_time).total_seconds()
        print(f"  diagram 0s = first user turn; transcript 0s = room join "
              f"(prelude: greeting + setup = {prelude:.3f}s before diagram 0s)")
    ws_parts = []
    if conv.stt_ws_connect_secs is not None:
        ws_parts.append(f"STT ws_connect: {conv.stt_ws_connect_secs:.3f}s")
    if conv.tts_ws_connect_secs is not None:
        ws_parts.append(f"TTS ws_connect: {conv.tts_ws_connect_secs:.3f}s")
    if ws_parts:
        print(f"  connections: {' │ '.join(ws_parts)}")
    print(f"{'═' * width}\n")

    # ── Diagram ───────────────────────────────────────────────────────────
    print(pad + "".join(tick_lbl))
    print(pad + "".join(tick_row))
    print(pad + "".join(minor_row))
    print("USER  " + _colorize_row(user_row, user_colors, color))
    print(pad + "".join(ann_row))
    print("PIPE  " + _colorize_row(pipe_row, pipe_colors, color))
    print("BOT   " + _colorize_row(bot_row, bot_colors, color))
    print()

    # ── Per-turn metrics ──────────────────────────────────────────────────
    def _f(v: Optional[float], unit: str = "s") -> str:
        return f"{v:.3f}{unit}" if v is not None else "n/a"

    prev_prompt  = None

    n_missing = 0
    for t in conv.turns:
        ub       = t.user_bot_latency
        perceived = t.perceived
        if perceived is None and ub is not None:
            n_missing += 1
        lbl      = _turn_label(t.num)

        # Latency block
        row = [
            f"perceived {_f(perceived)}",
            f"VAD {_f(t.user_turn_secs)}",
            f"user→bot {_f(ub)}",
        ]
        if t.user_speech_secs is not None:
            row.append(f"user speaking time {_f(t.user_speech_secs)}")
        if t.bot_speech_secs is not None:
            row.append(f"bot speaking time {_f(t.bot_speech_secs)}")
        row += [
            f"LLM TTFB {_f(t.llm_ttfb)}",
            f"txt agg {_f(t.txt_agg)}",
            f"TTS TTFB {_f(t.tts_ttfb)}",
        ]
        if t.llm_http_ttfb is not None:
            row.append(f"LLM http_ttfb {_f(t.llm_http_ttfb)}")
        if t.llm_processing_secs is not None:
            llm_ttfb = t.llm_ttfb or 0.0
            llm_gen = max(0.0, t.llm_processing_secs - llm_ttfb)
            row.append(f"LLM gen {_f(llm_gen)}")
            row.append(f"LLM call {_f(t.llm_call_secs)}")
        if t.stt_call_secs is not None:
            row.append(f"STT call {_f(t.stt_call_secs)}")
        if t.turn_complete is not None:
            status = "COMPLETE" if t.turn_complete else "INCOMPLETE"
            prob   = f" p={t.turn_probability:.2f}" if t.turn_probability is not None else ""
            turn_str = f"SmartTurn {status}{prob}"
            if color:
                c = _GREEN if t.turn_complete else _DIM_YEL
                row.append(f"{c}{turn_str}{_RESET}")
            else:
                row.append(turn_str)
        if t.tool_calls:
            tools_str = f"tools: {_fmt_tool_calls(t.tool_calls)}"
            row.append(f"{_YELLOW}{tools_str}{_RESET}" if color else tools_str)

        note = f"  [{_RED}{t.note}{_RESET}]" if (t.note and color) else (f"  [{t.note}]" if t.note else "")
        print(f"  {lbl}  " + "  │  ".join(row) + note)

        # Token / char block
        tok_parts = []
        if t.llm_prompt_tokens is not None:
            delta = f" (+{t.llm_prompt_tokens - prev_prompt})" if prev_prompt is not None else ""
            tok_parts.append(f"prompt {t.llm_prompt_tokens}{delta} tok")
            prev_prompt = t.llm_prompt_tokens
        if t.llm_cache_read_tokens is not None:
            pct = int(t.llm_cache_read_tokens / t.llm_prompt_tokens * 100) if t.llm_prompt_tokens else 0
            cache_str = f"cached {t.llm_cache_read_tokens} ({pct}%)"
            if color:
                tok_parts.append(f"{_CYAN}{cache_str}{_RESET}")
            else:
                tok_parts.append(cache_str)
        if t.llm_completion_tokens is not None:
            tok_parts.append(f"output {t.llm_completion_tokens} tok")
        # Prefer the true inter-token streaming rate (from GENERATING timestamps);
        # fall back to the coarse completion/window proxy only when unavailable.
        stream_rate = t.llm_stream_tok_per_sec
        if stream_rate is not None:
            tok_parts.append(f"{stream_rate:.0f} tok/s stream")
        elif t.llm_completion_tokens and t.llm_ttfb:
            gen_window = (ub or 0) - (t.llm_ttfb or 0)
            if gen_window > 0.05:
                tok_parts.append(f"{t.llm_completion_tokens / gen_window:.0f} tok/s")
        if t.llm_stream_ttfb is not None:
            tok_parts.append(f"stream TTFB {t.llm_stream_ttfb:.3f}s")
        if t.tts_chars > 0:
            tok_parts.append(f"TTS {t.tts_chars} chars")
        if t.tts_chars_per_sec is not None:
            tok_parts.append(f"{t.tts_chars_per_sec:.0f} chars/s")
        if t.llm_cost_usd is not None:
            split = []
            if t.llm_input_cost_usd:  split.append(f"in ${t.llm_input_cost_usd:.6f}")
            if t.llm_cache_cost_usd:  split.append(f"cache ${t.llm_cache_cost_usd:.6f}")
            if t.llm_output_cost_usd: split.append(f"out ${t.llm_output_cost_usd:.6f}")
            detail = f" ({' '.join(split)})" if split else ""
            tok_parts.append(f"LLM ${t.llm_cost_usd:.6f}{detail}")
        if t.tts_cost_usd is not None:
            tok_parts.append(f"TTS ${t.tts_cost_usd:.6f}")
        if t.total_cost_usd is not None:
            tok_parts.append(f"turn ${t.total_cost_usd:.6f}")
        if t.llm_error and color:
            tok_parts.append(f"{_RED}{_trunc(t.llm_error, 60)}{_RESET}")
        elif t.llm_error:
            tok_parts.append(_trunc(t.llm_error, 60))

        if tok_parts:
            print(f"     " + "  │  ".join(tok_parts))

    # ── Averages ──────────────────────────────────────────────────────────
    def _avgs(fn) -> Optional[float]:
        vals = [fn(t) for t in conv.turns if fn(t) is not None]
        return sum(vals) / len(vals) if vals else None

    ub_avg   = _avgs(lambda t: t.user_bot_latency)
    perc_avg = _avgs(lambda t: t.perceived)
    llm_avg  = _avgs(lambda t: t.llm_ttfb)
    agg_avg  = _avgs(lambda t: t.txt_agg)
    tts_avg  = _avgs(lambda t: t.tts_ttfb)
    uspk_avg = _avgs(lambda t: t.user_speech_secs)
    bspk_avg = _avgs(lambda t: t.bot_speech_secs)

    vad_avg = _avgs(lambda t: t.user_turn_secs)
    avg_row = [
        f"perceived {_f(perc_avg)}",
        f"VAD {_f(vad_avg)}",
        f"user→bot {_f(ub_avg)}",
    ]
    if uspk_avg is not None: avg_row.append(f"user speaking time {_f(uspk_avg)}")
    if bspk_avg is not None: avg_row.append(f"bot speaking time {_f(bspk_avg)}")
    llm_call_avg = _avgs(lambda t: t.llm_call_secs)
    stt_call_avg = _avgs(lambda t: t.stt_call_secs)
    avg_row += [
        f"LLM TTFB {_f(llm_avg)}",
        f"txt agg {_f(agg_avg)}",
        f"TTS TTFB {_f(tts_avg)}",
    ]
    if llm_call_avg is not None: avg_row.append(f"LLM call {_f(llm_call_avg)}")
    if stt_call_avg is not None: avg_row.append(f"STT call {_f(stt_call_avg)}")

    sep = "─" * (width - 4)
    print(f"\n  {sep}")
    print(f"  avg  " + "  │  ".join(avg_row))

    # Token/char/cost averages
    tok_avg = []
    pt_avg   = _avgs(lambda t: t.llm_prompt_tokens)
    cr_avg   = _avgs(lambda t: float(t.llm_cache_read_tokens) if t.llm_cache_read_tokens is not None else None)
    ct_avg   = _avgs(lambda t: t.llm_completion_tokens)
    ch_avg   = _avgs(lambda t: float(t.tts_chars) if t.tts_chars else None)
    cps_avg  = _avgs(lambda t: t.tts_chars_per_sec)
    srate_avg = _avgs(lambda t: t.llm_stream_tok_per_sec)
    llmc_avg = _avgs(lambda t: t.llm_cost_usd)
    ttsc_avg = _avgs(lambda t: t.tts_cost_usd)
    if pt_avg:  tok_avg.append(f"prompt {pt_avg:.0f} tok")
    if cr_avg:
        pct_avg = int(cr_avg / pt_avg * 100) if pt_avg else 0
        cache_str = f"cached {cr_avg:.0f} ({pct_avg}%)"
        tok_avg.append(f"{_CYAN}{cache_str}{_RESET}" if color else cache_str)
    if ct_avg:  tok_avg.append(f"output {ct_avg:.0f} tok")
    if srate_avg: tok_avg.append(f"{srate_avg:.0f} tok/s stream")
    if ch_avg:  tok_avg.append(f"TTS {ch_avg:.0f} chars")
    if cps_avg: tok_avg.append(f"{cps_avg:.0f} chars/s")
    if llmc_avg: tok_avg.append(f"LLM ø${llmc_avg:.6f}")
    if ttsc_avg: tok_avg.append(f"TTS ø${ttsc_avg:.6f}")
    if tok_avg:
        print(f"       " + "  │  ".join(tok_avg))

    # Session cost totals, broken out by component.
    def _sum(fn) -> float:
        return sum(v for v in (fn(t) for t in conv.turns) if v is not None)
    in_total    = _sum(lambda t: t.llm_input_cost_usd)
    cache_total = _sum(lambda t: t.llm_cache_cost_usd)
    out_total   = _sum(lambda t: t.llm_output_cost_usd)
    tts_total   = _sum(lambda t: t.tts_cost_usd)
    # STT bills per second streamed over the whole call, not per turn → use the
    # call wall-clock span (room join → last speech event) as the audio duration.
    stt_total = 0.0
    if conv.start_time is not None:
        call_secs = (t_end - conv.start_time).total_seconds()
        if call_secs > 0:
            stt_total = call_secs * _DEEPGRAM_USD_PER_AUDIO_SEC
    session_total = in_total + cache_total + out_total + tts_total + stt_total
    if session_total > 0:
        print(
            f"       session  LLM in ${in_total:.6f}  │  cache ${cache_total:.6f}  │  "
            f"out ${out_total:.6f}  │  TTS ${tts_total:.6f}  │  STT ${stt_total:.6f}  │  "
            f"total ${session_total:.6f}"
        )

    # Tools/scenario summary — distinct functions called across the call.
    all_tools = [tool for t in conv.turns for tool in t.tool_names]
    if all_tools:
        seen = dict.fromkeys(all_tools)  # preserve first-seen order, dedupe
        summary = ", ".join(f"{name}×{all_tools.count(name)}" if all_tools.count(name) > 1 else name
                            for name in seen)
        print(f"       tools used  {summary}")

    # ── Legend & formula ──────────────────────────────────────────────────
    print()
    if color:
        legend = (
            f"  {_GREEN}{_USER_SPK*2}{_RESET} user spk  "
            f"{_DIM_YEL}{_VAD_CH*2}{_RESET} VAD+TurnAnalyzer  "
            f"{_BLUE}{_BOT_SPK*2}{_RESET} bot spk  "
            f"{_YELLOW}{_LLM_CH*2}{_RESET} LLM TTFB  "
            f"{_CYAN}{_AGG_CH*2}{_RESET} txt agg  "
            f"{_MAGENTA}{_TTS_CH*2}{_RESET} TTS TTFB"
        )
    else:
        legend = (
            f"  {_USER_SPK*2} user spk  {_VAD_CH*2} VAD+TurnAnalyzer  "
            f"{_BOT_SPK*2} bot spk  {_LLM_CH*2} LLM TTFB  {_AGG_CH*2} txt agg  {_TTS_CH*2} TTS TTFB"
        )
    print(legend)
    print(f"  rows: USER = caller  │  PIPE = per-cycle LLM TTFB→txt agg→TTS TTFB (tool turns show 2+ cycles)  │  BOT = speech (gaps idle)")
    print(f"  (LLM generation overlaps the txt-agg band — shown as 'LLM gen' in the per-turn metrics, not as its own band)")
    print(f"  Perceived = user_turn_secs (observer) + User→Bot   [physical voice stop → bot start]")
    print(f"  User→Bot  = turn release → bot start  (= LLM TTFB + TxtAgg + TTS TTFB)")
    if n_missing:
        print(f"  note: {n_missing} turn(s) show perceived n/a — no observer breakdown logged "
              f"(VAD stop {stop_lbl} is NOT substituted)")

    # ── Transcript section ────────────────────────────────────────────────
    has_transcript = any(t.user_text or t.bot_text for t in conv.turns)
    if has_transcript:
        print(f"\n  {'─' * (width - 4)}")
        print(f"  Transcript")
        print(f"  {'─' * (width - 4)}")

        def _rel_ts(dt: Optional[datetime]) -> str:
            if dt is None or conv.start_time is None:
                return "      "
            secs = (dt - conv.start_time).total_seconds()
            return f"[{int(secs // 60):02d}:{int(secs % 60):02d}]"

        max_text = width - 22   # indent + label + timestamp
        for t in conv.turns:
            lbl = _turn_label(t.num)
            u = _trunc(t.user_text, max_text) if t.user_text else None
            b = _trunc(t.bot_text,  max_text) if t.bot_text  else None
            u_ts = _rel_ts(t.user_transcript_time or t.user_stop_time)
            b_ts = _rel_ts(t.bot_start_time)
            if u:
                if color:
                    print(f"  {lbl}  {_DIM}{u_ts}{_RESET} {_GREEN}user{_RESET}: \"{u}\"")
                else:
                    print(f"  {lbl}  {u_ts} user: \"{u}\"")
            if b:
                if color:
                    print(f"       {_DIM}{b_ts}{_RESET}  {_BLUE}bot{_RESET}:  \"{b}\"")
                else:
                    print(f"       {b_ts}  bot:  \"{b}\"")
            if t.llm_error:
                err_str = _trunc(t.llm_error, max_text)
                if color:
                    print(f"             {_RED}err{_RESET}:  {err_str}")
                else:
                    print(f"             err:  {err_str}")
            if u or b or t.llm_error:
                print()
    print()


# ── HTML export ──────────────────────────────────────────────────────────────

def display_html(conv: Conversation, label: str, out_path: str, width: int = 140) -> None:
    LABEL_W = 6
    plot_w  = width - LABEL_W
    pad     = " " * LABEL_W

    all_times = []
    for t in conv.turns:
        for dt in (t.user_start_time, t.user_stop_time, t.bot_start_time, t.bot_stop_time):
            if dt is not None:
                all_times.append(dt)
    if len(all_times) < 2:
        print(f"Not enough data to render HTML.", file=sys.stderr)
        return

    t_start    = min(all_times)
    t_end      = max(all_times)
    total_secs = (t_end - t_start).total_seconds()
    if total_secs < 0.1:
        return

    cps = plot_w / total_secs

    def px(dt: datetime) -> int:
        return max(0, min(plot_w - 1, int((dt - t_start).total_seconds() * cps)))

    # ── Build diagram rows (same logic as display_timeline) ────────────────
    user_segs:   list = []
    user_colors: list = []
    pipe_segs:   list = []
    pipe_colors: list = []
    bot_segs:    list = []
    bot_colors:  list = []

    for turn in conv.turns:
        if turn.user_start_time and turn.user_stop_time:
            if turn.user_turn_secs is not None:
                physical_stop = turn.user_stop_time - timedelta(seconds=turn.user_turn_secs)
                x0, x1, x2 = px(turn.user_start_time), px(physical_stop), px(turn.user_stop_time)
                if x1 >= x0:
                    user_segs.append((x0, x1, _USER_SPK));  user_colors.append((x0, x1, _GREEN))
                if x2 > x1:
                    user_segs.append((x1+1, x2, _VAD_CH)); user_colors.append((x1+1, x2, _DIM_YEL))
            else:
                x0, x1 = px(turn.user_start_time), px(turn.user_stop_time)
                user_segs.append((x0, x1, _USER_SPK)); user_colors.append((x0, x1, _GREEN))

        if turn.user_stop_time is None:
            continue

        for x0, x1, ch, col in _pipe_phases(turn, px):
            pipe_segs.append((x0, x1, ch)); pipe_colors.append((x0, x1, col))

        # Speech drawn per span so tool-call gaps stay idle (matches the metric).
        if turn.bot_speech_segments:
            for seg_start, seg_stop in turn.bot_speech_segments:
                bot_segs.append((px(seg_start), px(seg_stop), _BOT_SPK))
                bot_colors.append((px(seg_start), px(seg_stop), _BLUE))
        elif turn.bot_start_time and turn.bot_stop_time:
            x0, x1 = px(turn.bot_start_time), px(turn.bot_stop_time)
            bot_segs.append((x0, x1, _BOT_SPK)); bot_colors.append((x0, x1, _BLUE))

    user_row = _build_char_row(plot_w, user_segs)
    pipe_row = _build_char_row(plot_w, pipe_segs)
    bot_row  = _build_char_row(plot_w, bot_segs)

    ann_row = [" "] * plot_w
    for turn in conv.turns:
        if turn.user_start_time and turn.user_stop_time:
            mid = (px(turn.user_start_time) + px(turn.user_stop_time)) // 2
        elif turn.user_stop_time:
            mid = px(turn.user_stop_time)
        else:
            continue
        for i, ch in enumerate(_turn_label(turn.num)):
            if 0 <= mid + i < plot_w:
                ann_row[mid + i] = ch

    iv = _tick_interval(total_secs, plot_w)
    tick_row  = [" "] * plot_w
    tick_lbl  = [" "] * plot_w
    major_xs: set = set()
    n = 0
    while True:
        offset = n * iv
        if offset > total_secs + 0.5:
            break
        x = px(t_start + timedelta(seconds=offset))
        if 0 <= x < plot_w:
            tick_row[x] = "┼"; major_xs.add(x)
            for i, ch in enumerate(f"{offset:.0f}s"):
                if x + i < plot_w:
                    tick_lbl[x + i] = ch
        n += 1

    _minor_map = {1:1, 2:1, 5:1, 10:2, 15:3, 30:5, 60:10, 120:30}
    minor_iv  = _minor_map.get(iv, max(1, iv // 5))
    minor_row = ["─"] * plot_w
    if minor_iv < iv:
        m = 0
        while True:
            offset = m * minor_iv
            if offset > total_secs + 0.5:
                break
            x = px(t_start + timedelta(seconds=offset))
            if 0 <= x < plot_w:
                minor_row[x] = "┼" if x in major_xs else "·"
            m += 1

    # ── HTML rendering ─────────────────────────────────────────────────────
    lines: list[str] = []

    def p(text: str = "") -> None:
        lines.append(f"<p>{text}</p>")

    def _f(v: Optional[float], unit: str = "s") -> str:
        return f"{v:.3f}{unit}" if v is not None else "n/a"

    stop     = conv.vad_stop_secs
    stop_lbl = f"{stop:.2f}s" if stop else "n/a"
    started  = conv.start_time.strftime("%Y-%m-%d %H:%M") if conv.start_time else "unknown"
    sep      = "─" * (width - 4)

    p(_esc("═" * width))
    p(_esc(f"  {conv.room_id}  │  {started}  │  VAD stop: {stop_lbl}  │  "
           f"LLM: {conv.llm_provider_label}  │  {len(conv.turns)} turns  ({label})"))
    if conv.start_time is not None:
        prelude = (t_start - conv.start_time).total_seconds()
        p(_esc(f"  diagram 0s = first user turn; transcript 0s = room join "
               f"(prelude: {prelude:.3f}s)"))
    p(_esc("═" * width))
    p()
    p(_esc(pad + "".join(tick_lbl)))
    p(_esc(pad + "".join(tick_row)))
    p(_esc(pad + "".join(minor_row)))
    p(_esc("USER  ") + _colorize_row_html(user_row, user_colors))
    p(_esc(pad + "".join(ann_row)))
    p(_esc("PIPE  ") + _colorize_row_html(pipe_row, pipe_colors))
    p(_esc("BOT   ") + _colorize_row_html(bot_row, bot_colors))
    p()

    prev_prompt = None
    n_missing   = 0
    for t in conv.turns:
        ub        = t.user_bot_latency
        perceived = t.perceived
        if perceived is None and ub is not None:
            n_missing += 1
        lbl = _turn_label(t.num)

        row_parts = [_esc(f"perceived {_f(perceived)}"), _esc(f"user→bot {_f(ub)}")]
        if t.user_speech_secs is not None:
            row_parts.append(_esc(f"user speaking time {_f(t.user_speech_secs)}"))
        if t.bot_speech_secs is not None:
            row_parts.append(_esc(f"bot speaking time {_f(t.bot_speech_secs)}"))
        row_parts += [
            _esc(f"LLM TTFB {_f(t.llm_ttfb)}"),
            _esc(f"txt agg {_f(t.txt_agg)}"),
            _esc(f"TTS TTFB {_f(t.tts_ttfb)}"),
        ]
        if t.llm_processing_secs is not None:
            llm_gen = max(0.0, t.llm_processing_secs - (t.llm_ttfb or 0.0))
            row_parts.append(_esc(f"LLM gen {_f(llm_gen)}"))
        if t.turn_complete is not None:
            status = "COMPLETE" if t.turn_complete else "INCOMPLETE"
            prob   = f" p={t.turn_probability:.2f}" if t.turn_probability is not None else ""
            c = _GREEN if t.turn_complete else _DIM_YEL
            row_parts.append(_hspan(f"SmartTurn {status}{prob}", c))
        if t.tool_calls:
            row_parts.append(_hspan(f"tools: {_fmt_tool_calls(t.tool_calls)}", _YELLOW))
        note = ("  [" + _hspan(t.note, _RED) + "]") if t.note else ""
        p(f"  {_esc(lbl)}  " + "  │  ".join(row_parts) + note)

        tok = []
        if t.llm_prompt_tokens is not None:
            delta = f" (+{t.llm_prompt_tokens - prev_prompt})" if prev_prompt is not None else ""
            tok.append(_esc(f"prompt {t.llm_prompt_tokens}{delta} tok"))
            prev_prompt = t.llm_prompt_tokens
        if t.llm_cache_read_tokens is not None:
            pct = int(t.llm_cache_read_tokens / t.llm_prompt_tokens * 100) if t.llm_prompt_tokens else 0
            tok.append(_hspan(f"cached {t.llm_cache_read_tokens} ({pct}%)", _CYAN))
        if t.llm_completion_tokens is not None:
            tok.append(_esc(f"output {t.llm_completion_tokens} tok"))
        if t.llm_stream_tok_per_sec is not None:
            tok.append(_esc(f"{t.llm_stream_tok_per_sec:.0f} tok/s stream"))
        if t.llm_stream_ttfb is not None:
            tok.append(_esc(f"stream TTFB {t.llm_stream_ttfb:.3f}s"))
        if t.tts_chars > 0:
            tok.append(_esc(f"TTS {t.tts_chars} chars"))
        if t.tts_chars_per_sec is not None:
            tok.append(_esc(f"{t.tts_chars_per_sec:.0f} chars/s"))
        if t.llm_cost_usd is not None:
            split = []
            if t.llm_input_cost_usd:  split.append(f"in ${t.llm_input_cost_usd:.6f}")
            if t.llm_cache_cost_usd:  split.append(f"cache ${t.llm_cache_cost_usd:.6f}")
            if t.llm_output_cost_usd: split.append(f"out ${t.llm_output_cost_usd:.6f}")
            detail = f" ({' '.join(split)})" if split else ""
            tok.append(_esc(f"LLM ${t.llm_cost_usd:.6f}{detail}"))
        if t.tts_cost_usd is not None:
            tok.append(_esc(f"TTS ${t.tts_cost_usd:.6f}"))
        if t.total_cost_usd is not None:
            tok.append(_esc(f"turn ${t.total_cost_usd:.6f}"))
        if t.llm_error:
            tok.append(_hspan(_trunc(t.llm_error, 60), _RED))
        if tok:
            p("     " + "  │  ".join(tok))

    # Averages
    def _avgs(fn) -> Optional[float]:
        vals = [fn(t) for t in conv.turns if fn(t) is not None]
        return sum(vals) / len(vals) if vals else None

    p()
    p(_esc(f"  {sep}"))
    avg_parts = [_esc(f"perceived {_f(_avgs(lambda t: t.perceived))}"),
                 _esc(f"user→bot {_f(_avgs(lambda t: t.user_bot_latency))}")]
    uspk = _avgs(lambda t: t.user_speech_secs)
    bspk = _avgs(lambda t: t.bot_speech_secs)
    if uspk is not None: avg_parts.append(_esc(f"user speaking time {_f(uspk)}"))
    if bspk is not None: avg_parts.append(_esc(f"bot speaking time {_f(bspk)}"))
    avg_parts += [_esc(f"LLM TTFB {_f(_avgs(lambda t: t.llm_ttfb))}"),
                  _esc(f"txt agg {_f(_avgs(lambda t: t.txt_agg))}"),
                  _esc(f"TTS TTFB {_f(_avgs(lambda t: t.tts_ttfb))}")]
    p("  avg  " + "  │  ".join(avg_parts))

    tok_avg: list[str] = []
    pt_avg  = _avgs(lambda t: t.llm_prompt_tokens)
    cr_avg  = _avgs(lambda t: float(t.llm_cache_read_tokens) if t.llm_cache_read_tokens is not None else None)
    ct_avg  = _avgs(lambda t: t.llm_completion_tokens)
    ch_avg  = _avgs(lambda t: float(t.tts_chars) if t.tts_chars else None)
    cps_avg = _avgs(lambda t: t.tts_chars_per_sec)
    sr_avg  = _avgs(lambda t: t.llm_stream_tok_per_sec)
    if pt_avg: tok_avg.append(_esc(f"prompt {pt_avg:.0f} tok"))
    if cr_avg:
        pct_avg = int(cr_avg / pt_avg * 100) if pt_avg else 0
        tok_avg.append(_hspan(f"cached {cr_avg:.0f} ({pct_avg}%)", _CYAN))
    if ct_avg: tok_avg.append(_esc(f"output {ct_avg:.0f} tok"))
    if sr_avg: tok_avg.append(_esc(f"{sr_avg:.0f} tok/s stream"))
    if ch_avg: tok_avg.append(_esc(f"TTS {ch_avg:.0f} chars"))
    if cps_avg: tok_avg.append(_esc(f"{cps_avg:.0f} chars/s"))
    llmc = _avgs(lambda t: t.llm_cost_usd)
    ttsc = _avgs(lambda t: t.tts_cost_usd)
    if llmc: tok_avg.append(_esc(f"LLM ø${llmc:.6f}"))
    if ttsc: tok_avg.append(_esc(f"TTS ø${ttsc:.6f}"))
    if tok_avg:
        p("       " + "  │  ".join(tok_avg))

    def _sum(fn) -> float:
        return sum(v for v in (fn(t) for t in conv.turns) if v is not None)
    in_tot    = _sum(lambda t: t.llm_input_cost_usd)
    cache_tot = _sum(lambda t: t.llm_cache_cost_usd)
    out_tot   = _sum(lambda t: t.llm_output_cost_usd)
    tts_tot   = _sum(lambda t: t.tts_cost_usd)
    stt_tot   = 0.0
    if conv.start_time is not None:
        call_secs = (t_end - conv.start_time).total_seconds()
        if call_secs > 0:
            stt_tot = call_secs * _DEEPGRAM_USD_PER_AUDIO_SEC
    sess_tot = in_tot + cache_tot + out_tot + tts_tot + stt_tot
    if sess_tot > 0:
        p(_esc(f"       session  LLM in ${in_tot:.6f}  │  cache ${cache_tot:.6f}  │  "
               f"out ${out_tot:.6f}  │  TTS ${tts_tot:.6f}  │  STT ${stt_tot:.6f}  │  "
               f"total ${sess_tot:.6f}"))

    all_tools = [tool for t in conv.turns for tool in t.tool_names]
    if all_tools:
        seen = dict.fromkeys(all_tools)
        summary = ", ".join(f"{name}×{all_tools.count(name)}" if all_tools.count(name) > 1 else name
                            for name in seen)
        p(_esc(f"       tools used  {summary}"))

    # Legend
    p()
    legend = (
        _hspan(_USER_SPK * 2, _GREEN)   + _esc(" user spk  ") +
        _hspan(_VAD_CH * 2,  _DIM_YEL) + _esc(" VAD+TurnAnalyzer  ") +
        _hspan(_BOT_SPK * 2, _BLUE)    + _esc(" bot spk  ") +
        _hspan(_LLM_CH * 2,  _YELLOW)  + _esc(" LLM TTFB  ") +
        _hspan(_AGG_CH * 2,  _CYAN)    + _esc(" txt agg  ") +
        _hspan(_TTS_CH * 2,  _MAGENTA) + _esc(" TTS TTFB")
    )
    p("  " + legend)
    p(_esc("  Perceived = user_turn_secs (observer) + User→Bot   [physical voice stop → bot start]"))
    p(_esc("  User→Bot  = turn release → bot start  (= LLM TTFB + TxtAgg + TTS TTFB)"))

    # Transcript
    has_transcript = any(t.user_text or t.bot_text for t in conv.turns)
    if has_transcript:
        p()
        p(_esc(f"  {sep}"))
        p(_esc("  Transcript"))
        p(_esc(f"  {sep}"))

        def _rel_ts(dt: Optional[datetime]) -> str:
            if dt is None or conv.start_time is None:
                return "      "
            secs = (dt - conv.start_time).total_seconds()
            return f"[{int(secs // 60):02d}:{int(secs % 60):02d}]"

        max_text = width - 22
        for t in conv.turns:
            lbl  = _turn_label(t.num)
            u    = _trunc(t.user_text, max_text) if t.user_text else None
            b    = _trunc(t.bot_text,  max_text) if t.bot_text  else None
            u_ts = _rel_ts(t.user_transcript_time or t.user_stop_time)
            b_ts = _rel_ts(t.bot_start_time)
            if u:
                p(f"  {_esc(lbl)}  {_hspan(u_ts, _DIM)} {_hspan('user', _GREEN)}: &quot;{_esc(u)}&quot;")
            if b:
                p(f"       {_hspan(b_ts, _DIM)}  {_hspan('bot', _BLUE)}:  &quot;{_esc(b)}&quot;")
            if t.llm_error:
                p(f"             {_hspan('err', _RED)}:  {_esc(_trunc(t.llm_error, max_text))}")
            if u or b or t.llm_error:
                p()

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: 'Courier New', Courier, monospace; font-size: 13px;
       line-height: 1.5; background: #ffffff; color: #111827;
       padding: 20px; margin: 0; }}
p {{ margin: 0; white-space: pre; }}
</style>
</head>
<body>
{"".join(lines)}
</body>
</html>"""

    with open(out_path, "w") as fh:
        fh.write(html)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Report per-turn latency metrics from pipecat.log"
    )
    ap.add_argument(
        "-n", type=int, default=0, metavar="N",
        help="how many conversations back (0 = latest, 1 = previous, …)"
    )
    ap.add_argument(
        "--table", action="store_true",
        help="use original tabular format instead of river-flow timeline"
    )
    ap.add_argument(
        "--list", action="store_true",
        help="list all rooms/conversations found in the log and exit"
    )
    ap.add_argument(
        "--log", default=DEFAULT_LOG, metavar="FILE",
        help=f"log file to parse (default: {DEFAULT_LOG})"
    )
    ap.add_argument(
        "--html", action="store_true",
        help="write an HTML file instead of printing to terminal"
    )
    args = ap.parse_args()

    try:
        convs = parse_log(args.log)
    except FileNotFoundError:
        print(f"Error: log file not found: {args.log}", file=sys.stderr)
        sys.exit(1)

    if not convs:
        print("No conversations found in log.")
        sys.exit(1)

    if args.list:
        print(f"\n{'#':>4}  {'Room ID':<30}  {'Started':<22}  {'VAD stop':>8}  Turns")
        print(f"{'─'*4}  {'─'*30}  {'─'*22}  {'─'*8}  {'─'*5}")
        for i, c in enumerate(reversed(convs)):
            started = c.start_time.strftime("%Y-%m-%d %H:%M:%S") if c.start_time else "unknown"
            stop    = f"{c.vad_stop_secs:.2f}s" if c.vad_stop_secs else "  n/a"
            tag     = " ← latest" if i == 0 else ""
            print(f"{i:>4}  {c.room_id:<30}  {started:<22}  {stop:>8}  {len(c.turns)}{tag}")
        print()
        return

    n = args.n
    if n >= len(convs):
        print(
            f"Error: only {len(convs)} conversation(s) in log. "
            f"Valid -n range: 0–{len(convs)-1}.",
            file=sys.stderr,
        )
        sys.exit(1)

    conv  = convs[-(1 + n)]
    label = "latest" if n == 0 else f"n={n}, {n} back"

    if args.html:
        import os as _os
        out_path = _os.path.splitext(args.log)[0] + f"_{conv.room_id}.html"
        display_html(conv, label, out_path)
        print(f"HTML written to: {out_path}", file=sys.stderr)
        print(f"Open in browser → Ctrl+A → Ctrl+C → paste into Notion", file=sys.stderr)
    elif args.table:
        display(conv, label)
    else:
        display_timeline(conv, label)


if __name__ == "__main__":
    main()
