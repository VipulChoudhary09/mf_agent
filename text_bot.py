"""Text-mode testing harness for the mutual-fund advisor.

This uses the Google GenAI SDK directly and shares the same deterministic
``AdvisorFlow`` as the LiveKit voice agent. It does not use STT, TTS, WebRTC,
or per-turn RAG, so only ``GOOGLE_API_KEY`` is required.
"""

from __future__ import annotations

import json
import os
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from loguru import logger

from advisor_core import AdvisorFlow, ConversationState as State
from knowledge_base import load_all
from prompts import PROFILING_PROMPT

load_dotenv(".env.local", override=False)
load_dotenv(".env", override=False)

MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
MAX_RETRIES = 3
RETRY_DELAY = 2

logger.remove()
logger.add(
    sys.stderr,
    format="<dim>{time:HH:mm:ss}</dim> | <level>{level:<7}</level> | {message}",
    level="INFO",
)

EXTRACT_PROFILE_SCHEMA = types.FunctionDeclaration(
    name="extract_user_profile",
    description=(
        "Call this tool once the user has provided their investment timeframe, "
        "risk appetite, investment mode, and starting amount."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "horizon_years": types.Schema(
                type="INTEGER",
                description="The number of years the user plans to stay invested.",
            ),
            "risk_tolerance": types.Schema(
                type="STRING",
                enum=["low", "moderate", "high"],
                description="The user's comfort with investment volatility.",
            ),
            "growth_style": types.Schema(
                type="STRING",
                enum=[
                    "downside_protection",
                    "stable_bluechip",
                    "diversified_global",
                    "unspecified",
                ],
                description="The user's preferred portfolio style.",
            ),
            "initial_investment": types.Schema(
                type="INTEGER",
                description="Starting investment in Indian Rupees.",
            ),
            "investment_mode": types.Schema(
                type="STRING",
                enum=["lumpsum", "sip", "both", "unspecified"],
                description="One-time lumpsum, monthly SIP, both, or unknown.",
            ),
        },
        required=[
            "horizon_years",
            "risk_tolerance",
            "initial_investment",
            "investment_mode",
        ],
    ),
)
TOOLS = types.Tool(function_declarations=[EXTRACT_PROFILE_SCHEMA])


class TextBot:
    """Terminal advisor using Gemini function calling and shared core logic."""

    def __init__(self) -> None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for text mode")

        self.client = genai.Client(api_key=api_key)
        self.flow = AdvisorFlow(use_rag=False)
        self.state = self.flow.state.current
        self.history: list[types.Content] = []

        load_all()
        self._create_chat(PROFILING_PROMPT)

    def _create_chat(self, system_prompt: str) -> None:
        tools = [TOOLS] if self.state != State.RECOMMENDATION else []
        self.chat = self.client.chats.create(
            model=MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=tools,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
                temperature=0.3,
            ),
            history=self.history,
        )

    def _send_with_retry(self, message: str) -> object:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self.chat.send_message(message)
            except Exception as exc:
                error = str(exc)
                transient = any(
                    marker in error.lower()
                    for marker in ("remoteprotocolerror", "disconnected", "503", "429")
                )
                if attempt >= MAX_RETRIES or not transient:
                    raise
                delay = RETRY_DELAY * attempt
                logger.warning(
                    "Transient Gemini error on attempt {}/{}; retrying in {}s",
                    attempt,
                    MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)
        raise RuntimeError("Gemini retry loop ended unexpectedly")

    def _handle_function_call(self, fn_call: types.FunctionCall) -> types.Content:
        args = dict(fn_call.args) if fn_call.args else {}
        logger.info("Profile tool call: {}", json.dumps(args, ensure_ascii=False))

        transition = self.flow.handle_profile(args)
        self.state = transition.state

        # Preserve the curated conversation while recreating the chat with the
        # new state-specific system instruction.
        self.history = list(self.chat._curated_history)
        self.history.append(
            types.Content(role="model", parts=[types.Part(function_call=fn_call)])
        )
        fn_response = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fn_call.name,
                        response=transition.tool_result,
                    )
                )
            ],
        )
        self.history.append(fn_response)
        self._create_chat(transition.prompt)
        return fn_response

    def get_greeting(self) -> str:
        response = self._send_with_retry("Begin the conversation.")
        return self._resolve_response(response)

    def send_message(self, user_input: str) -> str:
        response = self._send_with_retry(user_input)
        return self._resolve_response(response)

    def _resolve_response(self, response: object) -> str:
        candidates = getattr(response, "candidates", None)
        if candidates and candidates[0].content.parts:
            for part in candidates[0].content.parts:
                if part.function_call:
                    self._handle_function_call(part.function_call)
                    follow_up = self._send_with_retry(
                        "Respond naturally based on the function result."
                    )
                    return getattr(follow_up, "text", None) or ""
        return getattr(response, "text", None) or ""


def main() -> None:
    print("\nPPFAS Mutual Fund Advisor, text mode. Type 'quit' to exit.\n")
    bot = TextBot()
    print(f"Agent: {bot.get_greeting()}\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "bye", "q"}:
                print("Agent: Thank you for speaking with PPFAS Mutual Fund.")
                return
            print(f"Agent: {bot.send_message(user_input)}\n")
        except KeyboardInterrupt:
            print("\nGoodbye.")
            return


if __name__ == "__main__":
    main()
