"""
Text-mode testing harness for the Mutual Fund Voice Agent.

Uses the google-genai SDK directly (NOT Pipecat) for simplicity.
Implements the full 3-state machine: PROFILING → (TIE_BREAKER?) → RECOMMENDATION.

Run with: python text_bot.py
Requires only GOOGLE_API_KEY in .env.

This lets you test the entire conversation flow without burning
Deepgram/Cartesia free-tier minutes.
"""

import os
import sys
import json
from enum import Enum

from dotenv import load_dotenv
from loguru import logger
from google import genai
from google.genai import types

from knowledge_base import load_all, get_scheme_context, get_comparison_table
from scoring import recommend_mutual_fund
from prompts import (
    PROFILING_PROMPT,
    FUND_NAMES,
    build_recommendation_prompt,
    build_tie_breaker_prompt,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

MODEL = "gemini-2.0-flash"

# Configure loguru — clean format for terminal
logger.remove()
logger.add(
    sys.stderr,
    format="<dim>{time:HH:mm:ss}</dim> | <level>{level:<7}</level> | {message}",
    level="DEBUG",
)

# ---------------------------------------------------------------------------
# Function calling schema (matches HANDOFF.md §5)
# ---------------------------------------------------------------------------

EXTRACT_PROFILE_SCHEMA = types.FunctionDeclaration(
    name="extract_user_profile",
    description=(
        "Call this tool once the user has provided enough information about "
        "their investment timeframe, risk appetite, and starting amount."
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
                description=(
                    "Low = fears losing money / wants capital protection. "
                    "Moderate = accepts some drops for growth. "
                    "High = comfortable with significant volatility."
                ),
            ),
            "growth_style": types.Schema(
                type="STRING",
                enum=[
                    "downside_protection",
                    "stable_bluechip",
                    "diversified_global",
                    "unspecified",
                ],
                description=(
                    "downside_protection = wants active debt/equity management to "
                    "reduce losses. stable_bluechip = wants low-cost exposure to "
                    "India's top 100 companies. diversified_global = wants exposure "
                    "to Indian AND international stocks like Google/Alphabet."
                ),
            ),
            "initial_investment": types.Schema(
                type="INTEGER",
                description="The initial amount in INR the user wants to invest.",
            ),
            "investment_mode": types.Schema(
                type="STRING",
                enum=["lumpsum", "sip", "both", "unspecified"],
                description=(
                    "Whether the user wants to invest a one-time lump sum, "
                    "set up a monthly SIP, or both."
                ),
            ),
        },
        required=["horizon_years", "risk_tolerance", "initial_investment"],
    ),
)

TOOLS = types.Tool(function_declarations=[EXTRACT_PROFILE_SCHEMA])


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class State(Enum):
    PROFILING = "profiling"
    TIE_BREAKER = "tie_breaker"
    RECOMMENDATION = "recommendation"


# ---------------------------------------------------------------------------
# Main bot class
# ---------------------------------------------------------------------------

class TextBot:
    """Terminal-based mutual fund advisor using Gemini with function calling."""

    def __init__(self):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key or api_key == "your_key_here":
            logger.error("GOOGLE_API_KEY not set in .env — cannot start")
            sys.exit(1)

        self.client = genai.Client(api_key=api_key)
        self.state = State.PROFILING
        self.history: list[types.Content] = []

        # Load knowledge base
        logger.info("Loading knowledge base...")
        load_all()
        logger.info("Knowledge base loaded ✓")

        # Create initial chat
        self._create_chat(PROFILING_PROMPT)

    def _create_chat(self, system_prompt: str) -> None:
        """Create (or recreate) a chat session with a new system prompt.

        On state transitions, we need to swap the system instruction.
        The google-genai SDK binds system_instruction at chat creation time,
        so we recreate the chat while preserving conversation history.
        """
        self.chat = self.client.chats.create(
            model=MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=[TOOLS],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
                temperature=0.7,
            ),
            history=self.history,
        )
        logger.debug(f"Chat created with state={self.state.value}")

    def _handle_function_call(
        self, fn_call: types.FunctionCall
    ) -> types.Content:
        """Process the extract_user_profile function call.

        Runs the scoring engine, transitions state, and returns a
        function response to feed back to the model.
        """
        args = dict(fn_call.args) if fn_call.args else {}
        logger.info(f"🔧 Function call: {fn_call.name}({json.dumps(args, indent=2)})")

        # Run scoring engine
        result = recommend_mutual_fund(args)
        logger.info(
            f"📊 Scores: {result['scores']} | "
            f"Winner: {result['winner']} | Tie: {result['is_tie']}"
        )

        if result["is_tie"] and self.state == State.PROFILING:
            # --- Transition to TIE_BREAKER ---
            self.state = State.TIE_BREAKER
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
            self.state = State.RECOMMENDATION
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

        # Save current history before recreating chat
        self.history = list(self.chat._curated_history)

        # Add the assistant's function call to history
        self.history.append(
            types.Content(
                role="model",
                parts=[types.Part(function_call=fn_call)],
            )
        )

        # Add function response to history
        fn_response = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fn_call.name,
                        response=response_data,
                    )
                )
            ],
        )
        self.history.append(fn_response)

        # Recreate chat with new system prompt + preserved history
        self._create_chat(new_prompt)

        return fn_response

    def get_greeting(self) -> str:
        """Get the initial greeting from the bot (bot speaks first)."""
        logger.info("🎙️ Bot generating greeting...")
        response = self.chat.send_message("Begin the conversation.")

        # Check for function call (shouldn't happen on greeting, but be safe)
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    self._handle_function_call(part.function_call)
                    # Get the model's response after function result
                    follow_up = self.chat.send_message("Continue the conversation.")
                    return follow_up.text or ""

        return response.text or ""

    def send_message(self, user_input: str) -> str:
        """Send a user message and get the bot's response.

        Handles function calls transparently — if the model wants to
        call extract_user_profile, we run the scoring engine, swap
        prompts, and return the model's recommendation.
        """
        response = self.chat.send_message(user_input)

        # Check if the model wants to call a function
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    # Process the function call (runs scoring, transitions state)
                    self._handle_function_call(part.function_call)

                    # Now get the model's natural language response
                    # (recommendation announcement or tie-breaker question)
                    follow_up = self.chat.send_message(
                        "Now respond to the user based on the function result."
                    )
                    return follow_up.text or ""

        return response.text or ""


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  PPFAS Mutual Fund Voice Agent (Text Mode)")
    print("  Type your messages below. Ctrl+C to quit.")
    print("=" * 60 + "\n")

    bot = TextBot()

    # Bot speaks first
    greeting = bot.get_greeting()
    print(f"\n🤖 Agent: {greeting}\n")

    while True:
        try:
            user_input = input("👤 You: ").strip()
            if not user_input:
                continue

            # Check for quit commands
            if user_input.lower() in ("quit", "exit", "bye", "q"):
                print("\n🤖 Agent: Thank you for speaking with PPFAS Mutual Fund. "
                      "Have a great day!\n")
                break

            response = bot.send_message(user_input)
            print(f"\n🤖 Agent: {response}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\n⚠️ Error: {e}. Please try again.\n")


if __name__ == "__main__":
    main()
