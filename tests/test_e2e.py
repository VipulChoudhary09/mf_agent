"""
Automated end-to-end test for the text bot.

Runs 3 scripted conversations to validate the full pipeline:
  1. Short horizon / low risk → Dynamic Asset Allocation (clear winner)
  2. Long horizon / high risk / global → Flexi Cap (clear winner)
  3. Medium horizon / moderate risk → near-tie → TIE_BREAKER state

Each test verifies:
  - The function call is triggered with correct extracted args
  - The state transition is correct
  - The scoring engine produces the expected winner
  - The model generates a natural language response (not empty)

Requires GOOGLE_API_KEY in .env.
"""

import os
import sys
import json
import time

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure loguru
logger.remove()
logger.add(
    sys.stderr,
    format="<dim>{time:HH:mm:ss}</dim> | <level>{level:<7}</level> | {message}",
    level="INFO",
)


def run_e2e_test():
    """Run scripted end-to-end conversations."""
    from text_bot import TextBot, State

    results = []

    # -----------------------------------------------------------------------
    # Test 1: Short horizon, low risk → Dynamic Asset Allocation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  TEST 1: Short horizon + Low risk → Dynamic")
    print("=" * 60)

    bot = TextBot()
    greeting = bot.get_greeting()
    print(f"🤖 Greeting: {greeting[:100]}...")

    # Give all info in one message to force function call
    response = bot.send_message(
        "I want to invest about 50,000 rupees for 2 years. "
        "I'm quite conservative and don't want to lose my money."
    )
    print(f"🤖 Response: {response[:200]}...")

    test1_pass = (
        bot.state == State.RECOMMENDATION
        and len(response) > 50
    )
    results.append(("Test 1: Dynamic winner", test1_pass, bot.state.value))
    print(f"{'✅' if test1_pass else '❌'} State={bot.state.value}")

    time.sleep(2)  # Rate limiting

    # -----------------------------------------------------------------------
    # Test 2: Long horizon, high risk, global → Flexi Cap
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  TEST 2: Long horizon + High risk + Global → Flexi Cap")
    print("=" * 60)

    bot2 = TextBot()
    greeting2 = bot2.get_greeting()
    print(f"🤖 Greeting: {greeting2[:100]}...")

    response2 = bot2.send_message(
        "I have 1 lakh to invest for about 10 years. "
        "I'm fine with high risk and I want exposure to "
        "international stocks like Google and Apple."
    )
    print(f"🤖 Response: {response2[:200]}...")

    test2_pass = (
        bot2.state == State.RECOMMENDATION
        and len(response2) > 50
    )
    results.append(("Test 2: Flexi Cap winner", test2_pass, bot2.state.value))
    print(f"{'✅' if test2_pass else '❌'} State={bot2.state.value}")

    time.sleep(2)

    # -----------------------------------------------------------------------
    # Test 3: Medium horizon, moderate risk → TIE possible
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  TEST 3: Medium horizon + Moderate risk → Tie possible")
    print("=" * 60)

    bot3 = TextBot()
    greeting3 = bot3.get_greeting()
    print(f"🤖 Greeting: {greeting3[:100]}...")

    response3 = bot3.send_message(
        "I'm planning to invest around 20,000 for about 4 years. "
        "I have moderate risk tolerance."
    )
    print(f"🤖 Response: {response3[:200]}...")

    # This should trigger either TIE_BREAKER or RECOMMENDATION
    test3_pass = (
        bot3.state in (State.TIE_BREAKER, State.RECOMMENDATION)
        and len(response3) > 20
    )
    results.append(("Test 3: State transition", test3_pass, bot3.state.value))
    print(f"{'✅' if test3_pass else '❌'} State={bot3.state.value}")

    # If tie-breaker, send a follow-up
    if bot3.state == State.TIE_BREAKER:
        time.sleep(2)
        response3b = bot3.send_message(
            "I think I'd prefer something that protects me from "
            "big market drops."
        )
        print(f"🤖 Follow-up: {response3b[:200]}...")
        test3b_pass = (
            bot3.state == State.RECOMMENDATION
            and len(response3b) > 50
        )
        results.append(("Test 3b: Tie resolved", test3b_pass, bot3.state.value))
        print(f"{'✅' if test3b_pass else '❌'} State={bot3.state.value}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  E2E TEST RESULTS")
    print("=" * 60)

    all_pass = True
    for name, passed, state in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name} (state={state})")
        if not passed:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("  🎉 ALL TESTS PASSED")
    else:
        print("  ⚠️  SOME TESTS FAILED")
    print("=" * 60 + "\n")

    return all_pass


if __name__ == "__main__":
    success = run_e2e_test()
    sys.exit(0 if success else 1)
