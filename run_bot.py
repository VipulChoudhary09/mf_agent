"""
Launcher for the PPFAS Mutual Fund Voice Agent.

Creates a temporary Daily.co room, prints the join URL,
and starts the Pipecat voice pipeline.

Usage: python run_bot.py
"""

import asyncio
import os
import sys

import httpx
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

# Configure loguru
logger.remove()
logger.add(
    sys.stderr,
    format="<dim>{time:HH:mm:ss}</dim> | <level>{level:<7}</level> | {message}",
    level="DEBUG",
)


async def create_daily_room() -> tuple[str, str]:
    """Create a temporary Daily room and return (room_url, token)."""
    api_key = os.environ.get("DAILY_API_KEY")
    if not api_key:
        logger.error("DAILY_API_KEY not set in .env")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient() as client:
        # Create room
        room_resp = await client.post(
            "https://api.daily.co/v1/rooms",
            headers=headers,
            json={
                "properties": {
                    "exp": int(asyncio.get_event_loop().time()) + 3600,
                    "enable_chat": False,
                    "enable_screenshare": False,
                    "max_participants": 2,
                }
            },
        )
        room_resp.raise_for_status()
        room_data = room_resp.json()
        room_url = room_data["url"]
        room_name = room_data["name"]

        logger.info(f"Created Daily room: {room_name}")

        # Create meeting token
        token_resp = await client.post(
            "https://api.daily.co/v1/meeting-tokens",
            headers=headers,
            json={
                "properties": {
                    "room_name": room_name,
                    "is_owner": True,
                }
            },
        )
        token_resp.raise_for_status()
        token = token_resp.json()["token"]

    return room_url, token


async def main():
    print("\n" + "=" * 60)
    print("  PPFAS Mutual Fund Voice Agent")
    print("=" * 60)

    # Validate all API keys are present
    required_keys = [
        "GOOGLE_API_KEY",
        "DEEPGRAM_API_KEY",
        "CARTESIA_API_KEY",
        "DAILY_API_KEY",
    ]
    missing = [k for k in required_keys if not os.environ.get(k)]
    if missing:
        logger.error(f"Missing API keys in .env: {', '.join(missing)}")
        sys.exit(1)

    logger.info("All API keys present ✓")

    # Create Daily room
    room_url, token = await create_daily_room()

    print(f"\n  🎙️  Join the call: {room_url}")
    print(f"  📋 Or copy: {room_url}")
    print(f"\n  Waiting for you to join...\n")

    # Run the bot
    from bot import run_bot
    await run_bot(room_url, token)


if __name__ == "__main__":
    asyncio.run(main())
