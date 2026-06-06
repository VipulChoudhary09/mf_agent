"""
FastAPI server for the PPFAS Mutual Fund Voice Agent.

Exposes a single POST /start endpoint that:
1. Creates a temporary Daily.co room
2. Spawns the bot in a background task
3. Returns the room URL for the caller to join

Usage:
    python server.py                          # starts on port 7860
    curl -s -X POST http://localhost:7860/start | jq .

Then parse the logs:
    python measure_latency.py                 # river flow (latest conversation)
    python measure_latency.py --table         # tabular format
"""

import asyncio
import os
import sys
import time

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

load_dotenv(override=True)

# ── Logging ──────────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    format="<dim>{time:HH:mm:ss}</dim> | <level>{level:<7}</level> | {message}",
    level="DEBUG",
)
# File sink for measure_latency.py — format matches parse_ts() expectations
logger.add(
    "pipecat.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {message}",
    level="DEBUG",
    rotation="10 MB",
    retention=3,
)

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="PPFAS Voice Agent",
    description="Start a voice call with the PPFAS Mutual Fund Advisor bot",
)

# Track active bots so we don't spawn unlimited ones
_active_bot_tasks: list[asyncio.Task] = []
MAX_CONCURRENT_BOTS = 3


async def _create_daily_room() -> tuple[str, str]:
    """Create a temporary Daily room and return (room_url, token)."""
    api_key = os.environ.get("DAILY_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="DAILY_API_KEY not configured")

    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient() as client:
        # Create room
        room_resp = await client.post(
            "https://api.daily.co/v1/rooms",
            headers=headers,
            json={
                "properties": {
                    "exp": int(time.time()) + 3600,  # 1 hour expiry
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


async def _run_bot_background(room_url: str, token: str):
    """Run the bot in the background. Catches exceptions to avoid crashing the server."""
    try:
        from bot import run_bot
        await run_bot(room_url, token)
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
    finally:
        logger.info("Bot session ended.")


@app.post("/start")
async def start_bot():
    """Create a Daily room and spawn the voice bot.

    Returns:
        JSON with room_url and instructions.
    """
    # Validate API keys
    required_keys = [
        "GOOGLE_API_KEY",
        "DEEPGRAM_API_KEY",
        "CARTESIA_API_KEY",
        "DAILY_API_KEY",
    ]
    missing = [k for k in required_keys if not os.environ.get(k)]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing API keys: {', '.join(missing)}",
        )

    # Clean up finished tasks
    _active_bot_tasks[:] = [t for t in _active_bot_tasks if not t.done()]

    if len(_active_bot_tasks) >= MAX_CONCURRENT_BOTS:
        raise HTTPException(
            status_code=429,
            detail=f"Max {MAX_CONCURRENT_BOTS} concurrent bots. Try again later.",
        )

    # Create room and spawn bot
    room_url, token = await _create_daily_room()

    task = asyncio.create_task(_run_bot_background(room_url, token))
    _active_bot_tasks.append(task)

    logger.info(f"🎙️ Bot spawned for room: {room_url}")

    return JSONResponse(
        content={
            "room_url": room_url,
            "instructions": (
                "The bot is joining the room now. "
                "Open room_url in your browser to start the conversation."
            ),
        }
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    active = len([t for t in _active_bot_tasks if not t.done()])
    return {"status": "ok", "active_bots": active}


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PPFAS Mutual Fund Voice Agent — Server")
    print("=" * 60)
    print("  POST /start  → create room + spawn bot")
    print("  GET  /health → health check")
    print("=" * 60 + "\n")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )
