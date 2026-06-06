"""
PPFAS Mutual Fund Voice Agent — Server

Single entry point for the voice agent. Handles:
  - Auto-indexing KB into ChromaDB on first run
  - Loading the knowledge base into memory
  - FastAPI server with /start endpoint for room creation

Usage:
    python server.py                              # starts on port 7860
    curl -s -X POST http://localhost:7860/start | jq .

Then parse the logs:
    python measure_latency.py                     # river flow (latest conversation)
    python measure_latency.py --table             # tabular format
"""

import asyncio
import os
import sys
import time
from pathlib import Path

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


# ── Startup: Auto-index KB + load knowledge base ────────────────────────────

KB_DIR = Path("KB/pdftomds")
CHROMA_DB_PATH = "./chroma_db"


def _ensure_kb_indexed():
    """Index KB into ChromaDB if the vector store is missing or empty."""
    needs_indexing = False

    if not Path(CHROMA_DB_PATH).exists():
        needs_indexing = True
        logger.info("[STARTUP] ChromaDB not found — will index KB")
    else:
        try:
            from rag.embedder import get_collection
            collection = get_collection(CHROMA_DB_PATH)
            count = collection.count()
            if count == 0:
                needs_indexing = True
                logger.info("[STARTUP] ChromaDB is empty — will index KB")
            else:
                logger.info(f"[STARTUP] ChromaDB ready — {count} chunks indexed ✓")
        except Exception as e:
            needs_indexing = True
            logger.warning(f"[STARTUP] ChromaDB check failed ({e}) — will re-index")

    if not needs_indexing:
        return

    if not KB_DIR.exists():
        logger.error(f"[STARTUP] KB directory not found: {KB_DIR}")
        logger.error("          Cannot index. Place KB markdown files in KB/pdftomds/")
        sys.exit(1)

    logger.info(f"[STARTUP] Indexing KB from {KB_DIR}...")

    from rag.chunker import chunk_all_kb_files
    from rag.embedder import reset_collection

    chunks = chunk_all_kb_files(KB_DIR)
    logger.info(f"[STARTUP] {len(chunks)} chunks created")

    collection = reset_collection(CHROMA_DB_PATH)

    # Batch add with rate limit retry (Gemini embedding API)
    BATCH_SIZE = 5
    MAX_RETRIES = 3
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        for attempt in range(MAX_RETRIES):
            try:
                collection.add(
                    ids=[c["id"] for c in batch],
                    documents=[c["text"] for c in batch],
                    metadatas=[c["metadata"] for c in batch],
                )
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 5 * (2 ** attempt)
                    logger.warning(f"[STARTUP] Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        embedded = min(i + BATCH_SIZE, len(chunks))
        if embedded < len(chunks):
            time.sleep(3)  # rate limit cooldown

    logger.info(f"[STARTUP] ✅ {collection.count()} chunks indexed into ChromaDB")


def _load_knowledge_base():
    """Pre-load knowledge base so bot spawns are instant."""
    from knowledge_base import load_all
    load_all()
    logger.info("[STARTUP] Knowledge base loaded ✓")


def _validate_api_keys():
    """Check all required API keys are present."""
    required = ["GOOGLE_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY", "DAILY_API_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        logger.error(f"[STARTUP] Missing API keys: {', '.join(missing)}")
        logger.error("          Add them to .env file")
        sys.exit(1)
    logger.info("[STARTUP] All API keys present ✓")


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
    print("  PPFAS Mutual Fund Voice Agent")
    print("=" * 60)

    # Startup checks (run before server starts)
    _validate_api_keys()
    _ensure_kb_indexed()
    _load_knowledge_base()

    print()
    print("  POST /start  → create room + spawn bot")
    print("  GET  /health → health check")
    print()
    print("  Start a call:")
    print('    curl -s -X POST http://localhost:7860/start | jq .')
    print("=" * 60 + "\n")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )

