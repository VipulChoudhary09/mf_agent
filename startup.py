"""Process initialization shared by the LiveKit worker and maintenance scripts."""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from pathlib import Path

from filelock import FileLock
from loguru import logger

KB_DIR = Path("KB/pdftomds")
CHROMA_DB_PATH = Path("chroma_db")


def env_flag(name: str, default: bool) -> bool:
    """Read a conventional boolean environment variable."""

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def missing_environment(names: Iterable[str]) -> list[str]:
    """Return required environment variable names that are unset."""

    return [name for name in names if not os.environ.get(name)]


def validate_environment(names: Iterable[str]) -> None:
    """Raise a value-only error without ever logging secret contents."""

    missing = missing_environment(names)
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )


def _index_is_ready(db_path: Path) -> bool:
    if not db_path.exists():
        return False
    try:
        from rag.embedder import get_collection

        return get_collection(str(db_path)).count() > 0
    except Exception as exc:
        logger.warning("ChromaDB readiness check failed: {}", type(exc).__name__)
        return False


def ensure_kb_indexed(
    *,
    kb_dir: Path = KB_DIR,
    db_path: Path = CHROMA_DB_PATH,
) -> None:
    """Create the local Chroma index once, safely across worker processes."""

    if _index_is_ready(db_path):
        return

    lock_path = Path(f"{db_path}.index.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # LiveKit prewarms isolated job processes. A file lock prevents each one
    # from independently rebuilding and charging for the same embeddings.
    with FileLock(str(lock_path), timeout=240):
        if _index_is_ready(db_path):
            return
        if not kb_dir.exists():
            raise RuntimeError(f"Knowledge-base directory not found: {kb_dir}")

        from rag.chunker import chunk_all_kb_files
        from rag.embedder import reset_collection

        chunks = chunk_all_kb_files(kb_dir)
        if not chunks:
            raise RuntimeError(f"No knowledge-base chunks found in {kb_dir}")

        logger.info("Indexing {} KB chunks into {}", len(chunks), db_path)
        collection = reset_collection(str(db_path))

        batch_size = 5
        max_retries = 3
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            last_error: Exception | None = None

            for attempt in range(max_retries):
                try:
                    collection.add(
                        ids=[chunk["id"] for chunk in batch],
                        documents=[chunk["text"] for chunk in batch],
                        metadatas=[chunk["metadata"] for chunk in batch],
                    )
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    message = str(exc)
                    if "429" not in message and "RESOURCE_EXHAUSTED" not in message:
                        raise
                    delay = 5 * (2**attempt)
                    logger.warning(
                        "Embedding rate limited; retrying batch in {} seconds",
                        delay,
                    )
                    time.sleep(delay)

            if last_error is not None:
                raise RuntimeError(
                    f"Failed to index KB batch after {max_retries} attempts"
                ) from last_error

            if start + batch_size < len(chunks):
                time.sleep(3)

        count = collection.count()
        if count == 0:
            raise RuntimeError("ChromaDB index completed with zero stored chunks")
        logger.info("ChromaDB ready with {} chunks", count)


def load_knowledge_base() -> None:
    """Load markdown sources into the existing in-memory KB cache."""

    from knowledge_base import load_all

    load_all()
    logger.info("Knowledge base loaded")
