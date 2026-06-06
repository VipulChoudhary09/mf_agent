"""ChromaDB client and Gemini embedding function setup.

Provides a configured ChromaDB collection with Gemini embeddings
for the PPFAS KB. Uses PersistentClient so vectors survive restarts.
"""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION_NAME = "ppfas_kb"
DEFAULT_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768  # MRL reduction — sufficient for <2K chunks


# ---------------------------------------------------------------------------
# Singleton client + collection
# ---------------------------------------------------------------------------

_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def _get_api_key() -> str:
    """Get Google API key from environment.

    ChromaDB's GoogleGeminiEmbeddingFunction looks for GOOGLE_API_KEY
    or GEMINI_API_KEY. We normalise to GEMINI_API_KEY for consistency
    with ChromaDB's expectation.
    """
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise EnvironmentError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY for Gemini embeddings"
        )
    # ChromaDB specifically checks GEMINI_API_KEY
    os.environ["GEMINI_API_KEY"] = key
    return key


def get_embedding_function() -> embedding_functions.GoogleGeminiEmbeddingFunction:
    """Create a Gemini embedding function for ChromaDB.

    Uses ``gemini-embedding-001`` with RETRIEVAL_DOCUMENT task type
    for indexing. Query-time task type is handled automatically by
    ChromaDB when calling ``collection.query()``.
    """
    _get_api_key()
    return embedding_functions.GoogleGeminiEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        task_type="RETRIEVAL_DOCUMENT",
    )


def get_client(db_path: str | Path = DEFAULT_DB_PATH) -> chromadb.ClientAPI:
    """Get or create the ChromaDB persistent client.

    Args:
        db_path: Directory for persistent vector storage.

    Returns:
        ChromaDB client instance.
    """
    global _client
    if _client is None:
        db_path = Path(db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(db_path))
    return _client


def get_collection(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> chromadb.Collection:
    """Get or create the PPFAS KB collection with Gemini embeddings.

    Args:
        db_path: Directory for persistent vector storage.

    Returns:
        ChromaDB collection ready for add/query operations.
    """
    global _collection
    if _collection is None:
        client = get_client(db_path)
        ef = get_embedding_function()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def reset_collection(db_path: str | Path = DEFAULT_DB_PATH) -> chromadb.Collection:
    """Delete and recreate the collection (for re-indexing).

    Args:
        db_path: Directory for persistent vector storage.

    Returns:
        Fresh, empty ChromaDB collection.
    """
    global _collection
    client = get_client(db_path)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # Collection doesn't exist yet
    _collection = None
    return get_collection(db_path)


def reset_singletons() -> None:
    """Reset module-level singletons (for testing)."""
    global _client, _collection
    _client = None
    _collection = None
