"""RAG (Retrieval-Augmented Generation) module for PPFAS Voice Agent.

Provides section-aware chunking, embedding, and retrieval of mutual fund
knowledge base documents using ChromaDB + Gemini embeddings.
"""

from rag.chunker import chunk_markdown_file, chunk_all_kb_files
from rag.retriever import retrieve

__all__ = ["chunk_markdown_file", "chunk_all_kb_files", "retrieve"]
