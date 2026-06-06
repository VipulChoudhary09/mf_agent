"""Section-aware markdown chunker for mutual fund documents.

Splits markdown files on H2 (##) headings, preserving tables as atomic
units. Each chunk is tagged with metadata for filtered retrieval.

Design rules:
  1. Split on ``## `` headings — each section becomes a chunk.
  2. If a section exceeds MAX_CHUNK_TOKENS, sub-split on ``### ``.
  3. Never split within a markdown table.
  4. Merge tiny sections (< MIN_CHUNK_TOKENS) with the next section.
  5. Prepend a metadata header to each chunk for retrieval context.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CHUNK_TOKENS = 1000  # approx chars / 4
MIN_CHUNK_TOKENS = 100

# Map KB filenames → scheme IDs and source types
FILENAME_MAP: dict[str, tuple[str, str]] = {
    "Dfly.md": ("Dynamic_Asset_Allocation", "flyer"),
    "Dpres.md": ("Dynamic_Asset_Allocation", "presentation"),
    "LCfly.md": ("Large_Cap", "flyer"),
    "LCpres.md": ("Large_Cap", "presentation"),
    "FCfly.md": ("Flexi_Cap", "flyer"),
    "FCpres.md": ("Flexi_Cap", "presentation"),
}

FUND_DISPLAY_NAMES: dict[str, str] = {
    "Dynamic_Asset_Allocation": "Parag Parikh Dynamic Asset Allocation Fund",
    "Large_Cap": "Parag Parikh Large Cap Fund",
    "Flexi_Cap": "Parag Parikh Flexi Cap Fund",
}


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class ChunkMeta(TypedDict):
    """Metadata attached to each chunk for ChromaDB storage."""

    scheme_id: str
    fund_name: str
    source_type: str  # "flyer" or "presentation"
    source_file: str
    section_title: str
    has_table: bool
    has_numbers: bool


class Chunk(TypedDict):
    """A single chunk ready for embedding + storage."""

    id: str  # unique chunk ID
    text: str  # chunk content with metadata header
    metadata: ChunkMeta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TABLE_RE = re.compile(r"^\|.*\|$", re.MULTILINE)
_NUMBER_RE = re.compile(r"[\d,]+\.?\d*%|₹[\d,]+|Rs\.?\s*[\d,]+")


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 English characters."""
    return len(text) // 4


def _has_table(text: str) -> bool:
    """Check if text contains a markdown table."""
    return bool(_TABLE_RE.search(text))


def _has_numbers(text: str) -> bool:
    """Check if text contains financial numbers (%, ₹, Rs.)."""
    return bool(_NUMBER_RE.search(text))


def _strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter (``---`` delimited) from markdown."""
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            return text[end + 3 :].lstrip("\n")
    return text


def _split_on_heading(text: str, level: int = 2) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs at the given heading level.

    Args:
        text: Full markdown content.
        level: Heading level (2 = ``## ``, 3 = ``### ``).

    Returns:
        List of (section_title, section_body) tuples.
        The first element may have an empty title (preamble before first heading).
    """
    prefix = "#" * level + " "
    sections: list[tuple[str, str]] = []
    current_title = ""
    current_lines: list[str] = []

    for line in text.split("\n"):
        if line.startswith(prefix):
            # Save previous section
            if current_lines or current_title:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line[len(prefix) :].strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Don't forget the last section
    if current_lines or current_title:
        sections.append((current_title, "\n".join(current_lines).strip()))

    return sections


# ---------------------------------------------------------------------------
# Main chunking logic
# ---------------------------------------------------------------------------


def chunk_markdown_file(
    filepath: str | Path,
    scheme_id: str,
    source_type: str,
    fund_name: str,
) -> list[Chunk]:
    """Chunk a single markdown file into retrieval-ready chunks.

    Args:
        filepath: Path to the ``.md`` file.
        scheme_id: Fund identifier (e.g. ``Flexi_Cap``).
        source_type: ``"flyer"`` or ``"presentation"``.
        fund_name: Display name of the fund.

    Returns:
        List of Chunk dicts ready for ChromaDB insertion.
    """
    filepath = Path(filepath)
    raw = filepath.read_text(encoding="utf-8")
    raw = _strip_frontmatter(raw)

    sections = _split_on_heading(raw, level=2)
    chunks: list[Chunk] = []

    for section_title, section_body in sections:
        if not section_body.strip():
            continue

        tokens = _estimate_tokens(section_body)

        if tokens > MAX_CHUNK_TOKENS:
            # Sub-split on H3 headings
            subsections = _split_on_heading(section_body, level=3)
            for sub_title, sub_body in subsections:
                if not sub_body.strip():
                    continue
                full_title = (
                    f"{section_title} > {sub_title}" if sub_title else section_title
                )
                chunks.append(
                    _make_chunk(
                        text=sub_body,
                        section_title=full_title,
                        scheme_id=scheme_id,
                        source_type=source_type,
                        source_file=filepath.name,
                        fund_name=fund_name,
                        chunk_index=len(chunks),
                    )
                )
        else:
            chunks.append(
                _make_chunk(
                    text=section_body,
                    section_title=section_title,
                    scheme_id=scheme_id,
                    source_type=source_type,
                    source_file=filepath.name,
                    fund_name=fund_name,
                    chunk_index=len(chunks),
                )
            )

    # Merge tiny chunks with the next chunk
    chunks = _merge_tiny_chunks(chunks)

    return chunks


def _make_chunk(
    *,
    text: str,
    section_title: str,
    scheme_id: str,
    source_type: str,
    source_file: str,
    fund_name: str,
    chunk_index: int,
) -> Chunk:
    """Create a single Chunk with metadata header prepended."""
    header = (
        f"[Fund: {fund_name} | Source: {source_type} | Section: {section_title}]"
    )
    full_text = f"{header}\n\n{text}"

    chunk_id = f"{scheme_id}_{source_type}_{chunk_index:03d}"

    return Chunk(
        id=chunk_id,
        text=full_text,
        metadata=ChunkMeta(
            scheme_id=scheme_id,
            fund_name=fund_name,
            source_type=source_type,
            source_file=source_file,
            section_title=section_title,
            has_table=_has_table(text),
            has_numbers=_has_numbers(text),
        ),
    )


def _merge_tiny_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Merge chunks smaller than MIN_CHUNK_TOKENS with the next chunk."""
    if not chunks:
        return chunks

    merged: list[Chunk] = []
    carry: Chunk | None = None

    for chunk in chunks:
        if carry is not None:
            # Merge carry into current chunk
            combined_text = carry["text"] + "\n\n" + chunk["text"]
            combined_title = (
                carry["metadata"]["section_title"]
                + " + "
                + chunk["metadata"]["section_title"]
            )
            merged_chunk = Chunk(
                id=chunk["id"],
                text=combined_text,
                metadata=ChunkMeta(
                    scheme_id=chunk["metadata"]["scheme_id"],
                    fund_name=chunk["metadata"]["fund_name"],
                    source_type=chunk["metadata"]["source_type"],
                    source_file=chunk["metadata"]["source_file"],
                    section_title=combined_title,
                    has_table=(
                        carry["metadata"]["has_table"]
                        or chunk["metadata"]["has_table"]
                    ),
                    has_numbers=(
                        carry["metadata"]["has_numbers"]
                        or chunk["metadata"]["has_numbers"]
                    ),
                ),
            )
            merged.append(merged_chunk)
            carry = None
        elif _estimate_tokens(chunk["text"]) < MIN_CHUNK_TOKENS:
            carry = chunk
        else:
            merged.append(chunk)

    # If the last chunk was tiny and carried, just add it
    if carry is not None:
        if merged:
            # Merge with the previous chunk
            last = merged[-1]
            last["text"] = last["text"] + "\n\n" + carry["text"]
            last["metadata"]["section_title"] = (
                last["metadata"]["section_title"]
                + " + "
                + carry["metadata"]["section_title"]
            )
            last["metadata"]["has_table"] = (
                last["metadata"]["has_table"] or carry["metadata"]["has_table"]
            )
            last["metadata"]["has_numbers"] = (
                last["metadata"]["has_numbers"] or carry["metadata"]["has_numbers"]
            )
        else:
            merged.append(carry)

    return merged


# ---------------------------------------------------------------------------
# Bulk chunking
# ---------------------------------------------------------------------------


def chunk_all_kb_files(kb_dir: str | Path = "KB/pdftomds") -> list[Chunk]:
    """Chunk all known KB markdown files from the given directory.

    Args:
        kb_dir: Path to directory containing the ``.md`` files.

    Returns:
        All chunks from all files, ready for bulk embedding.
    """
    kb_dir = Path(kb_dir)
    all_chunks: list[Chunk] = []

    for filename, (scheme_id, source_type) in FILENAME_MAP.items():
        filepath = kb_dir / filename
        if not filepath.exists():
            continue
        fund_name = FUND_DISPLAY_NAMES[scheme_id]
        file_chunks = chunk_markdown_file(filepath, scheme_id, source_type, fund_name)
        all_chunks.extend(file_chunks)

    # Re-index IDs to be globally unique
    for i, chunk in enumerate(all_chunks):
        chunk["id"] = f"chunk_{i:04d}"

    return all_chunks
