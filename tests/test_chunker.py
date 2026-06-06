"""Unit tests for rag/chunker.py — section-aware markdown chunking.

Tests:
  1. Correct chunk count from a known document
  2. No table is split across chunks
  3. Every chunk has valid metadata
  4. Tiny sections get merged
  5. Frontmatter is stripped
  6. Bulk chunking works across all files
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from rag.chunker import (
    Chunk,
    chunk_all_kb_files,
    chunk_markdown_file,
    _split_on_heading,
    _strip_frontmatter,
    _has_table,
    _has_numbers,
    _merge_tiny_chunks,
    _estimate_tokens,
    FILENAME_MAP,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

KB_DIR = Path(__file__).resolve().parent.parent / "KB" / "pdftomds"


@pytest.fixture
def sample_md(tmp_path: Path) -> Path:
    """Create a sample markdown file for testing."""
    content = textwrap.dedent("""\
    ---
    Title: Test Fund Flyer
    ---

    # TEST FUND

    Some intro text.

    ## Asset Allocation

    | Type | Allocation | Risk |
    |------|-----------|------|
    | Equity | 65-100% | High |
    | Debt | 0-35% | Low |

    ## Load Structure

    - **Entry Load:** Not Applicable
    - **Exit Load:** 1% if redeemed within 1 year. Rs. 5,000 minimum.

    ## Tiny Section

    One line.

    ## Performance

    | Period | Regular | Direct | Nifty 500 |
    |--------|---------|--------|-----------|
    | 1 Year | 3.77% | 4.44% | 3.96% |
    | 3 Year | 16.83% | 17.64% | 15.28% |
    | Since Inception | 17.80% | 18.63% | 14.21% |

    This fund has delivered ₹83,351 on an initial investment of ₹10,000.

    ## Contact

    PPFAS Asset Management.
    Tel: 91 22 6140 6555
    """)
    fp = tmp_path / "test_fund.md"
    fp.write_text(content)
    return fp


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_strip_frontmatter(self):
        text = "---\nTitle: Test\n---\n\n# Hello"
        assert _strip_frontmatter(text) == "# Hello"

    def test_strip_frontmatter_no_frontmatter(self):
        text = "# Hello\nWorld"
        assert _strip_frontmatter(text) == text

    def test_has_table_positive(self):
        assert _has_table("| A | B |\n|---|---|\n| 1 | 2 |")

    def test_has_table_negative(self):
        assert not _has_table("Just some text without pipes")

    def test_has_numbers_percentage(self):
        assert _has_numbers("Returns: 17.80%")

    def test_has_numbers_rupees(self):
        assert _has_numbers("Minimum investment: ₹5,000")

    def test_has_numbers_rs(self):
        assert _has_numbers("Rs. 10,000 invested")

    def test_has_numbers_negative(self):
        assert not _has_numbers("No numbers here at all")

    def test_estimate_tokens(self):
        text = "a" * 400
        assert _estimate_tokens(text) == 100

    def test_split_on_heading_h2(self):
        text = "## First\nContent 1\n## Second\nContent 2"
        sections = _split_on_heading(text, level=2)
        assert len(sections) == 2
        assert sections[0] == ("First", "Content 1")
        assert sections[1] == ("Second", "Content 2")

    def test_split_on_heading_with_preamble(self):
        text = "Intro text\n## First\nContent"
        sections = _split_on_heading(text, level=2)
        assert len(sections) == 2
        assert sections[0][0] == ""  # preamble has no title
        assert sections[1][0] == "First"


# ---------------------------------------------------------------------------
# Chunk tests
# ---------------------------------------------------------------------------


class TestChunkMarkdownFile:
    def test_produces_chunks(self, sample_md: Path):
        chunks = chunk_markdown_file(sample_md, "Test_Fund", "flyer", "Test Fund")
        assert len(chunks) > 0

    def test_all_chunks_have_metadata(self, sample_md: Path):
        chunks = chunk_markdown_file(sample_md, "Test_Fund", "flyer", "Test Fund")
        for chunk in chunks:
            meta = chunk["metadata"]
            assert meta["scheme_id"] == "Test_Fund"
            assert meta["source_type"] == "flyer"
            assert meta["source_file"] == "test_fund.md"
            assert meta["section_title"]  # non-empty
            assert isinstance(meta["has_table"], bool)
            assert isinstance(meta["has_numbers"], bool)

    def test_tables_not_split(self, sample_md: Path):
        """Verify that table rows stay within a single chunk."""
        chunks = chunk_markdown_file(sample_md, "Test_Fund", "flyer", "Test Fund")
        for chunk in chunks:
            text = chunk["text"]
            if "| Type |" in text or "| Period |" in text:
                # Table header found — verify rows are present too
                lines = [l for l in text.split("\n") if l.strip().startswith("|")]
                assert len(lines) >= 3, f"Table split! Only {len(lines)} rows in chunk"

    def test_table_metadata_flag(self, sample_md: Path):
        chunks = chunk_markdown_file(sample_md, "Test_Fund", "flyer", "Test Fund")
        table_chunks = [c for c in chunks if c["metadata"]["has_table"]]
        assert len(table_chunks) >= 2  # Asset Allocation + Performance

    def test_numbers_metadata_flag(self, sample_md: Path):
        chunks = chunk_markdown_file(sample_md, "Test_Fund", "flyer", "Test Fund")
        number_chunks = [c for c in chunks if c["metadata"]["has_numbers"]]
        assert len(number_chunks) >= 1  # Performance has %, ₹

    def test_frontmatter_stripped(self, sample_md: Path):
        chunks = chunk_markdown_file(sample_md, "Test_Fund", "flyer", "Test Fund")
        for chunk in chunks:
            assert "---\nTitle:" not in chunk["text"]

    def test_tiny_section_merged(self, sample_md: Path):
        """The 'Tiny Section' (one line) should be merged with adjacent."""
        chunks = chunk_markdown_file(sample_md, "Test_Fund", "flyer", "Test Fund")
        # Tiny Section alone would have title "Tiny Section" as sole section
        # After merge, it should be combined with next section
        solo_tiny = [
            c
            for c in chunks
            if c["metadata"]["section_title"] == "Tiny Section"
        ]
        assert len(solo_tiny) == 0, "Tiny section should have been merged"

    def test_metadata_header_in_text(self, sample_md: Path):
        chunks = chunk_markdown_file(sample_md, "Test_Fund", "flyer", "Test Fund")
        for chunk in chunks:
            assert chunk["text"].startswith("[Fund: Test Fund |")

    def test_unique_ids(self, sample_md: Path):
        chunks = chunk_markdown_file(sample_md, "Test_Fund", "flyer", "Test Fund")
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


# ---------------------------------------------------------------------------
# Real KB tests (skip if files not present)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not KB_DIR.exists(), reason="KB/pdftomds not found")
class TestRealKBChunking:
    def test_chunk_all_produces_chunks(self):
        chunks = chunk_all_kb_files(KB_DIR)
        # 6 files × ~8-16 sections = roughly 50-80 chunks
        assert len(chunks) >= 30, f"Expected ≥30 chunks, got {len(chunks)}"
        assert len(chunks) <= 120, f"Expected ≤120 chunks, got {len(chunks)}"

    def test_all_three_schemes_represented(self):
        chunks = chunk_all_kb_files(KB_DIR)
        schemes = {c["metadata"]["scheme_id"] for c in chunks}
        assert "Dynamic_Asset_Allocation" in schemes
        assert "Large_Cap" in schemes
        assert "Flexi_Cap" in schemes

    def test_both_source_types(self):
        chunks = chunk_all_kb_files(KB_DIR)
        source_types = {c["metadata"]["source_type"] for c in chunks}
        assert "flyer" in source_types
        assert "presentation" in source_types

    def test_globally_unique_ids(self):
        chunks = chunk_all_kb_files(KB_DIR)
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_no_empty_chunks(self):
        chunks = chunk_all_kb_files(KB_DIR)
        for chunk in chunks:
            # Strip the metadata header line
            content = chunk["text"].split("\n", 2)[-1].strip()
            assert len(content) > 0, f"Empty chunk: {chunk['id']}"
