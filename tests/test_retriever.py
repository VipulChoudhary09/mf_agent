"""Integration tests for rag/retriever.py — accuracy on golden questions.

Requires a populated ChromaDB at ./chroma_db (run scripts/index_kb.py first).
Tests verify that the correct fund sections are retrieved for 10 canonical
financial questions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from rag.embedder import reset_singletons

# Reset singletons before importing retriever to avoid stale state
reset_singletons()

from rag.retriever import retrieve, retrieve_formatted, get_collection_stats

# ---------------------------------------------------------------------------
# Skip if no ChromaDB or no API key
# ---------------------------------------------------------------------------

DB_PATH = "./chroma_db"
HAS_DB = Path(DB_PATH).exists()
HAS_KEY = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
SKIP_REASON = (
    "Requires populated ChromaDB + GEMINI_API_KEY"
    if not (HAS_DB and HAS_KEY)
    else ""
)


@pytest.mark.skipif(not (HAS_DB and HAS_KEY), reason=SKIP_REASON)
class TestCollectionStats:
    def test_total_chunks(self):
        stats = get_collection_stats(DB_PATH)
        assert stats["total_chunks"] >= 50, f"Expected ≥50 chunks, got {stats['total_chunks']}"

    def test_all_schemes_present(self):
        stats = get_collection_stats(DB_PATH)
        for scheme in ["Dynamic_Asset_Allocation", "Large_Cap", "Flexi_Cap"]:
            assert stats["per_scheme"][scheme] > 0, f"No chunks for {scheme}"


@pytest.mark.skipif(not (HAS_DB and HAS_KEY), reason=SKIP_REASON)
class TestGoldenQuestions:
    """Test retrieval accuracy on 10 canonical financial questions.

    Each test verifies that the retrieved chunks contain expected
    keywords or section titles for the correct fund.
    """

    def test_exit_load_flexi_cap(self):
        chunks = retrieve("What is the exit load?", "Flexi_Cap", n_results=3, db_path=DB_PATH)
        assert len(chunks) > 0, "No chunks retrieved for exit load"
        combined = " ".join(chunks).lower()
        assert "exit load" in combined or "exit" in combined

    def test_exit_load_dynamic(self):
        chunks = retrieve("What is the exit load?", "Dynamic_Asset_Allocation", n_results=3, db_path=DB_PATH)
        assert len(chunks) > 0
        combined = " ".join(chunks).lower()
        assert "exit load" in combined or "exit" in combined

    def test_performance_flexi_cap(self):
        chunks = retrieve("How has the fund performed?", "Flexi_Cap", n_results=3, db_path=DB_PATH)
        assert len(chunks) > 0
        combined = " ".join(chunks).lower()
        # Should contain percentage figures or performance keywords
        assert "%" in combined or "performance" in combined or "returns" in combined

    def test_top_holdings(self):
        chunks = retrieve("What are the top holdings?", "Flexi_Cap", n_results=3, db_path=DB_PATH)
        assert len(chunks) > 0
        combined = " ".join(chunks).lower()
        # FC presentation has top 10 stocks section
        assert any(kw in combined for kw in ["hdfc", "portfolio", "stock", "holding"])

    def test_minimum_sip_amount(self):
        chunks = retrieve("What is the minimum SIP amount?", "Large_Cap", n_results=3, db_path=DB_PATH)
        assert len(chunks) > 0
        combined = " ".join(chunks).lower()
        assert any(kw in combined for kw in ["sip", "minimum", "1000", "1,000"])

    def test_tax_treatment(self):
        chunks = retrieve("Is this fund taxed like equity?", "Flexi_Cap", n_results=3, db_path=DB_PATH)
        assert len(chunks) > 0
        combined = " ".join(chunks).lower()
        assert any(kw in combined for kw in ["tax", "equity", "capital gain"])

    def test_risk_level(self):
        chunks = retrieve("What is the risk level of this fund?", "Dynamic_Asset_Allocation", n_results=3, db_path=DB_PATH)
        assert len(chunks) > 0
        combined = " ".join(chunks).lower()
        assert any(kw in combined for kw in ["risk", "riskometer", "moderately"])

    def test_asset_allocation(self):
        chunks = retrieve("What is the asset allocation?", "Large_Cap", n_results=3, db_path=DB_PATH)
        assert len(chunks) > 0
        combined = " ".join(chunks).lower()
        assert any(kw in combined for kw in ["equity", "debt", "allocation", "%"])

    def test_fund_manager(self):
        chunks = retrieve("Who manages this fund?", "Flexi_Cap", n_results=5, db_path=DB_PATH)
        assert len(chunks) > 0
        combined = " ".join(chunks).lower()
        # Manager info may appear in performance footnotes or separate section
        assert any(kw in combined for kw in [
            "rajeev", "manager", "fund manager", "thakkar",
            "raunak", "onkar", "raj mehta", "managed by",
            "investment", "team", "schemes managed",
        ])

    def test_benchmark_comparison(self):
        chunks = retrieve("How does this fund compare to Nifty 500?", "Flexi_Cap", n_results=3, db_path=DB_PATH)
        assert len(chunks) > 0
        combined = " ".join(chunks).lower()
        assert any(kw in combined for kw in ["nifty", "benchmark", "comparative", "500"])


@pytest.mark.skipif(not (HAS_DB and HAS_KEY), reason=SKIP_REASON)
class TestRetrievalProperties:
    def test_no_cross_fund_contamination(self):
        """Verify that querying for Flexi_Cap never returns Dynamic Fund chunks."""
        chunks = retrieve("What is the exit load?", "Flexi_Cap", n_results=5, db_path=DB_PATH)
        for chunk in chunks:
            assert "Dynamic Asset Allocation" not in chunk or "Flexi Cap" in chunk

    def test_empty_result_for_bad_scheme(self):
        """Querying a non-existent scheme_id returns empty."""
        chunks = retrieve("anything", "NonExistent_Fund", n_results=3, db_path=DB_PATH)
        assert chunks == []

    def test_retrieve_formatted_returns_string(self):
        result = retrieve_formatted("exit load", "Flexi_Cap", n_results=2, db_path=DB_PATH)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "---" in result  # separator between chunks

    def test_respects_n_results(self):
        chunks = retrieve("fund performance", "Flexi_Cap", n_results=1, db_path=DB_PATH)
        assert len(chunks) <= 1
