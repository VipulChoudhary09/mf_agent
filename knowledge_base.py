"""
Knowledge Base Loader for PPFAS Mutual Fund Voice Agent.

Loads the 6 extracted markdown files (flyers + presentations) for the 3 PPFAS
schemes at startup. Provides per-scheme context for injection into the LLM
system prompt, and a hardcoded comparison table for cross-fund questions.

Total KB size: ~104KB — fits easily in Gemini's context window. No vector DB needed.
"""

import os
from pathlib import Path

# Base directory for knowledge base markdown files
KB_DIR = Path(__file__).parent / "KB" / "pdftomds"

# Mapping of scheme IDs to their source files (flyer + presentation)
SCHEME_FILES: dict[str, list[str]] = {
    "Dynamic_Asset_Allocation": ["Dfly.md", "Dpres.md"],
    "Large_Cap": ["LCfly.md", "LCpres.md"],
    "Flexi_Cap": ["FCfly.md", "FCpres.md"],
}

# Module-level cache — populated by load_all()
_cache: dict[str, str] = {}


def load_all() -> None:
    """Read all 6 MD files into the module-level cache.

    Call once at startup. Each scheme gets its flyer + presentation
    concatenated with a separator.
    """
    for scheme_id, filenames in SCHEME_FILES.items():
        parts = []
        for filename in filenames:
            filepath = KB_DIR / filename
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Knowledge base file missing: {filepath}"
                )
            parts.append(filepath.read_text(encoding="utf-8"))
        _cache[scheme_id] = (
            f"\n\n{'='*60}\n"
            f"=== {scheme_id.replace('_', ' ').upper()} — FLYER ===\n"
            f"{'='*60}\n\n"
            f"{parts[0]}"
            f"\n\n{'='*60}\n"
            f"=== {scheme_id.replace('_', ' ').upper()} — PRESENTATION ===\n"
            f"{'='*60}\n\n"
            f"{parts[1]}"
        )


def get_scheme_context(scheme_id: str) -> str:
    """Return combined flyer + presentation text for a scheme.

    Args:
        scheme_id: One of 'Dynamic_Asset_Allocation', 'Large_Cap', 'Flexi_Cap'

    Returns:
        The full markdown knowledge base text for this scheme.

    Raises:
        KeyError: If scheme_id is not found (call load_all() first).
    """
    if not _cache:
        raise RuntimeError("Knowledge base not loaded. Call load_all() first.")
    if scheme_id not in _cache:
        raise KeyError(
            f"Unknown scheme: {scheme_id}. "
            f"Valid IDs: {list(SCHEME_FILES.keys())}"
        )
    return _cache[scheme_id]


def get_comparison_table() -> str:
    """Return a hardcoded comparison table for cross-fund questions.

    This table is always available in the RECOMMENDATION prompt so the
    agent can answer "how does this compare to the other funds?" questions.
    """
    return """\
| Feature | Dynamic Asset Allocation | Large Cap | Flexi Cap |
|:--------|:------------------------|:----------|:----------|
| Inception | Feb 27, 2024 (~2 yrs) | Feb 4, 2026 (~4 mo) | May 24, 2013 (~13 yrs) |
| Risk Level | Moderately High | Very High | Very High |
| Benchmark | CRISIL Hybrid 50+50 Moderate | Nifty 100 TRI | Nifty 500 TRI |
| Min Lumpsum | ₹5,000 | ₹1,000 | ₹1,000 |
| Min SIP | ₹1,000/month | ₹1,000/month | ₹1,000/month |
| Exit Load | 1% within 1 yr | NIL | 2% ≤1yr, 1% ≤2yr |
| TER (Direct) | Not disclosed | 0.15% | Not disclosed |
| Global Stocks | No | No | Yes (e.g. Alphabet) |
| Since Inception CAGR (Direct) | 7.49% | N/A (too new) | 18.63% |
| Equity Range | 0-100% (dynamic) | 80-100% (Nifty 100) | 65-100% (flexible) |"""
