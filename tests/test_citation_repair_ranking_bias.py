"""Phase 9b red→green tests for citation ranking bias.

These scenarios express the *corrected* ranking semantics:

1. A cited year that disagrees by ≥2 years must dominate a slightly higher
   title similarity. A newer/older paper with a similar title must not
   outrank a candidate whose year matches the parsed citation year.
2. When the parsed citation supplies author surnames and the candidate has
   author data but zero surname overlap, a high upstream ``matchConfidence``
   must not outrank a candidate with matching surname overlap and a lower
   ``matchConfidence``.
3. An identifier (DOI/arXiv) near-match must outrank a pure title match.
   This is a regression pin — the current formula already honors it; the
   test fixes the invariant so Phase 9b's rebalance cannot accidentally
   weaken it.
4. Title similarity alone (without any corroborating year/author/identifier
   signal) must cap below the 0.9 "high-confidence" threshold used by
   ``resolve_citation``. Otherwise a title-only pattern match would be
   advertised as a confident resolution.

Scenarios 1 and 2 fail RED against the pre-Phase-9b formula: current
weights (title 0.35, year-delta=2 penalty only 0.08, upstream high bonus
0.15 with only a 0.06 cap on year conflict, and no gating for author-
surname disagreement) let a high-matchConfidence candidate with a shinier
title beat the canonical match. Phase 9b strengthens the year-disagreement
penalty and gates matchConfidence on author-surname corroboration.
"""

from __future__ import annotations

from typing import Any

import pytest

from paper_chaser_mcp.citation_repair.candidates import ParsedCitation
from paper_chaser_mcp.citation_repair.ranking import _rank_candidate

# ---------------------------------------------------------------------------
# Scenario 1 — Year-conflict dominance.
# ---------------------------------------------------------------------------


def test_year_conflict_beats_higher_title_similarity() -> None:
    """A 2+ year mismatch must outweigh a small title-similarity edge.

    Construction: the parsed citation supplies a year (2020) but neither
    surnames nor a venue, so only the year signal distinguishes the two
    candidates.

    * ``matching_year``: title_sim forced to 0.6, year = 2020 (exact).
    * ``wrong_year``:     title_sim forced to 1.0, year = 2022 (delta=2),
      ``matchConfidence="high"`` from upstream.

    Pre-Phase-9b the ``wrong_year`` candidate wins (score ≈ 0.371 vs
    0.331) because the year-delta-2 penalty is only -0.08 and the
    upstream high-confidence bonus contributes +0.06 even under year
    conflict. After Phase 9b, year disagreement ≥2 must dominate, so the
    ``matching_year`` candidate must rank higher.
    """
    parsed = ParsedCitation(
        original_text="neural networks 2020",
        normalized_text="neural networks 2020",
        year=2020,
        title_candidates=["neural networks"],
    )

    matching_year: dict[str, Any] = {
        "paperId": "matching-year",
        "title": "unrelated placeholder text",
        "titleSimilarity": 0.6,
        "year": 2020,
        "authors": [],
    }
    wrong_year: dict[str, Any] = {
        "paperId": "wrong-year",
        "title": "unrelated placeholder text",
        "titleSimilarity": 1.0,
        "year": 2022,
        "authors": [],
        "matchConfidence": "high",
    }

    matching_ranked = _rank_candidate(
        paper=matching_year,
        parsed=parsed,
        resolution_strategy="fuzzy_search",
        candidate_count=2,
        snippet_text=None,
    )
    wrong_ranked = _rank_candidate(
        paper=wrong_year,
        parsed=parsed,
        resolution_strategy="fuzzy_search",
        candidate_count=2,
        snippet_text=None,
    )

    assert matching_ranked.score > wrong_ranked.score, (
        "A candidate whose year matches the parsed citation must outrank a "
        "year-mismatched candidate even when the latter has a higher title "
        "similarity and upstream high matchConfidence."
    )


def test_large_year_gap_strongly_dominates_title_signal() -> None:
    """A 5+ year gap must carry at least a 0.15 penalty even for perfect titles.

    This pins the lower bound of the Phase 9b year-disagreement penalty:
    with everything else held constant (fuzzy_search strategy, no authors,
    no venue, no upstream bonus), a candidate with ``year_delta == 5``
    must score at least 0.15 below an otherwise identical candidate whose
    year matches. The 0.15 delta comes from combining:

    * losing the +0.08 year-match bonus, and
    * paying at least -0.15 for year disagreement ≥2.
    """
    parsed = ParsedCitation(
        original_text="neural networks 2020",
        normalized_text="neural networks 2020",
        year=2020,
        title_candidates=["neural networks"],
    )

    matching: dict[str, Any] = {
        "paperId": "match-year",
        "title": "placeholder",
        "titleSimilarity": 1.0,
        "year": 2020,
        "authors": [],
    }
    off_by_five: dict[str, Any] = {
        "paperId": "off-year",
        "title": "placeholder",
        "titleSimilarity": 1.0,
        "year": 2025,
        "authors": [],
    }

    matching_ranked = _rank_candidate(
        paper=matching,
        parsed=parsed,
        resolution_strategy="fuzzy_search",
        candidate_count=2,
        snippet_text=None,
    )
    off_ranked = _rank_candidate(
        paper=off_by_five,
        parsed=parsed,
        resolution_strategy="fuzzy_search",
        candidate_count=2,
        snippet_text=None,
    )

    assert matching_ranked.score - off_ranked.score >= 0.22, (
        "Year mismatch of 5 must cost at least 0.22 (year-match bonus +0.08 "
        "plus year-disagreement penalty ≥0.15) against an otherwise identical "
        "candidate with the same title."
    )


# ---------------------------------------------------------------------------
# Scenario 2 — Author-surname disagreement gates matchConfidence.
# ---------------------------------------------------------------------------


def test_author_surname_disagreement_gates_match_confidence() -> None:
    """Zero surname overlap must gate an upstream high ``matchConfidence``.

    Construction: the parsed citation declares surname ``smith``. Both
    candidates share the same year (neutralizing the year signal).

    * ``surname_match``: title_sim=0.7, authors contain "John Smith",
      ``matchConfidence="medium"``.
    * ``surname_mismatch``: title_sim=1.0, authors are "Jane Doe", zero
      overlap with the parsed surname, yet ``matchConfidence="high"``.

    Pre-Phase-9b ``surname_mismatch`` wins (≈0.621 vs 0.526) because the
    upstream high-confidence bonus (+0.15) is not gated when authors
    disagree but everything else is quiet. Phase 9b must either apply a
    ≥0.15 penalty for author-surname disagreement or damp the upstream
    bonus enough that the corroborated candidate ranks higher.
    """
    parsed = ParsedCitation(
        original_text="smith 2020 neural networks",
        normalized_text="smith 2020 neural networks",
        year=2020,
        title_candidates=["neural networks"],
        author_surnames=["smith"],
    )

    surname_match: dict[str, Any] = {
        "paperId": "surname-match",
        "title": "placeholder",
        "titleSimilarity": 0.7,
        "year": 2020,
        "authors": [{"name": "John Smith"}],
        "matchConfidence": "medium",
    }
    surname_mismatch: dict[str, Any] = {
        "paperId": "surname-mismatch",
        "title": "placeholder",
        "titleSimilarity": 1.0,
        "year": 2020,
        "authors": [{"name": "Jane Doe"}],
        "matchConfidence": "high",
    }

    match_ranked = _rank_candidate(
        paper=surname_match,
        parsed=parsed,
        resolution_strategy="fuzzy_search",
        candidate_count=2,
        snippet_text=None,
    )
    mismatch_ranked = _rank_candidate(
        paper=surname_mismatch,
        parsed=parsed,
        resolution_strategy="fuzzy_search",
        candidate_count=2,
        snippet_text=None,
    )

    assert match_ranked.score > mismatch_ranked.score, (
        "A candidate whose author surnames match the parsed citation must "
        "outrank a candidate with zero surname overlap even when the latter "
        "reports a higher upstream matchConfidence and a shinier title."
    )


# ---------------------------------------------------------------------------
# Scenario 3 — Identifier near-match beats pure title (regression pin).
# ---------------------------------------------------------------------------


def test_identifier_hit_beats_pure_title_match() -> None:
    """A DOI identifier hit must outrank a pure high-title-similarity hit.

    This is a regression pin: the existing formula already gives
    identifier hits a +0.55 head-start, but Phase 9b's rebalance of
    title/year/author weights must not accidentally erode that lead.
    """
    parsed = ParsedCitation(
        original_text="10.1038/171737a0",
        normalized_text="10.1038/171737a0",
        identifier="10.1038/171737a0",
        identifier_type="doi",
        title_candidates=["molecular structure of nucleic acids"],
    )

    identifier_paper: dict[str, Any] = {
        "paperId": "doi-hit",
        "title": "totally unrelated heading",
        "titleSimilarity": 0.5,
        "year": None,
        "authors": [],
        "externalIds": {"DOI": "10.1038/171737a0"},
    }
    title_paper: dict[str, Any] = {
        "paperId": "title-only",
        "title": "molecular structure of nucleic acids",
        "titleSimilarity": 0.95,
        "year": None,
        "authors": [],
    }

    identifier_ranked = _rank_candidate(
        paper=identifier_paper,
        parsed=parsed,
        resolution_strategy="identifier",
        candidate_count=2,
        snippet_text=None,
    )
    title_ranked = _rank_candidate(
        paper=title_paper,
        parsed=parsed,
        resolution_strategy="fuzzy_search",
        candidate_count=2,
        snippet_text=None,
    )

    assert identifier_ranked.score > title_ranked.score
    assert "identifier" in identifier_ranked.matched_fields


# ---------------------------------------------------------------------------
# Scenario 4 — Title similarity alone cannot reach the high-confidence band.
# ---------------------------------------------------------------------------


def test_pure_title_match_cannot_reach_high_confidence_threshold() -> None:
    """Title similarity with no corroboration must stay below 0.9.

    With no year, no author overlap, no identifier hit, and even an
    upstream ``matchConfidence="high"``, a 1.0 title match must not
    reach the 0.9 high-confidence band that ``resolve_citation`` uses
    to escalate to an authoritative answer. Phase 9b's matchConfidence
    ceiling for uncorroborated candidates keeps this invariant true.
    """
    parsed = ParsedCitation(
        original_text="some obscure paper",
        normalized_text="some obscure paper",
        title_candidates=["some obscure paper"],
    )

    title_only: dict[str, Any] = {
        "paperId": "title-only",
        "title": "some obscure paper",
        "titleSimilarity": 1.0,
        "year": None,
        "authors": [],
        "matchConfidence": "high",
    }

    ranked = _rank_candidate(
        paper=title_only,
        parsed=parsed,
        resolution_strategy="fuzzy_search",
        candidate_count=1,
        snippet_text=None,
    )

    assert ranked.score < 0.9, (
        "A pure title-similarity match with no year, no author overlap, and "
        "no identifier hit must not reach the 0.9 high-confidence band even "
        "when upstream reports matchConfidence='high'."
    )
    # And the title similarity we measured must indeed be 1.0 — otherwise
    # the assertion above is vacuous.
    assert ranked.title_similarity == pytest.approx(1.0)
