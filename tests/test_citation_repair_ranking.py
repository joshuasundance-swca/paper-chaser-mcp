"""TDD tests for the extracted citation_repair/ranking.py module.

Phase 2 Track B Step 2 (RED→GREEN). These tests target
``paper_chaser_mcp.citation_repair.ranking`` directly so they:

1. Force the extraction (the module must exist + export the named symbols).
2. Pin today's scoring behavior — including the known title-weight vs
   year-mismatch-penalty imbalance that Phase 9b is scheduled to fix. Tests
   named ``*_pinned_for_phase_9b`` exist *specifically* so Phase 9b's fix
   lights them up as intentional red gates; do not "fix" them here. Tests
   named without that suffix encode today's correctness invariants that are
   expected to stay green forever.

The helpers under test (``_rank_candidate``, ``_author_overlap``,
``_year_delta``, ``_venue_overlap``, ``_identifier_hit``, ``_snippet_alignment``,
``_token_overlap_ratio``, ``_publication_preference_score``) and the
``RankedCitationCandidate`` dataclass are also importable via the
``paper_chaser_mcp.citation_repair`` package facade; a handful of aliasing
assertions at the bottom of this file pin that re-export so Phase 9b cannot
break it silently.
"""

from __future__ import annotations

from typing import Any

import pytest

from paper_chaser_mcp.citation_repair import _core, parse_citation
from paper_chaser_mcp.citation_repair import ranking as ranking_module
from paper_chaser_mcp.citation_repair.ranking import (
    RankedCitationCandidate,
    _author_overlap,
    _identifier_hit,
    _publication_preference_score,
    _rank_candidate,
    _snippet_alignment,
    _token_overlap_ratio,
    _venue_overlap,
    _year_delta,
)

# ---------------------------------------------------------------------------
# RankedCitationCandidate dataclass
# ---------------------------------------------------------------------------


def test_ranked_citation_candidate_construction_basic() -> None:
    """The dataclass stores every scoring-engine output field verbatim."""
    paper = {"paperId": "p1", "title": "A Title"}
    ranked = RankedCitationCandidate(
        paper=paper,
        score=0.42,
        resolution_strategy="fuzzy_search",
        matched_fields=["title"],
        conflicting_fields=["year"],
        title_similarity=0.9,
        year_delta=3,
        author_overlap=2,
        candidate_count=5,
        why_selected="matched on title via fuzzy_search.",
    )

    assert ranked.paper is paper
    assert ranked.score == pytest.approx(0.42)
    assert ranked.resolution_strategy == "fuzzy_search"
    assert ranked.matched_fields == ["title"]
    assert ranked.conflicting_fields == ["year"]
    assert ranked.title_similarity == pytest.approx(0.9)
    assert ranked.year_delta == 3
    assert ranked.author_overlap == 2
    assert ranked.candidate_count == 5
    assert "fuzzy_search" in ranked.why_selected


def test_ranked_citation_candidate_allows_none_for_optional_scalars() -> None:
    """``year_delta`` and ``candidate_count`` accept ``None`` (unknown/missing)."""
    ranked = RankedCitationCandidate(
        paper={},
        score=0.0,
        resolution_strategy="snippet_recovery",
        matched_fields=[],
        conflicting_fields=[],
        title_similarity=0.0,
        year_delta=None,
        author_overlap=0,
        candidate_count=None,
        why_selected="weak fallback",
    )

    assert ranked.year_delta is None
    assert ranked.candidate_count is None


def test_ranked_citation_candidate_has_slots() -> None:
    """Pin ``slots=True`` so the dataclass stays memory-friendly."""
    ranked = RankedCitationCandidate(
        paper={},
        score=0.0,
        resolution_strategy="identifier",
        matched_fields=[],
        conflicting_fields=[],
        title_similarity=0.0,
        year_delta=None,
        author_overlap=0,
        candidate_count=None,
        why_selected="",
    )
    with pytest.raises(AttributeError):
        ranked.extra_field = "should not be settable"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# _rank_candidate: scoring-engine invariants
# ---------------------------------------------------------------------------


def _parsed_watson_crick():
    return parse_citation("Watson Crick 1953 molecular structure nucleic acids Nature")


def test_rank_candidate_identifier_hit_outranks_pure_title_match() -> None:
    """Exact DOI/identifier hit must outrank a same-year, same-venue title match."""
    parsed = parse_citation("10.1038/171737a0 Watson Crick nucleic acids")

    identifier_paper = {
        "paperId": "doi-paper",
        "title": "Completely unrelated headline",
        "year": 1953,
        "authors": [{"name": "Somebody Else"}],
        "venue": "Unrelated",
        "externalIds": {"DOI": "10.1038/171737a0"},
    }
    title_paper = {
        "paperId": "title-paper",
        "title": "Molecular structure of nucleic acids",
        "year": 1953,
        "authors": [{"name": "James Watson"}, {"name": "Francis Crick"}],
        "venue": "Nature",
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


def test_rank_candidate_title_match_outranks_snippet_only_match() -> None:
    """A title-similarity hit must outrank a pure snippet-alignment hit."""
    parsed = _parsed_watson_crick()

    title_paper: dict[str, Any] = {
        "paperId": "titled",
        "title": "Molecular structure of nucleic acids",
        "year": None,  # neutralize year bonus/penalty to isolate title vs. snippet
        "authors": [],
        "venue": "",
    }
    snippet_paper: dict[str, Any] = {
        "paperId": "snippet-only",
        "title": "Something totally different",
        "abstract": "Watson and Crick 1953 discovered molecular structure nucleic acids Nature",
        "year": None,
        "authors": [],
        "venue": "",
    }

    title_ranked = _rank_candidate(
        paper=title_paper,
        parsed=parsed,
        resolution_strategy="fuzzy_search",
        candidate_count=2,
        snippet_text=None,
    )
    snippet_ranked = _rank_candidate(
        paper=snippet_paper,
        parsed=parsed,
        resolution_strategy="snippet_recovery",
        candidate_count=2,
        snippet_text="Watson Crick 1953 molecular structure nucleic acids Nature",
    )

    assert title_ranked.score > snippet_ranked.score


def test_rank_candidate_year_mismatch_reduces_score() -> None:
    """A large year delta must penalize the score vs. an exact-year twin."""
    parsed = _parsed_watson_crick()

    exact_year = {
        "paperId": "exact",
        "title": "molecular structure of nucleic acids",
        "year": 1953,
        "authors": [{"name": "James Watson"}],
        "venue": "Nature",
    }
    wrong_year = {
        "paperId": "wrong",
        "title": "molecular structure of nucleic acids",
        "year": 2020,
        "authors": [{"name": "James Watson"}],
        "venue": "Nature",
    }

    exact_ranked = _rank_candidate(
        paper=exact_year,
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

    assert exact_ranked.score > wrong_ranked.score


def test_rank_candidate_handles_paper_missing_authors_field() -> None:
    """``_rank_candidate`` must not crash when the paper has no authors."""
    parsed = _parsed_watson_crick()

    paper = {
        "paperId": "missing-authors",
        "title": "Molecular structure of nucleic acids",
        "year": 1953,
        "venue": "Nature",
    }

    ranked = _rank_candidate(
        paper=paper,
        parsed=parsed,
        resolution_strategy="fuzzy_search",
        candidate_count=1,
        snippet_text=None,
    )
    assert ranked.author_overlap == 0
    assert 0.0 <= ranked.score <= 1.0


def test_rank_candidate_applies_upstream_high_confidence_bonus() -> None:
    """``matchConfidence == "high"`` from the provider must nudge the score up."""
    parsed = _parsed_watson_crick()

    base_paper = {
        "paperId": "base",
        "title": "molecular structure of nucleic acids",
        "year": 1953,
        "authors": [{"name": "James Watson"}, {"name": "Francis Crick"}],
        "venue": "Nature",
    }
    high_confidence = dict(base_paper, matchConfidence="high")

    base_ranked = _rank_candidate(
        paper=base_paper,
        parsed=parsed,
        resolution_strategy="fuzzy_search",
        candidate_count=1,
        snippet_text=None,
    )
    high_ranked = _rank_candidate(
        paper=high_confidence,
        parsed=parsed,
        resolution_strategy="fuzzy_search",
        candidate_count=1,
        snippet_text=None,
    )
    assert high_ranked.score >= base_ranked.score


def test_rank_candidate_score_is_clamped_to_unit_interval() -> None:
    """Scores are always clamped into ``[0.0, 1.0]`` inclusive."""
    parsed = _parsed_watson_crick()

    over_boosted_paper = {
        "paperId": "boosted",
        "title": "molecular structure of nucleic acids",
        "year": 1953,
        "authors": [{"name": "James Watson"}, {"name": "Francis Crick"}],
        "venue": "Nature",
        "matchConfidence": "high",
        "externalIds": {"DOI": "10.1038/171737a0"},
    }
    parsed_with_identifier = parse_citation("10.1038/171737a0 Watson Crick molecular structure")

    ranked = _rank_candidate(
        paper=over_boosted_paper,
        parsed=parsed_with_identifier,
        resolution_strategy="identifier",
        candidate_count=1,
        snippet_text="Watson Crick 1953 molecular structure nucleic acids Nature",
    )
    assert 0.0 <= ranked.score <= 1.0

    low_paper: dict[str, Any] = {"paperId": "empty", "title": "", "year": None, "authors": [], "venue": ""}
    low_ranked = _rank_candidate(
        paper=low_paper,
        parsed=parsed,
        resolution_strategy="sparse_metadata",
        candidate_count=1,
        snippet_text=None,
    )
    assert 0.0 <= low_ranked.score <= 1.0


# ---------------------------------------------------------------------------
# Phase 9b weight pins
#
# The names below end in ``_pinned_for_phase_9b`` so Phase 9b (ranking bias
# fix) can update them in one diff and it is obvious from the test name
# whether Phase 9b should be allowed to flip them. Changing these pins in
# earlier phases would mask the very bias Phase 9b is supposed to fix.
# ---------------------------------------------------------------------------


def test_rank_candidate_title_weight_pinned_for_phase_9b() -> None:
    """Pin the title-similarity weight at 0.35.

    Given a paper with a perfect title match, no identifier hit, no authors,
    no year signal, no venue, no snippet, ``sparse_metadata`` strategy
    (source_confidence=0.65), the resulting score should be exactly the title
    component plus the source-confidence component:

        title_similarity (1.0) * 0.35 + source_confidence (0.65) * 0.05
        = 0.35 + 0.0325 = 0.3825

    Phase 9b is expected to rebalance this weight relative to the
    year-mismatch penalty; this pin will need to be updated in the same
    commit that lands the rebalance.
    """
    parsed = parse_citation("molecular structure of nucleic acids")
    paper: dict[str, Any] = {
        "paperId": "title-only",
        "title": "molecular structure of nucleic acids",
        "year": None,
        "authors": [],
        "venue": "",
    }

    ranked = _rank_candidate(
        paper=paper,
        parsed=parsed,
        resolution_strategy="sparse_metadata",
        candidate_count=1,
        snippet_text=None,
    )
    assert ranked.score == pytest.approx(0.35 + 0.65 * 0.05, abs=1e-6)
    assert ranked.title_similarity == pytest.approx(1.0)


def test_rank_candidate_match_confidence_high_bonus_pinned_for_phase_9b() -> None:
    """Pin the upstream ``matchConfidence=="high"`` bonus at 0.15.

    With a minimal paper (no title, no identifier, no author overlap, no
    year, no venue, no snippet) on ``sparse_metadata`` strategy the score
    should equal the source_confidence component (0.65 * 0.05 = 0.0325)
    plus the 0.15 high-confidence bonus = 0.1825. No other term can fire.
    """
    parsed = parse_citation("unknown paper")
    paper: dict[str, Any] = {
        "paperId": "blank",
        "title": "",
        "year": None,
        "authors": [],
        "venue": "",
        "matchConfidence": "high",
    }
    ranked = _rank_candidate(
        paper=paper,
        parsed=parsed,
        resolution_strategy="sparse_metadata",
        candidate_count=1,
        snippet_text=None,
    )
    assert ranked.score == pytest.approx(0.65 * 0.05 + 0.15, abs=1e-6)


def test_rank_candidate_match_confidence_medium_bonus_pinned_for_phase_9b() -> None:
    """Pin the upstream ``matchConfidence=="medium"`` bonus at 0.08."""
    parsed = parse_citation("unknown paper")
    paper: dict[str, Any] = {
        "paperId": "blank-med",
        "title": "",
        "year": None,
        "authors": [],
        "venue": "",
        "matchConfidence": "medium",
    }
    ranked = _rank_candidate(
        paper=paper,
        parsed=parsed,
        resolution_strategy="sparse_metadata",
        candidate_count=1,
        snippet_text=None,
    )
    assert ranked.score == pytest.approx(0.65 * 0.05 + 0.08, abs=1e-6)


def test_rank_candidate_year_mismatch_penalty_bounded_for_phase_9b() -> None:
    """Pin the year-mismatch penalty shape: capped at delta=10 + >5 kicker.

    The penalty terms today are:
        - subtract ``min(year_delta, 10) * 0.04``
        - subtract an additional ``0.06`` when ``year_delta > 5``

    For a perfect-title, no-other-signal paper with year_delta = 50 the
    penalty should saturate at delta=10:

        title_similarity * 0.35
        + source_confidence(fuzzy_search=0.82) * 0.05
        - min(50, 10) * 0.04
        - 0.06   # year_delta > 5 kicker
        = 0.35 + 0.041 - 0.4 - 0.06 = -0.069   → clamped to 0.0

    This pin documents the known imbalance: large year deltas can drive the
    score fully to zero even when the title matches perfectly. Phase 9b
    should soften or rebalance this and update the pin.
    """
    parsed = parse_citation("some paper 1950 nature")
    paper = {
        "paperId": "way-off-year",
        "title": "some paper 1950 nature",
        "year": 2000,  # delta = 50
        "authors": [],
        "venue": "",
    }
    ranked = _rank_candidate(
        paper=paper,
        parsed=parsed,
        resolution_strategy="fuzzy_search",
        candidate_count=1,
        snippet_text=None,
    )
    assert ranked.score == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Ranking-score helpers
# ---------------------------------------------------------------------------


def test_author_overlap_counts_surnames_case_insensitive() -> None:
    parsed = parse_citation("Watson Crick 1953 Nature molecular structure nucleic acids")
    paper = {
        "authors": [{"name": "James D. Watson"}, {"name": "Francis H.C. Crick"}, {"name": "Rosalind Franklin"}],
    }
    assert _author_overlap(parsed, paper) == 2


def test_author_overlap_returns_zero_when_parsed_has_no_authors() -> None:
    parsed = parse_citation("molecular structure")
    assert not parsed.author_surnames  # defensive precondition
    assert _author_overlap(parsed, {"authors": [{"name": "Someone"}]}) == 0


def test_year_delta_returns_absolute_difference() -> None:
    parsed = parse_citation("title 2000 venue")
    assert _year_delta(parsed, {"year": 2010}) == 10
    assert _year_delta(parsed, {"year": 1990}) == 10
    assert _year_delta(parsed, {"year": 2000}) == 0


def test_year_delta_returns_none_when_either_side_is_missing() -> None:
    parsed_no_year = parse_citation("some title no year venue")
    assert _year_delta(parsed_no_year, {"year": 2010}) is None

    parsed = parse_citation("title 2000 venue")
    assert _year_delta(parsed, {"year": None}) is None
    assert _year_delta(parsed, {}) is None
    assert _year_delta(parsed, {"year": "not-a-year"}) is None


def test_venue_overlap_matches_substring_either_direction() -> None:
    parsed = parse_citation("some title 2019 Nature Communications")
    assert _venue_overlap(parsed, {"venue": "Nature Communications"})
    assert _venue_overlap(parsed, {"venue": "Nature"})  # paper venue inside hint


def test_venue_overlap_returns_false_when_parsed_has_no_venue_hints() -> None:
    parsed = parse_citation("molecular structure")
    assert not parsed.venue_hints
    assert _venue_overlap(parsed, {"venue": "Nature"}) is False


def test_identifier_hit_doi_normalization() -> None:
    parsed = parse_citation("10.1038/171737a0 some paper")
    paper = {"externalIds": {"DOI": "10.1038/171737a0"}}
    assert _identifier_hit(parsed, paper) is True

    parsed_with_prefix = parse_citation("doi:10.1038/171737a0 some paper")
    assert _identifier_hit(parsed_with_prefix, {"externalIds": {"DOI": "10.1038/171737a0"}})


def test_identifier_hit_returns_false_when_parsed_has_no_identifier() -> None:
    parsed = parse_citation("no identifier in here")
    assert parsed.identifier is None
    assert _identifier_hit(parsed, {"paperId": "abc", "externalIds": {"DOI": "10.1038/x"}}) is False


def test_snippet_alignment_is_zero_without_snippet() -> None:
    parsed = _parsed_watson_crick()
    assert _snippet_alignment(parsed, {"title": "whatever"}, snippet_text=None) == 0.0


def test_snippet_alignment_is_positive_on_token_overlap() -> None:
    parsed = _parsed_watson_crick()
    paper = {"title": "Molecular structure of nucleic acids", "abstract": "1953 Nature paper"}
    score = _snippet_alignment(parsed, paper, snippet_text="molecular structure nucleic acids")
    assert 0.0 < score <= 1.0


def test_token_overlap_ratio_basic() -> None:
    assert _token_overlap_ratio("hello world python", "hello world other") == pytest.approx(2 / 3)


def test_token_overlap_ratio_empty_inputs_are_zero() -> None:
    assert _token_overlap_ratio("", "something here") == 0.0
    assert _token_overlap_ratio("something here", "") == 0.0


def test_publication_preference_score_rewards_journal_articles_with_doi() -> None:
    journal = {
        "doi": "10.1000/x",
        "venue": "Some Journal",
        "publicationTypes": ["journal-article"],
    }
    preprint = {
        "doi": "",
        "venue": "arXiv",
        "publicationTypes": ["preprint"],
        "source": "arxiv",
    }
    assert _publication_preference_score(journal) > _publication_preference_score(preprint)


def test_publication_preference_score_penalizes_arxiv_preprints() -> None:
    preprint = {
        "doi": "",
        "venue": "arXiv",
        "publicationTypes": ["preprint"],
        "source": "arxiv",
    }
    assert _publication_preference_score(preprint) < 0.0


# ---------------------------------------------------------------------------
# Re-export / facade pins
# ---------------------------------------------------------------------------


def test_ranking_symbols_are_still_reachable_via_citation_repair_facade() -> None:
    """Every ranking symbol must remain reachable via the package facade.

    Phase 9b and later work may refactor internal callers, but the public
    ``paper_chaser_mcp.citation_repair`` import surface must keep these
    names resolvable so existing tests + callers keep working.
    """
    import paper_chaser_mcp.citation_repair as facade

    assert facade.RankedCitationCandidate is RankedCitationCandidate
    assert facade._rank_candidate is _rank_candidate


def test_ranking_symbols_are_exported_by_core_shim_for_backward_compat() -> None:
    """``_core`` re-exports the ranking helpers so in-module callers keep working."""
    assert _core._rank_candidate is _rank_candidate
    assert _core._author_overlap is _author_overlap
    assert _core._year_delta is _year_delta
    assert _core._venue_overlap is _venue_overlap
    assert _core._identifier_hit is _identifier_hit
    assert _core._snippet_alignment is _snippet_alignment
    assert _core._token_overlap_ratio is _token_overlap_ratio
    assert _core._publication_preference_score is _publication_preference_score
    assert _core.RankedCitationCandidate is RankedCitationCandidate


def test_ranking_module_advertises_public_helpers_via___all__() -> None:
    """The ranking submodule should declare the helpers it owns in ``__all__``."""
    assert hasattr(ranking_module, "__all__")
    assert "_rank_candidate" in ranking_module.__all__
    assert "RankedCitationCandidate" in ranking_module.__all__
