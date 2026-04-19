"""Tests for Phase 5 citation repair fixes.

Covers three areas:
1. Year penalty strengthening in _rank_candidate()
2. conflictingFields / confidence classification consistency
3. Title similarity length-difference penalty
"""

from __future__ import annotations

from paper_chaser_mcp.citation_repair import (
    ParsedCitation,
    _classify_resolution_confidence,
    _rank_candidate,
    _title_similarity,
    build_match_metadata,
    normalize_citation_text,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parsed(
    *,
    text: str = "some citation",
    year: int | None = None,
    title_candidates: list[str] | None = None,
    author_surnames: list[str] | None = None,
    venue_hints: list[str] | None = None,
) -> ParsedCitation:
    return ParsedCitation(
        original_text=text,
        normalized_text=normalize_citation_text(text),
        year=year,
        title_candidates=title_candidates or [],
        author_surnames=author_surnames or [],
        venue_hints=venue_hints or [],
    )


def _make_paper(
    *,
    title: str = "A Paper Title",
    year: int | None = None,
    authors: list[dict[str, str]] | None = None,
    venue: str | None = None,
    matchConfidence: str | None = None,
) -> dict:
    paper: dict = {"title": title}
    if year is not None:
        paper["year"] = year
    if authors is not None:
        paper["authors"] = authors
    if venue is not None:
        paper["venue"] = venue
    if matchConfidence is not None:
        paper["matchConfidence"] = matchConfidence
    return paper


# ===================================================================
# 1. Year penalty in _rank_candidate
# ===================================================================


class TestYearPenaltyStrength:
    """_rank_candidate should penalize large year mismatches more heavily."""

    def test_year_delta_3_penalty_greater_than_before(self) -> None:
        """A 3-year mismatch should produce a meaningfully lower score
        than an exact year match, all else equal."""
        parsed = _make_parsed(
            text="Some title about ecosystems 2010",
            year=2010,
            title_candidates=["Some title about ecosystems"],
        )
        paper_exact = _make_paper(title="Some title about ecosystems", year=2010)
        paper_off3 = _make_paper(title="Some title about ecosystems", year=2013)

        exact = _rank_candidate(
            paper=paper_exact,
            parsed=parsed,
            resolution_strategy="fuzzy_search",
            candidate_count=5,
            snippet_text=None,
        )
        off3 = _rank_candidate(
            paper=paper_off3,
            parsed=parsed,
            resolution_strategy="fuzzy_search",
            candidate_count=5,
            snippet_text=None,
        )
        # Year-exact gives +0.08; 3-year-off should give meaningful penalty.
        # Before fix: gap = 0.08 - (-0.06) = 0.14  (old penalty max -0.06)
        # After fix:  gap = 0.08 - (-0.12) = 0.20  (new penalty -0.04*3)
        gap = exact.score - off3.score
        assert gap >= 0.18, f"Score gap for 3-year delta too small: {gap:.4f}"

    def test_year_delta_10_penalised_more_than_delta_3(self) -> None:
        """A 10-year mismatch must be penalized MORE than a 3-year mismatch.
        Before the fix, both were capped at -0.06."""
        parsed = _make_parsed(
            text="Climate change impacts 2000",
            year=2000,
            title_candidates=["Climate change impacts"],
        )
        paper_off3 = _make_paper(title="Climate change impacts", year=2003)
        paper_off10 = _make_paper(title="Climate change impacts", year=2010)

        off3 = _rank_candidate(
            paper=paper_off3,
            parsed=parsed,
            resolution_strategy="fuzzy_search",
            candidate_count=5,
            snippet_text=None,
        )
        off10 = _rank_candidate(
            paper=paper_off10,
            parsed=parsed,
            resolution_strategy="fuzzy_search",
            candidate_count=5,
            snippet_text=None,
        )
        assert off3.score > off10.score, (
            f"10-year mismatch ({off10.score:.4f}) should score lower than 3-year ({off3.score:.4f})"
        )

    def test_year_delta_gt5_has_hard_penalty(self) -> None:
        """Year delta > 5 should receive an additional hard penalty,
        making it clearly worse than delta == 5."""
        parsed = _make_parsed(
            text="Biodiversity assessment 2005",
            year=2005,
            title_candidates=["Biodiversity assessment"],
        )
        paper_off5 = _make_paper(title="Biodiversity assessment", year=2010)
        paper_off6 = _make_paper(title="Biodiversity assessment", year=2011)

        off5 = _rank_candidate(
            paper=paper_off5,
            parsed=parsed,
            resolution_strategy="fuzzy_search",
            candidate_count=5,
            snippet_text=None,
        )
        off6 = _rank_candidate(
            paper=paper_off6,
            parsed=parsed,
            resolution_strategy="fuzzy_search",
            candidate_count=5,
            snippet_text=None,
        )
        gap = off5.score - off6.score
        # The hard penalty at >5 should create a noticeable cliff.
        assert gap >= 0.04, (
            f"Hard penalty cliff at delta>5 too small: off5={off5.score:.4f}, off6={off6.score:.4f}, gap={gap:.4f}"
        )

    def test_upstream_high_confidence_cannot_override_large_year_mismatch(self) -> None:
        """A candidate with matchConfidence='high' but 10-year mismatch
        must score lower than a candidate with no upstream confidence but
        exact year match."""
        parsed = _make_parsed(
            text="Ocean acidification effects 2010",
            year=2010,
            title_candidates=["Ocean acidification effects"],
        )
        paper_wrong_year = _make_paper(
            title="Ocean acidification effects",
            year=2000,
            matchConfidence="high",
        )
        paper_right_year = _make_paper(
            title="Ocean acidification effects",
            year=2010,
            matchConfidence=None,
        )

        wrong = _rank_candidate(
            paper=paper_wrong_year,
            parsed=parsed,
            resolution_strategy="fuzzy_search",
            candidate_count=5,
            snippet_text=None,
        )
        right = _rank_candidate(
            paper=paper_right_year,
            parsed=parsed,
            resolution_strategy="fuzzy_search",
            candidate_count=5,
            snippet_text=None,
        )
        assert right.score > wrong.score, (
            f"Upstream 'high' with 10-year gap ({wrong.score:.4f}) should not beat exact year ({right.score:.4f})"
        )


# ===================================================================
# 2. conflictingFields / confidence consistency
# ===================================================================


class TestConflictingFieldsConfidenceConsistency:
    """When title is in conflictingFields, confidence should be capped
    at 'medium' at most."""

    def test_title_conflicting_caps_confidence_at_medium(self) -> None:
        """If conflictingFields contains 'title', confidence must not be 'high'."""
        # Scenario: identifier match + exact title match gives "high" normally,
        # but if title is conflicting it should cap at medium.
        result = _classify_resolution_confidence(
            best_score=0.90,
            runner_up_score=0.60,
            matched_fields=["identifier", "author", "year"],
            conflicting_fields=["title"],
            resolution_strategy="identifier_doi",
        )
        assert result != "high", f"Confidence should be capped at 'medium' when title is conflicting, got '{result}'"

    def test_title_conflicting_with_high_score_still_capped(self) -> None:
        """Even with best_score >= 0.82, large gap, and only 1 conflicting
        field (title), confidence should be at most 'medium'."""
        result = _classify_resolution_confidence(
            best_score=0.88,
            runner_up_score=0.50,
            matched_fields=["author", "year", "venue"],
            conflicting_fields=["title"],
            resolution_strategy="fuzzy_search",
        )
        assert result in ("medium", "low"), f"Expected 'medium' or 'low' with title conflict, got '{result}'"

    def test_no_title_conflict_allows_high(self) -> None:
        """Verify 'high' is still reachable when title is NOT conflicting."""
        result = _classify_resolution_confidence(
            best_score=0.90,
            runner_up_score=0.60,
            matched_fields=["title", "author", "year"],
            conflicting_fields=[],
            resolution_strategy="fuzzy_search",
        )
        assert result == "high"

    def test_title_in_conflicting_from_rank_candidate_propagates(self) -> None:
        """When _rank_candidate marks title as conflicting (titleSimilarity < 0.72),
        subsequent confidence classification should respect the cap."""
        parsed = _make_parsed(
            text="Distinct paper title 2015",
            year=2015,
            title_candidates=["Distinct paper title"],
        )
        paper = _make_paper(
            title="Completely different subject matter",
            year=2015,
            matchConfidence="high",
        )
        ranked = _rank_candidate(
            paper=paper,
            parsed=parsed,
            resolution_strategy="fuzzy_search",
            candidate_count=3,
            snippet_text=None,
        )
        assert "title" in ranked.conflicting_fields
        # Now check that classify_resolution_confidence with these fields is capped
        confidence = _classify_resolution_confidence(
            best_score=ranked.score,
            runner_up_score=0.0,
            matched_fields=ranked.matched_fields,
            conflicting_fields=ranked.conflicting_fields,
            resolution_strategy="fuzzy_search",
        )
        assert confidence != "high", f"Confidence should not be 'high' when title is conflicting, got '{confidence}'"

    def test_high_title_similarity_in_fuzzy_search_promotes_to_medium(self) -> None:
        metadata = build_match_metadata(
            query="Planetary boundaries exploration",
            paper={"title": "Planetary boundaries exploration"},
            candidate_count=5,
            resolution_strategy="fuzzy_search",
        )

        assert metadata["titleSimilarity"] >= 0.95
        assert metadata["matchConfidence"] in {"medium", "high"}

    def test_high_title_similarity_plus_year_match_can_promote_to_high(self) -> None:
        parsed = _make_parsed(
            text="Planetary boundaries exploration 2009",
            year=2009,
            title_candidates=["Planetary boundaries exploration"],
        )
        paper = _make_paper(title="Planetary boundaries exploration", year=2009)
        ranked = _rank_candidate(
            paper=paper,
            parsed=parsed,
            resolution_strategy="fuzzy_search",
            candidate_count=4,
            snippet_text=None,
        )

        confidence = _classify_resolution_confidence(
            best_score=ranked.score,
            runner_up_score=0.25,
            matched_fields=ranked.matched_fields,
            conflicting_fields=ranked.conflicting_fields,
            resolution_strategy="fuzzy_search",
            title_similarity=ranked.title_similarity,
        )

        assert ranked.title_similarity >= 0.95
        assert confidence == "high"


# ===================================================================
# 3. Title similarity length-difference penalty
# ===================================================================


class TestTitleSimilarityLengthPenalty:
    """Short titles with high token overlap but big length difference
    should get penalized."""

    def test_exact_match_unchanged(self) -> None:
        """Exact title match should still return ~1.0."""
        parsed = _make_parsed(
            text="Human domination of Earth's ecosystems",
            title_candidates=["Human domination of Earth's ecosystems"],
        )
        paper = _make_paper(title="Human domination of Earth's ecosystems")
        sim = _title_similarity(parsed, paper)
        assert sim >= 0.95, f"Exact match should be >= 0.95, got {sim:.4f}"

    def test_short_candidate_vs_long_paper_penalised(self) -> None:
        """A very short candidate matching a long paper title via token
        overlap should be penalized for the length difference."""
        parsed = _make_parsed(
            text="Ecosystems",
            title_candidates=["Ecosystems"],
        )
        paper = _make_paper(title="Human domination of Earth's ecosystems in the modern era of change")
        sim = _title_similarity(parsed, paper)
        # Without length penalty, token overlap could give this high scores.
        # With penalty, it should be clearly < 0.7.
        assert sim < 0.70, f"Short candidate vs long paper should be penalized, got {sim:.4f}"

    def test_long_candidate_vs_short_paper_penalised(self) -> None:
        """A long candidate against a short paper title should also be penalized."""
        parsed = _make_parsed(
            text="Human domination of Earth's ecosystems in the modern era Vitousek 1997 Science",
            title_candidates=["Human domination of Earth's ecosystems in the modern era Vitousek 1997 Science"],
        )
        paper = _make_paper(title="Ecosystems")
        sim = _title_similarity(parsed, paper)
        assert sim < 0.70, f"Long candidate vs short paper should be penalized, got {sim:.4f}"

    def test_similar_length_titles_not_penalised(self) -> None:
        """Titles of similar length should not be penalized."""
        parsed = _make_parsed(
            text="Planetary boundaries exploration",
            title_candidates=["Planetary boundaries exploration"],
        )
        paper = _make_paper(title="Planetary boundaries exploration")
        sim = _title_similarity(parsed, paper)
        assert sim >= 0.95, f"Same-length match should be >= 0.95, got {sim:.4f}"

    def test_moderate_length_difference_moderate_penalty(self) -> None:
        """Moderate length difference should produce moderate penalty."""
        parsed = _make_parsed(
            text="Biodiversity loss drivers",
            title_candidates=["Biodiversity loss drivers"],
        )
        paper = _make_paper(title="Biodiversity loss and its drivers in the tropics")
        sim = _title_similarity(parsed, paper)
        # All tokens in the short title overlap: without penalty could be 1.0.
        # With penalty the length difference should pull it down moderately.
        assert sim < 0.90, f"Moderate length diff should reduce score, got {sim:.4f}"
