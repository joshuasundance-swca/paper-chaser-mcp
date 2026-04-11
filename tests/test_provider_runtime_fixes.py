"""Tests for Phase 3 provider runtime and graceful degradation fixes.

Covers:
1. canAnswerFollowUp should be capability-based (requires evidence, not just session)
2. fullTextUrlFound / accessStatus consistency in source records
3. SerpApi venue parsing robustness
"""

from __future__ import annotations

from typing import Any

import paper_chaser_mcp.dispatch as dispatch_module
from paper_chaser_mcp.clients.serpapi.normalize import normalize_organic_result

# ---------------------------------------------------------------------------
# 1. canAnswerFollowUp – capability-based, not permission-based
# ---------------------------------------------------------------------------


class TestCanAnswerFollowUpCapabilityBased:
    """canAnswerFollowUp should require both a session AND actual evidence."""

    def test_session_with_sources_can_follow_up(self) -> None:
        state = dispatch_module._guided_result_state(
            status="answered",
            sources=[{"sourceId": "s1", "title": "A paper"}],
            evidence_gaps=[],
            search_session_id="sess-123",
        )
        assert state["canAnswerFollowUp"] is True

    def test_session_without_sources_cannot_follow_up(self) -> None:
        """A session exists but no evidence sources — follow-up is useless."""
        state = dispatch_module._guided_result_state(
            status="partial",
            sources=[],
            evidence_gaps=["no evidence found"],
            search_session_id="sess-456",
        )
        assert state["canAnswerFollowUp"] is False

    def test_no_session_cannot_follow_up(self) -> None:
        state = dispatch_module._guided_result_state(
            status="answered",
            sources=[{"sourceId": "s1", "title": "A paper"}],
            evidence_gaps=[],
            search_session_id=None,
        )
        assert state["canAnswerFollowUp"] is False

    def test_no_session_no_sources_cannot_follow_up(self) -> None:
        state = dispatch_module._guided_result_state(
            status="failed",
            sources=[],
            evidence_gaps=["provider failure"],
            search_session_id=None,
        )
        assert state["canAnswerFollowUp"] is False


# ---------------------------------------------------------------------------
# 2. fullTextUrlFound / accessStatus consistency
# ---------------------------------------------------------------------------


class TestFullTextAccessStatusConsistency:
    """When fullTextUrlFound is False, accessStatus must not be full_text_verified."""

    def test_paper_with_full_text_gets_full_text_verified(self) -> None:
        paper: dict[str, Any] = {
            "title": "Some Paper",
            "fullTextUrlFound": True,
        }
        record = dispatch_module._guided_source_record_from_paper("query", paper, index=1)
        assert record["fullTextUrlFound"] is True
        assert record["accessStatus"] == "full_text_verified"

    def test_paper_without_full_text_does_not_get_full_text_verified(self) -> None:
        paper: dict[str, Any] = {
            "title": "Some Paper",
            "fullTextUrlFound": False,
        }
        record = dispatch_module._guided_source_record_from_paper("query", paper, index=1)
        assert record["fullTextUrlFound"] is False
        assert record["accessStatus"] != "full_text_verified"

    def test_paper_with_no_fulltext_key_does_not_get_full_text_verified(self) -> None:
        paper: dict[str, Any] = {
            "title": "Some Paper",
        }
        record = dispatch_module._guided_source_record_from_paper("query", paper, index=1)
        assert record["fullTextUrlFound"] is False
        assert record["accessStatus"] != "full_text_verified"

    def test_paper_with_explicit_access_status_preserved(self) -> None:
        """If the paper already has an explicit accessStatus, honour it."""
        paper: dict[str, Any] = {
            "title": "Some Paper",
            "fullTextUrlFound": False,
            "accessStatus": "abstract_only",
        }
        record = dispatch_module._guided_source_record_from_paper("query", paper, index=1)
        assert record["accessStatus"] == "abstract_only"

    def test_paper_with_abstract_observed_gets_abstract_only(self) -> None:
        paper: dict[str, Any] = {
            "title": "Some Paper",
            "fullTextUrlFound": False,
            "abstractObserved": True,
        }
        record = dispatch_module._guided_source_record_from_paper("query", paper, index=1)
        assert record["accessStatus"] in {"abstract_only", "access_unverified"}
        assert record["accessStatus"] != "full_text_verified"


# ---------------------------------------------------------------------------
# 3. SerpApi venue parsing robustness
# ---------------------------------------------------------------------------


class TestSerpApiVenueParsing:
    """Venue parsing should not produce garbage from author-heavy summaries."""

    def _make_result(self, summary: str, **extra: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "title": "Test Paper",
            "publication_info": {"summary": summary},
        }
        base.update(extra)
        return base

    def test_normal_author_venue_year(self) -> None:
        """Standard format: 'A Smith, B Jones - Nature, 2023'."""
        result = self._make_result("A Smith, B Jones - Nature, 2023")
        paper = normalize_organic_result(result)
        assert paper is not None
        assert paper["venue"] == "Nature"
        assert paper["year"] == 2023

    def test_no_dash_in_summary(self) -> None:
        """Summary with no dash should not produce author names as venue."""
        result = self._make_result("John Smith, Jane Doe, Robert Brown")
        paper = normalize_organic_result(result)
        assert paper is not None
        # Venue should be None or at least not look like author names
        if paper["venue"] is not None:
            # Should not be the raw author string
            assert paper["venue"] != "John Smith, Jane Doe, Robert Brown"

    def test_multiple_dashes_in_summary(self) -> None:
        """Summary with multiple dashes: 'A Smith - B Jones - Nature, 2023'."""
        result = self._make_result("A Smith - B Jones - Nature, 2023")
        paper = normalize_organic_result(result)
        assert paper is not None
        # After split(" - ", 1) takes last part: "B Jones - Nature, 2023"
        # Venue should still be reasonable
        if paper["venue"] is not None:
            assert len(paper["venue"]) < 80

    def test_very_long_venue_candidate_rejected(self) -> None:
        """Excessively long venue candidates should be rejected."""
        long_authors = ", ".join(f"Author{i}" for i in range(20))
        result = self._make_result(f"{long_authors}, 2023")
        paper = normalize_organic_result(result)
        assert paper is not None
        # Should not produce a very long garbage venue
        if paper["venue"] is not None:
            assert len(paper["venue"]) < 100

    def test_author_pattern_not_used_as_venue(self) -> None:
        """When dash splits yield something that looks like author names, reject it."""
        result = self._make_result("J Smith, K Lee - M Johnson, R Davis, 2021")
        paper = normalize_organic_result(result)
        assert paper is not None
        # "M Johnson, R Davis" looks like authors, not a venue
        if paper["venue"] is not None:
            # Venue should not contain multiple comma-separated capitalized names
            assert paper["venue"] != "M Johnson, R Davis"

    def test_empty_summary(self) -> None:
        result = self._make_result("")
        paper = normalize_organic_result(result)
        assert paper is not None
        assert paper["venue"] is None
        assert paper["year"] is None

    def test_year_only_summary(self) -> None:
        result = self._make_result("2023")
        paper = normalize_organic_result(result)
        assert paper is not None
        assert paper["year"] == 2023
        assert paper["venue"] is None
