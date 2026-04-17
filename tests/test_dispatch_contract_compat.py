"""Regression tests for ws-dispatch-contract-trust (findings #2, #3, #5).

These pin the backward-compatible contract for guided dispatch output:

* ``fullTextObserved`` is emitted alongside ``fullTextUrlFound`` (finding #2).
* Scholarly papers with basic descriptive metadata (title + author/venue) keep
  the ``verified_metadata`` verification status even when no DOI is present
  (finding #3).
* ``subjectChainGaps`` from planner ``strategyMetadata`` flows into the
  machine-readable ``confidenceSignals`` and ``trustSummary`` (finding #5).
"""

from __future__ import annotations

from paper_chaser_mcp.dispatch import (
    _guided_confidence_signals,
    _guided_source_record_from_paper,
    _guided_source_record_from_structured_source,
    _guided_trust_summary,
)

# ---------------------------------------------------------------------------
# Finding #2 — fullTextObserved legacy alias retained
# ---------------------------------------------------------------------------


class TestFullTextObservedDualEmit:
    def test_paper_record_dual_emits_both_keys(self) -> None:
        record = _guided_source_record_from_paper(
            "query",
            {
                "title": "Example Paper",
                "authors": [{"name": "Researcher"}],
                "doi": "10.1/ex",
                "fullTextUrlFound": True,
            },
            index=1,
        )
        assert record["fullTextUrlFound"] is True
        assert record["fullTextObserved"] is True

    def test_paper_record_dual_emits_when_false(self) -> None:
        record = _guided_source_record_from_paper(
            "query",
            {
                "title": "Example Paper",
                "authors": [{"name": "Researcher"}],
                "doi": "10.1/ex",
            },
            index=1,
        )
        # ``strip_null_fields`` preserves False booleans so both keys remain.
        assert record.get("fullTextUrlFound") is False
        assert record.get("fullTextObserved") is False

    def test_structured_source_record_dual_emits(self) -> None:
        record = _guided_source_record_from_structured_source(
            {
                "sourceId": "src-1",
                "title": "Example",
                "sourceType": "scholarly_article",
                "fullTextUrlFound": True,
            },
            index=1,
        )
        assert record["fullTextUrlFound"] is True
        assert record["fullTextObserved"] is True

    def test_legacy_input_key_mirrored_to_both_outputs(self) -> None:
        record = _guided_source_record_from_structured_source(
            {
                "sourceId": "src-1",
                "title": "Example",
                "sourceType": "scholarly_article",
                "fullTextObserved": True,
            },
            index=1,
        )
        assert record["fullTextUrlFound"] is True
        assert record["fullTextObserved"] is True


# ---------------------------------------------------------------------------
# Finding #3 — scholarly default verification status
# ---------------------------------------------------------------------------


class TestScholarlyVerifiedMetadataDefault:
    def test_doi_less_scholarly_with_author_is_verified_metadata(self) -> None:
        record = _guided_source_record_from_paper(
            "query",
            {
                "title": "A Scholarly Article Without DOI",
                "authors": [{"name": "Some Author"}],
                "sourceType": "scholarly_article",
            },
            index=1,
        )
        assert record["verificationStatus"] == "verified_metadata"

    def test_doi_less_scholarly_with_venue_is_verified_metadata(self) -> None:
        record = _guided_source_record_from_paper(
            "query",
            {
                "title": "Another Paper",
                "venue": "Nature Communications",
                "sourceType": "scholarly_article",
            },
            index=1,
        )
        assert record["verificationStatus"] == "verified_metadata"

    def test_scholarly_default_source_type_is_applied(self) -> None:
        # sourceType omitted — defaults to "scholarly_article" per existing code.
        record = _guided_source_record_from_paper(
            "query",
            {
                "title": "Another Paper",
                "authors": ["First Author"],
            },
            index=1,
        )
        assert record["sourceType"] == "scholarly_article"
        assert record["verificationStatus"] == "verified_metadata"

    def test_scholarly_without_descriptive_metadata_stays_unverified(self) -> None:
        record = _guided_source_record_from_paper(
            "query",
            {
                "title": "Bare Title Only",
                "sourceType": "scholarly_article",
            },
            index=1,
        )
        # Title alone is not enough — author OR venue required to claim
        # ``verified_metadata``. Stays ``unverified`` per finding #3 contract.
        assert record["verificationStatus"] == "unverified"

    def test_explicit_verification_status_is_respected(self) -> None:
        record = _guided_source_record_from_paper(
            "query",
            {
                "title": "Explicit",
                "authors": [{"name": "X"}],
                "sourceType": "scholarly_article",
                "verificationStatus": "verified_primary_source",
            },
            index=1,
        )
        assert record["verificationStatus"] == "verified_primary_source"


# ---------------------------------------------------------------------------
# Finding #5 — subjectChainGaps threading into trust signals
# ---------------------------------------------------------------------------


class TestSubjectChainGapsThreading:
    def test_confidence_signals_surfaces_subject_chain_gaps(self) -> None:
        signals = _guided_confidence_signals(
            status="partial",
            sources=[],
            evidence_gaps=[],
            subject_chain_gaps=["missing subject anchor: species identity"],
        )
        assert signals.get("subjectChainGaps") == ["missing subject anchor: species identity"]

    def test_confidence_signals_uses_gap_as_trust_revision_fallback(self) -> None:
        signals = _guided_confidence_signals(
            status="partial",
            sources=[],
            evidence_gaps=[],
            subject_chain_gaps=["planner saw no subject anchor"],
        )
        # With no other degradation reason or evidence gap, the first subject
        # chain gap should populate ``trustRevisionReason`` rather than leaving
        # it empty.
        assert signals.get("trustRevisionReason") == "planner saw no subject anchor"

    def test_confidence_signals_omits_key_when_empty(self) -> None:
        signals = _guided_confidence_signals(
            status="succeeded",
            sources=[],
            evidence_gaps=[],
        )
        assert "subjectChainGaps" not in signals

    def test_confidence_signals_handles_none_without_crash(self) -> None:
        signals = _guided_confidence_signals(
            status="succeeded",
            sources=[],
            evidence_gaps=[],
            subject_chain_gaps=None,
        )
        assert "subjectChainGaps" not in signals

    def test_trust_summary_surfaces_subject_chain_gaps(self) -> None:
        summary = _guided_trust_summary(
            [],
            [],
            subject_chain_gaps=["gap a", "gap b"],
        )
        assert summary.get("subjectChainGaps") == ["gap a", "gap b"]

    def test_trust_summary_omits_when_none(self) -> None:
        summary = _guided_trust_summary([], [])
        assert "subjectChainGaps" not in summary

    def test_trust_summary_filters_blank_entries(self) -> None:
        summary = _guided_trust_summary(
            [],
            [],
            subject_chain_gaps=["", "  ", "real gap"],
        )
        assert summary.get("subjectChainGaps") == ["real gap"]
