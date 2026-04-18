"""Integration tests for cross-cutting remediation fixes.

These verify that the guided-tool response contract holds when
multiple remediation concerns interact: failure summaries, provider
sets, evidence/lead splits, author dedup, verification status,
answerability classification, and refusal detection.

All tests are deterministic — no API keys required.
"""

from __future__ import annotations

from typing import Any

import pytest

from paper_chaser_mcp.dispatch import (
    _assign_verification_status,
    _guided_citation_from_paper,
    _guided_failure_summary,
    _guided_trust_summary,
)
from paper_chaser_mcp.guided_semantic import (
    build_evidence_records,
    classify_answerability,
    classify_source,
)
from paper_chaser_mcp.models.common import CoverageSummary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source(
    *,
    source_id: str = "src-1",
    topical_relevance: str = "on_topic",
    verification_status: str = "verified_metadata",
    is_primary_source: bool = False,
    full_text_url_found: bool = False,
    access_status: str = "access_unverified",
    source_type: str = "scholarly_article",
    title: str = "Test Paper",
    **extra: object,
) -> dict[str, Any]:
    return {
        "sourceId": source_id,
        "sourceAlias": source_id,
        "topicalRelevance": topical_relevance,
        "verificationStatus": verification_status,
        "isPrimarySource": is_primary_source,
        "fullTextUrlFound": full_text_url_found,
        "accessStatus": access_status,
        "sourceType": source_type,
        "title": title,
        "citation": {"title": title, "authors": [], "year": "2024"},
        **extra,
    }


# ---------------------------------------------------------------------------
# 1. Zero-result research maintains schema invariants
# ---------------------------------------------------------------------------


class TestZeroResultSchemaInvariants:
    """When all providers return 0 results the response must abstain
    cleanly rather than claim no_failure."""

    def test_failure_summary_outcome_not_no_failure(self) -> None:
        summary = _guided_failure_summary(
            failure_summary=None,
            status="abstained",
            sources=[],
            evidence_gaps=[],
        )
        assert summary["outcome"] != "no_failure"
        assert summary["outcome"] == "partial_success"

    def test_answerability_is_insufficient_with_no_evidence(self) -> None:
        result = classify_answerability(
            status="abstained",
            evidence=[],
            leads=[],
            evidence_gaps=[],
        )
        assert result == "insufficient"

    def test_empty_providers_produce_empty_succeeded(self) -> None:
        coverage = CoverageSummary(
            providersAttempted=["semantic_scholar", "openalex"],
            providersSucceeded=[],
            providersFailed=[],
            providersZeroResults=["semantic_scholar", "openalex"],
        )
        assert len(coverage.providers_succeeded) == 0
        assert set(coverage.providers_zero_results) == {"semantic_scholar", "openalex"}

    def test_trust_summary_zero_counts(self) -> None:
        trust = _guided_trust_summary([], [])
        assert trust["verifiedSourceCount"] == 0
        assert trust["verifiedPrimarySourceCount"] == 0
        assert trust["verifiedMetadataSourceCount"] == 0

    def test_evidence_split_empty(self) -> None:
        evidence, leads = build_evidence_records(sources=[], leads=[])
        assert evidence == []
        assert leads == []

    def test_zero_results_all_invariants_together(self) -> None:
        """End-to-end: zero results → abstained status, partial_success
        outcome, insufficient answerability, empty evidence/leads."""
        sources: list[dict[str, Any]] = []
        status = "abstained"
        failure = _guided_failure_summary(
            failure_summary=None,
            status=status,
            sources=sources,
            evidence_gaps=[],
        )
        answerability = classify_answerability(
            status=status,
            evidence=[],
            leads=[],
            evidence_gaps=[],
        )
        trust = _guided_trust_summary(sources, [])
        evidence, leads = build_evidence_records(sources=sources, leads=[])

        assert failure["outcome"] == "partial_success"
        assert answerability == "insufficient"
        assert trust["verifiedSourceCount"] == 0
        assert evidence == []
        assert leads == []


# ---------------------------------------------------------------------------
# 2. Mixed provider results maintain disjointness
# ---------------------------------------------------------------------------


class TestMixedProviderDisjointness:
    """When semantic_scholar returns 3 results but openalex returns 0,
    the two provider sets must be disjoint."""

    def test_succeeded_and_zero_results_are_disjoint(self) -> None:
        coverage = CoverageSummary(
            providersAttempted=["semantic_scholar", "openalex"],
            providersSucceeded=["semantic_scholar"],
            providersFailed=[],
            providersZeroResults=["openalex"],
        )
        overlap = set(coverage.providers_succeeded) & set(coverage.providers_zero_results)
        assert overlap == set(), f"Overlap found: {overlap}"

    def test_failed_provider_not_in_succeeded(self) -> None:
        coverage = CoverageSummary(
            providersAttempted=["semantic_scholar", "openalex", "scholarapi"],
            providersSucceeded=["semantic_scholar"],
            providersFailed=["scholarapi"],
            providersZeroResults=["openalex"],
        )
        assert "scholarapi" not in coverage.providers_succeeded
        assert "openalex" not in coverage.providers_succeeded

    def test_active_provider_set_excludes_suppressed(self) -> None:
        from paper_chaser_mcp.provider_runtime import (
            ProviderDiagnosticsRegistry,
            ProviderOutcomeEnvelope,
            policy_for_provider,
        )

        registry = ProviderDiagnosticsRegistry()
        for _ in range(5):
            envelope = ProviderOutcomeEnvelope(
                provider="scholarapi",
                endpoint="/search",
                status_bucket="provider_error",
                latency_ms=100,
            )
            registry.record(envelope, policy=policy_for_provider("scholarapi"))

        enabled = {"semantic_scholar": True, "openalex": True, "scholarapi": True}
        active_set = sorted(p for p, on in enabled.items() if on and not registry.is_suppressed(p))
        assert "scholarapi" not in active_set
        assert "semantic_scholar" in active_set
        assert "openalex" in active_set


# ---------------------------------------------------------------------------
# 3. Follow-up response only includes referenced sources
# ---------------------------------------------------------------------------


class TestFollowUpSourceFiltering:
    """A follow-up answer that references only 2 of 5 sources should
    produce exactly 2 items after filtering."""

    def test_evidence_records_from_subset(self) -> None:
        all_sources = [_make_source(source_id=f"src-{i}", topical_relevance="on_topic") for i in range(5)]
        referenced_ids = {"src-1", "src-3"}
        filtered = [s for s in all_sources if s["sourceId"] in referenced_ids]
        evidence, leads = build_evidence_records(sources=filtered, leads=[])
        assert len(evidence) == 2
        evidence_ids = {e["evidenceId"] for e in evidence}
        assert evidence_ids == referenced_ids

    def test_unreferenced_sources_excluded(self) -> None:
        all_sources = [_make_source(source_id=f"src-{i}", topical_relevance="on_topic") for i in range(5)]
        referenced_ids = {"src-0", "src-4"}
        filtered = [s for s in all_sources if s["sourceId"] in referenced_ids]
        evidence, _ = build_evidence_records(sources=filtered, leads=[])
        for record in evidence:
            assert record["evidenceId"] in referenced_ids


# ---------------------------------------------------------------------------
# 4. Author dedup works end-to-end in citation fields
# ---------------------------------------------------------------------------


class TestAuthorDedupEndToEnd:
    """Verify that _guided_citation_from_paper deduplicates authors
    whose names differ only by given-name completeness."""

    def test_initial_vs_full_name_deduped(self) -> None:
        paper: dict[str, Any] = {
            "title": "Test Paper",
            "authors": [
                {"name": "J. Smith"},
                {"name": "James Smith"},
                {"name": "A. Jones"},
            ],
        }
        citation = _guided_citation_from_paper(paper, canonical_url=None)
        assert citation is not None
        authors = citation["authors"]
        # Smith forms collapse → 1 entry; Jones stays → 2 total
        assert len(authors) == 2, f"Expected 2, got {len(authors)}: {authors}"
        surnames = [a.split()[-1] for a in authors]
        assert "Smith" in surnames
        assert "Jones" in surnames

    def test_longest_form_kept(self) -> None:
        paper: dict[str, Any] = {
            "title": "Paper",
            "authors": [
                {"name": "R. Feynman"},
                {"name": "Richard Feynman"},
            ],
        }
        citation = _guided_citation_from_paper(paper, canonical_url=None)
        assert citation is not None
        assert len(citation["authors"]) == 1
        assert citation["authors"][0] == "Richard Feynman"

    def test_three_distinct_authors_preserved(self) -> None:
        paper: dict[str, Any] = {
            "title": "Paper",
            "authors": [
                {"name": "Alice Smith"},
                {"name": "Bob Jones"},
                {"name": "Carol Williams"},
            ],
        }
        citation = _guided_citation_from_paper(paper, canonical_url=None)
        assert citation is not None
        assert len(citation["authors"]) == 3


# ---------------------------------------------------------------------------
# 5. Verification status properly assigned across source types
# ---------------------------------------------------------------------------


class TestVerificationStatusAssignment:
    """Verify that _assign_verification_status returns the correct
    status for different source type / DOI combinations."""

    def test_scholarly_article_with_doi(self) -> None:
        status = _assign_verification_status(
            source_type="scholarly_article",
            has_doi=True,
            has_doi_resolution=True,
        )
        assert status == "verified_metadata"

    def test_scholarly_article_without_doi(self) -> None:
        status = _assign_verification_status(
            source_type="scholarly_article",
            has_doi=False,
            has_doi_resolution=False,
        )
        assert status == "unverified"

    @pytest.mark.parametrize(
        "source_type",
        [
            "regulatory_document",
            "primary_source",
            "government_document",
            "federal_register_rule",
        ],
    )
    def test_regulatory_types_get_verified_primary_source(self, source_type: str) -> None:
        status = _assign_verification_status(
            source_type=source_type,
            has_doi=False,
            has_doi_resolution=False,
            body_text_embedded=True,
        )
        assert status == "verified_primary_source"

    def test_unknown_type_without_doi_unverified(self) -> None:
        status = _assign_verification_status(
            source_type="unknown",
            has_doi=False,
            has_doi_resolution=False,
        )
        assert status == "unverified"

    def test_unknown_type_with_doi_verified_metadata(self) -> None:
        status = _assign_verification_status(
            source_type="unknown",
            has_doi=True,
            has_doi_resolution=True,
        )
        assert status == "verified_metadata"

    def test_trust_summary_reflects_verification(self) -> None:
        """End-to-end: source types → verification → trust summary counts."""
        sources = [
            _make_source(
                source_id="article-1",
                verification_status="verified_metadata",
                source_type="scholarly_article",
            ),
            _make_source(
                source_id="reg-1",
                verification_status="verified_primary_source",
                source_type="federal_register_rule",
                is_primary_source=True,
            ),
            _make_source(
                source_id="unv-1",
                verification_status="unverified",
                source_type="unknown",
            ),
        ]
        trust = _guided_trust_summary(sources, [])
        assert trust["verifiedPrimarySourceCount"] == 1
        assert trust["verifiedMetadataSourceCount"] == 1
        assert trust["verifiedSourceCount"] == 2


# ---------------------------------------------------------------------------
# 6. Evidence / lead split respects verification status
# ---------------------------------------------------------------------------


class TestEvidenceLeadSplitVerification:
    """Evidence/lead classification must consider both topical relevance
    AND verification status."""

    def test_on_topic_verified_is_evidence(self) -> None:
        source = _make_source(
            topical_relevance="on_topic",
            verification_status="verified_metadata",
        )
        decision = classify_source(source)
        assert decision.include_as == "evidence"

    def test_on_topic_unverified_is_lead(self) -> None:
        source = _make_source(
            topical_relevance="on_topic",
            verification_status="unverified",
        )
        decision = classify_source(source)
        assert decision.include_as == "lead"
        assert decision.why_not_verified is not None

    def test_off_topic_verified_is_lead(self) -> None:
        source = _make_source(
            topical_relevance="off_topic",
            verification_status="verified_metadata",
        )
        decision = classify_source(source)
        assert decision.include_as == "lead"

    def test_off_topic_unverified_is_lead(self) -> None:
        source = _make_source(
            topical_relevance="off_topic",
            verification_status="unverified",
        )
        decision = classify_source(source)
        assert decision.include_as == "lead"

    def test_weak_match_verified_is_lead(self) -> None:
        source = _make_source(
            topical_relevance="weak_match",
            verification_status="verified_metadata",
        )
        decision = classify_source(source)
        assert decision.include_as == "lead"

    def test_build_evidence_records_respects_split(self) -> None:
        """End-to-end through build_evidence_records."""
        sources = [
            _make_source(
                source_id="ev-1",
                topical_relevance="on_topic",
                verification_status="verified_metadata",
            ),
            _make_source(
                source_id="lead-1",
                topical_relevance="on_topic",
                verification_status="unverified",
            ),
            _make_source(
                source_id="lead-2",
                topical_relevance="off_topic",
                verification_status="verified_metadata",
            ),
        ]
        evidence, leads = build_evidence_records(sources=sources, leads=[])
        assert len(evidence) == 1
        assert evidence[0]["evidenceId"] == "ev-1"
        assert len(leads) == 2
        lead_ids = {ld["evidenceId"] for ld in leads}
        assert lead_ids == {"lead-1", "lead-2"}


# ---------------------------------------------------------------------------
# 7. Refusal detection prevents false grounded status
# ---------------------------------------------------------------------------


class TestRefusalPreventsFalseGrounded:
    """classify_answerability must downgrade to insufficient when the
    answer text contains refusal language, even with evidence present."""

    _EVIDENCE = [{"sourceId": "s1", "title": "Paper A"}]

    def test_cannot_determine_downgrades(self) -> None:
        result = classify_answerability(
            status="succeeded",
            evidence=self._EVIDENCE,
            leads=[],
            evidence_gaps=[],
            answer_text="I cannot determine the precise mechanism from these sources.",
        )
        assert result == "insufficient"

    def test_unable_to_answer_downgrades(self) -> None:
        result = classify_answerability(
            status="answered",
            evidence=self._EVIDENCE,
            leads=[],
            evidence_gaps=[],
            answer_text="Unfortunately, I am unable to answer this question conclusively.",
        )
        assert result == "insufficient"

    def test_cannot_provide_a_definitive_downgrades(self) -> None:
        result = classify_answerability(
            status="succeeded",
            evidence=self._EVIDENCE,
            leads=[],
            evidence_gaps=[],
            answer_text="I cannot provide a definitive answer based on the available literature.",
        )
        assert result == "insufficient"

    def test_substantive_answer_stays_grounded(self) -> None:
        result = classify_answerability(
            status="succeeded",
            evidence=self._EVIDENCE,
            leads=[],
            evidence_gaps=[],
            answer_text="The study demonstrates a clear causal link between habitat loss and population decline.",
        )
        assert result == "grounded"

    def test_deterministic_fallback_gap_stays_limited(self) -> None:
        """Even without refusal text, a deterministic fallback gap
        should cap answerability at limited."""
        result = classify_answerability(
            status="succeeded",
            evidence=self._EVIDENCE,
            leads=[],
            evidence_gaps=["deterministic_synthesis_fallback"],
            answer_text="The evidence suggests a moderate effect.",
        )
        assert result == "limited"

    def test_refusal_with_no_evidence_is_insufficient(self) -> None:
        result = classify_answerability(
            status="succeeded",
            evidence=[],
            leads=[],
            evidence_gaps=[],
            answer_text="I cannot determine the answer.",
        )
        assert result == "insufficient"


# ---------------------------------------------------------------------------
# 8. Cross-cutting: full pipeline from source → evidence → answerability
# ---------------------------------------------------------------------------


class TestCrossCuttingPipeline:
    """Verify that verification assignment, evidence/lead split, trust
    summary, failure summary, and answerability all agree."""

    def test_verified_on_topic_pipeline(self) -> None:
        """A verified on-topic source → evidence → grounded."""
        vs = _assign_verification_status(
            source_type="scholarly_article",
            has_doi=True,
            has_doi_resolution=True,
        )
        source = _make_source(
            source_id="pipe-1",
            verification_status=vs,
            topical_relevance="on_topic",
        )
        evidence, leads = build_evidence_records(sources=[source], leads=[])
        assert len(evidence) == 1
        assert len(leads) == 0

        trust = _guided_trust_summary([source], [])
        assert trust["verifiedSourceCount"] == 1

        answerability = classify_answerability(
            status="succeeded",
            evidence=evidence,
            leads=leads,
            evidence_gaps=[],
            answer_text="The data clearly supports the hypothesis.",
        )
        assert answerability == "grounded"

        failure = _guided_failure_summary(
            failure_summary=None,
            status="succeeded",
            sources=[source],
            evidence_gaps=[],
        )
        assert failure["outcome"] == "no_failure"

    def test_unverified_on_topic_pipeline(self) -> None:
        """An unverified on-topic source → lead → limited."""
        vs = _assign_verification_status(
            source_type="unknown",
            has_doi=False,
            has_doi_resolution=False,
        )
        assert vs == "unverified"

        source = _make_source(
            source_id="pipe-2",
            verification_status=vs,
            topical_relevance="on_topic",
        )
        evidence, leads = build_evidence_records(sources=[source], leads=[])
        assert len(evidence) == 0
        assert len(leads) == 1

        answerability = classify_answerability(
            status="succeeded",
            evidence=evidence,
            leads=leads,
            evidence_gaps=[],
        )
        assert answerability == "limited"

    def test_regulatory_source_pipeline(self) -> None:
        """A regulatory source → verified_primary_source → evidence → grounded."""
        vs = _assign_verification_status(
            source_type="federal_register_rule",
            has_doi=False,
            has_doi_resolution=False,
            body_text_embedded=True,
        )
        assert vs == "verified_primary_source"

        source = _make_source(
            source_id="reg-pipe",
            verification_status=vs,
            topical_relevance="on_topic",
            source_type="federal_register_rule",
            is_primary_source=True,
        )
        evidence, leads = build_evidence_records(sources=[source], leads=[])
        assert len(evidence) == 1

        trust = _guided_trust_summary([source], [])
        assert trust["verifiedPrimarySourceCount"] == 1

        answerability = classify_answerability(
            status="succeeded",
            evidence=evidence,
            leads=leads,
            evidence_gaps=[],
            answer_text="The regulation establishes critical habitat boundaries.",
        )
        assert answerability == "grounded"

    def test_refusal_overrides_otherwise_grounded_pipeline(self) -> None:
        """Even with verified evidence, refusal text → insufficient."""
        source = _make_source(
            source_id="refusal-pipe",
            verification_status="verified_metadata",
            topical_relevance="on_topic",
        )
        evidence, leads = build_evidence_records(sources=[source], leads=[])
        assert len(evidence) == 1

        answerability = classify_answerability(
            status="succeeded",
            evidence=evidence,
            leads=leads,
            evidence_gaps=[],
            answer_text="I cannot determine whether the effect is significant.",
        )
        assert answerability == "insufficient"
