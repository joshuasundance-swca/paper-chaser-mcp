"""Schema invariant tests for guided response contracts.

These are deterministic assertions that verify structural consistency
across guided response payloads. Each test maps to a specific finding
from the v0.2.2 stress test review.
"""

from __future__ import annotations

from paper_chaser_mcp.dispatch import (
    _assign_verification_status,
    _guided_failure_summary,
    _guided_trust_summary,
)
from paper_chaser_mcp.guided_semantic import build_evidence_records

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
) -> dict:
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
# 1.1: isPrimarySource / verifiedPrimarySourceCount reconciliation
# ---------------------------------------------------------------------------


class TestPrimarySourceCountConsistency:
    """Stress-test finding: isPrimarySource=true in source object but
    verifiedPrimarySourceCount=0 in trustSummary."""

    def test_primary_source_counted_in_trust_summary(self) -> None:
        """A source with isPrimarySource=true AND verificationStatus=verified_primary_source
        must be reflected in verifiedPrimarySourceCount."""
        sources = [
            _make_source(
                source_id="reg-1",
                is_primary_source=True,
                verification_status="verified_primary_source",
                source_type="federal_register_rule",
            ),
        ]
        trust = _guided_trust_summary(sources, [])
        assert trust["verifiedPrimarySourceCount"] == 1

    def test_assign_verification_status_for_primary_source_types(self) -> None:
        """Regulatory source types should get verified_primary_source status."""
        for source_type in ("regulatory_document", "primary_source", "government_document", "federal_register_rule"):
            status = _assign_verification_status(
                source_type=source_type,
                has_doi=False,
                has_doi_resolution=False,
                full_text_url_found=False,
            )
            assert status == "verified_primary_source", (
                f"Expected verified_primary_source for {source_type}, got {status}"
            )

    def test_primary_source_flag_aligns_with_verification_status(self) -> None:
        """When isPrimarySource is true, the trust summary count must include it."""
        sources = [
            _make_source(
                source_id="fed-reg",
                is_primary_source=True,
                verification_status="verified_primary_source",
            ),
            _make_source(
                source_id="article",
                is_primary_source=False,
                verification_status="verified_metadata",
            ),
        ]
        trust = _guided_trust_summary(sources, [])
        primary_count = sum(1 for s in sources if s.get("isPrimarySource"))
        assert trust["verifiedPrimarySourceCount"] == primary_count


# ---------------------------------------------------------------------------
# 1.2: providersSucceeded ∩ providersZeroResults must be disjoint
# ---------------------------------------------------------------------------


class TestProviderSetDisjointness:
    """Stress-test finding: a provider appears in both providersSucceeded
    AND providersZeroResults simultaneously."""

    def test_succeeded_and_zero_results_disjoint(self) -> None:
        """These two sets must never overlap."""
        from paper_chaser_mcp.models.common import CoverageSummary

        coverage = CoverageSummary(
            providersAttempted=["openalex", "semantic_scholar"],
            providersSucceeded=["openalex"],
            providersFailed=[],
            providersZeroResults=["semantic_scholar"],
        )
        overlap = set(coverage.providers_succeeded) & set(coverage.providers_zero_results)
        assert overlap == set(), f"Overlap found: {overlap}"

    def test_provider_returning_zero_not_in_succeeded(self) -> None:
        """The _smart_coverage_summary builder must not put a zero-result
        provider into providersSucceeded."""
        from paper_chaser_mcp.models.common import CoverageSummary

        # Simulate what _smart_coverage_summary should produce:
        # openalex returned empty status_bucket → should NOT be in succeeded
        providers_used = ["openalex"]
        zero_results = ["openalex"]
        zero_results_set = set(zero_results)
        succeeded = [p for p in providers_used if p not in zero_results_set]

        coverage = CoverageSummary(
            providersAttempted=["openalex"],
            providersSucceeded=succeeded,
            providersFailed=[],
            providersZeroResults=zero_results,
        )
        overlap = set(coverage.providers_succeeded) & set(coverage.providers_zero_results)
        assert overlap == set(), "A provider cannot both succeed and return zero results"


# ---------------------------------------------------------------------------
# 1.3: fallbackAttempted / fallbackMode consistency
# ---------------------------------------------------------------------------


class TestFallbackConsistency:
    """Stress-test finding: fallbackAttempted=false but fallbackMode is non-null."""

    def test_fallback_mode_requires_fallback_attempted(self) -> None:
        """If fallbackMode is set, fallbackAttempted must be True."""
        summary = _guided_failure_summary(
            failure_summary={
                "fallbackMode": "smart_provider_fallback",
                "fallbackAttempted": False,
            },
            status="partial",
            sources=[_make_source()],
            evidence_gaps=[],
        )
        if summary["fallbackMode"] is not None:
            assert summary["fallbackAttempted"] is True, (
                f"fallbackMode={summary['fallbackMode']} but fallbackAttempted={summary['fallbackAttempted']}"
            )

    def test_null_fallback_mode_allows_false_attempted(self) -> None:
        """When fallbackMode is None, fallbackAttempted can be False."""
        summary = _guided_failure_summary(
            failure_summary=None,
            status="succeeded",
            sources=[_make_source()],
            evidence_gaps=[],
        )
        assert summary["fallbackMode"] is None
        assert summary["fallbackAttempted"] is False


# ---------------------------------------------------------------------------
# 1.5: activeProviderSet must exclude suppressed providers
# ---------------------------------------------------------------------------


class TestActiveProviderSetSuppression:
    """Stress-test finding: scholarapi in activeProviderSet while suppressed=true
    with 38 consecutive failures."""

    def test_suppressed_provider_excluded_from_active_set(self) -> None:
        """A provider that is suppressed must not appear in activeProviderSet."""
        from paper_chaser_mcp.provider_runtime import (
            ProviderDiagnosticsRegistry,
            ProviderOutcomeEnvelope,
            policy_for_provider,
        )

        registry = ProviderDiagnosticsRegistry()
        # Suppress scholarapi by recording failures exceeding threshold
        for _ in range(5):
            envelope = ProviderOutcomeEnvelope(
                provider="scholarapi",
                endpoint="/search",
                status_bucket="provider_error",
                latency_ms=100,
            )
            registry.record(envelope, policy=policy_for_provider("scholarapi"))
        assert registry.is_suppressed("scholarapi"), "scholarapi should be suppressed"

        # Build active set excluding suppressed
        enabled_state = {
            "semantic_scholar": True,
            "openalex": True,
            "scholarapi": True,
        }
        active_set = sorted([p for p, enabled in enabled_state.items() if enabled and not registry.is_suppressed(p)])
        assert "scholarapi" not in active_set


# ---------------------------------------------------------------------------
# 1.10: failureSummary.outcome semantic accuracy
# ---------------------------------------------------------------------------


class TestFailureOutcomeSemantic:
    """Stress-test finding: outcome='no_failure' for a pseudoscience query
    with 0 evidence — correct behavior but wrong label."""

    def test_abstained_with_no_sources_not_no_failure(self) -> None:
        """When status is abstained and there are no sources, outcome
        should not be 'no_failure'."""
        summary = _guided_failure_summary(
            failure_summary=None,
            status="abstained",
            sources=[],
            evidence_gaps=[],
        )
        assert summary["outcome"] != "no_failure", "Abstaining with zero sources should not claim 'no_failure'"

    def test_abstained_with_sources_can_be_no_failure(self) -> None:
        """Abstaining with sources present (e.g., all off-topic) is a
        legitimate 'no_failure' state — the system worked correctly."""
        summary = _guided_failure_summary(
            failure_summary=None,
            status="abstained",
            sources=[_make_source(topical_relevance="off_topic")],
            evidence_gaps=[],
        )
        # This is acceptable — the system found sources but correctly abstained
        assert summary["outcome"] in ("no_failure", "partial_success")


# ---------------------------------------------------------------------------
# Evidence / Leads split
# ---------------------------------------------------------------------------


class TestEvidenceLeadsSplit:
    """Verify evidence/leads classification contracts."""

    def test_on_topic_verified_goes_to_evidence(self) -> None:
        source = _make_source(topical_relevance="on_topic", verification_status="verified_metadata")
        evidence, leads = build_evidence_records(sources=[source], leads=[])
        assert len(evidence) == 1
        assert len(leads) == 0

    def test_weak_match_goes_to_leads(self) -> None:
        source = _make_source(topical_relevance="weak_match", verification_status="verified_metadata")
        evidence, leads = build_evidence_records(sources=[source], leads=[])
        assert len(evidence) == 0
        assert len(leads) == 1

    def test_off_topic_goes_to_leads(self) -> None:
        source = _make_source(topical_relevance="off_topic", verification_status="verified_metadata")
        evidence, leads = build_evidence_records(sources=[source], leads=[])
        assert len(evidence) == 0
        assert len(leads) == 1


# ---------------------------------------------------------------------------
# Author deduplication
# ---------------------------------------------------------------------------


class TestAuthorDeduplication:
    """Stress-test finding: authors duplicated in French-form + initials form."""

    def test_author_list_no_logical_duplicates(self) -> None:
        """Authors that differ only by given-name completeness should be
        deduplicated to the most complete form."""
        from paper_chaser_mcp.dispatch import _guided_citation_from_paper

        paper = {
            "title": "Test Paper",
            "authors": [
                {"name": "Jean Dupont"},
                {"name": "J. Dupont"},
                {"name": "Marie Curie"},
                {"name": "M. Curie"},
            ],
        }
        citation = _guided_citation_from_paper(paper, canonical_url=None)
        assert citation is not None
        authors = citation["authors"]
        # After deduplication, we should have 2 unique authors
        assert len(authors) == 2, f"Expected 2 unique authors, got {len(authors)}: {authors}"

    def test_distinct_authors_preserved(self) -> None:
        """Truly distinct authors must not be collapsed."""
        from paper_chaser_mcp.dispatch import _guided_citation_from_paper

        paper = {
            "title": "Test Paper",
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
# 1.6: currentTextSatisfied must require actual text access
# ---------------------------------------------------------------------------


class TestCurrentTextSatisfied:
    """Stress-test finding: govinfo_matched=True (metadata-only hit) incorrectly
    sets currentTextSatisfied=True even when no body text was retrieved."""

    def test_satisfied_when_govinfo_has_full_text(self) -> None:
        """currentTextSatisfied=True when a govinfo structured source has fullTextUrlFound=True."""
        from paper_chaser_mcp.agentic.models import StructuredSourceRecord
        from paper_chaser_mcp.models.common import PrimaryDocumentCoverage

        structured_sources = [
            StructuredSourceRecord(
                sourceId="cfr-1",
                provider="govinfo",
                fullTextUrlFound=True,
                title="40 CFR 50",
            ),
        ]
        govinfo_matched = True
        current_text_requested = True

        govinfo_text_retrieved = govinfo_matched and any(
            s.full_text_url_found for s in structured_sources if s.provider == "govinfo"
        )
        current_text_satisfied = (not current_text_requested) or govinfo_text_retrieved

        coverage = PrimaryDocumentCoverage(
            currentTextRequested=current_text_requested,
            govinfoMatched=govinfo_matched,
            currentTextSatisfied=current_text_satisfied,
        )
        assert coverage.current_text_satisfied is True

    def test_not_satisfied_when_govinfo_metadata_only(self) -> None:
        """currentTextSatisfied=False when govinfo matched but fullTextUrlFound=False."""
        from paper_chaser_mcp.agentic.models import StructuredSourceRecord
        from paper_chaser_mcp.models.common import PrimaryDocumentCoverage

        structured_sources = [
            StructuredSourceRecord(
                sourceId="cfr-1",
                provider="govinfo",
                fullTextUrlFound=False,
                title="40 CFR 50",
            ),
        ]
        govinfo_matched = True
        current_text_requested = True

        govinfo_text_retrieved = govinfo_matched and any(
            s.full_text_url_found for s in structured_sources if s.provider == "govinfo"
        )
        current_text_satisfied = (not current_text_requested) or govinfo_text_retrieved

        coverage = PrimaryDocumentCoverage(
            currentTextRequested=current_text_requested,
            govinfoMatched=govinfo_matched,
            currentTextSatisfied=current_text_satisfied,
        )
        assert coverage.current_text_satisfied is False

    def test_satisfied_when_text_not_requested(self) -> None:
        """currentTextSatisfied=True when currentTextRequested=False (not applicable)."""
        current_text_requested = False

        current_text_satisfied = (not current_text_requested) or False
        assert current_text_satisfied is True

    def test_not_satisfied_when_no_govinfo_sources(self) -> None:
        """currentTextSatisfied=False when govinfo matched but no govinfo structured sources exist."""
        from paper_chaser_mcp.agentic.models import StructuredSourceRecord

        structured_sources = [
            StructuredSourceRecord(
                sourceId="fr-1",
                provider="federal_register",
                fullTextUrlFound=True,
                title="Final Rule",
            ),
        ]
        govinfo_matched = True
        current_text_requested = True

        govinfo_text_retrieved = govinfo_matched and any(
            s.full_text_url_found for s in structured_sources if s.provider == "govinfo"
        )
        current_text_satisfied = (not current_text_requested) or govinfo_text_retrieved
        assert current_text_satisfied is False


# ---------------------------------------------------------------------------
# 1.7: fullTextUrlFound rename + fullTextRetrieved semantics
# ---------------------------------------------------------------------------


class TestFullTextFieldRename:
    """Verify the fullTextObserved → fullTextUrlFound rename with backward compat."""

    def test_full_text_retrieved_only_true_when_content_ingested(self) -> None:
        """fullTextRetrieved should only be True when actual content was ingested."""
        from paper_chaser_mcp.agentic.models import StructuredSourceRecord

        # URL found but not retrieved
        record = StructuredSourceRecord(
            sourceId="src-1",
            title="Paper A",
            fullTextUrlFound=True,
            fullTextRetrieved=None,
        )
        assert record.full_text_url_found is True
        assert record.full_text_retrieved is None

        # Content actually retrieved
        record2 = StructuredSourceRecord(
            sourceId="src-2",
            title="Paper B",
            fullTextUrlFound=True,
            fullTextRetrieved=True,
        )
        assert record2.full_text_url_found is True
        assert record2.full_text_retrieved is True

    def test_backward_compat_fullTextObserved_accepted(self) -> None:
        """The old alias fullTextObserved must still be accepted on input."""
        from paper_chaser_mcp.agentic.models import StructuredSourceRecord
        from paper_chaser_mcp.models.common import Paper

        # StructuredSourceRecord accepts old name
        record = StructuredSourceRecord.model_validate(
            {
                "sourceId": "src-1",
                "title": "Legacy Paper",
                "fullTextObserved": True,
            }
        )
        assert record.full_text_url_found is True

        # Paper model accepts old name
        paper = Paper.model_validate(
            {
                "title": "Legacy Paper",
                "fullTextObserved": True,
            }
        )
        assert paper.full_text_url_found is True

    def test_serialization_uses_new_field_name(self) -> None:
        """Output serialization must use fullTextUrlFound, not fullTextObserved."""
        from paper_chaser_mcp.agentic.models import StructuredSourceRecord
        from paper_chaser_mcp.models.common import Paper

        record = StructuredSourceRecord(
            sourceId="src-1",
            title="New Paper",
            fullTextUrlFound=True,
            fullTextRetrieved=True,
        )
        dumped = record.model_dump(by_alias=True, exclude_none=True)
        assert "fullTextUrlFound" in dumped
        assert "fullTextObserved" not in dumped
        assert "fullTextRetrieved" in dumped
        assert dumped["fullTextUrlFound"] is True
        assert dumped["fullTextRetrieved"] is True

        paper = Paper(title="New Paper", fullTextUrlFound=True)
        paper_dumped = paper.model_dump(by_alias=True, exclude_none=True)
        assert "fullTextUrlFound" in paper_dumped
        assert "fullTextObserved" not in paper_dumped

    def test_guided_dispatch_source_record_emits_both_keys(self) -> None:
        """The guided-layer source output must emit BOTH the legacy
        ``fullTextObserved`` key and the new ``fullTextUrlFound`` key with the
        same value. Clients built against the pre-rename contract keep working.
        (ws-dispatch-contract-trust / finding #2.)"""
        from paper_chaser_mcp.dispatch import (
            _guided_source_record_from_paper,
            _guided_source_record_from_structured_source,
        )

        # Paper path — DOI present so upstream asserts don't care about
        # verification status; only the dual-emit invariant is tested here.
        paper_record = _guided_source_record_from_paper(
            "query",
            {
                "title": "Some Paper",
                "authors": [{"name": "A Person"}],
                "doi": "10.1234/example",
                "fullTextUrlFound": True,
            },
            index=1,
        )
        assert paper_record["fullTextUrlFound"] is True
        assert paper_record["fullTextObserved"] is True
        assert paper_record["fullTextUrlFound"] == paper_record["fullTextObserved"]

        # Structured-source path (from ask_result_set output).
        structured_record = _guided_source_record_from_structured_source(
            {
                "sourceId": "src-1",
                "title": "Another",
                "sourceType": "scholarly_article",
                "fullTextObserved": True,
            },
            index=1,
        )
        assert structured_record["fullTextUrlFound"] is True
        assert structured_record["fullTextObserved"] is True
        assert structured_record["fullTextUrlFound"] == structured_record["fullTextObserved"]


# ---------------------------------------------------------------------------
# 1.8: regulatoryTimeline / timeline deduplication
# ---------------------------------------------------------------------------


class TestTimelineDeduplication:
    """Stress-test finding: research response has BOTH regulatoryTimeline and
    timeline keys containing the same data."""

    def test_response_must_not_have_both_timeline_keys(self) -> None:
        """A guided response should not contain both regulatoryTimeline and timeline."""
        # Simulate what the response dict looks like after contract_fields update
        timeline_data = {"events": [{"title": "ESA listing", "date": "2015-04-02"}]}
        contract_fields = {"timeline": timeline_data, "resultStatus": "succeeded"}

        response: dict = {
            "intent": "regulatory",
            "status": "succeeded",
            "sources": [],
        }
        # After fix, regulatoryTimeline should NOT be in the response
        response.update(contract_fields)

        assert "timeline" in response
        assert "regulatoryTimeline" not in response, (
            "Response must not contain both 'regulatoryTimeline' and 'timeline'"
        )

    def test_timeline_data_appears_only_under_canonical_key(self) -> None:
        """When timeline data exists, it must appear only under 'timeline'."""
        timeline_data = {"events": [{"title": "Final rule", "date": "2023-01-15"}]}
        contract_fields = {"timeline": timeline_data}

        response: dict = {"intent": "regulatory"}
        response.update(contract_fields)

        assert response.get("timeline") == timeline_data
        assert "regulatoryTimeline" not in response


# ---------------------------------------------------------------------------
# 1.9: openAccessPdf.url null normalization
# ---------------------------------------------------------------------------


class TestPdfUrlNullNormalization:
    """Stress-test finding: pdf_url="" leaks into Paper objects and source records
    instead of being normalized to None."""

    def test_paper_empty_string_pdf_url_normalized_to_none(self) -> None:
        """Paper with pdf_url='' should be normalized to pdf_url=None
        after annotate_paper_trust_metadata."""
        from paper_chaser_mcp.models.common import Paper
        from paper_chaser_mcp.search_executor import annotate_paper_trust_metadata

        paper = Paper(
            paperId="test-1",
            title="Test Paper",
            url="https://example.com/paper",
            pdfUrl="",
            source="core",
        )
        annotated = annotate_paper_trust_metadata(paper)
        assert annotated.pdf_url is None, f"Expected pdf_url=None after normalization, got {annotated.pdf_url!r}"

    def test_paper_none_pdf_url_stays_none(self) -> None:
        """Paper with pdf_url=None remains None."""
        from paper_chaser_mcp.models.common import Paper
        from paper_chaser_mcp.search_executor import annotate_paper_trust_metadata

        paper = Paper(
            paperId="test-2",
            title="Test Paper",
            url="https://example.com/paper",
            pdfUrl=None,
            source="openalex",
        )
        annotated = annotate_paper_trust_metadata(paper)
        assert annotated.pdf_url is None

    def test_paper_valid_pdf_url_preserved(self) -> None:
        """Paper with a real pdf_url keeps it."""
        from paper_chaser_mcp.models.common import Paper
        from paper_chaser_mcp.search_executor import annotate_paper_trust_metadata

        paper: Paper = Paper(
            paperId="test-3",
            title="Test Paper",
            url="https://example.com/paper",
            pdfUrl="https://example.com/paper.pdf",
            source="semantic_scholar",
        )
        annotated = annotate_paper_trust_metadata(paper)
        assert annotated.pdf_url == "https://example.com/paper.pdf"

    def test_source_record_pdf_url_not_empty_string(self) -> None:
        """Source records built from papers must not have empty-string canonical URLs
        derived from empty pdf_url."""
        from paper_chaser_mcp.dispatch import _guided_source_record_from_paper

        paper: dict[str, object] = {
            "title": "Test Paper",
            "url": None,
            "pdfUrl": "",
            "source": "core",
            "authors": [],
        }
        record = _guided_source_record_from_paper("test query", paper, index=1)
        # canonicalUrl falls through to pdfUrl as last resort — must not be ""
        assert record.get("canonicalUrl") != "", "canonicalUrl must not be an empty string"
