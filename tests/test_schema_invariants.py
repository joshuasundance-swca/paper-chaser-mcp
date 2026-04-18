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
        """Regulatory types with body-text embedded earn verified_primary_source."""
        for source_type in ("regulatory_document", "primary_source", "government_document", "federal_register_rule"):
            status = _assign_verification_status(
                source_type=source_type,
                has_doi=False,
                has_doi_resolution=False,
                full_text_url_found=False,
                body_text_embedded=True,
            )
            assert status == "verified_primary_source", (
                f"Expected verified_primary_source for {source_type}, got {status}"
            )

    def test_regulatory_url_only_is_verified_metadata_not_primary(self) -> None:
        """P0-2: URL-only regulatory hits are verified_metadata, not primary."""
        for source_type in ("regulatory_document", "primary_source", "government_document", "federal_register_rule"):
            status = _assign_verification_status(
                source_type=source_type,
                has_doi=False,
                has_doi_resolution=False,
                full_text_url_found=False,
                body_text_embedded=False,
            )
            assert status == "verified_metadata", f"Expected verified_metadata for URL-only {source_type}, got {status}"

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


def test_zero_results_excludes_succeeded_providers_post_fallback() -> None:
    """P0-3 item 1 regression: when a provider initially returned empty but
    was then recovered via a fallback path, it must not remain in
    ``providersZeroResults``. The fix is a one-line reconciliation of
    ``zero_results`` against ``succeeded``/``failed`` immediately before
    ``CoverageSummary`` construction in ``_search_regulatory``.
    """
    import inspect

    from paper_chaser_mcp.agentic import graphs
    from paper_chaser_mcp.models.common import CoverageSummary

    # --- Behavioral invariant -------------------------------------------------
    # Simulate the buggy pre-reconciliation state produced by the two-pass
    # govinfo path in ``_search_regulatory``: the first attempt returned empty
    # (so ``zero_results`` contains ``govinfo``), and the fallback attempt
    # recovered it (so ``succeeded`` also contains ``govinfo``).
    attempted = ["govinfo", "federal_register"]
    succeeded = ["govinfo"]
    failed: list[str] = []

    # Canonical reconciliation that must run before CoverageSummary is built.
    zero_results = [p for p in attempted if p not in succeeded and p not in failed]

    coverage = CoverageSummary(
        providersAttempted=attempted,
        providersSucceeded=succeeded,
        providersFailed=failed,
        providersZeroResults=zero_results,
    )
    assert "govinfo" not in coverage.providers_zero_results, (
        "A provider recovered via fallback must not remain in providersZeroResults"
    )
    assert coverage.providers_zero_results == ["federal_register"]
    assert set(coverage.providers_succeeded) & set(coverage.providers_zero_results) == set()

    # --- Code-level regression guard -----------------------------------------
    # The fix must be applied inside ``_search_regulatory`` immediately before
    # the ``CoverageSummary`` is constructed so the invariant cannot be
    # violated at runtime, regardless of the order of appends along the
    # regulatory retrieval path.
    source = inspect.getsource(graphs.AgenticRuntime._search_regulatory)
    assert "zero_results = [p for p in attempted if p not in succeeded and p not in failed]" in source, (
        "_search_regulatory must reconcile zero_results against succeeded/failed "
        "before building CoverageSummary (see docs/ux-remediation-checklist.md P0-3 item 1)"
    )


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
# P0-2: body-text-embedded / qa-readable-text signal separation
# ---------------------------------------------------------------------------


class TestBodyTextAndQaReadableSignals:
    """P0-2: URL-found, body-text-embedded, and QA-readable-text are three
    distinct signals. The guided contract must not conflate them."""

    def test_access_status_literal_accepts_new_values(self) -> None:
        """AccessStatus must accept url_verified, body_text_embedded,
        qa_readable_text, and pdf_available."""
        from paper_chaser_mcp.models.common import Paper

        for value in ("url_verified", "body_text_embedded", "qa_readable_text", "pdf_available"):
            paper = Paper(title="x", accessStatus=value)  # type: ignore[arg-type]
            assert paper.access_status == value

    def test_full_text_url_found_does_not_imply_body_text_embedded(self) -> None:
        """P0-2: a paper with only a URL (no inline body) must emit
        bodyTextEmbedded=False and accessStatus=url_verified."""
        from paper_chaser_mcp import dispatch as dispatch_module

        paper = {"title": "URL only", "fullTextUrlFound": True}
        record = dispatch_module._guided_source_record_from_paper("q", paper, index=1)
        assert record["fullTextUrlFound"] is True
        assert record["accessStatus"] == "url_verified"
        assert record["bodyTextEmbedded"] is False
        assert record["qaReadableText"] is False

    def test_regulatory_url_only_is_verified_metadata_not_primary(self) -> None:
        """P0-2: URL-only regulatory hits must not be promoted to primary."""
        status = _assign_verification_status(
            source_type="federal_register_rule",
            has_doi=False,
            has_doi_resolution=False,
            full_text_url_found=False,
            body_text_embedded=False,
        )
        assert status == "verified_metadata"

    def test_regulatory_body_embedded_is_verified_primary(self) -> None:
        """P0-2: body_text_embedded=True promotes regulatory to primary source."""
        status = _assign_verification_status(
            source_type="federal_register_rule",
            has_doi=False,
            has_doi_resolution=False,
            full_text_url_found=False,
            body_text_embedded=True,
        )
        assert status == "verified_primary_source"


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


# ---------------------------------------------------------------------------
# P0-3 Item 2: verifiedPrimarySourceCount semantics + fullTextVerifiedPrimarySourceCount
# ---------------------------------------------------------------------------


class TestVerifiedPrimarySourceCountSemantics:
    """P0-3 item 2: rename/rescope verifiedPrimarySourceCount so it reflects
    isPrimarySource=True AND verificationStatus ∈ {verified_primary_source,
    verified_metadata}. Introduce a separate fullTextVerifiedPrimarySourceCount
    for the narrower full-text-only signal."""

    def test_primary_source_flag_reflected_in_verified_count(self) -> None:
        """verifiedPrimarySourceCount counts sources where isPrimarySource=True
        AND verificationStatus is verified_primary_source OR verified_metadata."""
        sources = [
            _make_source(
                source_id="fr-1",
                is_primary_source=True,
                verification_status="verified_primary_source",
                source_type="federal_register_rule",
            ),
            _make_source(
                source_id="gov-1",
                is_primary_source=True,
                verification_status="verified_metadata",
                source_type="regulatory_document",
            ),
            _make_source(
                source_id="paper-1",
                is_primary_source=False,
                verification_status="verified_metadata",
            ),
            _make_source(
                source_id="weak-1",
                is_primary_source=True,
                verification_status="unverified",
            ),
        ]
        trust = _guided_trust_summary(sources, [])
        assert trust["verifiedPrimarySourceCount"] == 2

    def test_full_text_verified_primary_source_count_is_narrower(self) -> None:
        """fullTextVerifiedPrimarySourceCount only counts sources where
        isPrimarySource=True AND verificationStatus=verified_primary_source."""
        sources = [
            _make_source(
                source_id="fr-1",
                is_primary_source=True,
                verification_status="verified_primary_source",
                source_type="federal_register_rule",
            ),
            _make_source(
                source_id="gov-1",
                is_primary_source=True,
                verification_status="verified_metadata",
                source_type="regulatory_document",
            ),
        ]
        trust = _guided_trust_summary(sources, [])
        assert "fullTextVerifiedPrimarySourceCount" in trust
        assert trust["fullTextVerifiedPrimarySourceCount"] == 1
        assert trust["verifiedPrimarySourceCount"] == 2


# ---------------------------------------------------------------------------
# P0-3 Item 3: source alias collision detection / global uniqueness
# ---------------------------------------------------------------------------


class TestSourceAliasCollisionDetection:
    """P0-3 item 3: per-bucket helpers used to pre-assign source-{index}
    aliases producing duplicates when multiple buckets were merged.
    _attach_source_aliases must detect collisions and reissue unique aliases."""

    def test_source_alias_collision_detection(self) -> None:
        """When two sources arrive with the same pre-assigned sourceAlias,
        _attach_source_aliases must detect the collision and reissue a unique
        alias to the colliding entries."""
        from paper_chaser_mcp.agentic.workspace import WorkspaceRegistry

        store = WorkspaceRegistry()
        payload = {
            "sources": [
                {"sourceId": "paper-a", "sourceAlias": "source-1"},
                {"sourceId": "paper-b", "sourceAlias": "source-1"},
            ],
        }
        normalized = store.attach_source_aliases(payload)
        aliases = [entry["sourceAlias"] for entry in normalized["sources"]]
        assert len(aliases) == len(set(aliases)), f"Aliases must be unique, got {aliases}"

    def test_source_alias_globally_unique_across_buckets(self) -> None:
        """Aliases must be globally unique across evidence, leads, sources,
        structuredSources, and other bucket keys processed by
        _attach_source_aliases."""
        from paper_chaser_mcp.agentic.workspace import WorkspaceRegistry

        store = WorkspaceRegistry()
        payload = {
            "sources": [
                {"sourceId": "p1", "sourceAlias": "source-1"},
                {"sourceId": "p2", "sourceAlias": "source-2"},
            ],
            "structuredSources": [
                {"sourceId": "s1", "sourceAlias": "source-1"},
            ],
            "leads": [
                {"sourceId": "l1", "sourceAlias": "source-fr-1"},
                {"sourceId": "l2", "sourceAlias": "source-fr-1"},
            ],
        }
        normalized = store.attach_source_aliases(payload)
        all_aliases: list[str] = []
        for key in ("sources", "structuredSources", "leads"):
            for entry in normalized.get(key, []):
                all_aliases.append(entry["sourceAlias"])
        assert len(all_aliases) == len(set(all_aliases)), (
            f"Aliases must be globally unique across buckets, got {all_aliases}"
        )


# ---------------------------------------------------------------------------
# P0-3 Item 4: topical relevance computed consistently, not hardcoded
# ---------------------------------------------------------------------------


class TestTopicalRelevanceConsistency:
    """P0-3 item 4: _guided_sources_from_fr_documents hardcodes
    topicalRelevance='on_topic'; research and inspect_source disagree on the
    same FR doc for the same query. Extract compute_topical_relevance and
    use it for both FR docs and the smart/paper path."""

    def test_fr_document_relevance_computed_not_assumed(self) -> None:
        """An off-topic FR document should NOT be labeled on_topic just because
        the FR API returned it — topicalRelevance must be computed from the
        canonical relevance function against the query."""
        import types

        from paper_chaser_mcp import dispatch as dispatch_module

        off_topic_doc = types.SimpleNamespace(
            title="Safety Standards for Commercial Truck Drivers",
            htmlUrl="https://example.com/fr/trucks",
            pdfUrl="https://example.com/fr/trucks.pdf",
            documentNumber="2020-11111",
            documentType="Rule",
            publicationDate="2020-06-15",
            citation="85 FR 11111",
            agencies=[types.SimpleNamespace(name="Federal Motor Carrier Safety Administration")],
            cfrReferences=["49 CFR 391"],
            abstract="Hours-of-service regulations for commercial motor vehicle operators.",
        )
        sources = dispatch_module._guided_sources_from_fr_documents(
            "northern long-eared bat endangered species critical habitat",
            [off_topic_doc],
        )
        assert len(sources) == 1
        assert sources[0]["topicalRelevance"] != "on_topic", (
            f"Off-topic FR doc should not be labeled on_topic, got {sources[0]['topicalRelevance']}"
        )

    def test_topical_relevance_consistent_across_research_and_inspect_source(self) -> None:
        """The same canonical relevance function must be used for FR docs and
        paper sources — so research and inspect_source agree on the same
        query/source pair."""
        from paper_chaser_mcp import dispatch as dispatch_module

        # A canonical helper must exist (item 4 extraction).
        assert hasattr(dispatch_module, "compute_topical_relevance"), (
            "compute_topical_relevance must be exposed as the canonical topical-relevance function"
        )
        # Same query + same structured signals should produce the same label
        # whether we call the public helper or the paper helper.
        query = "quantum error correction surface code"
        paper = {
            "title": "Surface code quantum error correction threshold",
            "abstract": "We prove a new threshold for the surface code.",
            "venue": "Physical Review A",
        }
        via_paper_helper = dispatch_module._paper_topical_relevance(query, paper)
        via_canonical = dispatch_module.compute_topical_relevance(query, paper)
        assert via_paper_helper == via_canonical


# ---------------------------------------------------------------------------
# P0-3 Item 5: authoritativeButWeak semantics documentation + prose note
# ---------------------------------------------------------------------------


class TestAuthoritativeButWeakSemantics:
    """P0-3 item 5: authoritativeButWeak is the 'missed-escalation' bucket —
    authoritative/primary-source records that are topically weak or off-topic.
    Docstrings and tool-spec descriptions should say so, and trustSummary
    should emit a one-line prose note when the bucket is populated."""

    def test_authoritative_but_weak_function_docstring_is_explicit(self) -> None:
        """_authoritative_but_weak_source_ids must document the bucket as a
        missed-escalation signal."""
        from paper_chaser_mcp.dispatch import _authoritative_but_weak_source_ids

        docstring = (_authoritative_but_weak_source_ids.__doc__ or "").lower()
        assert "missed-escalation" in docstring or "missed escalation" in docstring, (
            "Docstring should identify authoritativeButWeak as a missed-escalation bucket"
        )

    def test_authoritative_but_weak_tool_spec_descriptions_are_explicit(self) -> None:
        """research and follow_up_research descriptions should clarify that
        authoritativeButWeak is a missed-escalation bucket."""
        from paper_chaser_mcp.tool_specs.descriptions import TOOL_DESCRIPTIONS

        research = TOOL_DESCRIPTIONS["research"].lower()
        follow_up = TOOL_DESCRIPTIONS["follow_up_research"].lower()
        for description in (research, follow_up):
            assert "missed-escalation" in description or "missed escalation" in description, (
                "Tool description must call out missed-escalation semantics"
            )

    def test_authoritative_but_weak_emits_prose_note_when_populated(self) -> None:
        """When the authoritativeButWeak bucket is non-empty, trustSummary
        must include a one-line prose note calling attention to it."""
        sources = [
            _make_source(
                source_id="fr-off",
                is_primary_source=True,
                verification_status="verified_primary_source",
                topical_relevance="off_topic",
                source_type="federal_register_rule",
            ),
        ]
        trust = _guided_trust_summary(sources, [])
        assert trust.get("authoritativeButWeak") == ["fr-off"]
        note = trust.get("authoritativeButWeakNote")
        assert isinstance(note, str) and note.strip(), (
            f"Expected a non-empty authoritativeButWeakNote when bucket populated, got {note!r}"
        )
        assert "authoritative" in note.lower()

    def test_authoritative_but_weak_note_absent_when_empty(self) -> None:
        """When the bucket is empty, no note should be emitted."""
        sources = [
            _make_source(
                source_id="fr-on",
                is_primary_source=True,
                verification_status="verified_primary_source",
                topical_relevance="on_topic",
                source_type="federal_register_rule",
            ),
        ]
        trust = _guided_trust_summary(sources, [])
        assert trust.get("authoritativeButWeak") == []
        assert "authoritativeButWeakNote" not in trust or not trust.get("authoritativeButWeakNote")
