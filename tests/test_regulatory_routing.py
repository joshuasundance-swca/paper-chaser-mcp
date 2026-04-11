"""Tests for regulatory routing verification status and source type classification.

Phase 6 fixes: verification status over-promotion and regulatory source type brittleness.
"""

from __future__ import annotations

import pytest

from paper_chaser_mcp.dispatch import (
    _assign_verification_status,
    _guided_source_record_from_paper,
    _guided_source_record_from_structured_source,
)
from paper_chaser_mcp.guided_semantic import classify_source

# ---------------------------------------------------------------------------
# 1. _assign_verification_status — core logic
# ---------------------------------------------------------------------------


class TestAssignVerificationStatus:
    """Verify that _assign_verification_status returns the correct status."""

    def test_unknown_source_no_doi_returns_unverified(self) -> None:
        """A source with unknown type and no DOI must be 'unverified'."""
        result = _assign_verification_status(source_type="unknown", has_doi=False)
        assert result == "unverified"

    def test_scholarly_article_no_doi_returns_unverified(self) -> None:
        """A scholarly article without DOI should be 'unverified'."""
        result = _assign_verification_status(source_type="scholarly_article", has_doi=False)
        assert result == "unverified"

    def test_scholarly_article_with_doi_returns_verified_metadata(self) -> None:
        """A scholarly article with DOI should be 'verified_metadata'."""
        result = _assign_verification_status(source_type="scholarly_article", has_doi=True)
        assert result == "verified_metadata"

    def test_doi_with_resolution_returns_verified_metadata(self) -> None:
        """Any source type with DOI + resolution should return 'verified_metadata'."""
        result = _assign_verification_status(source_type="unknown", has_doi=True, has_doi_resolution=True)
        assert result == "verified_metadata"

    def test_regulatory_document_returns_verified_primary_source(self) -> None:
        result = _assign_verification_status(source_type="regulatory_document")
        assert result == "verified_primary_source"

    def test_government_document_returns_verified_primary_source(self) -> None:
        result = _assign_verification_status(source_type="government_document")
        assert result == "verified_primary_source"

    def test_federal_register_rule_returns_verified_primary_source(self) -> None:
        result = _assign_verification_status(source_type="federal_register_rule")
        assert result == "verified_primary_source"

    def test_primary_source_returns_verified_primary_source(self) -> None:
        result = _assign_verification_status(source_type="primary_source")
        assert result == "verified_primary_source"

    def test_full_text_url_found_regulatory_returns_verified_primary_source(self) -> None:
        result = _assign_verification_status(source_type="regulatory_document", full_text_url_found=True)
        assert result == "verified_primary_source"

    # --- Regulatory source type variations (brittleness fix) ---

    def test_government_report_returns_verified_primary_source(self) -> None:
        """'government_report' should be recognized as a regulatory source."""
        result = _assign_verification_status(source_type="government_report")
        assert result == "verified_primary_source"

    def test_regulatory_guidance_returns_verified_primary_source(self) -> None:
        result = _assign_verification_status(source_type="regulatory_guidance")
        assert result == "verified_primary_source"

    def test_executive_order_returns_verified_primary_source(self) -> None:
        result = _assign_verification_status(source_type="executive_order")
        assert result == "verified_primary_source"

    def test_legislation_returns_verified_primary_source(self) -> None:
        result = _assign_verification_status(source_type="legislation")
        assert result == "verified_primary_source"

    def test_agency_report_returns_verified_primary_source(self) -> None:
        result = _assign_verification_status(source_type="agency_report")
        assert result == "verified_primary_source"

    def test_congressional_report_returns_verified_primary_source(self) -> None:
        result = _assign_verification_status(source_type="congressional_report")
        assert result == "verified_primary_source"

    def test_empty_source_type_no_doi_returns_unverified(self) -> None:
        """Empty string source type with no DOI should be 'unverified'."""
        result = _assign_verification_status(source_type="", has_doi=False)
        assert result == "unverified"

    def test_blog_post_no_doi_returns_unverified(self) -> None:
        """An unrecognized source type like 'blog_post' with no DOI should be unverified."""
        result = _assign_verification_status(source_type="blog_post", has_doi=False)
        assert result == "unverified"


# ---------------------------------------------------------------------------
# 2. Source record builders — verification status flows correctly
# ---------------------------------------------------------------------------


class TestSourceRecordVerificationFlow:
    """Verify that source records get correct verification status."""

    def test_paper_without_doi_gets_unverified(self) -> None:
        """A paper dict without DOI should produce unverified status in record."""
        paper = {
            "title": "Some Paper Without DOI",
            "source": "unknown_provider",
        }
        record = _guided_source_record_from_paper("test query", paper, index=0)
        assert record["verificationStatus"] == "unverified"

    def test_paper_with_doi_gets_verified_metadata(self) -> None:
        """A paper with DOI should produce verified_metadata."""
        paper = {
            "title": "Paper With DOI",
            "doi": "10.1234/test",
            "source": "semantic_scholar",
        }
        record = _guided_source_record_from_paper("test query", paper, index=0)
        assert record["verificationStatus"] == "verified_metadata"

    def test_structured_source_regulatory_type(self) -> None:
        """A structured source with regulatory_document type gets verified_primary_source."""
        source = {
            "sourceType": "regulatory_document",
            "title": "EPA Final Rule",
        }
        record = _guided_source_record_from_structured_source(source, index=0)
        assert record["verificationStatus"] == "verified_primary_source"

    def test_structured_source_unknown_no_doi_gets_unverified(self) -> None:
        """A structured source with unknown type and no DOI should be unverified."""
        source = {
            "sourceType": "unknown",
            "title": "Random Thing",
        }
        record = _guided_source_record_from_structured_source(source, index=0)
        assert record["verificationStatus"] == "unverified"

    def test_structured_source_explicit_status_preserved(self) -> None:
        """If the source already has verificationStatus, it should be preserved."""
        source = {
            "sourceType": "unknown",
            "title": "Manually Verified",
            "verificationStatus": "verified_metadata",
        }
        record = _guided_source_record_from_structured_source(source, index=0)
        assert record["verificationStatus"] == "verified_metadata"


# ---------------------------------------------------------------------------
# 3. classify_source — downstream impact of verification status
# ---------------------------------------------------------------------------


class TestClassifySourceVerification:
    """Verify classify_source correctly handles unverified sources."""

    def test_on_topic_verified_is_evidence(self) -> None:
        source = {
            "sourceId": "s1",
            "topicalRelevance": "on_topic",
            "verificationStatus": "verified_metadata",
        }
        decision = classify_source(source)
        assert decision.include_as == "evidence"

    def test_on_topic_unverified_is_lead(self) -> None:
        """An on-topic but unverified source should be classified as a lead, not evidence."""
        source = {
            "sourceId": "s2",
            "topicalRelevance": "on_topic",
            "verificationStatus": "unverified",
        }
        decision = classify_source(source)
        assert decision.include_as == "lead"
        assert "unverified" in (decision.why_not_verified or "").lower()

    def test_on_topic_no_verification_is_lead(self) -> None:
        """Source with no verificationStatus should default to unverified → lead."""
        source = {
            "sourceId": "s3",
            "topicalRelevance": "on_topic",
        }
        decision = classify_source(source)
        assert decision.include_as == "lead"

    def test_on_topic_verified_primary_source_is_evidence(self) -> None:
        source = {
            "sourceId": "s4",
            "topicalRelevance": "on_topic",
            "verificationStatus": "verified_primary_source",
        }
        decision = classify_source(source)
        assert decision.include_as == "evidence"


# ---------------------------------------------------------------------------
# 4. Regulatory source type set coverage
# ---------------------------------------------------------------------------


class TestRegulatorySourceTypeSet:
    """Ensure the regulatory source type set captures common variations."""

    @pytest.mark.parametrize(
        "source_type",
        [
            "regulatory_document",
            "primary_source",
            "government_document",
            "federal_register_rule",
            "government_report",
            "regulatory_guidance",
            "executive_order",
            "legislation",
            "agency_report",
            "congressional_report",
            "treaty",
            "statute",
            "policy_document",
        ],
    )
    def test_regulatory_types_get_primary_source_status(self, source_type: str) -> None:
        result = _assign_verification_status(source_type=source_type)
        assert result == "verified_primary_source", (
            f"Source type '{source_type}' should be verified_primary_source, got '{result}'"
        )
