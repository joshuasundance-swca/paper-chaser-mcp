"""Typed Federal Register and CFR response models."""

from __future__ import annotations

from pydantic import Field

from .common import ApiModel, VerificationStatus


class FederalRegisterAgency(ApiModel):
    """Normalized Federal Register agency metadata."""

    name: str | None = None
    slug: str | None = None


class FederalRegisterDocument(ApiModel):
    """Normalized Federal Register document summary."""

    document_number: str | None = Field(default=None, alias="documentNumber")
    title: str | None = None
    abstract: str | None = None
    document_type: str | None = Field(default=None, alias="documentType")
    publication_date: str | None = Field(default=None, alias="publicationDate")
    start_page: int | None = Field(default=None, alias="startPage")
    end_page: int | None = Field(default=None, alias="endPage")
    citation: str | None = None
    agencies: list[FederalRegisterAgency] = Field(default_factory=list)
    body_html_url: str | None = Field(default=None, alias="bodyHtmlUrl")
    html_url: str | None = Field(default=None, alias="htmlUrl")
    pdf_url: str | None = Field(default=None, alias="pdfUrl")
    raw_text_url: str | None = Field(default=None, alias="rawTextUrl")
    public_inspection_pdf_url: str | None = Field(default=None, alias="publicInspectionPdfUrl")
    cfr_references: list[str] = Field(default_factory=list, alias="cfrReferences")
    govinfo_link: str | None = Field(default=None, alias="govInfoLink")
    govinfo_package_id: str | None = Field(default=None, alias="govInfoPackageId")
    govinfo_granule_id: str | None = Field(default=None, alias="govInfoGranuleId")
    is_primary_source: bool | None = Field(default=True, alias="isPrimarySource")
    verification_status: VerificationStatus | None = Field(default="verified_metadata", alias="verificationStatus")


class FederalRegisterSearchResponse(ApiModel):
    """Search response for Federal Register discovery."""

    total: int = 0
    data: list[FederalRegisterDocument] = Field(default_factory=list)


class FederalRegisterDocumentTextResponse(ApiModel):
    """Authoritative or fallback text extraction for one Federal Register document."""

    document: FederalRegisterDocument
    markdown: str | None = None
    content_source: str | None = Field(default=None, alias="contentSource")
    content_type: str | None = Field(default=None, alias="contentType")
    authoritative_url: str | None = Field(default=None, alias="authoritativeUrl")
    is_primary_source: bool = Field(default=True, alias="isPrimarySource")
    verification_status: VerificationStatus = Field(default="verified_primary_source", alias="verificationStatus")
    warnings: list[str] = Field(default_factory=list)


class CfrTextResponse(ApiModel):
    """Resolved CFR part or section text."""

    title_number: int = Field(alias="titleNumber")
    part_number: int = Field(alias="partNumber")
    section_number: str | None = Field(default=None, alias="sectionNumber")
    revision_year: int | None = Field(default=None, alias="revisionYear")
    effective_date: str | None = Field(default=None, alias="effectiveDate")
    citation: str | None = None
    package_id: str | None = Field(default=None, alias="packageId")
    granule_id: str | None = Field(default=None, alias="granuleId")
    resolved_volume: int | None = Field(default=None, alias="resolvedVolume")
    content_source: str | None = Field(default=None, alias="contentSource")
    content_type: str | None = Field(default=None, alias="contentType")
    source_url: str | None = Field(default=None, alias="sourceUrl")
    markdown: str | None = None
    is_primary_source: bool = Field(default=True, alias="isPrimarySource")
    verification_status: VerificationStatus = Field(default="verified_primary_source", alias="verificationStatus")
    warnings: list[str] = Field(default_factory=list)


class RegulatoryTimelineEvent(ApiModel):
    """One dated event in a regulatory history response."""

    event_date: str | None = Field(default=None, alias="eventDate")
    event_type: str = Field(alias="eventType")
    title: str | None = None
    source_type: str = Field(default="primary_regulatory", alias="sourceType")
    verification_status: VerificationStatus = Field(default="verified_metadata", alias="verificationStatus")
    citation: str | None = None
    canonical_url: str | None = Field(default=None, alias="canonicalUrl")
    provider: str | None = None
    note: str | None = None


class RegulatoryTimeline(ApiModel):
    """Chronological regulatory history with explicit gaps."""

    query: str
    subject: str | None = None
    events: list[RegulatoryTimelineEvent] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list, alias="evidenceGaps")
