"""Typed ECOS species and document response models."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .common import ApiModel

EcosDocumentKind = Literal[
    "recovery_plan",
    "five_year_review",
    "biological_opinion",
    "federal_register",
    "other_recovery_doc",
    "conservation_plan_link",
]
EcosExtractionStatus = Literal[
    "ok",
    "too_large",
    "unsupported_type",
    "empty_or_image_only",
    "fetch_failed",
]


class EcosSpeciesHit(ApiModel):
    """Compact species hit returned from ECOS search."""

    species_id: str = Field(alias="speciesId")
    common_name: str | None = Field(default=None, alias="commonName")
    scientific_name: str | None = Field(default=None, alias="scientificName")
    status_category: str | None = Field(default=None, alias="statusCategory")
    listing_status: str | None = Field(default=None, alias="listingStatus")
    group: str | None = None
    lead_agency: str | None = Field(default=None, alias="leadAgency")
    profile_url: str | None = Field(default=None, alias="profileUrl")


class EcosSpecies(ApiModel):
    """Normalized top-level ECOS species identity."""

    species_id: str = Field(alias="speciesId")
    common_name: str | None = Field(default=None, alias="commonName")
    scientific_name: str | None = Field(default=None, alias="scientificName")
    family: str | None = None
    group: str | None = None
    kingdom: str | None = None
    tsn: str | None = None
    uri: str | None = None
    profile_url: str | None = Field(default=None, alias="profileUrl")


class EcosSpeciesEntity(ApiModel):
    """Per-entity ECOS listing metadata."""

    entity_id: int | None = Field(default=None, alias="entityId")
    abbreviation: str | None = Field(default=None, alias="abbreviation")
    agency: str | None = None
    alt_status: str | None = Field(default=None, alias="altStatus")
    description: str | None = None
    dps: str | None = None
    status: str | None = None
    status_category: str | None = Field(default=None, alias="statusCategory")
    listing_date: str | None = Field(default=None, alias="listingDate")
    lead_fws_region: str | None = Field(default=None, alias="leadFwsRegion")
    recovery_priority_number: str | None = Field(
        default=None,
        alias="recoveryPriorityNumber",
    )
    more_info_url: str | None = Field(default=None, alias="moreInfoUrl")
    current_range_countries: list[str] = Field(
        default_factory=list,
        alias="currentRangeCountries",
    )
    current_range_states: list[str] = Field(
        default_factory=list,
        alias="currentRangeStates",
    )
    current_range_counties: list[str] = Field(
        default_factory=list,
        alias="currentRangeCounties",
    )
    current_range_refuges: list[str] = Field(
        default_factory=list,
        alias="currentRangeRefuges",
    )
    range_envelope: str | None = Field(default=None, alias="rangeEnvelope")
    range_shapefile: str | None = Field(default=None, alias="rangeShapefile")
    shapefile_last_updated: str | None = Field(
        default=None,
        alias="shapefileLastUpdated",
    )


class EcosRangeSummary(ApiModel):
    """Summarized range fields surfaced from species and entities."""

    historical_range_states: list[str] = Field(
        default_factory=list,
        alias="historicalRangeStates",
    )
    current_range_countries: list[str] = Field(
        default_factory=list,
        alias="currentRangeCountries",
    )
    current_range_states: list[str] = Field(
        default_factory=list,
        alias="currentRangeStates",
    )
    current_range_counties: list[str] = Field(
        default_factory=list,
        alias="currentRangeCounties",
    )
    current_range_refuges: list[str] = Field(
        default_factory=list,
        alias="currentRangeRefuges",
    )


class EcosDocument(ApiModel):
    """Normalized ECOS document or document-like link."""

    document_kind: EcosDocumentKind | None = Field(default=None, alias="documentKind")
    title: str | None = None
    url: str | None = None
    document_date: str | None = Field(default=None, alias="documentDate")
    entity_id: int | None = Field(default=None, alias="entityId")
    source_id: int | None = Field(default=None, alias="sourceId")
    recovery_plan_id: int | None = Field(default=None, alias="recoveryPlanId")
    publication_id: int | None = Field(default=None, alias="publicationId")
    document_type: str | None = Field(default=None, alias="documentType")
    qualifier: str | None = None
    publication_page: str | None = Field(default=None, alias="publicationPage")
    document_types: list[str] = Field(default_factory=list, alias="documentTypes")
    category: str | None = None
    event_code: str | None = Field(default=None, alias="eventCode")
    lead_agencies: str | None = Field(default=None, alias="leadAgencies")
    lead_offices: str | None = Field(default=None, alias="leadOffices")
    activities: str | None = None
    locations: str | None = None
    work_types: str | None = Field(default=None, alias="workTypes")
    plan_id: int | None = Field(default=None, alias="planId")
    plan_type: str | None = Field(default=None, alias="planType")


class EcosDocumentGroups(ApiModel):
    """Grouped species dossier documents."""

    recovery_plans: list[EcosDocument] = Field(
        default_factory=list,
        alias="recoveryPlans",
    )
    five_year_reviews: list[EcosDocument] = Field(
        default_factory=list,
        alias="fiveYearReviews",
    )
    biological_opinions: list[EcosDocument] = Field(
        default_factory=list,
        alias="biologicalOpinions",
    )
    federal_register_documents: list[EcosDocument] = Field(
        default_factory=list,
        alias="federalRegisterDocuments",
    )
    other_recovery_docs: list[EcosDocument] = Field(
        default_factory=list,
        alias="otherRecoveryDocs",
    )


class EcosSpeciesProfile(ApiModel):
    """Single-species ECOS dossier."""

    species: EcosSpecies
    species_entities: list[EcosSpeciesEntity] = Field(
        default_factory=list,
        alias="speciesEntities",
    )
    life_history: str | None = Field(default=None, alias="lifeHistory")
    range_summary: EcosRangeSummary = Field(
        default_factory=EcosRangeSummary,
        alias="range",
    )
    documents: EcosDocumentGroups = Field(default_factory=EcosDocumentGroups)
    conservation_plan_links: list[EcosDocument] = Field(
        default_factory=list,
        alias="conservationPlanLinks",
    )


class EcosSpeciesDocumentListResponse(ApiModel):
    """Flattened species-document inventory."""

    species_id: str = Field(alias="speciesId")
    total: int = 0
    document_kinds_applied: list[EcosDocumentKind] = Field(
        default_factory=list,
        alias="documentKindsApplied",
    )
    data: list[EcosDocument] = Field(default_factory=list)


class EcosDocumentTextResponse(ApiModel):
    """Fetched ECOS document text converted to Markdown."""

    document: EcosDocument
    markdown: str | None = None
    content_type: str | None = Field(default=None, alias="contentType")
    extraction_status: EcosExtractionStatus = Field(alias="extractionStatus")
    warnings: list[str] = Field(default_factory=list)
