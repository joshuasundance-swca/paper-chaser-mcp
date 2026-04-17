"""Shared Pydantic models for request validation and normalized payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, RootModel, model_validator

# Mirrors :data:`paper_chaser_mcp.models.tools.KnownItemResolutionState`. We redefine
# the Literal alias locally to avoid a ``common <- tools <- ecos <- common`` import
# cycle; the two aliases are structurally identical (Literal types are nominal by
# value, not by name).
KnownItemResolutionState = Literal[
    "resolved_exact",
    "resolved_probable",
    "needs_disambiguation",
]


class ApiModel(BaseModel):
    """Base model that preserves unknown provider fields during normalization."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class Pagination(BaseModel):
    """Unified pagination envelope present on every list response.

    ``has_more`` tells the caller whether additional pages exist.
    ``next_cursor`` is an opaque token to pass as ``cursor`` in the next tool
    call to retrieve the following page.  Its internal encoding is
    provider-specific and may change between releases; treat it as a black box.

    ``extra="ignore"`` is intentional: the parent model's ``model_validator``
    always recomputes this object from typed fields (``next`` / ``token``), so
    any leftover camelCase keys that arrive when a dict is round-tripped through
    ``dump_jsonable`` are safely discarded without raising.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    has_more: bool = Field(default=False, serialization_alias="hasMore")
    next_cursor: str | None = Field(default=None, serialization_alias="nextCursor")


class Author(ApiModel):
    """Normalized author payload."""

    name: str | None = None


class CrossrefWorkSummary(ApiModel):
    """Normalized Crossref work subset used by explicit enrichment tools."""

    doi: str | None = None
    title: str | None = None
    authors: list[Author] = Field(default_factory=list)
    venue: str | None = None
    publisher: str | None = None
    publication_type: str | None = Field(default=None, alias="publicationType")
    publication_date: str | None = Field(default=None, alias="publicationDate")
    year: int | None = None
    url: str | None = None
    citation_count: int | None = Field(default=None, alias="citationCount")


class CrossrefEnrichment(ApiModel):
    """Additive paper enrichment fields sourced from Crossref."""

    doi: str | None = None
    publisher: str | None = None
    venue: str | None = None
    publication_type: str | None = Field(default=None, alias="publicationType")
    publication_date: str | None = Field(default=None, alias="publicationDate")
    year: int | None = None
    url: str | None = None
    citation_count: int | None = Field(default=None, alias="citationCount")


class UnpaywallEnrichment(ApiModel):
    """Additive paper enrichment fields sourced from Unpaywall."""

    doi: str | None = None
    is_oa: bool | None = Field(default=None, alias="isOa")
    oa_status: str | None = Field(default=None, alias="oaStatus")
    best_oa_url: str | None = Field(default=None, alias="bestOaUrl")
    pdf_url: str | None = Field(default=None, alias="pdfUrl")
    license: str | None = None
    journal_is_in_doaj: bool | None = Field(
        default=None,
        alias="journalIsInDoaj",
    )


class OpenAlexEnrichment(ApiModel):
    """Additive paper enrichment fields sourced from OpenAlex."""

    source_id: str | None = Field(default=None, alias="sourceId")
    doi: str | None = None
    venue: str | None = None
    publication_type: str | None = Field(default=None, alias="publicationType")
    publication_date: str | None = Field(default=None, alias="publicationDate")
    year: int | None = None
    url: str | None = None
    pdf_url: str | None = Field(default=None, alias="pdfUrl")
    citation_count: int | None = Field(default=None, alias="citationCount")


class ScholarApiContentAccess(ApiModel):
    """Structured content-access metadata surfaced from ScholarAPI."""

    paper_id: str | None = Field(default=None, alias="paperId")
    has_text: bool | None = Field(default=None, alias="hasText")
    has_pdf: bool | None = Field(default=None, alias="hasPdf")
    indexed_at: str | None = Field(default=None, alias="indexedAt")
    journal_publisher: str | None = Field(default=None, alias="journalPublisher")
    journal_issn: str | None = Field(default=None, alias="journalIssn")
    journal_issue: str | None = Field(default=None, alias="journalIssue")
    journal_pages: str | None = Field(default=None, alias="journalPages")


class PaperContentAccess(ApiModel):
    """Optional structured access metadata attached to a normalized paper."""

    scholarapi: ScholarApiContentAccess | None = None


class PaperEnrichments(ApiModel):
    """Optional additive enrichment payload attached to a normalized paper."""

    crossref: CrossrefEnrichment | None = None
    unpaywall: UnpaywallEnrichment | None = None
    openalex: OpenAlexEnrichment | None = None


SourceType = Literal[
    "primary_regulatory",
    "scholarly_article",
    "repository_record",
    "mirror",
    "secondary_summary",
    "unknown",
]
VerificationStatus = Literal[
    "verified_primary_source",
    "verified_metadata",
    "search_hit_only",
    "unverified",
]
AccessStatus = Literal[
    "full_text_verified",
    "full_text_retrieved",
    "abstract_only",
    "oa_verified",
    "oa_uncertain",
    "access_unverified",
    "mirror_only",
]
OpenAccessRoute = Literal[
    "canonical_open_access",
    "repository_open_access",
    "mirror_only",
    "non_oa_or_unconfirmed",
    "unknown",
]
LikelyCompleteness = Literal["likely_complete", "partial", "unknown", "incomplete"]
FailureOutcome = Literal[
    "total_failure",
    "partial_success",
    "fallback_success",
    "no_failure",
    "needs_clarification",
]


class CitationRecord(ApiModel):
    """Export-ready citation metadata shared across guided source payloads."""

    authors: list[str] = Field(default_factory=list)
    year: str | None = None
    title: str | None = None
    journal_or_publisher: str | None = Field(default=None, alias="journalOrPublisher")
    doi: str | None = None
    url: str | None = None
    source_type: SourceType | None = Field(default=None, alias="sourceType")
    confidence: Literal["high", "medium", "low"] | None = None


class PrimaryDocumentCoverage(ApiModel):
    """Regulatory primary-document retrieval coverage for current-text workflows."""

    current_text_requested: bool = Field(default=False, alias="currentTextRequested")
    govinfo_attempted: bool = Field(default=False, alias="govinfoAttempted")
    govinfo_matched: bool = Field(default=False, alias="govinfoMatched")
    cfr_attempted: bool = Field(default=False, alias="cfrAttempted")
    cfr_matched: bool = Field(default=False, alias="cfrMatched")
    federal_register_attempted: bool = Field(default=False, alias="federalRegisterAttempted")
    federal_register_matched: bool = Field(default=False, alias="federalRegisterMatched")
    history_only: bool = Field(default=False, alias="historyOnly")
    current_text_satisfied: bool = Field(default=False, alias="currentTextSatisfied")


class CoverageSummary(ApiModel):
    """Structured provider-coverage summary for one retrieval run."""

    providers_attempted: list[str] = Field(default_factory=list, alias="providersAttempted")
    providers_succeeded: list[str] = Field(default_factory=list, alias="providersSucceeded")
    providers_failed: list[str] = Field(default_factory=list, alias="providersFailed")
    providers_zero_results: list[str] = Field(default_factory=list, alias="providersZeroResults")
    date_searched: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        alias="dateSearched",
    )
    likely_completeness: LikelyCompleteness = Field(default="unknown", alias="likelyCompleteness")
    search_mode: str | None = Field(default=None, alias="searchMode")
    retrieval_notes: list[str] = Field(default_factory=list, alias="retrievalNotes")
    summary_line: str | None = Field(default=None, alias="summaryLine")
    primary_document_coverage: PrimaryDocumentCoverage | None = Field(
        default=None,
        alias="primaryDocumentCoverage",
    )


class FailureSummary(ApiModel):
    """Compact, user-facing explanation of degraded or failed retrieval."""

    outcome: FailureOutcome | None = None
    what_failed: str | None = Field(default=None, alias="whatFailed")
    what_still_worked: str | None = Field(default=None, alias="whatStillWorked")
    fallback_attempted: bool = Field(default=False, alias="fallbackAttempted")
    fallback_mode: str | None = Field(default=None, alias="fallbackMode")
    primary_path_failure_reason: str | None = Field(default=None, alias="primaryPathFailureReason")
    completeness_impact: str | None = Field(default=None, alias="completenessImpact")
    recommended_next_action: str | None = Field(default=None, alias="recommendedNextAction")


class NormalizationRepair(ApiModel):
    """One guided-input normalization repair applied before tool execution."""

    field: str
    from_value: str = Field(default="", alias="from")
    to_value: str = Field(default="", alias="to")
    reason: str


class InputNormalization(ApiModel):
    """Structured record of guided-input normalization and warnings."""

    normalized_query: str | None = Field(default=None, alias="normalizedQuery")
    normalized_focus: str | None = Field(default=None, alias="normalizedFocus")
    normalized_venue: str | None = Field(default=None, alias="normalizedVenue")
    normalized_year: str | None = Field(default=None, alias="normalizedYear")
    normalized_search_session_id: str | None = Field(default=None, alias="normalizedSearchSessionId")
    normalized_question: str | None = Field(default=None, alias="normalizedQuestion")
    normalized_source_id: str | None = Field(default=None, alias="normalizedSourceId")
    repairs: list[NormalizationRepair] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MachineFailure(ApiModel):
    """Structured machine-failure payload for guided wrappers."""

    category: str
    error_type: str = Field(alias="errorType")
    error: str
    retryable: bool = True
    best_next_internal_action: str | None = Field(default=None, alias="bestNextInternalAction")


class SourceResolution(ApiModel):
    """Resolution metadata for guided source inspection."""

    requested_source_id: str | None = Field(default=None, alias="requestedSourceId")
    resolved_source_id: str | None = Field(default=None, alias="resolvedSourceId")
    match_type: str | None = Field(default=None, alias="matchType")
    available_source_ids: list[str] = Field(default_factory=list, alias="availableSourceIds")


class SessionCandidate(ApiModel):
    """Compact candidate session record used for guided disambiguation."""

    search_session_id: str = Field(alias="searchSessionId")
    source_tool: str = Field(alias="sourceTool")
    query: str | None = None
    summary: str | None = None
    age_seconds: int = Field(default=0, alias="ageSeconds")
    source_count: int = Field(default=0, alias="sourceCount")


class SessionResolution(ApiModel):
    """How the guided wrapper resolved or failed to resolve searchSessionId."""

    requested_search_session_id: str | None = Field(default=None, alias="requestedSearchSessionId")
    resolved_search_session_id: str | None = Field(default=None, alias="resolvedSearchSessionId")
    resolution_mode: str = Field(alias="resolutionMode")
    warnings: list[str] = Field(default_factory=list)
    candidates: list[SessionCandidate] = Field(default_factory=list)


class GuidedExecutionProvenance(ApiModel):
    """Compact guided-facing explanation of how the server produced a result."""

    execution_mode: str = Field(alias="executionMode")
    answer_source: str | None = Field(default=None, alias="answerSource")
    server_policy_applied: str = Field(alias="serverPolicyApplied")
    latency_profile_applied: str | None = Field(default=None, alias="latencyProfileApplied")
    allow_paid_providers: bool | None = Field(default=None, alias="allowPaidProviders")
    provider_budget_applied: dict[str, Any] = Field(default_factory=dict, alias="providerBudgetApplied")
    configured_smart_provider: str | None = Field(default=None, alias="configuredSmartProvider")
    active_smart_provider: str | None = Field(default=None, alias="activeSmartProvider")
    deterministic_fallback_used: bool = Field(default=False, alias="deterministicFallbackUsed")
    escalation_attempted: bool = Field(default=False, alias="escalationAttempted")
    escalation_reason: str | None = Field(default=None, alias="escalationReason")
    passes_run: int = Field(default=0, alias="passesRun")
    pass_modes: list[str] = Field(default_factory=list, alias="passModes")


class AbstentionDetails(ApiModel):
    """Actionable rationale for guided abstention or insufficient-evidence states."""

    category: str
    reason: str
    inspectable_source_count: int = Field(default=0, alias="inspectableSourceCount")
    on_topic_source_count: int = Field(default=0, alias="onTopicSourceCount")
    weak_match_count: int = Field(default=0, alias="weakMatchCount")
    off_topic_count: int = Field(default=0, alias="offTopicCount")
    can_inspect_sources: bool = Field(default=False, alias="canInspectSources")
    refinement_hints: list[str] = Field(default_factory=list, alias="refinementHints")


class GuidedResultState(ApiModel):
    """Compact guided-result state used by the guided wrappers."""

    status: str
    groundedness: str
    has_inspectable_sources: bool = Field(default=False, alias="hasInspectableSources")
    can_answer_follow_up: bool = Field(default=False, alias="canAnswerFollowUp")
    best_next_internal_action: str = Field(alias="bestNextInternalAction")
    missing_evidence_type: str = Field(alias="missingEvidenceType")


class ConfidenceSignals(ApiModel):
    """Additive trust and synthesis cues for guided responses."""

    evidence_quality_profile: Literal["high", "medium", "low"] | None = Field(
        default=None,
        alias="evidenceQualityProfile",
    )
    synthesis_mode: str | None = Field(default=None, alias="synthesisMode")
    trust_revision_reason: str | None = Field(default=None, alias="trustRevisionReason")
    evidence_use_plan_applied: bool | None = Field(default=None, alias="evidenceUsePlanApplied")
    fallback_explanation: str | None = Field(default=None, alias="fallbackExplanation")
    source_scope_label: str | None = Field(default=None, alias="sourceScopeLabel")
    source_scope_reason: str | None = Field(default=None, alias="sourceScopeReason")


class EvidenceUsePlan(ApiModel):
    """Additive synthesis-planning payload for grounded follow-up answers."""

    answer_subtype: str = Field(alias="answerSubtype")
    directly_responsive_ids: list[str] = Field(default_factory=list, alias="directlyResponsiveIds")
    unsupported_components: list[str] = Field(default_factory=list, alias="unsupportedComponents")
    retrieval_sufficiency: Literal["sufficient", "thin", "insufficient"] = Field(
        default="thin",
        alias="retrievalSufficiency",
    )
    confidence: Literal["high", "medium", "low"] = "medium"


class RuntimeSummary(ApiModel):
    """Effective runtime state for debugging and support."""

    effective_profile: str = Field(alias="effectiveProfile")
    transport_mode: str = Field(alias="transportMode")
    smart_layer_enabled: bool = Field(alias="smartLayerEnabled")
    active_provider_set: list[str] = Field(default_factory=list, alias="activeProviderSet")
    disabled_provider_set: list[str] = Field(default_factory=list, alias="disabledProviderSet")
    configured_smart_provider: str | None = Field(default=None, alias="configuredSmartProvider")
    active_smart_provider: str | None = Field(default=None, alias="activeSmartProvider")
    provider_order_effective: list[str] = Field(default_factory=list, alias="providerOrderEffective")
    tools_hidden: bool = Field(default=False, alias="toolsHidden")
    session_ttl_seconds: int | None = Field(default=None, alias="sessionTtlSeconds")
    embeddings_enabled: bool | None = Field(default=None, alias="embeddingsEnabled")
    guided_policy: str | None = Field(default=None, alias="guidedPolicy")
    guided_research_latency_profile: str | None = Field(default=None, alias="guidedResearchLatencyProfile")
    guided_follow_up_latency_profile: str | None = Field(default=None, alias="guidedFollowUpLatencyProfile")
    guided_allow_paid_providers: bool | None = Field(default=None, alias="guidedAllowPaidProviders")
    guided_escalation_enabled: bool | None = Field(default=None, alias="guidedEscalationEnabled")
    guided_escalation_max_passes: int | None = Field(default=None, alias="guidedEscalationMaxPasses")
    guided_escalation_allow_paid_providers: bool | None = Field(
        default=None,
        alias="guidedEscalationAllowPaidProviders",
    )
    version: str | None = None
    warnings: list[str] = Field(default_factory=list)


class Paper(ApiModel):
    """Normalized paper payload used across providers."""

    paper_id: str | None = Field(default=None, alias="paperId")
    title: str | None = None
    abstract: str | None = None
    year: int | None = None
    authors: list[Author] = Field(default_factory=list)
    citation_count: int | None = Field(default=None, alias="citationCount")
    reference_count: int | None = Field(default=None, alias="referenceCount")
    influential_citation_count: int | None = Field(
        default=None,
        alias="influentialCitationCount",
    )
    venue: str | None = None
    publication_types: Any = Field(default=None, alias="publicationTypes")
    publication_date: str | None = Field(default=None, alias="publicationDate")
    url: str | None = None
    pdf_url: str | None = Field(default=None, alias="pdfUrl")
    source: str | None = None
    source_id: str | None = Field(default=None, alias="sourceId")
    canonical_id: str | None = Field(default=None, alias="canonicalId")
    recommended_expansion_id: str | None = Field(
        default=None,
        alias="recommendedExpansionId",
        description=(
            "Semantic Scholar-compatible identifier to prefer for expansion "
            "tools such as get_paper_citations, get_paper_references, and "
            "get_paper_authors. Present only when the paper already exposes a "
            "portable DOI, arXiv ID, or Semantic Scholar paperId."
        ),
    )
    expansion_id_status: Literal["portable", "not_portable"] | None = Field(
        default=None,
        alias="expansionIdStatus",
        description=(
            "Whether this paper already exposes a Semantic Scholar-compatible "
            "expansion identifier. 'portable' means "
            "recommendedExpansionId can be passed directly to Semantic Scholar "
            "expansion tools. 'not_portable' means visible IDs are still "
            "provider-specific brokered identifiers and a DOI or "
            "Semantic-Scholar-native lookup is required first."
        ),
    )
    scholar_result_id: str | None = Field(
        default=None,
        alias="scholarResultId",
        description=(
            "SerpApi Google Scholar result_id for this paper. Set only on "
            "serpapi_google_scholar results; None otherwise. Pass as result_id "
            "to get_paper_citation_formats to retrieve MLA, APA, BibTeX, and "
            "other export formats. Use this field — not sourceId — because "
            "sourceId may be a cluster_id or cites_id when result_id is absent."
        ),
    )
    enrichments: PaperEnrichments | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "Optional additive enrichment payload. Crossref can improve DOI, "
            "publisher, venue, and publication metadata, Unpaywall can surface "
            "open-access and PDF availability, and OpenAlex can add citation, "
            "venue, and access-adjacent metadata without changing the base "
            "paper-search contract."
        ),
    )
    content_access: PaperContentAccess | None = Field(
        default=None,
        alias="contentAccess",
        exclude_if=lambda value: value is None,
        description=(
            "Optional structured content-access metadata. This is kept separate "
            "from bibliographic enrichment so access/full-text availability can "
            "be surfaced without implying new metadata provenance."
        ),
    )
    source_type: SourceType | None = Field(default=None, alias="sourceType")
    verification_status: VerificationStatus | None = Field(default=None, alias="verificationStatus")
    access_status: AccessStatus | None = Field(default=None, alias="accessStatus")
    canonical_url: str | None = Field(default=None, alias="canonicalUrl")
    retrieved_url: str | None = Field(default=None, alias="retrievedUrl")
    confidence: Literal["high", "medium", "low"] | None = None
    is_primary_source: bool | None = Field(default=None, alias="isPrimarySource")
    full_text_url_found: bool | None = Field(
        default=None,
        alias="fullTextUrlFound",
        validation_alias=AliasChoices("fullTextUrlFound", "fullTextObserved"),
    )
    full_text_retrieved: bool | None = Field(default=None, alias="fullTextRetrieved")
    abstract_observed: bool | None = Field(default=None, alias="abstractObserved")
    open_access_route: OpenAccessRoute | None = Field(default=None, alias="openAccessRoute")


class BrokerMetadata(BaseModel):
    """Metadata describing the brokered nature of a ``search_papers`` response.

    ``mode`` is always ``"brokered_single_page"`` for ``search_papers``,
    indicating the response is a single-page best-effort result rather than a
    provider-native continuation stream.

    ``provider_used`` identifies which provider supplied the returned results.
    Possible values are ``"core"``, ``"semantic_scholar"``, ``"arxiv"``,
    ``"serpapi_google_scholar"``, ``"scholarapi"``, or ``"none"`` when no
    provider returned results.

    ``result_status`` gives a first-class signal about the outcome of the search:
    - ``"returned_results"``: at least one provider returned results.
    - ``"no_results"``: all attempted providers returned empty result sets (genuine
      no-results for this query).
    - ``"provider_failed"``: every active provider encountered an upstream error;
      no results were returned.  This is distinct from ``"no_results"`` — it
      indicates a transient provider outage rather than a query with no matches.
      Inspect ``attemptedProviders`` for error details and retry later or pivot to
      a different tool.

    ``continuation_supported`` is always ``False`` for ``search_papers``; for
    paginated retrieval use ``search_papers_bulk`` or other provider-specific
    tools.

    ``result_quality`` characterises the match strength of the returned results:
    - ``"strong"``: Semantic Scholar used semantic/relevance ranking; results are
      likely topically relevant.
    - ``"low_relevance"``: Semantic Scholar returned results but one or more
      distinctive query tokens were absent from all result titles and abstracts,
      suggesting the results may not be relevant to the full query. Treat with
      caution; consider rephrasing or trying a different provider.
    - ``"lexical"``: CORE or arXiv used keyword/lexical matching only; results
      contain the query terms but may not be topically relevant, especially for
      unusual or nonsense queries.
        - ``"unknown"``: SerpApi, ScholarAPI, or no-result path; match quality is
            not determined.

        ``paid_provider_used`` is ``True`` when the winning provider path is
        paywalled, so callers can distinguish free broker results from explicit paid
        paths without inferring that from provider names.

    ``bulk_search_is_provider_pivot`` is ``True`` when ``provider_used`` is not
    ``"semantic_scholar"``, meaning that calling ``search_papers_bulk`` next
    would be a provider pivot to Semantic Scholar rather than a continuation of
    the same provider used here.  It is ``False`` when Semantic Scholar already
    supplied the current results, so ``search_papers_bulk`` is the closest
    continuation path.
    """

    model_config = ConfigDict(populate_by_name=True)
    mode: str = Field(default="brokered_single_page", serialization_alias="mode")
    result_status: Literal["returned_results", "no_results", "provider_failed"] = Field(
        default="no_results",
        serialization_alias="resultStatus",
        description=(
            "First-class outcome signal. 'returned_results' means at least one "
            "provider returned data. 'no_results' means all providers returned empty "
            "result sets (genuine empty query). 'provider_failed' means every active "
            "provider raised an upstream error — treat this as a transient outage, "
            "not an empty query; retry later or pivot to search_papers."
        ),
    )
    provider_used: str = Field(serialization_alias="providerUsed")
    paid_provider_used: bool = Field(
        default=False,
        serialization_alias="paidProviderUsed",
        description=(
            "True when the provider that produced these brokered results is a paid "
            "provider path such as SerpApi or ScholarAPI."
        ),
    )
    continuation_supported: bool = Field(
        default=False,
        serialization_alias="continuationSupported",
    )
    attempted_providers: list["BrokerAttempt"] = Field(
        default_factory=list,
        serialization_alias="attemptedProviders",
    )
    semantic_scholar_only_filters: list[str] = Field(
        default_factory=list,
        serialization_alias="semanticScholarOnlyFilters",
    )
    recommended_pagination_tool: str = Field(
        default="search_papers_bulk",
        serialization_alias="recommendedPaginationTool",
    )
    result_quality: Literal["strong", "low_relevance", "lexical", "unknown"] = Field(
        default="unknown",
        serialization_alias="resultQuality",
        description=(
            "Match-strength signal for the returned results. 'strong' means "
            "Semantic Scholar semantic ranking was used and results are likely "
            "topically relevant. 'low_relevance' means Semantic Scholar was used "
            "but distinctive query tokens were absent from all returned results — "
            "results may be weakly relevant; verify before trusting. "
            "'lexical' means keyword-only matching was used "
            "(CORE or arXiv) — results contain query terms but may not be "
            "topically relevant, especially for unusual or nonsense queries. "
            "'unknown' means match quality is not determined (SerpApi path or "
            "no results)."
        ),
    )
    bulk_search_is_provider_pivot: bool = Field(
        default=True,
        serialization_alias="bulkSearchIsProviderPivot",
        description=(
            "True when search_papers_bulk would switch to Semantic Scholar "
            "rather than continuing from the provider that supplied these results. "
            "False only when Semantic Scholar already supplied the current results."
        ),
    )
    next_step_hint: str = Field(
        default=(
            "Inspect the results. To get more pages use search_papers_bulk. "
            "To expand from a paper use get_paper_citations or get_paper_references."
        ),
        serialization_alias="nextStepHint",
    )


class BrokerAttempt(BaseModel):
    """One provider decision in the ``search_papers`` broker chain."""

    model_config = ConfigDict(populate_by_name=True)
    provider: str
    status: Literal["returned_results", "returned_no_results", "failed", "skipped"]
    reason: str | None = None


class SearchResponse(ApiModel):
    """Unified response for the ``search_papers`` tool.

    ``search_papers`` is a best-effort convenience tool that tries multiple
    providers in order.  It does **not** support cursor-based pagination because
    different providers use incompatible continuation mechanisms; mixing pages
    from different backends would produce incorrect results.  For paginated
    retrieval use ``search_papers_bulk`` (Semantic Scholar) or other
    provider-specific tools.

    ``broker_metadata`` is populated by the broker when results are returned;
    it is ``None`` only for internal helper paths that do not go through the
    full broker fallback chain.
    """

    total: int = 0
    offset: int = 0
    data: list[Paper] = Field(default_factory=list)
    broker_metadata: BrokerMetadata | None = Field(
        default=None,
        serialization_alias="brokerMetadata",
    )
    coverage_summary: CoverageSummary | None = Field(
        default=None,
        serialization_alias="coverageSummary",
    )
    failure_summary: FailureSummary | None = Field(
        default=None,
        serialization_alias="failureSummary",
    )


class SemanticSearchResponse(ApiModel):
    """Semantic Scholar search response."""

    total: int = 0
    offset: int = 0
    next: int | None = None
    data: list[Paper] = Field(default_factory=list)
    pagination: Pagination = Field(
        default_factory=lambda: Pagination(has_more=False),
    )

    @model_validator(mode="after")
    def _compute_pagination(self) -> "SemanticSearchResponse":
        self.pagination = Pagination(
            has_more=self.next is not None,
            next_cursor=str(self.next) if self.next is not None else None,
        )
        return self


class BulkSearchResponse(ApiModel):
    """Semantic Scholar bulk search response (token-paginated).

    The raw provider ``token`` is used internally to build the structured
    ``pagination.nextCursor`` and is excluded from the serialized response.
    Callers must use ``pagination.nextCursor`` as the single continuation handle.
    """

    total: int = 0
    token: str | None = Field(default=None, exclude=True)
    data: list[Paper] = Field(default_factory=list)
    pagination: Pagination = Field(
        default_factory=lambda: Pagination(has_more=False),
    )

    @model_validator(mode="after")
    def _compute_pagination(self) -> "BulkSearchResponse":
        self.pagination = Pagination(
            has_more=self.token is not None,
            next_cursor=self.token,
        )
        return self


class CoreSearchResponse(ApiModel):
    """CORE search response normalized to shared paper entries."""

    total: int = 0
    entries: list[Paper] = Field(default_factory=list)


class ArxivSearchResponse(ApiModel):
    """arXiv search response normalized to shared paper entries."""

    total_results: int = Field(default=0, alias="totalResults")
    entries: list[Paper] = Field(default_factory=list)


class AuthorProfile(ApiModel):
    """Semantic Scholar author response."""

    author_id: str | None = Field(default=None, alias="authorId")
    name: str | None = None
    affiliations: list[str] = Field(default_factory=list)
    homepage: str | None = None
    paper_count: int | None = Field(default=None, alias="paperCount")
    citation_count: int | None = Field(default=None, alias="citationCount")
    h_index: int | None = Field(default=None, alias="hIndex")


class AuthorListResponse(ApiModel):
    """Semantic Scholar offset-paginated author list response."""

    total: int = 0
    offset: int = 0
    next: int | None = None
    data: list[AuthorProfile] = Field(default_factory=list)
    pagination: Pagination = Field(
        default_factory=lambda: Pagination(has_more=False),
    )

    @model_validator(mode="after")
    def _compute_pagination(self) -> "AuthorListResponse":
        self.pagination = Pagination(
            has_more=self.next is not None,
            next_cursor=str(self.next) if self.next is not None else None,
        )
        return self


class BatchAuthorResponse(RootModel[list[AuthorProfile]]):
    """Semantic Scholar batch author lookup response."""


class PaperListResponse(ApiModel):
    """Provider response containing a list of papers under `data`.

    The ``offset`` and ``next`` fields mirror what Semantic Scholar returns for
    offset-paginated sub-resource endpoints (citations, references, author papers).
    ``next`` is the offset to pass in the following call; if absent the current
    page is the last one.
    """

    offset: int = 0
    next: int | None = None
    data: list[Paper] = Field(default_factory=list)
    pagination: Pagination = Field(
        default_factory=lambda: Pagination(has_more=False),
    )

    @model_validator(mode="after")
    def _compute_pagination(self) -> "PaperListResponse":
        self.pagination = Pagination(
            has_more=self.next is not None,
            next_cursor=str(self.next) if self.next is not None else None,
        )
        return self


class PaperAuthorListResponse(ApiModel):
    """Offset-paginated author list for a specific paper."""

    total: int = 0
    offset: int = 0
    next: int | None = None
    data: list[AuthorProfile] = Field(default_factory=list)
    pagination: Pagination = Field(
        default_factory=lambda: Pagination(has_more=False),
    )

    @model_validator(mode="after")
    def _compute_pagination(self) -> "PaperAuthorListResponse":
        self.pagination = Pagination(
            has_more=self.next is not None,
            next_cursor=str(self.next) if self.next is not None else None,
        )
        return self


class SnippetPaper(ApiModel):
    """Minimal paper info embedded in a snippet search result."""

    paper_id: str | None = Field(default=None, alias="paperId")
    title: str | None = None
    year: int | None = None
    url: str | None = None


class SnippetObject(ApiModel):
    """The `snippet` sub-object returned by ``/snippet/search``.

    Per the API guide the snippet result wraps snippet text inside a nested
    ``snippet`` object that can include ``text``, ``snippetKind``, ``section``,
    and annotation data.  Modelling it as a sub-object (instead of hoisting
    ``text`` to the top level) preserves all returned fields.
    """

    text: str | None = None
    snippet_kind: str | None = Field(default=None, alias="snippetKind")
    section: str | None = None


class SnippetResult(ApiModel):
    """One result from ``/snippet/search``.

    The API returns ``{"snippet": {...}, "score": ..., "paper": {...}}``.
    """

    score: float | None = None
    snippet: SnippetObject | None = None
    paper: SnippetPaper | None = None


class SnippetSearchResponse(ApiModel):
    """Response from `/snippet/search`."""

    data: list[SnippetResult] = Field(default_factory=list)


class RecommendationResponse(ApiModel):
    """Semantic Scholar recommendations response."""

    recommended_papers: list[Paper] = Field(
        default_factory=list,
        alias="recommendedPapers",
    )


class CitationFormat(ApiModel):
    """One text citation format (e.g. MLA, APA) returned by
    get_paper_citation_formats."""

    title: str = ""
    snippet: str = ""


class ExportLink(ApiModel):
    """One export download link (e.g. BibTeX, EndNote) returned by
    get_paper_citation_formats."""

    name: str = ""
    link: str = ""


class CitationFormatsResponse(ApiModel):
    """Response from the ``get_paper_citation_formats`` tool.

    ``result_id`` echoes the Scholar result_id used for the lookup.
    ``citations`` contains text citation strings (MLA, APA, Chicago, etc.).
    ``export_links`` contains structured-format download links (BibTeX, etc.).
    ``provider`` identifies the upstream service that supplied the data.
    """

    result_id: str = Field(serialization_alias="resultId")
    citations: list[CitationFormat] = Field(default_factory=list)
    export_links: list[ExportLink] = Field(
        default_factory=list,
        serialization_alias="exportLinks",
    )
    provider: str = "serpapi_google_scholar"


class CitationResolutionCandidate(ApiModel):
    """One ranked candidate returned by citation repair."""

    paper: Paper
    score: float = 0.0
    resolution_strategy: str = Field(
        default="",
        alias="resolutionStrategy",
    )
    matched_fields: list[str] = Field(
        default_factory=list,
        alias="matchedFields",
    )
    conflicting_fields: list[str] = Field(
        default_factory=list,
        alias="conflictingFields",
    )
    title_similarity: float = Field(
        default=0.0,
        alias="titleSimilarity",
    )
    year_delta: int | None = Field(
        default=None,
        alias="yearDelta",
    )
    author_overlap: int = Field(
        default=0,
        alias="authorOverlap",
    )
    candidate_count: int | None = Field(
        default=None,
        alias="candidateCount",
    )
    why_selected: str = Field(
        default="",
        alias="whySelected",
    )


class CitationResolutionResponse(ApiModel):
    """Structured response from the citation-repair workflow."""

    best_match: CitationResolutionCandidate | None = Field(
        default=None,
        alias="bestMatch",
    )
    alternatives: list[CitationResolutionCandidate] = Field(default_factory=list)
    resolution_confidence: Literal["high", "medium", "low"] = Field(
        default="low",
        alias="resolutionConfidence",
    )
    resolution_strategy: str = Field(
        default="none",
        alias="resolutionStrategy",
    )
    matched_fields: list[str] = Field(
        default_factory=list,
        alias="matchedFields",
    )
    conflicting_fields: list[str] = Field(
        default_factory=list,
        alias="conflictingFields",
    )
    normalized_citation: str = Field(
        default="",
        alias="normalizedCitation",
    )
    extracted_fields: dict[str, Any] = Field(
        default_factory=dict,
        alias="extractedFields",
    )
    inferred_fields: dict[str, Any] = Field(
        default_factory=dict,
        alias="inferredFields",
    )
    candidate_count: int = Field(
        default=0,
        alias="candidateCount",
    )
    known_item_resolution_state: KnownItemResolutionState | None = Field(
        default=None,
        alias="knownItemResolutionState",
        description=(
            "Execution-provenance label for resolve_reference outcomes. "
            "One of resolved_exact, resolved_probable, needs_disambiguation."
        ),
    )
    message: str = ""


class BatchPaperResponse(RootModel[list[Paper]]):
    """Semantic Scholar batch lookup response."""


class DoiResolution(ApiModel):
    """Resolved DOI state used by explicit enrichment workflows."""

    resolved_doi: str | None = Field(default=None, alias="resolvedDoi")
    resolution_source: str | None = Field(default=None, alias="resolutionSource")


class CrossrefEnrichmentResult(ApiModel):
    """Structured response from the explicit Crossref enrichment tool."""

    provider: str = "crossref"
    found: bool = False
    resolved_doi: str | None = Field(default=None, alias="resolvedDoi")
    work: CrossrefWorkSummary | None = None
    enrichment: CrossrefEnrichment | None = None


class UnpaywallEnrichmentResult(ApiModel):
    """Structured response from the explicit Unpaywall enrichment tool."""

    provider: str = "unpaywall"
    found: bool = False
    doi: str | None = None
    is_oa: bool | None = Field(default=None, alias="isOa")
    oa_status: str | None = Field(default=None, alias="oaStatus")
    best_oa_url: str | None = Field(default=None, alias="bestOaUrl")
    pdf_url: str | None = Field(default=None, alias="pdfUrl")
    license: str | None = None
    journal_is_in_doaj: bool | None = Field(
        default=None,
        alias="journalIsInDoaj",
    )
    enrichment: UnpaywallEnrichment | None = None


class OpenAlexEnrichmentResult(ApiModel):
    """Structured response from OpenAlex enrichment."""

    provider: str = "openalex"
    found: bool = False
    lookup_id: str | None = Field(default=None, alias="lookupId")
    resolved_doi: str | None = Field(default=None, alias="resolvedDoi")
    enrichment: OpenAlexEnrichment | None = None


class ScholarApiContentAccessResult(ApiModel):
    """Structured response from ScholarAPI content-access augmentation."""

    provider: str = "scholarapi"
    found: bool = False
    paper_id: str | None = Field(default=None, alias="paperId")
    content_access: ScholarApiContentAccess | None = Field(default=None, alias="contentAccess")


class PaperEnrichmentResponse(ApiModel):
    """Combined Crossref, Unpaywall, and OpenAlex enrichment response."""

    doi_resolution: DoiResolution = Field(
        default_factory=DoiResolution,
        alias="doiResolution",
    )
    crossref: CrossrefEnrichmentResult | None = None
    unpaywall: UnpaywallEnrichmentResult | None = None
    openalex: OpenAlexEnrichmentResult | None = None
    enrichments: PaperEnrichments | None = None
    scholarapi: ScholarApiContentAccessResult | None = None
    content_access: PaperContentAccess | None = Field(default=None, alias="contentAccess")


def dump_jsonable(value: Any) -> Any:
    """Serialize nested Pydantic models while leaving plain JSON values intact."""

    if isinstance(value, BaseModel):
        return value.model_dump(by_alias=True)
    if isinstance(value, list):
        return [dump_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: dump_jsonable(item) for key, item in value.items()}
    return value
