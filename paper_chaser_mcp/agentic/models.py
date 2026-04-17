"""Structured models for additive smart-tool workflows."""

from __future__ import annotations

from typing import Any, Final, Literal

from pydantic import AliasChoices, Field

from ..models.common import ApiModel, CitationRecord, CoverageSummary, FailureSummary, OpenAccessRoute, Paper
from ..models.regulations import RegulatoryTimeline
from ..models.tools import KnownItemResolutionState

IntentLabel = Literal[
    "discovery",
    "review",
    "known_item",
    "author",
    "citation",
    "regulatory",
]

PlannerQueryType = Literal[
    "broad_concept",
    "known_item",
    "citation_repair",
    "regulatory",
    "author",
    "review",
]

PlannerFirstPassMode = Literal["targeted", "broad", "mixed"]

RETRIEVAL_MODE_TARGETED: Final[PlannerFirstPassMode] = "targeted"
RETRIEVAL_MODE_BROAD: Final[PlannerFirstPassMode] = "broad"
RETRIEVAL_MODE_MIXED: Final[PlannerFirstPassMode] = "mixed"

# Canonical regulatory-intent split surfaced on PlannerDecision and
# SearchStrategyMetadata. Populated LLM-first from planner signals (intent,
# retrievalHypotheses, candidateConcepts) with a deterministic fallback in
# ``planner.py``. See docs/environmental-science-remediation-plan.md §B.
RegulatoryIntentLabel = Literal[
    "current_cfr_text",
    "rulemaking_history",
    "species_dossier",
    "guidance_lookup",
    "hybrid_regulatory_plus_literature",
    "unspecified",
]

# Document families practitioners expect from regulatory retrieval. Used by
# ``SubjectCard.requestedDocumentFamily`` and the document-family ranking boost
# in ``graphs._rank_regulatory_documents``.
DocumentFamilyLabel = Literal[
    "recovery_plan",
    "critical_habitat",
    "listing_rule",
    "consultation_guidance",
    "cfr_current_text",
    "rulemaking_notice",
    "programmatic_agreement",
    "tribal_policy",
    "agency_guidance",
    "unspecified",
]

SubjectCardConfidence = Literal["high", "medium", "low", "deterministic_fallback"]


class SubjectCard(ApiModel):
    """LLM-resolved subject card for regulatory / known-item retrieval.

    Populated LLM-first from planner outputs (``entity_card``,
    ``candidate_concepts``, ``anchor_value``). When the planner ran in
    deterministic-fallback mode, the card is still emitted but
    ``confidence`` is set to ``"deterministic_fallback"``.
    """

    common_name: str | None = Field(default=None, alias="commonName")
    scientific_name: str | None = Field(default=None, alias="scientificName")
    agency: str | None = None
    requested_document_family: DocumentFamilyLabel | None = Field(
        default=None,
        alias="requestedDocumentFamily",
    )
    subject_terms: list[str] = Field(default_factory=list, alias="subjectTerms")
    confidence: SubjectCardConfidence = "deterministic_fallback"
    source: Literal["planner_llm", "deterministic_fallback", "hybrid"] = "deterministic_fallback"


class AgentHints(ApiModel):
    """Agent-facing next-step cues included on read-tool responses."""

    next_tool_candidates: list[str] = Field(
        default_factory=list,
        alias="nextToolCandidates",
    )
    why_this_next_step: str = Field(
        default="",
        alias="whyThisNextStep",
    )
    safe_retry: str = Field(default="", alias="safeRetry")
    warnings: list[str] = Field(default_factory=list)


class Clarification(ApiModel):
    """Bounded clarification cue for clients that cannot do elicitation."""

    reason: str
    question: str
    options: list[str] = Field(default_factory=list)
    can_proceed_without_answer: bool = Field(
        default=True,
        alias="canProceedWithoutAnswer",
    )


class StructuredToolError(ApiModel):
    """Structured additive error response for disabled or expired smart flows."""

    error: str
    message: str
    fallback_tools: list[str] = Field(
        default_factory=list,
        alias="fallbackTools",
    )
    agent_hints: AgentHints | None = Field(
        default=None,
        alias="agentHints",
    )


class IntentCandidate(ApiModel):
    """One plausible intent considered during planning."""

    intent: IntentLabel
    confidence: Literal["high", "medium", "low"] = "low"
    source: Literal["explicit", "planner", "heuristic", "hybrid", "fallback"] = "planner"
    rationale: str = ""


class SearchStrategyMetadata(ApiModel):
    """Transparent search-planning metadata surfaced to external agents."""

    intent: IntentLabel = "discovery"
    intent_source: Literal[
        "explicit",
        "planner",
        "heuristic_override",
        "hybrid_agreement",
        "fallback_recovery",
    ] = Field(default="planner", alias="intentSource")
    intent_confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        alias="intentConfidence",
    )
    intent_candidates: list[IntentCandidate] = Field(
        default_factory=list,
        alias="intentCandidates",
    )
    secondary_intents: list[IntentLabel] = Field(
        default_factory=list,
        alias="secondaryIntents",
    )
    routing_confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        alias="routingConfidence",
    )
    query_specificity: Literal["high", "medium", "low"] = Field(
        default="medium",
        alias="querySpecificity",
    )
    ambiguity_level: Literal["low", "medium", "high"] = Field(
        default="low",
        alias="ambiguityLevel",
    )
    query_type: PlannerQueryType = Field(
        default="broad_concept",
        alias="queryType",
    )
    regulatory_subintent: str | None = Field(default=None, alias="regulatorySubintent")
    regulatory_intent: RegulatoryIntentLabel | None = Field(
        default=None,
        alias="regulatoryIntent",
        description=(
            "Canonical regulatory-intent split (current_cfr_text, rulemaking_history, "
            "species_dossier, guidance_lookup, hybrid_regulatory_plus_literature, "
            "unspecified). LLM-first with deterministic fallback."
        ),
    )
    entity_card: dict[str, Any] | None = Field(default=None, alias="entityCard")
    subject_card: SubjectCard | None = Field(default=None, alias="subjectCard")
    subject_chain_gaps: list[str] = Field(default_factory=list, alias="subjectChainGaps")
    intent_family: str | None = Field(default=None, alias="intentFamily")
    breadth_estimate: int = Field(
        default=2,
        ge=1,
        le=4,
        alias="breadthEstimate",
    )
    first_pass_mode: PlannerFirstPassMode = Field(
        default="targeted",
        alias="firstPassMode",
    )
    intent_rationale: str = Field(
        default="",
        alias="intentRationale",
    )
    latency_profile: Literal["fast", "balanced", "deep"] = Field(
        default="deep",
        alias="latencyProfile",
    )
    normalized_query: str = Field(
        default="",
        alias="normalizedQuery",
    )
    query_variants_tried: list[str] = Field(
        default_factory=list,
        alias="queryVariantsTried",
    )
    retrieval_hypotheses: list[str] = Field(
        default_factory=list,
        alias="retrievalHypotheses",
    )
    search_angles: list[str] = Field(
        default_factory=list,
        alias="searchAngles",
    )
    uncertainty_flags: list[str] = Field(
        default_factory=list,
        alias="uncertaintyFlags",
    )
    accepted_expansions: list[str] = Field(
        default_factory=list,
        alias="acceptedExpansions",
    )
    rejected_expansions: list[str] = Field(
        default_factory=list,
        alias="rejectedExpansions",
    )
    speculative_expansions: list[str] = Field(
        default_factory=list,
        alias="speculativeExpansions",
    )
    providers_used: list[str] = Field(
        default_factory=list,
        alias="providersUsed",
    )
    paid_providers_used: list[str] = Field(
        default_factory=list,
        alias="paidProvidersUsed",
    )
    configured_smart_provider: str = Field(
        default="deterministic",
        alias="configuredSmartProvider",
    )
    active_smart_provider: str = Field(
        default="deterministic",
        alias="activeSmartProvider",
    )
    planner_model: str = Field(
        default="",
        alias="plannerModel",
    )
    planner_model_source: str = Field(
        default="configured",
        alias="plannerModelSource",
    )
    synthesis_model: str = Field(
        default="",
        alias="synthesisModel",
    )
    synthesis_model_source: str = Field(
        default="configured",
        alias="synthesisModelSource",
    )
    result_coverage: str = Field(
        default="narrow",
        alias="resultCoverage",
    )
    drift_warnings: list[str] = Field(
        default_factory=list,
        alias="driftWarnings",
    )
    recovery_attempted: bool = Field(
        default=False,
        alias="recoveryAttempted",
    )
    recovery_path: list[str] = Field(
        default_factory=list,
        alias="recoveryPath",
    )
    recovery_reason: str | None = Field(
        default=None,
        alias="recoveryReason",
    )
    stopped_recovery_because: str | None = Field(
        default=None,
        alias="stoppedRecoveryBecause",
    )
    anchor_type: str | None = Field(
        default=None,
        alias="anchorType",
    )
    anchor_strength: Literal["high", "medium", "low"] | None = Field(
        default=None,
        alias="anchorStrength",
    )
    anchored_subject: str | None = Field(
        default=None,
        alias="anchoredSubject",
    )
    known_item_resolution_state: KnownItemResolutionState | None = Field(
        default=None,
        alias="knownItemResolutionState",
        description=(
            "Execution-provenance label for known-item / resolve_reference outcomes. "
            "One of resolved_exact, resolved_probable, needs_disambiguation."
        ),
    )
    normalization_warnings: list[str] = Field(
        default_factory=list,
        alias="normalizationWarnings",
    )
    repaired_inputs: dict[str, Any] = Field(
        default_factory=dict,
        alias="repairedInputs",
    )
    provider_budget_applied: dict[str, Any] = Field(
        default_factory=dict,
        alias="providerBudgetApplied",
    )
    provider_outcomes: list[dict[str, Any]] = Field(
        default_factory=list,
        alias="providerOutcomes",
    )
    stage_timings_ms: dict[str, int] = Field(
        default_factory=dict,
        alias="stageTimingsMs",
    )
    best_next_internal_action: str | None = Field(
        default=None,
        alias="bestNextInternalAction",
    )
    ranking_diagnostics: list[dict[str, Any]] = Field(
        default_factory=list,
        alias="rankingDiagnostics",
    )


class StructuredSourceRecord(ApiModel):
    """Trust-graded source record for smart responses."""

    source_id: str | None = Field(default=None, alias="sourceId")
    source_alias: str | None = Field(default=None, alias="sourceAlias")
    title: str | None = None
    provider: str | None = None
    source_type: str | None = Field(default=None, alias="sourceType")
    verification_status: str | None = Field(default=None, alias="verificationStatus")
    access_status: str | None = Field(default=None, alias="accessStatus")
    topical_relevance: Literal["on_topic", "weak_match", "off_topic"] | None = Field(
        default=None,
        alias="topicalRelevance",
    )
    confidence: str | None = None
    is_primary_source: bool | None = Field(default=None, alias="isPrimarySource")
    canonical_url: str | None = Field(default=None, alias="canonicalUrl")
    retrieved_url: str | None = Field(default=None, alias="retrievedUrl")
    full_text_url_found: bool | None = Field(
        default=None,
        alias="fullTextUrlFound",
        validation_alias=AliasChoices("fullTextUrlFound", "fullTextObserved"),
    )
    full_text_retrieved: bool | None = Field(default=None, alias="fullTextRetrieved")
    abstract_observed: bool | None = Field(default=None, alias="abstractObserved")
    open_access_route: OpenAccessRoute | None = Field(default=None, alias="openAccessRoute")
    citation_text: str | None = Field(default=None, alias="citationText")
    citation: CitationRecord | None = None
    date: str | None = None
    note: str | None = None
    why_classified_as_weak_match: str | None = Field(default=None, alias="whyClassifiedAsWeakMatch")
    lead_reason: str | None = Field(default=None, alias="leadReason")
    why_not_verified: str | None = Field(default=None, alias="whyNotVerified")
    relevance_source: Literal["llm", "llm_retry", "deterministic_tier", "hybrid"] | None = Field(
        default=None,
        alias="relevanceSource",
    )
    relevance_confidence: float | None = Field(default=None, alias="relevanceConfidence")
    relevance_reason: str | None = Field(default=None, alias="relevanceReason")
    classification_rationale: str | None = Field(default=None, alias="classificationRationale")
    document_family_match: str | None = Field(default=None, alias="documentFamilyMatch")
    document_family_boost: float | None = Field(default=None, alias="documentFamilyBoost")


class ScoreBreakdown(ApiModel):
    """Scoring signals behind a smart-ranked hit."""

    fused_rank_score: float = Field(
        default=0.0,
        alias="fusedRankScore",
    )
    query_similarity: float = Field(
        default=0.0,
        alias="querySimilarity",
    )
    concept_coverage_bonus: float = Field(
        default=0.0,
        alias="conceptCoverageBonus",
    )
    provider_consensus_bonus: float = Field(
        default=0.0,
        alias="providerConsensusBonus",
    )
    query_facet_coverage: float = Field(
        default=0.0,
        alias="queryFacetCoverage",
    )
    query_term_coverage: float = Field(
        default=0.0,
        alias="queryTermCoverage",
    )
    query_anchor_coverage: float = Field(
        default=0.0,
        alias="queryAnchorCoverage",
    )
    title_facet_coverage: float = Field(
        default=0.0,
        alias="titleFacetCoverage",
    )
    title_term_coverage: float = Field(
        default=0.0,
        alias="titleTermCoverage",
    )
    title_anchor_coverage: float = Field(
        default=0.0,
        alias="titleAnchorCoverage",
    )
    citation_recency_prior: float = Field(
        default=0.0,
        alias="citationRecencyPrior",
    )
    drift_penalty: float = Field(default=0.0, alias="driftPenalty")
    query_facet_penalty: float = Field(default=0.0, alias="queryFacetPenalty")
    final_score: float = Field(default=0.0, alias="finalScore")


class SmartPaperHit(ApiModel):
    """One smart-ranked paper hit that wraps the existing normalized Paper shape."""

    paper: Paper
    rank: int
    why_matched: str = Field(default="", alias="whyMatched")
    matched_concepts: list[str] = Field(
        default_factory=list,
        alias="matchedConcepts",
    )
    retrieved_by: list[str] = Field(
        default_factory=list,
        alias="retrievedBy",
    )
    topical_relevance: Literal["on_topic", "weak_match", "off_topic"] | None = Field(
        default=None,
        alias="topicalRelevance",
    )
    relevance_source: Literal["llm", "llm_retry", "deterministic_tier", "hybrid"] | None = Field(
        default=None,
        alias="relevanceSource",
    )
    relevance_confidence: float | None = Field(default=None, alias="relevanceConfidence")
    relevance_reason: str | None = Field(default=None, alias="relevanceReason")
    classification_rationale: str | None = Field(default=None, alias="classificationRationale")
    score_breakdown: ScoreBreakdown = Field(
        default_factory=ScoreBreakdown,
        alias="scoreBreakdown",
    )


class SmartSearchResponse(ApiModel):
    """Result payload for search_papers_smart."""

    results: list[SmartPaperHit] = Field(default_factory=list)
    search_session_id: str = Field(alias="searchSessionId")
    strategy_metadata: SearchStrategyMetadata = Field(
        alias="strategyMetadata",
    )
    next_step_hint: str = Field(alias="nextStepHint")
    agent_hints: AgentHints = Field(alias="agentHints")
    resource_uris: list[str] = Field(
        default_factory=list,
        alias="resourceUris",
    )
    verified_findings: list[str] = Field(default_factory=list, alias="verifiedFindings")
    likely_unverified: list[str] = Field(default_factory=list, alias="likelyUnverified")
    answerability: Literal["grounded", "limited", "insufficient"] = "limited"
    routing_summary: dict[str, Any] | None = Field(default=None, alias="routingSummary")
    evidence: list[StructuredSourceRecord] = Field(default_factory=list)
    leads: list[StructuredSourceRecord] = Field(default_factory=list)
    candidate_leads: list[StructuredSourceRecord] = Field(default_factory=list, alias="candidateLeads")
    evidence_gaps: list[str] = Field(default_factory=list, alias="evidenceGaps")
    structured_sources: list[StructuredSourceRecord] = Field(default_factory=list, alias="structuredSources")
    coverage_summary: CoverageSummary | None = Field(default=None, alias="coverageSummary")
    failure_summary: FailureSummary | None = Field(default=None, alias="failureSummary")
    regulatory_timeline: RegulatoryTimeline | None = Field(default=None, alias="regulatoryTimeline")
    clarification: Clarification | None = None
    result_status: str = Field(default="succeeded", alias="resultStatus")
    has_inspectable_sources: bool = Field(default=False, alias="hasInspectableSources")
    best_next_internal_action: str = Field(default="follow_up_research", alias="bestNextInternalAction")


class EvidenceItem(ApiModel):
    """Evidence citation attached to grounded answers."""

    evidence_id: str | None = Field(default=None, alias="evidenceId")
    paper: Paper
    excerpt: str = ""
    why_relevant: str = Field(default="", alias="whyRelevant")
    relevance_score: float = Field(default=0.0, alias="relevanceScore")


class AskResultSetResponse(ApiModel):
    """Grounded answer over a saved result set."""

    answer: str | None = None
    answer_status: Literal["answered", "abstained", "insufficient_evidence"] = Field(
        default="answered",
        alias="answerStatus",
    )
    evidence: list[EvidenceItem] = Field(default_factory=list)
    unsupported_asks: list[str] = Field(
        default_factory=list,
        alias="unsupportedAsks",
    )
    follow_up_questions: list[str] = Field(
        default_factory=list,
        alias="followUpQuestions",
    )
    answerability: Literal["grounded", "limited", "insufficient"] = "limited"
    selected_evidence_ids: list[str] = Field(default_factory=list, alias="selectedEvidenceIds")
    selected_lead_ids: list[str] = Field(default_factory=list, alias="selectedLeadIds")
    confidence: Literal["high", "medium", "low"] = "medium"
    search_session_id: str = Field(alias="searchSessionId")
    agent_hints: AgentHints = Field(alias="agentHints")
    resource_uris: list[str] = Field(
        default_factory=list,
        alias="resourceUris",
    )
    verified_findings: list[str] = Field(default_factory=list, alias="verifiedFindings")
    likely_unverified: list[str] = Field(default_factory=list, alias="likelyUnverified")
    candidate_leads: list[StructuredSourceRecord] = Field(default_factory=list, alias="candidateLeads")
    evidence_gaps: list[str] = Field(default_factory=list, alias="evidenceGaps")
    structured_sources: list[StructuredSourceRecord] = Field(default_factory=list, alias="structuredSources")
    coverage_summary: CoverageSummary | None = Field(default=None, alias="coverageSummary")
    failure_summary: FailureSummary | None = Field(default=None, alias="failureSummary")
    provider_used: str = Field(default="deterministic", alias="providerUsed")
    degradation_reason: str | None = Field(default=None, alias="degradationReason")
    evidence_use_plan: dict[str, Any] | None = Field(default=None, alias="evidenceUsePlan")
    classification_provenance: dict[str, Any] | None = Field(
        default=None,
        alias="classificationProvenance",
    )
    degraded_classification: bool | None = Field(default=None, alias="degradedClassification")


class LandscapeTheme(ApiModel):
    """A thematic cluster within a saved result set."""

    title: str
    summary: str
    representative_papers: list[Paper] = Field(
        default_factory=list,
        alias="representativePapers",
    )
    matched_concepts: list[str] = Field(
        default_factory=list,
        alias="matchedConcepts",
    )


class LandscapeResponse(ApiModel):
    """Theme map for a saved result set."""

    themes: list[LandscapeTheme] = Field(default_factory=list)
    representative_papers: list[Paper] = Field(
        default_factory=list,
        alias="representativePapers",
    )
    gaps: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    suggested_next_searches: list[str] = Field(
        default_factory=list,
        alias="suggestedNextSearches",
    )
    search_session_id: str = Field(alias="searchSessionId")
    agent_hints: AgentHints = Field(alias="agentHints")
    resource_uris: list[str] = Field(
        default_factory=list,
        alias="resourceUris",
    )
    verified_findings: list[str] = Field(default_factory=list, alias="verifiedFindings")
    likely_unverified: list[str] = Field(default_factory=list, alias="likelyUnverified")
    candidate_leads: list[StructuredSourceRecord] = Field(default_factory=list, alias="candidateLeads")
    evidence_gaps: list[str] = Field(default_factory=list, alias="evidenceGaps")
    structured_sources: list[StructuredSourceRecord] = Field(default_factory=list, alias="structuredSources")
    coverage_summary: CoverageSummary | None = Field(default=None, alias="coverageSummary")
    failure_summary: FailureSummary | None = Field(default=None, alias="failureSummary")


class GraphNode(ApiModel):
    """Node in the additive research graph."""

    id: str
    kind: Literal["paper", "author"]
    label: str
    score: float = 0.0
    attributes: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(ApiModel):
    """Directed graph edge for citations, references, or authorship."""

    source: str
    target: str
    relation: Literal["cites", "references", "authored_by", "wrote"]
    weight: float = 1.0


class ResearchGraphResponse(ApiModel):
    """Response from expand_research_graph."""

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    frontier: list[GraphNode] = Field(default_factory=list)
    next_step_hint: str = Field(alias="nextStepHint")
    search_session_id: str | None = Field(
        default=None,
        alias="searchSessionId",
    )
    agent_hints: AgentHints = Field(alias="agentHints")
    resource_uris: list[str] = Field(
        default_factory=list,
        alias="resourceUris",
    )


class PlannerDecision(ApiModel):
    """Planner output for smart search orchestration."""

    intent: IntentLabel = "discovery"
    intent_source: Literal[
        "explicit",
        "planner",
        "heuristic_override",
        "hybrid_agreement",
        "fallback_recovery",
    ] = Field(default="planner", alias="intentSource")
    intent_confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        alias="intentConfidence",
    )
    intent_candidates: list[IntentCandidate] = Field(
        default_factory=list,
        alias="intentCandidates",
    )
    secondary_intents: list[IntentLabel] = Field(
        default_factory=list,
        alias="secondaryIntents",
    )
    routing_confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        alias="routingConfidence",
    )
    query_specificity: Literal["high", "medium", "low"] = Field(
        default="medium",
        alias="querySpecificity",
    )
    ambiguity_level: Literal["low", "medium", "high"] = Field(
        default="low",
        alias="ambiguityLevel",
    )
    query_type: PlannerQueryType = Field(
        default="broad_concept",
        alias="queryType",
    )
    regulatory_subintent: str | None = Field(default=None, alias="regulatorySubintent")
    regulatory_intent: RegulatoryIntentLabel | None = Field(
        default=None,
        alias="regulatoryIntent",
        description=(
            "Canonical regulatory-intent split surfaced to strategy metadata. "
            "LLM-first from planner signals with deterministic fallback."
        ),
    )
    entity_card: dict[str, Any] | None = Field(default=None, alias="entityCard")
    subject_card: SubjectCard | None = Field(default=None, alias="subjectCard")
    subject_chain_gaps: list[str] = Field(default_factory=list, alias="subjectChainGaps")
    intent_family: str | None = Field(default=None, alias="intentFamily")
    breadth_estimate: int = Field(
        default=2,
        ge=1,
        le=4,
        alias="breadthEstimate",
    )
    first_pass_mode: PlannerFirstPassMode = Field(
        default="targeted",
        alias="firstPassMode",
    )
    intent_rationale: str = Field(
        default="",
        alias="intentRationale",
    )
    constraints: dict[str, str] = Field(default_factory=dict)
    seed_identifiers: list[str] = Field(
        default_factory=list,
        alias="seedIdentifiers",
    )
    candidate_concepts: list[str] = Field(
        default_factory=list,
        alias="candidateConcepts",
    )
    provider_plan: list[str] = Field(
        default_factory=list,
        alias="providerPlan",
    )
    authority_first: bool = Field(
        default=True,
        alias="authorityFirst",
    )
    anchor_type: str | None = Field(
        default=None,
        alias="anchorType",
    )
    anchor_value: str | None = Field(
        default=None,
        alias="anchorValue",
    )
    required_primary_sources: list[str] = Field(
        default_factory=list,
        alias="requiredPrimarySources",
    )
    success_criteria: list[str] = Field(
        default_factory=list,
        alias="successCriteria",
    )
    search_angles: list[str] = Field(
        default_factory=list,
        alias="searchAngles",
    )
    retrieval_hypotheses: list[str] = Field(
        default_factory=list,
        alias="retrievalHypotheses",
    )
    uncertainty_flags: list[str] = Field(
        default_factory=list,
        alias="uncertaintyFlags",
    )
    follow_up_mode: Literal["qa", "claim_check", "comparison"] = Field(
        default="qa",
        alias="followUpMode",
    )


class ExpansionCandidate(ApiModel):
    """One candidate query expansion."""

    variant: str
    source: Literal["from_input", "from_retrieved_evidence", "speculative", "hypothesis"]
    rationale: str = ""
    provider_plan: list[str] = Field(default_factory=list, alias="providerPlan")
