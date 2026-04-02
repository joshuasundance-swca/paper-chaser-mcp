"""Structured models for additive smart-tool workflows."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from ..models.common import ApiModel, CoverageSummary, FailureSummary, Paper
from ..models.regulations import RegulatoryTimeline


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


class SearchStrategyMetadata(ApiModel):
    """Transparent search-planning metadata surfaced to external agents."""

    intent: str = "discovery"
    latency_profile: Literal["fast", "balanced", "deep"] = Field(
        default="balanced",
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


class StructuredSourceRecord(ApiModel):
    """Trust-graded source record for smart responses."""

    title: str | None = None
    provider: str | None = None
    source_type: str | None = Field(default=None, alias="sourceType")
    verification_status: str | None = Field(default=None, alias="verificationStatus")
    access_status: str | None = Field(default=None, alias="accessStatus")
    confidence: str | None = None
    is_primary_source: bool | None = Field(default=None, alias="isPrimarySource")
    canonical_url: str | None = Field(default=None, alias="canonicalUrl")
    retrieved_url: str | None = Field(default=None, alias="retrievedUrl")
    full_text_observed: bool | None = Field(default=None, alias="fullTextObserved")
    abstract_observed: bool | None = Field(default=None, alias="abstractObserved")
    citation: str | None = None
    date: str | None = None
    note: str | None = None


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
    evidence_gaps: list[str] = Field(default_factory=list, alias="evidenceGaps")
    structured_sources: list[StructuredSourceRecord] = Field(default_factory=list, alias="structuredSources")
    coverage_summary: CoverageSummary | None = Field(default=None, alias="coverageSummary")
    failure_summary: FailureSummary | None = Field(default=None, alias="failureSummary")
    regulatory_timeline: RegulatoryTimeline | None = Field(default=None, alias="regulatoryTimeline")
    clarification: Clarification | None = None


class EvidenceItem(ApiModel):
    """Evidence citation attached to grounded answers."""

    paper: Paper
    excerpt: str = ""
    why_relevant: str = Field(default="", alias="whyRelevant")
    relevance_score: float = Field(default=0.0, alias="relevanceScore")


class AskResultSetResponse(ApiModel):
    """Grounded answer over a saved result set."""

    answer: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    unsupported_asks: list[str] = Field(
        default_factory=list,
        alias="unsupportedAsks",
    )
    follow_up_questions: list[str] = Field(
        default_factory=list,
        alias="followUpQuestions",
    )
    confidence: Literal["high", "medium", "low"] = "medium"
    search_session_id: str = Field(alias="searchSessionId")
    agent_hints: AgentHints = Field(alias="agentHints")
    resource_uris: list[str] = Field(
        default_factory=list,
        alias="resourceUris",
    )
    verified_findings: list[str] = Field(default_factory=list, alias="verifiedFindings")
    likely_unverified: list[str] = Field(default_factory=list, alias="likelyUnverified")
    evidence_gaps: list[str] = Field(default_factory=list, alias="evidenceGaps")
    structured_sources: list[StructuredSourceRecord] = Field(default_factory=list, alias="structuredSources")
    coverage_summary: CoverageSummary | None = Field(default=None, alias="coverageSummary")
    failure_summary: FailureSummary | None = Field(default=None, alias="failureSummary")


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

    intent: Literal[
        "discovery",
        "review",
        "known_item",
        "author",
        "citation",
        "regulatory",
    ] = "discovery"
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
    follow_up_mode: Literal["qa", "claim_check", "comparison"] = Field(
        default="qa",
        alias="followUpMode",
    )


class ExpansionCandidate(ApiModel):
    """One candidate query expansion."""

    variant: str
    source: Literal["from_input", "from_retrieved_evidence", "speculative"]
    rationale: str = ""
