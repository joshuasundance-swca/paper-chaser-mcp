"""Typed helpers for the guided evidence-first response contract."""

from __future__ import annotations

import re
from typing import Any, Literal, cast

from pydantic import Field

from .models.common import ApiModel

_VERIFIED_STATUSES = {"verified_primary_source", "verified_metadata"}
_STOPWORD_SOURCE_REFS = {"a", "an", "id", "is", "it", "of", "the", "this"}


class RegulatoryAnchor(ApiModel):
    """Normalized regulatory routing anchor for guided responses."""

    anchor_type: str | None = Field(default=None, alias="anchorType")
    anchor_value: str | None = Field(default=None, alias="anchorValue")
    required_primary_sources: list[str] = Field(default_factory=list, alias="requiredPrimarySources")
    success_criteria: list[str] = Field(default_factory=list, alias="successCriteria")
    subject_terms: list[str] = Field(default_factory=list, alias="subjectTerms")


class ProviderPlan(ApiModel):
    """Provider routing plan surfaced to guided callers."""

    providers: list[str] = Field(default_factory=list)
    authority_first: bool = Field(default=True, alias="authorityFirst")
    rationale: str = ""


class RoutingDecision(ApiModel):
    """Guided routing summary built from smart strategy metadata."""

    intent: str = "discovery"
    confidence: Literal["high", "medium", "low"] = "medium"
    query_specificity: Literal["high", "medium", "low"] = Field(
        default="medium",
        alias="querySpecificity",
    )
    ambiguity_level: Literal["low", "medium", "high"] = Field(
        default="low",
        alias="ambiguityLevel",
    )
    secondary_intents: list[str] = Field(default_factory=list, alias="secondaryIntents")
    retrieval_hypotheses: list[str] = Field(default_factory=list, alias="retrievalHypotheses")
    rationale: str = ""
    anchor: RegulatoryAnchor | None = None
    provider_plan: ProviderPlan = Field(default_factory=ProviderPlan, alias="providerPlan")


class GuidedEvidenceRecord(ApiModel):
    """Evidence or lead item for the guided response contract."""

    evidence_id: str = Field(alias="evidenceId")
    source_alias: str | None = Field(default=None, alias="sourceAlias")
    title: str | None = None
    provider: str | None = None
    source_type: str | None = Field(default=None, alias="sourceType")
    is_primary_source: bool | None = Field(default=None, alias="isPrimarySource")
    verification_status: str | None = Field(default=None, alias="verificationStatus")
    access_status: str | None = Field(default=None, alias="accessStatus")
    topical_relevance: str | None = Field(default=None, alias="topicalRelevance")
    canonical_url: str | None = Field(default=None, alias="canonicalUrl")
    retrieved_url: str | None = Field(default=None, alias="retrievedUrl")
    citation: dict[str, Any] | None = None
    date: str | None = None
    why_included: str = Field(default="", alias="whyIncluded")
    why_not_verified: str | None = Field(default=None, alias="whyNotVerified")


class EvidenceDecision(ApiModel):
    """Typed inclusion decision for one candidate evidence record."""

    evidence_id: str = Field(alias="evidenceId")
    include_as: Literal["evidence", "lead", "excluded"] = Field(alias="includeAs")
    why_included: str = Field(default="", alias="whyIncluded")
    why_not_verified: str | None = Field(default=None, alias="whyNotVerified")


class FollowUpDecision(ApiModel):
    """Evidence-selection decision for a guided follow-up."""

    answer_from_session: bool = Field(default=False, alias="answerFromSession")
    selected_evidence_ids: list[str] = Field(default_factory=list, alias="selectedEvidenceIds")
    selected_lead_ids: list[str] = Field(default_factory=list, alias="selectedLeadIds")
    unsupported_asks: list[str] = Field(default_factory=list, alias="unsupportedAsks")
    rationale: str = ""


class GuidedAnswer(ApiModel):
    """Top-level answerability contract for guided responses."""

    result_status: str = Field(default="partial", alias="resultStatus")
    answerability: Literal["grounded", "limited", "insufficient"] = "insufficient"
    summary: str = ""
    evidence_gaps: list[str] = Field(default_factory=list, alias="evidenceGaps")
    next_actions: list[str] = Field(default_factory=list, alias="nextActions")


def explicit_source_reference(question: str) -> str | None:
    """Extract only explicit source references, not incidental words after 'source'."""

    normalized_question = " ".join(str(question or "").split())
    if not normalized_question:
        return None
    patterns = (
        re.compile(r"\bsource\s*id\s*[:=]?\s*['\"]?(?P<id>[A-Za-z0-9._:/#-]{2,})['\"]?", re.IGNORECASE),
        re.compile(r"\bevidence\s*id\s*[:=]?\s*['\"]?(?P<id>[A-Za-z0-9._:/#-]{2,})['\"]?", re.IGNORECASE),
        re.compile(r"\blead\s*id\s*[:=]?\s*['\"]?(?P<id>[A-Za-z0-9._:/#-]{2,})['\"]?", re.IGNORECASE),
        re.compile(r"\bsource\s*[:=]\s*['\"]?(?P<id>[A-Za-z0-9._:/#-]{2,})['\"]?", re.IGNORECASE),
        re.compile(r"\binspect\s+source\s+['\"]?(?P<id>[A-Za-z0-9._:/#-]{2,})['\"]?", re.IGNORECASE),
    )
    for pattern in patterns:
        match = pattern.search(normalized_question)
        if not match:
            continue
        candidate = " ".join(match.group("id").split())
        if candidate and candidate.lower() not in _STOPWORD_SOURCE_REFS:
            return candidate
    return None


def build_routing_decision(
    *,
    query: str,
    intent: str,
    strategy_metadata: dict[str, Any] | None,
    coverage_summary: dict[str, Any] | None,
) -> RoutingDecision:
    """Build a typed routing decision from smart runtime metadata."""

    metadata = strategy_metadata or {}
    anchor_type = str(metadata.get("anchorType") or "").strip() or None
    anchor_value = str(metadata.get("anchoredSubject") or "").strip() or None
    provider_plan = list(metadata.get("providerPlan") or [])
    attempted = list((coverage_summary or {}).get("providersAttempted") or [])
    providers = [provider for provider in provider_plan if provider] or [provider for provider in attempted if provider]
    if not providers:
        providers = _default_provider_plan_for_anchor(anchor_type, intent)
    confidence_value = str(metadata.get("routingConfidence") or metadata.get("intentConfidence") or "medium")
    if confidence_value not in {"high", "medium", "low"}:
        confidence_value = "medium"
    confidence = cast(Literal["high", "medium", "low"], confidence_value)
    query_specificity_value = str(metadata.get("querySpecificity") or "medium")
    if query_specificity_value not in {"high", "medium", "low"}:
        query_specificity_value = "medium"
    ambiguity_level_value = str(metadata.get("ambiguityLevel") or "low")
    if ambiguity_level_value not in {"low", "medium", "high"}:
        ambiguity_level_value = "low"
    rationale = str(metadata.get("intentRationale") or "").strip()
    if not rationale:
        rationale = "Guided routing reused the smart runtime strategy metadata."
    secondary_intents = [str(item).strip() for item in metadata.get("secondaryIntents") or [] if str(item).strip()]
    retrieval_hypotheses = [
        str(item).strip() for item in metadata.get("retrievalHypotheses") or [] if str(item).strip()
    ]
    anchor = None
    if anchor_type or anchor_value or intent == "regulatory":
        anchor = RegulatoryAnchor(
            anchorType=anchor_type,
            anchorValue=anchor_value or _fallback_anchor_value(query=query, anchor_type=anchor_type),
            requiredPrimarySources=_required_primary_sources(anchor_type),
            successCriteria=_success_criteria(anchor_type),
            subjectTerms=[anchor_value] if anchor_value else [],
        )
    return RoutingDecision(
        intent=intent,
        confidence=confidence,
        querySpecificity=cast(Literal["high", "medium", "low"], query_specificity_value),
        ambiguityLevel=cast(Literal["low", "medium", "high"], ambiguity_level_value),
        secondaryIntents=secondary_intents[:4],
        retrievalHypotheses=retrieval_hypotheses[:5],
        rationale=rationale,
        anchor=anchor,
        providerPlan=ProviderPlan(
            providers=providers,
            authorityFirst=True,
            rationale="Authority-first routing is preferred for guided trust-sensitive retrieval.",
        ),
    )


def build_evidence_records(
    *,
    sources: list[dict[str, Any]],
    leads: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert guided sources into evidence and lead records."""

    evidence_items: list[dict[str, Any]] = []
    lead_items: list[dict[str, Any]] = []
    seen_lead_ids: set[str] = set()

    for source in sources:
        decision = classify_source(source)
        record = _guided_evidence_record_from_source(source, decision=decision).model_dump(by_alias=True)
        if decision.include_as == "evidence":
            evidence_items.append(record)
        elif decision.include_as == "lead":
            evidence_id = str(record.get("evidenceId") or "").strip()
            if evidence_id and evidence_id not in seen_lead_ids:
                seen_lead_ids.add(evidence_id)
                lead_items.append(record)

    for lead in leads:
        decision = classify_source(lead)
        if decision.include_as == "excluded":
            continue
        record = _guided_evidence_record_from_source(
            lead,
            decision=EvidenceDecision(
                evidenceId=str(lead.get("sourceId") or lead.get("sourceAlias") or "lead"),
                includeAs="lead",
                whyIncluded=decision.why_included or "Retained as background or unresolved lead.",
                whyNotVerified=decision.why_not_verified,
            ),
        ).model_dump(by_alias=True)
        evidence_id = str(record.get("evidenceId") or "").strip()
        if (
            evidence_id
            and evidence_id not in seen_lead_ids
            and not any(evidence_id == str(item.get("evidenceId") or "") for item in evidence_items)
        ):
            seen_lead_ids.add(evidence_id)
            lead_items.append(record)

    return evidence_items, lead_items


def classify_source(source: dict[str, Any]) -> EvidenceDecision:
    """Classify one guided source as evidence, lead, or excluded."""

    evidence_id = str(source.get("sourceId") or source.get("sourceAlias") or "source").strip()
    topical_relevance = str(source.get("topicalRelevance") or "weak_match")
    verification_status = str(source.get("verificationStatus") or "unverified")

    if topical_relevance == "on_topic" and verification_status in _VERIFIED_STATUSES:
        return EvidenceDecision(
            evidenceId=evidence_id,
            includeAs="evidence",
            whyIncluded="On-topic source retained to support the grounded guided answer.",
            whyNotVerified=None,
        )

    why_not_verified = None
    if verification_status not in _VERIFIED_STATUSES:
        why_not_verified = f"Verification status was {verification_status or 'unverified'}."
    elif topical_relevance != "on_topic":
        why_not_verified = f"Topical relevance was {topical_relevance or 'unknown'}."

    return EvidenceDecision(
        evidenceId=evidence_id,
        includeAs="lead",
        whyIncluded="Retained as related context or unresolved supporting material.",
        whyNotVerified=why_not_verified,
    )


def classify_answerability(
    *,
    status: str,
    evidence: list[dict[str, Any]],
    leads: list[dict[str, Any]],
    evidence_gaps: list[str],
) -> str:
    """Map source state into the simplified answerability ladder."""

    if status in {"succeeded", "answered"} and evidence:
        return "grounded"
    if evidence or leads or evidence_gaps:
        return "limited"
    return "insufficient"


def build_follow_up_decision(
    *,
    question: str,
    session_state: dict[str, Any],
    facets: set[str],
) -> FollowUpDecision:
    """Select evidence ids for guided follow-up introspection without token matching."""

    sources = [source for source in session_state.get("sources") or [] if isinstance(source, dict)]
    leads = [lead for lead in session_state.get("unverifiedLeads") or [] if isinstance(lead, dict)]
    explicit_reference = explicit_source_reference(question)
    selected_evidence_ids: list[str] = []
    selected_lead_ids: list[str] = []

    if explicit_reference:
        lowered = explicit_reference.lower()
        for source in sources:
            source_id = str(source.get("sourceId") or "").strip()
            source_alias = str(source.get("sourceAlias") or "").strip()
            if lowered in {source_id.lower(), source_alias.lower()}:
                selected_evidence_ids.append(source_id or source_alias)
                break
        if not selected_evidence_ids:
            for lead in leads:
                lead_id = str(lead.get("sourceId") or "").strip()
                lead_alias = str(lead.get("sourceAlias") or "").strip()
                if lowered in {lead_id.lower(), lead_alias.lower()}:
                    selected_lead_ids.append(lead_id or lead_alias)
                    break

    metadata_question = any(
        marker in question.lower() for marker in ("author", "authors", "venue", "journal", "publisher", "doi", "year")
    )
    if not selected_evidence_ids and metadata_question and len(sources) == 1:
        selected_evidence_ids.append(str(sources[0].get("sourceId") or sources[0].get("sourceAlias") or ""))

    if not selected_evidence_ids and not selected_lead_ids and "source_overview" in facets:
        selected_evidence_ids.extend(
            [
                str(source.get("sourceId") or source.get("sourceAlias") or "").strip()
                for source in sources[:3]
                if str(source.get("sourceId") or source.get("sourceAlias") or "").strip()
            ]
        )
        selected_lead_ids.extend(
            [
                str(lead.get("sourceId") or lead.get("sourceAlias") or "").strip()
                for lead in leads[:3]
                if str(lead.get("sourceId") or lead.get("sourceAlias") or "").strip()
            ]
        )

    if not selected_evidence_ids and not selected_lead_ids and "relevance_triage" in facets:
        selected_evidence_ids.extend(
            [
                str(source.get("sourceId") or source.get("sourceAlias") or "").strip()
                for source in sources[:5]
                if str(source.get("sourceId") or source.get("sourceAlias") or "").strip()
            ]
        )
        selected_lead_ids.extend(
            [
                str(lead.get("sourceId") or lead.get("sourceAlias") or "").strip()
                for lead in leads[:5]
                if str(lead.get("sourceId") or lead.get("sourceAlias") or "").strip()
            ]
        )

    return FollowUpDecision(
        answerFromSession=bool(facets or selected_evidence_ids or selected_lead_ids),
        selectedEvidenceIds=selected_evidence_ids,
        selectedLeadIds=selected_lead_ids,
        unsupportedAsks=[],
        rationale="Guided follow-up selected saved evidence ids without substring-based title matching.",
    )


def _guided_evidence_record_from_source(
    source: dict[str, Any],
    *,
    decision: EvidenceDecision,
) -> GuidedEvidenceRecord:
    citation = source.get("citation")
    return GuidedEvidenceRecord(
        evidenceId=str(source.get("sourceId") or source.get("sourceAlias") or "source"),
        sourceAlias=source.get("sourceAlias"),
        title=source.get("title"),
        provider=source.get("provider"),
        sourceType=source.get("sourceType"),
        isPrimarySource=source.get("isPrimarySource"),
        verificationStatus=source.get("verificationStatus"),
        accessStatus=source.get("accessStatus"),
        topicalRelevance=source.get("topicalRelevance"),
        canonicalUrl=source.get("canonicalUrl"),
        retrievedUrl=source.get("retrievedUrl"),
        citation=citation if isinstance(citation, dict) else None,
        date=source.get("date"),
        whyIncluded=decision.why_included,
        whyNotVerified=decision.why_not_verified,
    )


def _default_provider_plan_for_anchor(anchor_type: str | None, intent: str) -> list[str]:
    if anchor_type == "cfr_citation":
        return ["govinfo", "federal_register", "ecos"]
    if anchor_type in {"fr_citation", "document_number"}:
        return ["federal_register", "govinfo"]
    if anchor_type in {"species_common_name", "species_scientific_name"}:
        return ["ecos", "federal_register", "govinfo"]
    if anchor_type == "agency_guidance_title":
        return ["govinfo", "federal_register"]
    if intent == "regulatory":
        return ["ecos", "federal_register", "govinfo"]
    return []


def _required_primary_sources(anchor_type: str | None) -> list[str]:
    if anchor_type == "cfr_citation":
        return ["govinfo"]
    if anchor_type in {"fr_citation", "document_number"}:
        return ["federal_register", "govinfo"]
    if anchor_type in {"species_common_name", "species_scientific_name"}:
        return ["ecos"]
    if anchor_type == "agency_guidance_title":
        return ["govinfo", "federal_register"]
    return []


def _success_criteria(anchor_type: str | None) -> list[str]:
    if anchor_type == "cfr_citation":
        return ["current_text_required"]
    if anchor_type in {"fr_citation", "document_number"}:
        return ["timeline_required"]
    if anchor_type in {"species_common_name", "species_scientific_name"}:
        return ["dossier_required"]
    if anchor_type == "agency_guidance_title":
        return ["guidance_doc_required"]
    return []


def _fallback_anchor_value(*, query: str, anchor_type: str | None) -> str | None:
    if anchor_type == "cfr_citation":
        match = re.search(r"\b\d+\s*(?:CFR|F\.?\s*R\.?)\s*\d+(?:\.\d+)?\b", query, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return None
