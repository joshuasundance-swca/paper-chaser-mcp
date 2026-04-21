"""Guided trust/confidence/status helpers (Phase 3 extraction).

Extracted from :mod:`paper_chaser_mcp.dispatch._core`. This submodule owns
the helpers that turn a list of guided source records (plus failure and
coverage hints) into ``trustSummary`` / ``confidenceSignals`` /
``machineFailure`` / ``resultState`` payloads and compute the
``research_status`` verdict for the guided workflow.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from ...agentic.provider_helpers import generate_evidence_gaps_without_llm
from ...models.common import ConfidenceSignals, GuidedResultState, MachineFailure

# Forward references to helpers/constants still living in ``_core``. The
# imports below rely on ``_core``'s bottom-of-module re-export block pulling
# this submodule in *after* the top-level defs have executed.
from .._core import (  # noqa: E402 — forward refs; see note above
    _authoritative_but_weak_source_ids,
    _evidence_quality_detail,
    _synthesis_path,
    _trust_revision_narrative,
)
from .sources import (
    _guided_dedupe_source_records,
    _guided_merge_source_record_sets,
    _guided_merge_source_records,
    _guided_source_matches_reference,
    _guided_source_record_from_paper,
    _guided_source_record_from_structured_source,
    _guided_source_records_share_surface,
)

logger = logging.getLogger(__name__)


def _guided_trust_summary(
    sources: list[dict[str, Any]],
    evidence_gaps: list[str],
    *,
    classification_provenance: dict[str, Any] | None = None,
    subject_chain_gaps: list[str] | None = None,
) -> dict[str, Any]:
    # verifiedPrimarySourceCount now reflects the true primary-source bucket:
    # records marked ``isPrimarySource=True`` whose verification status is
    # either ``verified_primary_source`` or ``verified_metadata``. This fixes
    # the prior miscount that treated any ``verified_primary_source`` record
    # as primary regardless of the ``isPrimarySource`` flag (P0-3 item 2).
    verified_primary_source_count = sum(
        1
        for source in sources
        if source.get("isPrimarySource") is True
        and source.get("verificationStatus") in {"verified_primary_source", "verified_metadata"}
    )
    # Narrower full-text signal: isPrimarySource=True AND full-text verified.
    full_text_verified_primary_source_count = sum(
        1
        for source in sources
        if source.get("isPrimarySource") is True and source.get("verificationStatus") == "verified_primary_source"
    )
    # Retain the broader status-only counts for the combined verifiedSourceCount
    # total (keeps the overall denominator stable for downstream clients).
    status_verified_primary_source_count = sum(
        1 for source in sources if source.get("verificationStatus") == "verified_primary_source"
    )
    verified_metadata_source_count = sum(
        1 for source in sources if source.get("verificationStatus") == "verified_metadata"
    )
    on_topic_source_count = sum(1 for source in sources if source.get("topicalRelevance") == "on_topic")
    weak_match_reasons = [
        str(source.get("whyClassifiedAsWeakMatch") or source.get("note") or "").strip()
        for source in sources
        if source.get("topicalRelevance") == "weak_match"
        and str(source.get("whyClassifiedAsWeakMatch") or source.get("note") or "").strip()
    ]
    off_topic_reasons = [
        str(source.get("whyClassifiedAsWeakMatch") or source.get("note") or "").strip()
        for source in sources
        if source.get("topicalRelevance") == "off_topic"
        and str(source.get("whyClassifiedAsWeakMatch") or source.get("note") or "").strip()
    ]
    weak_match_rationales = [
        str(source.get("classificationRationale") or "").strip()
        for source in sources
        if source.get("topicalRelevance") == "weak_match" and str(source.get("classificationRationale") or "").strip()
    ]
    off_topic_rationales = [
        str(source.get("classificationRationale") or "").strip()
        for source in sources
        if source.get("topicalRelevance") == "off_topic" and str(source.get("classificationRationale") or "").strip()
    ]
    if verified_primary_source_count > 0 and on_topic_source_count > 0:
        strength_explanation = "Verified primary sources provide direct on-topic support."
    elif verified_metadata_source_count > 0 and on_topic_source_count > 0:
        strength_explanation = (
            "On-topic support is present, but some records remain metadata-verified rather than full-text verified."
        )
    elif weak_match_reasons:
        strength_explanation = (
            "The saved evidence is related, but the strongest remaining records are still scope-limited."
        )
    elif off_topic_reasons:
        strength_explanation = "Available sources are mostly off-topic for the saved query."
    else:
        strength_explanation = "No strong verified support was recorded for the saved query."
    authoritative_but_weak = _authoritative_but_weak_source_ids(sources)
    authoritative_but_weak_count = len(authoritative_but_weak)
    strong_on_topic_count = on_topic_source_count
    weak_match_bucket_count = sum(1 for source in sources if source.get("topicalRelevance") == "weak_match")
    off_topic_bucket_count = sum(1 for source in sources if source.get("topicalRelevance") == "off_topic")
    breakdown_fragments: list[str] = []
    if strong_on_topic_count:
        noun = "source" if strong_on_topic_count == 1 else "sources"
        breakdown_fragments.append(f"{strong_on_topic_count} strong on-topic verified {noun}")
    if authoritative_but_weak_count:
        noun = "source" if authoritative_but_weak_count == 1 else "sources"
        breakdown_fragments.append(f"{authoritative_but_weak_count} authoritative but weak-match {noun}")
        remaining_weak = max(weak_match_bucket_count - authoritative_but_weak_count, 0)
        if remaining_weak:
            noun = "lead" if remaining_weak == 1 else "leads"
            breakdown_fragments.append(f"{remaining_weak} other weak-match {noun}")
    elif weak_match_bucket_count:
        noun = "lead" if weak_match_bucket_count == 1 else "leads"
        breakdown_fragments.append(f"{weak_match_bucket_count} weak-match {noun}")
    if off_topic_bucket_count:
        noun = "lead" if off_topic_bucket_count == 1 else "leads"
        breakdown_fragments.append(f"{off_topic_bucket_count} off-target {noun}")
    if breakdown_fragments:
        strength_explanation = f"{strength_explanation} Breakdown: {', '.join(breakdown_fragments)}."
    summary = {
        "verifiedSourceCount": status_verified_primary_source_count + verified_metadata_source_count,
        "verifiedPrimarySourceCount": verified_primary_source_count,
        "fullTextVerifiedPrimarySourceCount": full_text_verified_primary_source_count,
        "verifiedMetadataSourceCount": verified_metadata_source_count,
        "onTopicSourceCount": on_topic_source_count,
        "weakMatchCount": sum(1 for source in sources if source.get("topicalRelevance") == "weak_match"),
        "offTopicCount": sum(1 for source in sources if source.get("topicalRelevance") == "off_topic"),
        "evidenceGapCount": len(evidence_gaps),
        "rationaleByBucket": {
            "weakMatch": weak_match_reasons[:3],
            "offTopic": off_topic_reasons[:3],
        },
        "classificationRationaleByBucket": {
            "weakMatch": weak_match_rationales[:3],
            "offTopic": off_topic_rationales[:3],
        },
        "authoritativeButWeak": authoritative_but_weak,
        "strengthExplanation": strength_explanation,
    }
    if authoritative_but_weak_count:
        # Missed-escalation prose note (P0-3 item 5): call out that these
        # authoritative records are NOT grounded evidence and subject-chain
        # grounding may be weak, so agents do not silently fold them in.
        noun = "source" if authoritative_but_weak_count == 1 else "sources"
        summary["authoritativeButWeakNote"] = (
            f"{authoritative_but_weak_count} authoritative {noun} found but not topically responsive "
            "(missed-escalation bucket); subject-chain grounding may be weak — do not treat as "
            "grounded evidence without a disambiguation or primary-source follow-up."
        )
    top_rationale: str | None = None
    if off_topic_rationales:
        top_rationale = f"Off-topic example: {off_topic_rationales[0]}"
    elif weak_match_rationales:
        top_rationale = f"Weak-match example: {weak_match_rationales[0]}"
    if top_rationale:
        summary["trustRationale"] = f"{strength_explanation} {top_rationale}".strip()[:280]
    else:
        summary["trustRationale"] = strength_explanation
    if classification_provenance and classification_provenance.get("total"):
        summary["classificationProvenance"] = classification_provenance
        summary["degradedClassification"] = bool(classification_provenance.get("degradedClassification"))
    # ws-dispatch-contract-trust (finding #5): surface planner subject-chain
    # gaps in machine-readable trust signals, not just the prose rationale
    # emitted by ``_compose_why_classified_weak_match``. Clients reading
    # ``trustSummary``/``confidenceSignals`` previously missed the same reason
    # shown in the human sentence.
    _subject_chain_gaps = [str(item).strip() for item in subject_chain_gaps or [] if str(item).strip()]
    if _subject_chain_gaps:
        summary["subjectChainGaps"] = _subject_chain_gaps
    return summary


def _guided_confidence_signals(
    *,
    status: str,
    sources: list[dict[str, Any]],
    evidence_gaps: list[str],
    degradation_reason: str | None = None,
    synthesis_mode: str | None = None,
    source: dict[str, Any] | None = None,
    evidence_use_plan_applied: bool = False,
    subject_chain_gaps: list[str] | None = None,
) -> dict[str, Any]:
    verified_on_topic_primary = sum(
        1
        for item in sources
        if item.get("topicalRelevance") == "on_topic" and item.get("verificationStatus") == "verified_primary_source"
    )
    verified_on_topic = sum(
        1
        for item in sources
        if item.get("topicalRelevance") == "on_topic"
        and item.get("verificationStatus") in {"verified_primary_source", "verified_metadata"}
    )
    if verified_on_topic_primary > 0 or verified_on_topic >= 3:
        evidence_quality_profile = "high"
    elif verified_on_topic > 0:
        evidence_quality_profile = "medium"
    else:
        evidence_quality_profile = "low"

    trust_revision_reason = degradation_reason or (evidence_gaps[0] if evidence_gaps else None)
    # ws-dispatch-contract-trust (finding #5): if no other revision reason
    # exists, surface the first subject-chain gap so the machine-readable
    # signal doesn't go empty when the only trust deficit is a planner gap.
    _subject_chain_gaps = [str(item).strip() for item in subject_chain_gaps or [] if str(item).strip()]
    if trust_revision_reason is None and _subject_chain_gaps:
        trust_revision_reason = _subject_chain_gaps[0]
    fallback_explanation = None
    if degradation_reason == "deterministic_synthesis_fallback":
        fallback_explanation = (
            "A deterministic fallback answered the follow-up because model-backed synthesis was unavailable."
        )
    if synthesis_mode is None:
        if status in {"answered", "succeeded"} and degradation_reason is None:
            synthesis_mode = "grounded"
        elif status in {"answered", "succeeded", "partial"}:
            synthesis_mode = "limited"
        else:
            synthesis_mode = "insufficient"

    source_scope_label = None
    source_scope_reason = None
    if source is None and len(sources) == 1:
        source = sources[0]
    if source is not None:
        topical_relevance = str(source.get("topicalRelevance") or "").strip()
        source_scope_reason = str(source.get("whyClassifiedAsWeakMatch") or source.get("note") or "").strip() or None
        if topical_relevance == "on_topic":
            source_scope_label = "directly_responsive"
        elif topical_relevance == "weak_match" and source.get("verificationStatus") == "verified_primary_source":
            source_scope_label = "authoritative_but_scope_limited"
        elif topical_relevance == "weak_match":
            source_scope_label = "related_but_incomplete"
        elif topical_relevance == "off_topic":
            source_scope_label = "off_topic"

    result = ConfidenceSignals(
        evidenceQualityProfile=cast(Any, evidence_quality_profile),
        synthesisMode=synthesis_mode,
        trustRevisionReason=trust_revision_reason,
        evidenceUsePlanApplied=evidence_use_plan_applied,
        fallbackExplanation=fallback_explanation,
        sourceScopeLabel=source_scope_label,
        sourceScopeReason=source_scope_reason,
    ).model_dump(by_alias=True, exclude_none=True)

    # Workstream C (ws-trust-ux-deepen): additive detail fields that expose the
    # richer WS-C enums without breaking the existing ``evidenceQualityProfile``
    # / ``synthesisMode`` contract.
    result["evidenceProfileDetail"] = _evidence_quality_detail(sources)
    result["synthesisPath"] = _synthesis_path(
        status=status,
        sources=sources,
        evidence_gaps=evidence_gaps,
        synthesis_mode=synthesis_mode,
    )
    narrative = _trust_revision_narrative(
        sources=sources,
        evidence_gaps=evidence_gaps,
        degradation_reason=degradation_reason,
    )
    if narrative:
        result["trustRevisionNarrative"] = narrative
    # ws-dispatch-contract-trust (finding #5): expose subject-chain gaps as a
    # first-class additive field so clients reading ``confidenceSignals`` get
    # the same reason already surfaced in prose by
    # ``_compose_why_classified_weak_match``.
    if _subject_chain_gaps:
        result["subjectChainGaps"] = _subject_chain_gaps
    return result


def _guided_sources_all_off_topic(sources: list[dict[str, Any]] | None) -> bool:
    """Return True when ``sources`` is non-empty and every entry is ``off_topic``.

    Seventh rubber-duck pass (finding 2): shared predicate used by
    ``_guided_failure_summary`` / ``_guided_next_actions`` / ``_guided_result_state``
    so their cross-field routing stays consistent when the current response
    contains only off-topic sources.
    """

    items = [source for source in (sources or []) if isinstance(source, dict)]
    if not items:
        return False
    return all(str(source.get("topicalRelevance") or "").strip().lower() == "off_topic" for source in items)


def _guided_failure_summary(
    *,
    failure_summary: dict[str, Any] | None,
    status: str,
    sources: list[dict[str, Any]],
    evidence_gaps: list[str],
    all_sources_off_topic: bool = False,
) -> dict[str, Any]:
    if failure_summary is not None:
        summary = dict(failure_summary)
    else:
        summary = {}
    outcome = str(summary.get("outcome") or "").strip()
    if not outcome:
        if status == "abstained" and not sources:
            outcome = "partial_success"
        elif status == "abstained":
            outcome = "no_failure"
        elif summary.get("fallbackAttempted"):
            outcome = "fallback_success"
        else:
            outcome = "no_failure"
    # Seventh rubber-duck pass (finding 2): when every current source is
    # off_topic, inspect_source cannot rescue the result — route to research
    # so the default recommendation agrees with _guided_result_state's
    # all-off-topic routing.
    effective_has_inspectable = bool(sources) and not all_sources_off_topic
    recommended_next_action = summary.get("recommendedNextAction")
    if not recommended_next_action:
        recommended_next_action = "inspect_source" if effective_has_inspectable else "research"
    completeness_impact = summary.get("completenessImpact")
    if not completeness_impact and evidence_gaps:
        completeness_impact = evidence_gaps[0]
    what_still_worked = summary.get("whatStillWorked")
    if not what_still_worked:
        if effective_has_inspectable:
            what_still_worked = "The guided run still returned inspectable sources."
        elif sources:
            # Ninth rubber-duck pass (finding 2): an all-off-topic pool is not
            # "inspectable" in the routing sense — do not claim otherwise.
            what_still_worked = "The guided run returned sources, but all were off-topic for the question."
        else:
            what_still_worked = (
                "No provider failures were recorded, but the evidence was not strong enough to ground a result."
            )
    fallback_attempted = bool(summary.get("fallbackAttempted"))
    fallback_mode = summary.get("fallbackMode")
    # Invariant: if fallbackMode is non-null, fallbackAttempted must be True
    if fallback_mode is not None and not fallback_attempted:
        fallback_attempted = True
    return {
        "outcome": outcome,
        "whatFailed": summary.get("whatFailed"),
        "whatStillWorked": what_still_worked,
        "fallbackAttempted": fallback_attempted,
        "fallbackMode": fallback_mode,
        "primaryPathFailureReason": summary.get("primaryPathFailureReason"),
        "completenessImpact": completeness_impact,
        "recommendedNextAction": recommended_next_action,
    }


def _guided_result_meaning(
    *,
    status: str,
    verified_findings: list[dict[str, Any]],
    evidence_gaps: list[str],
    coverage: dict[str, Any] | None,
    failure_summary: dict[str, Any],
    source_count: int = 0,
    all_sources_off_topic: bool = False,
) -> str:
    if verified_findings:
        return f"This result contains {len(verified_findings)} verified finding(s) grounded in the returned sources."
    if status == "partial":
        if source_count <= 0 or all_sources_off_topic:
            if all_sources_off_topic:
                return (
                    "This result returned sources, but all were off-topic for the query. "
                    "Tighten the anchor (exact title, DOI, species, agency, venue, or year range) and rerun research."
                )
            return (
                "This result is currently metadata-only and did not include inspectable sources. "
                "Use follow_up_research or rerun research with a tighter anchor."
            )
        return "This result found some relevant evidence, but the trust or coverage state is still incomplete."
    if status == "needs_disambiguation":
        return "This result needs a stronger anchor before the server can produce a grounded answer."
    if status == "abstained":
        return "This result did not find sufficiently trustworthy evidence to support a grounded answer."
    if failure_summary.get("outcome") not in {None, "no_failure"}:
        return "This result reflects degraded retrieval and should be treated as a partial recovery path."
    summary_line = str((coverage or {}).get("summaryLine") or "").strip()
    return summary_line or "This result should be reviewed source by source before relying on it."


def _guided_deterministic_fallback_used(provider_bundle: Any | None) -> bool:
    if provider_bundle is None or not hasattr(provider_bundle, "selection_metadata"):
        return False
    try:
        selection = provider_bundle.selection_metadata()
    except Exception:
        return False
    configured = str(selection.get("configuredSmartProvider") or "").strip()
    active = str(selection.get("activeSmartProvider") or "").strip()
    return bool(configured and configured != "deterministic" and active == "deterministic")


def _guided_partial_recovery_possible(
    *,
    coverage_summary: dict[str, Any] | None,
    failure_summary: dict[str, Any] | None,
) -> bool:
    coverage = coverage_summary or {}
    failed = [str(provider).strip() for provider in (coverage.get("providersFailed") or []) if str(provider).strip()]
    if failed:
        return True
    failure = failure_summary or {}
    return bool(
        failure.get("fallbackAttempted") or failure.get("fallbackMode") or failure.get("primaryPathFailureReason")
    )


async def _guided_research_status(
    *,
    query: str,
    intent: str,
    sources: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    unverified_leads_count: int,
    coverage_summary: dict[str, Any] | None,
    failure_summary: dict[str, Any] | None,
    clarification: dict[str, Any] | None,
    provider_bundle: Any | None = None,
) -> tuple[str, str | None]:
    if clarification is not None:
        return "needs_disambiguation", None
    if intent == "known_item" and findings:
        return "succeeded", None
    if intent == "regulatory":
        primary_document_coverage = cast(
            dict[str, Any] | None,
            (coverage_summary or {}).get("primaryDocumentCoverage"),
        )
        primary_sources = [
            source
            for source in sources
            if source.get("topicalRelevance") == "on_topic" and bool(source.get("isPrimarySource"))
        ]
        if primary_document_coverage is not None and primary_document_coverage.get("currentTextRequested"):
            if primary_document_coverage.get("currentTextSatisfied"):
                return ("partial" if failure_summary is not None else "succeeded"), None
            if primary_sources:
                return "partial", None
            if unverified_leads_count > 0:
                return "partial", None
            return "abstained", None
        if primary_sources:
            return ("partial" if failure_summary is not None else "succeeded"), None
        if unverified_leads_count > 0:
            return "partial", None
        return ("needs_disambiguation" if sources else "abstained"), None
    if len(findings) >= 2:
        base_status = "partial" if failure_summary is not None else "succeeded"
    else:
        on_topic_verified = sum(
            1
            for source in sources
            if source.get("topicalRelevance") == "on_topic"
            and source.get("verificationStatus") in {"verified_primary_source", "verified_metadata"}
        )
        if on_topic_verified >= 5:
            base_status = "partial" if failure_summary is not None else "succeeded"
        elif sources:
            base_status = "partial"
        elif unverified_leads_count > 0:
            base_status = "partial"
        else:
            base_status = "abstained"

    if (
        base_status == "abstained"
        and _guided_deterministic_fallback_used(provider_bundle)
        and _guided_partial_recovery_possible(
            coverage_summary=coverage_summary,
            failure_summary=failure_summary,
        )
    ):
        return (
            "partial",
            (
                "Configured smart provider was unavailable, so guided research stayed on "
                "deterministic fallback while retrieval remained incomplete."
            ),
        )

    adequacy_reason: str | None = None
    verified_sources = [
        source
        for source in sources
        if source.get("topicalRelevance") == "on_topic"
        and source.get("verificationStatus") in {"verified_primary_source", "verified_metadata"}
    ]
    if (
        base_status == "partial"
        and provider_bundle is not None
        and verified_sources
        and intent not in {"known_item", "regulatory"}
    ):
        try:
            adequacy = await provider_bundle.aassess_result_adequacy(
                query=query,
                intent=intent,
                verified_sources=verified_sources,
                evidence_gaps=[],
            )
            adequacy_label = str(adequacy.get("adequacy") or "partial")
            adequacy_reason = str(adequacy.get("reason") or "").strip() or None
            if adequacy_label == "succeeded":
                base_status = "succeeded"
            elif adequacy_label == "insufficient":
                base_status = "abstained"
        except Exception:
            adequacy_reason = None
    return base_status, adequacy_reason


def _guided_deterministic_evidence_gaps(
    *,
    query: str,
    intent: str,
    sources: list[dict[str, Any]],
    existing_evidence_gaps: list[str],
    retrieval_hypotheses: list[str],
    coverage_summary: dict[str, Any] | None,
    timeline: dict[str, Any] | None,
    anchor_type: str | None,
) -> list[str]:
    return generate_evidence_gaps_without_llm(
        query=query,
        intent=intent,
        sources=sources,
        evidence_gaps=existing_evidence_gaps,
        retrieval_hypotheses=retrieval_hypotheses,
        coverage_summary=coverage_summary,
        timeline=timeline,
        anchor_type=anchor_type,
    )


async def _guided_generate_evidence_gaps(
    *,
    query: str,
    intent: str,
    sources: list[dict[str, Any]],
    existing_evidence_gaps: list[str],
    coverage_summary: dict[str, Any] | None,
    strategy_metadata: dict[str, Any] | None,
    timeline: dict[str, Any] | None,
    provider_bundle: Any | None,
) -> list[str]:
    metadata = strategy_metadata or {}
    retrieval_hypotheses = [
        str(item).strip() for item in metadata.get("retrievalHypotheses") or [] if str(item).strip()
    ]
    anchor_type = str(metadata.get("anchorType") or "").strip() or None
    deterministic_gaps = _guided_deterministic_evidence_gaps(
        query=query,
        intent=intent,
        sources=sources,
        existing_evidence_gaps=existing_evidence_gaps,
        retrieval_hypotheses=retrieval_hypotheses,
        coverage_summary=coverage_summary,
        timeline=timeline,
        anchor_type=anchor_type,
    )
    if provider_bundle is None or not hasattr(provider_bundle, "agenerate_evidence_gaps"):
        return deterministic_gaps
    try:
        model_gaps = await provider_bundle.agenerate_evidence_gaps(
            query=query,
            intent=intent,
            sources=sources,
            evidence_gaps=existing_evidence_gaps,
            retrieval_hypotheses=retrieval_hypotheses,
            coverage_summary=coverage_summary,
            timeline=timeline,
            anchor_type=anchor_type,
        )
        cleaned_model_gaps = [str(gap).strip() for gap in model_gaps or [] if str(gap).strip()]
        if cleaned_model_gaps:
            return cleaned_model_gaps
    except Exception:
        logger.debug("Guided evidence-gap generation failed; using deterministic fallback.")
    return deterministic_gaps


def _guided_machine_failure_payload(
    *,
    search_session_id: str | None,
    error: Exception,
    normalization: dict[str, Any] | None = None,
    execution_provenance: dict[str, Any] | None = None,
    saved_session_has_sources: bool = False,
    saved_session_all_off_topic: bool = False,
) -> dict[str, Any]:
    evidence_gaps = ["Smart runtime returned an invalid or unstructured result payload, so guided output was degraded."]
    # Fifth rubber-duck pass (finding 2): when a saved session is still
    # inspectable, recommend inspect_source rather than research so the
    # failure payload agrees with the result-state routing used elsewhere.
    saved_session_inspectable = saved_session_has_sources and not saved_session_all_off_topic
    recommended_next_action = "inspect_source" if saved_session_inspectable else "research"
    failure_summary = _guided_failure_summary(
        failure_summary={
            "outcome": "total_failure",
            "whatFailed": "smart_runtime_structural_failure",
            "whatStillWorked": "The guided wrapper recovered and returned a machine-readable failure state.",
            "fallbackAttempted": False,
            "fallbackMode": None,
            "primaryPathFailureReason": str(type(error).__name__),
            "completenessImpact": evidence_gaps[0],
            "recommendedNextAction": recommended_next_action,
        },
        status="partial",
        sources=[],
        evidence_gaps=evidence_gaps,
    )
    payload: dict[str, Any] = {
        "intent": "discovery",
        "status": "partial",
        "searchSessionId": search_session_id,
        "summary": "Smart retrieval failed structurally; the server returned a safe machine-readable failure state.",
        "verifiedFindings": [],
        "sources": [],
        "unverifiedLeads": [],
        "evidenceGaps": evidence_gaps,
        "trustSummary": _guided_trust_summary([], evidence_gaps),
        "coverage": None,
        "failureSummary": failure_summary,
        "resultMeaning": _guided_result_meaning(
            status="partial",
            verified_findings=[],
            evidence_gaps=evidence_gaps,
            coverage=None,
            failure_summary=failure_summary,
            source_count=0,
        ),
        "nextActions": _guided_next_actions(
            search_session_id=search_session_id,
            status="partial",
            has_sources=False,
            saved_session_inspectable=saved_session_inspectable,
        ),
        "resultState": _guided_result_state(
            status="partial",
            sources=[],
            evidence_gaps=evidence_gaps,
            search_session_id=search_session_id,
            saved_session_has_sources=saved_session_has_sources,
            saved_session_all_off_topic=saved_session_all_off_topic,
        ),
        "machineFailure": MachineFailure(
            category="smart_runtime_structural_failure",
            errorType=type(error).__name__,
            error=str(error),
            retryable=True,
            bestNextInternalAction=recommended_next_action,
        ).model_dump(by_alias=True, exclude_none=True),
    }
    if execution_provenance is not None:
        payload["executionProvenance"] = execution_provenance
    from .research import _guided_normalization_payload  # noqa: E402 — avoid circular import

    normalization_payload = _guided_normalization_payload(normalization or {})
    if normalization_payload is not None:
        payload["inputNormalization"] = normalization_payload
    return payload


def _guided_summary(
    intent: str,
    status: str,
    findings: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    *,
    routing_summary: dict[str, Any] | None = None,
    pass_modes: list[str] | None = None,
) -> str:
    all_sources_off_topic = _guided_sources_all_off_topic(sources)
    if findings:
        top_claim = str(findings[0].get("claim") or "").strip()
        additional_count = max(len(findings) - 1, 0)
        if additional_count:
            summary = f"Top result: {top_claim}. Verified support includes {additional_count} additional source(s)."
        else:
            summary = f"Top result: {top_claim}."
    elif sources and all_sources_off_topic:
        # R12 finding: when every source is off_topic, do NOT tell callers to
        # "inspect the source" — that contradicts research-next routing.
        summary = (
            "The search returned sources, but every candidate was off-topic for the request. "
            "Tighten the anchor (exact title, DOI, species, agency, venue, or year range) and rerun research."
        )
    elif sources:
        top_title = str(sources[0].get("title") or sources[0].get("sourceId") or "").strip()
        if top_title:
            summary = (
                f"Top result: {top_title}. Evidence is still partial or mixed, "
                "so inspect the source before relying on it."
            )
        else:
            summary = (
                "The search found some source leads, but the evidence stayed too weak, off-topic, or incomplete "
                "for a grounded summary."
            )
    elif status == "needs_disambiguation":
        summary = "The request needs a more specific anchor before the system can build a grounded result."
    else:
        summary = "No sufficiently trustworthy evidence was found for a grounded result."

    notes: list[str] = []
    routing = routing_summary if isinstance(routing_summary, dict) else {}
    query_specificity = str(routing.get("querySpecificity") or "").strip()
    ambiguity_level = str(routing.get("ambiguityLevel") or "").strip()
    hypotheses = [str(item).strip() for item in routing.get("retrievalHypotheses") or [] if str(item).strip()]
    if hypotheses:
        hypothesis_label = "hypothesis" if len(hypotheses) == 1 else "hypotheses"
        notes.append(
            "The query was broad or ambiguous, so the server explored "
            f"{len(hypotheses)} bounded retrieval {hypothesis_label}."
        )
    elif query_specificity == "low" or ambiguity_level in {"medium", "high"}:
        notes.append("The query stayed broad or ambiguous, so the server blended nearby retrieval routes.")
    if pass_modes and "regulatory" in pass_modes and "review" in pass_modes:
        notes.append("This result blends regulatory and literature passes.")
    if notes:
        summary = f"{summary} {' '.join(dict.fromkeys(notes))}"
    return summary


def _guided_next_actions(
    *,
    search_session_id: str | None,
    status: str,
    has_sources: bool,
    calling_tool: str | None = None,
    saved_session_inspectable: bool = False,
    all_sources_off_topic: bool = False,
) -> list[str]:
    actions: list[str] = []
    # Sixth rubber-duck pass (finding 2): even when the current response has no
    # sources, a saved session can still hold inspectable candidates. Surface
    # inspect_source in nextActions so it agrees with the failure-summary and
    # machine-failure bestNextInternalAction routing.
    # Seventh rubber-duck pass (finding 2): when the current response has
    # sources but every one is off_topic, inspect_source is not productive —
    # treat the current response as empty for inspect-routing purposes so
    # nextActions agrees with bestNextInternalAction ("research").
    effective_has_inspectable = has_sources and not all_sources_off_topic
    inspect_relevant = effective_has_inspectable or saved_session_inspectable
    if search_session_id and inspect_relevant and calling_tool != "inspect_source":
        actions.append(
            f"Use inspect_source with searchSessionId='{search_session_id}' and one sourceId to inspect evidence."
        )
    if search_session_id:
        actions.append(
            f"Use follow_up_research with searchSessionId='{search_session_id}' to ask one grounded follow-up question."
        )
    if status in {"abstained", "needs_disambiguation"}:
        actions.append("Narrow the request with a specific title, DOI, species name, agency, venue, or year range.")
    if status == "partial":
        actions.append("Refine the request to reduce evidence gaps before treating the result as settled.")
    actions.append(
        "Use get_runtime_status if behavior differs across environments and you need the active runtime truth."
    )
    return actions[:4]


def _guided_missing_evidence_type(
    *,
    status: str,
    evidence_gaps: list[str],
    sources: list[dict[str, Any]],
) -> str:
    if status in {"succeeded", "answered"}:
        return "none"
    joined_gaps = " ".join(str(gap).lower() for gap in evidence_gaps)
    if "off-topic" in joined_gaps:
        return "off_topic_only"
    if any(marker in joined_gaps for marker in ("clarif", "anchor", "disambiguation")):
        return "anchor_missing"
    if any(marker in joined_gaps for marker in ("provider", "timeout", "failed", "error")):
        return "provider_gap"
    if not sources:
        return "no_sources"
    return "coverage_gap"


def _guided_best_next_internal_action(
    *,
    status: str,
    has_sources: bool,
    search_session_id: str | None,
    saved_session_has_sources: bool = False,
    saved_session_all_off_topic: bool = False,
    all_sources_off_topic: bool = False,
) -> str:
    normalized_status = str(status or "").strip().lower()
    weak_statuses = {
        "abstained",
        "needs_disambiguation",
        "failed",
        "insufficient_evidence",
        "partial",
    }
    # When every returned source is off-topic, inspect_source cannot rescue the
    # result; the agent should refine the query instead of chasing irrelevant
    # evidence.
    if has_sources and search_session_id and not all_sources_off_topic:
        return "inspect_source"
    # The CURRENT response may have no sources (e.g., smart runtime unavailable
    # or inspect_source retry with a wrong sourceId), yet the SAVED SESSION may
    # still be inspectable. In that case prefer inspect_source over research so
    # resultState agrees with the failureSummary's recommended retry. If every
    # saved candidate is off_topic, fall through to research instead so the
    # 9ee3168 "all off-topic → research" guarantee still holds.
    if not has_sources and saved_session_has_sources and not saved_session_all_off_topic and search_session_id:
        return "inspect_source"
    # Without inspectable sources, neither inspect_source nor another follow_up_research
    # over the same (empty) session can progress: keep the guidance aligned with the
    # failureSummary's recommended retry instead of looping the agent.
    if not has_sources and normalized_status in weak_statuses:
        return "research"
    if all_sources_off_topic:
        return "research"
    if search_session_id:
        return "follow_up_research"
    if normalized_status in weak_statuses:
        return "research"
    return "research"


def _guided_result_state(
    *,
    status: str,
    sources: list[dict[str, Any]],
    evidence_gaps: list[str],
    search_session_id: str | None,
    saved_session_has_sources: bool = False,
    saved_session_all_off_topic: bool = False,
    saved_session_inspectable_override: bool | None = None,
) -> dict[str, Any]:
    has_sources = bool(sources)
    all_sources_off_topic = has_sources and all(
        str(source.get("topicalRelevance") or "").strip().lower() == "off_topic"
        for source in sources
        if isinstance(source, dict)
    )
    normalized_status = str(status or "").strip() or "unknown"
    if normalized_status in {"succeeded", "answered"} and has_sources and not all_sources_off_topic:
        groundedness = "grounded"
    elif normalized_status == "answered":
        groundedness = "partial"
    elif normalized_status in {"partial", "insufficient_evidence"}:
        groundedness = "partial"
    elif normalized_status in {"abstained", "needs_disambiguation", "failed"}:
        groundedness = "insufficient_evidence"
    else:
        groundedness = "unknown"
    # Ninth rubber-duck pass (finding 3): when the status claims success but
    # every returned source is off_topic, the result is not actually grounded
    # in on-topic evidence. Downgrade groundedness to "insufficient_evidence"
    # so it agrees with bestNextInternalAction="research".
    if all_sources_off_topic and normalized_status in {"succeeded", "answered"}:
        groundedness = "insufficient_evidence"
    # Fifth rubber-duck pass (finding 3): hasInspectableSources must agree with
    # bestNextInternalAction. When the current response is empty but the saved
    # session still carries on_topic/weak_match evidence, the saved candidates
    # remain reachable via inspect_source, so the flag has to reflect that.
    saved_session_inspectable = (
        saved_session_inspectable_override
        if saved_session_inspectable_override is not None
        else (saved_session_has_sources and not saved_session_all_off_topic)
    )
    current_inspectable = has_sources and not all_sources_off_topic
    inspectable_sources = current_inspectable or saved_session_inspectable
    # Ninth rubber-duck pass (finding 1): canAnswerFollowUp must reflect whether
    # inspectable evidence is actually reachable. An all-off-topic current pool
    # cannot ground a follow-up, while a saved-session-inspectable case can.
    missing_evidence_type = _guided_missing_evidence_type(
        status=normalized_status,
        evidence_gaps=evidence_gaps,
        sources=sources,
    )
    # Ninth rubber-duck pass (finding 3): normalize missingEvidenceType for the
    # all-off-topic success/answered case so it reflects the off-topic gap
    # instead of advertising "none".
    if all_sources_off_topic and missing_evidence_type == "none":
        missing_evidence_type = "off_topic_only"
    state = GuidedResultState(
        status=normalized_status,
        groundedness=groundedness,
        hasInspectableSources=inspectable_sources,
        canAnswerFollowUp=bool(search_session_id) and inspectable_sources,
        bestNextInternalAction=_guided_best_next_internal_action(
            status=normalized_status,
            has_sources=has_sources,
            search_session_id=search_session_id,
            saved_session_has_sources=saved_session_has_sources,
            saved_session_all_off_topic=saved_session_all_off_topic,
            all_sources_off_topic=all_sources_off_topic,
        ),
        missingEvidenceType=missing_evidence_type,
    )
    return state.model_dump(by_alias=True, exclude_none=True)


def _guided_record_source_candidates(record: Any) -> list[dict[str, Any]]:
    payload = record.payload if isinstance(record.payload, dict) else {}
    has_explicit_source_payload = any(
        isinstance(payload.get(key), list) and bool(payload.get(key))
        for key in ("evidence", "sources", "structuredSources", "leads", "candidateLeads", "unverifiedLeads")
    )
    payload_sources = [source for source in payload.get("sources") or [] if isinstance(source, dict)]
    structured_sources = [
        _guided_source_record_from_structured_source(source, index=index)
        for index, source in enumerate(payload.get("structuredSources") or [], start=1)
        if isinstance(source, dict)
    ]
    evidence_sources = [
        _guided_source_record_from_structured_source(source, index=index)
        for index, source in enumerate(payload.get("evidence") or [], start=1)
        if isinstance(source, dict)
    ]
    lead_sources = [
        _guided_source_record_from_structured_source(source, index=index)
        for key in ("leads", "candidateLeads", "unverifiedLeads")
        for index, source in enumerate(payload.get(key) or [], start=1)
        if isinstance(source, dict)
    ]
    query = str(record.query or payload.get("query") or "")
    explicit_candidates = _guided_dedupe_source_records(
        _guided_merge_source_record_sets(
            payload_sources,
            structured_sources,
            evidence_sources,
            lead_sources,
        )
    )
    paper_sources = [
        _guided_source_record_from_paper(query, paper, index=index)
        for index, paper in enumerate(getattr(record, "papers", []) or [], start=1)
        if isinstance(paper, dict)
    ]
    if not paper_sources:
        return explicit_candidates
    if not has_explicit_source_payload:
        return _guided_dedupe_source_records(_guided_merge_source_record_sets(explicit_candidates, paper_sources))

    augmented_candidates = list(explicit_candidates)
    for paper_source in paper_sources:
        paper_source_id = str(paper_source.get("sourceId") or "").strip()
        if not paper_source_id:
            continue
        if any(_guided_source_matches_reference(candidate, paper_source_id) for candidate in augmented_candidates):
            continue
        merged_candidate = paper_source
        for candidate in explicit_candidates:
            if _guided_source_records_share_surface(candidate, paper_source):
                merged_candidate = _guided_merge_source_records(paper_source, candidate)
                break
        augmented_candidates.append(merged_candidate)
    return _guided_dedupe_source_records(augmented_candidates)


def _guided_follow_up_status(status: str | None) -> str:
    normalized = str(status or "").strip()
    if normalized in {"succeeded", "partial", "needs_disambiguation", "abstained", "failed"}:
        return normalized
    if normalized == "answered":
        return "succeeded"
    if normalized == "insufficient_evidence":
        return "partial"
    return "partial"
