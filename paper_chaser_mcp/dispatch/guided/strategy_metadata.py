"""Guided strategy-metadata helpers (Phase 3 extraction).

Extracted from :mod:`paper_chaser_mcp.dispatch._core`. This submodule owns
the helpers that shape ``strategyMetadata`` / ``executionProvenance`` /
``abstentionDetails`` / review-pass decisions used across guided tools.
"""

from __future__ import annotations

import re
from typing import Any, cast

from ...agentic.planner import (
    detect_literature_intent,
    detect_regulatory_intent,
    looks_like_exact_title,
)
from ...citation_repair import looks_like_citation_query, looks_like_paper_identifier
from ...models.common import AbstentionDetails, GuidedExecutionProvenance
from ..normalization import _guided_normalize_whitespace

# Forward references to helpers/constants still living in ``_core``. Valid
# because this module is imported at the bottom of ``_core.py`` after all
# top-level defs have executed.
from .._core import (  # noqa: E402 — see note above; forward refs
    _GUIDED_LITERATURE_TERMS,
    _GUIDED_REFERENCE_GENERIC_CANDIDATE_WORDS,
    _GUIDED_REFERENCE_UNCERTAINTY_MARKERS,
    GUIDED_POLICY_NAME,
)
from .trust import _guided_missing_evidence_type, _guided_sources_all_off_topic



def _guided_execution_provenance_payload(
    *,
    execution_mode: str,
    answer_source: str | None = None,
    latency_profile_applied: str | None = None,
    allow_paid_providers: bool | None = None,
    provider_budget_applied: dict[str, Any] | None = None,
    strategy_metadata: dict[str, Any] | None = None,
    escalation_attempted: bool = False,
    escalation_reason: str | None = None,
    passes_run: int = 0,
    pass_modes: list[str] | None = None,
) -> dict[str, Any]:
    metadata = strategy_metadata if isinstance(strategy_metadata, dict) else {}
    configured_provider = _guided_normalize_whitespace(metadata.get("configuredSmartProvider")) or None
    active_provider = _guided_normalize_whitespace(metadata.get("activeSmartProvider")) or None
    latency_profile = latency_profile_applied or _guided_normalize_whitespace(metadata.get("latencyProfile")) or None
    budget_payload = provider_budget_applied or cast(dict[str, Any], metadata.get("providerBudgetApplied") or {})
    deterministic_fallback_used = bool(
        active_provider == "deterministic" and configured_provider not in {None, "deterministic"}
    )
    provenance = GuidedExecutionProvenance(
        executionMode=execution_mode,
        answerSource=answer_source,
        serverPolicyApplied=GUIDED_POLICY_NAME,
        latencyProfileApplied=latency_profile,
        allowPaidProviders=allow_paid_providers,
        providerBudgetApplied=budget_payload,
        configuredSmartProvider=configured_provider,
        activeSmartProvider=active_provider,
        deterministicFallbackUsed=deterministic_fallback_used,
        escalationAttempted=escalation_attempted,
        escalationReason=escalation_reason,
        passesRun=passes_run,
        passModes=pass_modes or [],
    )
    return provenance.model_dump(by_alias=True, exclude_none=True)




def _guided_live_strategy_metadata(
    *,
    agentic_runtime: Any,
    strategy_metadata: dict[str, Any] | None = None,
    latency_profile: str | None = None,
) -> dict[str, Any]:
    merged = dict(strategy_metadata or {})
    provider_bundle = getattr(agentic_runtime, "_provider_bundle", None)
    if provider_bundle is not None and hasattr(provider_bundle, "selection_metadata"):
        try:
            selection = provider_bundle.selection_metadata()
            if isinstance(selection, dict):
                merged.update(selection)
        except Exception as exc:
            merged.setdefault("selectionMetadataError", type(exc).__name__)
    if latency_profile and not merged.get("latencyProfile"):
        merged["latencyProfile"] = latency_profile
    return merged




def _guided_abstention_details_payload(
    *,
    status: str,
    sources: list[dict[str, Any]],
    evidence_gaps: list[str],
    trust_summary: dict[str, Any],
) -> dict[str, Any] | None:
    if status not in {"abstained", "needs_disambiguation", "insufficient_evidence", "partial"}:
        return None
    category = _guided_missing_evidence_type(status=status, evidence_gaps=evidence_gaps, sources=sources)
    all_sources_off_topic = _guided_sources_all_off_topic(sources)
    # R11 finding 1: when every source is off_topic, agent-facing hints must
    # not tell the caller to inspect the returned sources — they aren't useful.
    # Force the off_topic_only category so the hints route to tighten-query.
    if sources and all_sources_off_topic and category != "off_topic_only":
        category = "off_topic_only"
    weak_match_count = int(trust_summary.get("weakMatchCount") or 0)
    off_topic_count = int(trust_summary.get("offTopicCount") or 0)
    on_topic_source_count = int(trust_summary.get("onTopicSourceCount") or 0)
    if status == "partial" and category == "coverage_gap":
        if weak_match_count and weak_match_count >= max(on_topic_source_count, 1):
            category = "weak_topical_match"
        elif on_topic_source_count and on_topic_source_count < 3:
            category = "narrow_evidence_pool"
    if category == "anchor_missing":
        refinement_hints = ["Add a specific title, DOI, species name, agency, venue, or year range."]
    elif category == "off_topic_only":
        refinement_hints = ["Tighten the query to the exact topic or anchored subject you need."]
    elif category == "provider_gap":
        refinement_hints = [
            "Retry later or compare get_runtime_status if provider behavior differs across environments.",
        ]
    elif category == "weak_topical_match":
        refinement_hints = [
            "Add a year range or venue to reduce weak topical matches.",
            "Specify the exact species (common or scientific name), agency, or concept you care about.",
            "Try resolve_reference if you have a DOI, arXiv id, URL, or full citation.",
        ]
    elif category == "narrow_evidence_pool":
        refinement_hints = [
            "Broaden the query with synonyms or a wider year range to recover more evidence.",
            "Run follow_up_research on the saved session to reuse the grounded sources you already have.",
            "Try resolve_reference if you have a DOI, arXiv id, URL, or full citation.",
        ]
    elif status == "partial" and sources:
        refinement_hints = [
            "Inspect the returned sources before treating the result as settled.",
            "Add a specific anchor (title, DOI, species, agency, venue, or year range) and retry research.",
        ]
    elif sources:
        refinement_hints = ["Inspect the returned sources before treating the result as settled."]
    else:
        refinement_hints = ["Narrow the request so the server can recover a stronger initial anchor."]
    effective_inspectable_count = 0 if all_sources_off_topic else len(sources)
    can_inspect = bool(sources) and not all_sources_off_topic
    details = AbstentionDetails(
        category=category,
        reason=(
            evidence_gaps[0] if evidence_gaps else "The current evidence was not strong enough to ground an answer."
        ),
        inspectableSourceCount=effective_inspectable_count,
        onTopicSourceCount=on_topic_source_count,
        weakMatchCount=weak_match_count,
        offTopicCount=off_topic_count,
        canInspectSources=can_inspect,
        refinementHints=refinement_hints,
    )
    return details.model_dump(by_alias=True, exclude_none=True)




def _guided_provider_budget_payload(*, allow_paid_providers: bool) -> dict[str, Any]:
    return {"allowPaidProviders": bool(allow_paid_providers)}




def _guided_strategy_metadata_from_runs(smart_runs: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    merged_lists: dict[str, list[str]] = {
        "providerPlan": [],
        "providersUsed": [],
        "secondaryIntents": [],
        "queryVariantsTried": [],
        "retrievalHypotheses": [],
    }
    confidence_rank = {"high": 3, "medium": 2, "low": 1}
    specificity_rank = {"high": 3, "medium": 2, "low": 1}
    ambiguity_rank = {"low": 1, "medium": 2, "high": 3}
    routing_confidences: list[str] = []
    intent_confidences: list[str] = []
    query_specificities: list[str] = []
    ambiguity_levels: list[str] = []

    for smart in smart_runs:
        metadata = smart.get("strategyMetadata")
        if not isinstance(metadata, dict):
            continue
        for field in (
            "intent",
            "intentRationale",
            "anchorType",
            "anchoredSubject",
            "configuredSmartProvider",
            "activeSmartProvider",
            "providerBudgetApplied",
            "latencyProfile",
            # Phase 4/5 planner classification signals — pass through the first
            # non-empty value so downstream routingSummary serialization can
            # expose them to agents.
            "intentFamily",
            "regulatoryIntent",
            "subjectCard",
        ):
            value = metadata.get(field)
            if value not in (None, "", [], {}) and field not in merged:
                merged[field] = value
        for field in merged_lists:
            for item in metadata.get(field) or []:
                text = str(item).strip()
                if text and text not in merged_lists[field]:
                    merged_lists[field].append(text)
        for item in metadata.get("subjectChainGaps") or []:
            text = str(item).strip()
            if text:
                subject_chain_gaps = merged.setdefault("subjectChainGaps", [])
                if isinstance(subject_chain_gaps, list) and text not in subject_chain_gaps:
                    subject_chain_gaps.append(text)

        intent_confidence = str(metadata.get("intentConfidence") or "").strip()
        if intent_confidence in confidence_rank:
            intent_confidences.append(intent_confidence)
        routing_confidence = str(metadata.get("routingConfidence") or "").strip()
        if routing_confidence in confidence_rank:
            routing_confidences.append(routing_confidence)
        query_specificity = str(metadata.get("querySpecificity") or "").strip()
        if query_specificity in specificity_rank:
            query_specificities.append(query_specificity)
        ambiguity_level = str(metadata.get("ambiguityLevel") or "").strip()
        if ambiguity_level in ambiguity_rank:
            ambiguity_levels.append(ambiguity_level)

    for field, values in merged_lists.items():
        if values:
            merged[field] = values
    if intent_confidences:
        merged["intentConfidence"] = min(intent_confidences, key=lambda value: confidence_rank[value])
    if routing_confidences:
        merged["routingConfidence"] = min(routing_confidences, key=lambda value: confidence_rank[value])
    if query_specificities:
        merged["querySpecificity"] = min(query_specificities, key=lambda value: specificity_rank[value])
    if ambiguity_levels:
        merged["ambiguityLevel"] = max(ambiguity_levels, key=lambda value: ambiguity_rank[value])
    return merged




def _guided_should_add_review_pass(
    *,
    initial_intent: str,
    query: str,
    focus: str | None,
    primary_smart: dict[str, Any],
    pass_modes: list[str],
) -> tuple[bool, str | None]:
    if "review" in pass_modes:
        return False, None
    if initial_intent == "mixed":
        return True, "mixed_intent_query"
    if initial_intent != "regulatory":
        return False, None
    if _guided_is_agency_guidance_query(query):
        return False, None

    metadata = cast(
        dict[str, Any],
        primary_smart.get("strategyMetadata") if isinstance(primary_smart.get("strategyMetadata"), dict) else {},
    )
    secondary_intents = {str(item).strip() for item in metadata.get("secondaryIntents") or [] if str(item).strip()}
    query_specificity = str(metadata.get("querySpecificity") or "").strip()
    ambiguity_level = str(metadata.get("ambiguityLevel") or "").strip()
    retrieval_hypotheses = [
        str(item).strip() for item in metadata.get("retrievalHypotheses") or [] if str(item).strip()
    ]
    # LLM-first: when the planner classifies the query as hybrid regulatory+literature
    # (either from the LLM or its deterministic fallback), honor that signal before any
    # keyword heuristic -- but only when an independent query-side literature cue
    # corroborates the hybrid label. Planners occasionally hallucinate
    # ``hybrid_regulatory_plus_literature`` for regulation-only asks (e.g.
    # "what does EPA require for stormwater discharges?"); trusting that label
    # in isolation causes the guided workflow to tack on an unnecessary review
    # pass. Corroboration is present when the query mentions literature-shaped
    # terms, the planner's secondaryIntents include review/literature, or
    # retrievalHypotheses carry explicit cross-domain literature cues.
    regulatory_intent = str(metadata.get("regulatoryIntent") or "").strip()
    if regulatory_intent == "hybrid_regulatory_plus_literature":
        has_literature_corroboration = (
            _guided_mentions_literature(query, focus)
            or "review" in secondary_intents
            or "literature" in secondary_intents
            or any(
                any(
                    marker in hypothesis.lower()
                    for marker in (
                        "literature",
                        "peer-review",
                        "peer review",
                        "peer-reviewed",
                        "systematic review",
                        "meta-analysis",
                        "hybrid_policy_science",
                    )
                )
                for hypothesis in retrieval_hypotheses
            )
        )
        if has_literature_corroboration:
            return True, "planner_hybrid_regulatory_plus_literature"
    if "review" in secondary_intents:
        return True, "review_secondary_intent_detected"
    if detect_literature_intent(query, focus) and (query_specificity == "low" or ambiguity_level in {"medium", "high"}):
        return True, "broad_regulatory_query_with_literature_signal"
    if ambiguity_level == "high" and retrieval_hypotheses:
        return True, "ambiguous_regulatory_query_with_retrieval_hypotheses"
    return False, None




def _guided_review_pass_overrides(
    *,
    query: str,
    focus: str | None,
    primary_smart: dict[str, Any],
) -> dict[str, Any]:
    metadata = cast(
        dict[str, Any],
        primary_smart.get("strategyMetadata") if isinstance(primary_smart.get("strategyMetadata"), dict) else {},
    )
    anchored_subject = _guided_normalize_whitespace(metadata.get("anchoredSubject"))
    existing_focus = _guided_normalize_whitespace(focus)
    if not anchored_subject:
        return {}

    lowered_query = _guided_normalize_whitespace(query).lower()
    lowered_focus = existing_focus.lower()
    if anchored_subject.lower() in lowered_query or anchored_subject.lower() in lowered_focus:
        return {}
    merged_focus = " ".join(part for part in [existing_focus, anchored_subject] if part)
    return {"focus": merged_focus} if merged_focus else {}




def _guided_is_agency_guidance_query(query: str) -> bool:
    normalized = _guided_normalize_whitespace(query).lower()
    if "guidance" not in normalized:
        return False
    return any(marker in normalized for marker in ("agency", "epa", "fda", "guidance for industry"))




def _guided_should_escalate_research(
    *,
    intent: str,
    status: str,
    sources: list[dict[str, Any]],
    verified_findings: list[dict[str, Any]],
    clarification: dict[str, Any] | None,
    pass_modes: list[str],
    max_passes: int,
) -> bool:
    if clarification is not None:
        return False
    if len(pass_modes) >= max_passes:
        return False
    if intent in {"known_item", "mixed", "regulatory"}:
        return False
    if verified_findings:
        return False
    if sources:
        return False
    return status in {"abstained", "partial"} and "review" not in pass_modes




def _guided_is_known_item_query(query: str) -> bool:
    return looks_like_paper_identifier(query) or looks_like_citation_query(query) or looks_like_exact_title(query)




def _guided_mentions_literature(query: str, focus: str | None = None) -> bool:
    normalized = " ".join(part for part in [query, focus or ""] if part).lower()
    if not normalized:
        return False
    if any(term in normalized for term in _GUIDED_LITERATURE_TERMS):
        return True
    if "scholarship" in normalized:
        return True
    return bool(re.search(r"\b(?:doi|systematic review|meta-analysis|peer-reviewed|scientific reports?)\b", normalized))




def _guided_is_mixed_intent_query(
    query: str,
    focus: str | None = None,
    *,
    planner_regulatory_intent: str | None = None,
) -> bool:
    # LLM-first: when the planner has already classified this query as
    # ``hybrid_regulatory_plus_literature`` (either from the LLM or from the
    # planner's deterministic fallback), trust that signal directly and skip the
    # keyword heuristic. Callers without planner context (the current default at
    # the top of guided research dispatch, before any smart pass has run) fall
    # back to the deterministic regulatory + literature keyword check below.
    if planner_regulatory_intent == "hybrid_regulatory_plus_literature":
        return True
    return detect_regulatory_intent(query, focus) and _guided_mentions_literature(query, focus)




def _guided_reference_signal_words(candidate: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9'/-]*", candidate.lower())
    return [
        word
        for word in words
        if word not in _GUIDED_REFERENCE_GENERIC_CANDIDATE_WORDS
        and word not in _GUIDED_REFERENCE_UNCERTAINTY_MARKERS
        and word not in {"a", "an", "and", "for", "in", "of", "on", "the", "to", "with"}
    ]


def _guided_merge_coverage_summaries(*coverages: dict[str, Any] | None) -> dict[str, Any] | None:
    usable = [coverage for coverage in coverages if isinstance(coverage, dict)]
    if not usable:
        return None
    if len(usable) == 1:
        return dict(usable[0])

    def _merge_list(field: str) -> list[Any]:
        merged: list[Any] = []
        seen_values: set[str] = set()
        for coverage in usable:
            for item in coverage.get(field) or []:
                marker = repr(item)
                if marker in seen_values:
                    continue
                seen_values.add(marker)
                merged.append(item)
        return merged

    likely_completeness = "unknown"
    for candidate, normalized_value in (
        ("incomplete", "incomplete"),
        ("partial", "partial"),
        ("likely_complete", "likely_complete"),
        ("complete", "likely_complete"),
        ("unknown", "unknown"),
        ("none", "unknown"),
    ):
        if any(str(coverage.get("likelyCompleteness") or "") == candidate for coverage in usable):
            likely_completeness = normalized_value
            break

    providers_attempted = _merge_list("providersAttempted")
    providers_succeeded = _merge_list("providersSucceeded")
    succeeded_markers = {repr(item) for item in providers_succeeded}
    providers_failed = [item for item in _merge_list("providersFailed") if repr(item) not in succeeded_markers]
    failed_markers = {repr(item) for item in providers_failed}
    providers_zero_results = [
        item
        for item in _merge_list("providersZeroResults")
        if repr(item) not in succeeded_markers and repr(item) not in failed_markers
    ]

    merged: dict[str, Any] = {
        "providersAttempted": providers_attempted,
        "providersSucceeded": providers_succeeded,
        "providersFailed": providers_failed,
        "providersZeroResults": providers_zero_results,
        "likelyCompleteness": likely_completeness,
        "searchMode": "guided_hybrid_research",
        "retrievalNotes": _merge_list("retrievalNotes"),
    }
    primary_document_coverage = usable[0].get("primaryDocumentCoverage")
    for coverage in usable:
        if coverage.get("primaryDocumentCoverage"):
            primary_document_coverage = coverage.get("primaryDocumentCoverage")
            break
    if primary_document_coverage is not None:
        merged["primaryDocumentCoverage"] = primary_document_coverage
    merged["summaryLine"] = (
        f"{len(merged['providersAttempted'])} provider(s) searched across blended literature and regulatory passes, "
        f"{len(merged['providersFailed'])} failed, {len(merged['providersZeroResults'])} returned zero results, "
        f"likely completeness: {likely_completeness}."
    )
    return merged




def _guided_merge_failure_summaries(*summaries: dict[str, Any] | None) -> dict[str, Any] | None:
    usable = [summary for summary in summaries if isinstance(summary, dict)]
    if not usable:
        return None
    if len(usable) == 1:
        return dict(usable[0])
    what_failed = (
        "; ".join(
            str(summary.get("whatFailed") or "").strip()
            for summary in usable
            if str(summary.get("whatFailed") or "").strip()
        )
        or None
    )
    what_still_worked_parts = [
        str(summary.get("whatStillWorked") or "").strip()
        for summary in usable
        if str(summary.get("whatStillWorked") or "").strip()
    ]
    completeness_parts = [
        str(summary.get("completenessImpact") or "").strip()
        for summary in usable
        if str(summary.get("completenessImpact") or "").strip()
    ]
    return {
        "outcome": "fallback_success" if any(summary.get("fallbackAttempted") for summary in usable) else "no_failure",
        "whatFailed": what_failed,
        "whatStillWorked": " ".join(dict.fromkeys(what_still_worked_parts)) or None,
        "fallbackAttempted": any(bool(summary.get("fallbackAttempted")) for summary in usable),
        "fallbackMode": "guided_hybrid_research",
        "primaryPathFailureReason": "; ".join(
            str(summary.get("primaryPathFailureReason") or "").strip()
            for summary in usable
            if str(summary.get("primaryPathFailureReason") or "").strip()
        )
        or None,
        "completenessImpact": " ".join(dict.fromkeys(completeness_parts)) or None,
        "recommendedNextAction": "review_partial_results",
    }
