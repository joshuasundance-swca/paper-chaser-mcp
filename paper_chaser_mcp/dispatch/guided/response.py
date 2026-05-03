"""Guided response finalization helpers (Phase 3 extraction)."""

from __future__ import annotations

import logging
from typing import Any

from ...guided_semantic import (
    build_evidence_records,
    build_routing_decision,
    classify_answerability,
)
from .._core import (
    _COMPACT_NULL_OK_FIELDS,
    _FOLLOW_UP_COMPACT_FIELDS,
    _LEGACY_GUIDED_FIELDS,
    _LLM_ANSWERABILITY_MAP,
    _RESEARCH_COMPACT_FIELDS,
    _apply_follow_up_response_mode,
    _compact_suppressed_source_rationales,
    _compact_suppressed_source_summaries,
)
from .trust import _guided_confidence_signals, _guided_follow_up_status

logger = logging.getLogger(__name__)


def _guided_compact_response_if_needed(*, tool_name: str, response: dict[str, Any]) -> dict[str, Any]:
    compact_fields: set[str] | None = None
    if tool_name == "follow_up_research":
        answer_status = str(response.get("answerStatus") or "").strip()
        if answer_status == "insufficient_evidence":
            compact_fields = _FOLLOW_UP_COMPACT_FIELDS
    elif tool_name == "research":
        status = str(response.get("status") or response.get("resultStatus") or "").strip()
        if status == "abstained":
            compact_fields = _RESEARCH_COMPACT_FIELDS

    if compact_fields is None:
        return response

    compacted = {key: value for key, value in response.items() if key in compact_fields}
    for key in list(compacted):
        if key in _COMPACT_NULL_OK_FIELDS:
            continue
        value = compacted[key]
        if value is None or value == [] or value == {}:
            compacted.pop(key, None)
    if any(key in response for key in ("evidence", "leads", *sorted(_LEGACY_GUIDED_FIELDS))):
        compacted["sourcesSuppressed"] = True
    suppressed_rationales = _compact_suppressed_source_rationales(response)
    if suppressed_rationales:
        compacted["suppressedSourceRationales"] = suppressed_rationales
    suppressed_summaries = _compact_suppressed_source_summaries(response)
    if suppressed_summaries:
        compacted["suppressedSourceSummaries"] = suppressed_summaries
    compacted["legacyFieldsIncluded"] = False
    return compacted


def _guided_finalize_response(
    *,
    tool_name: str,
    response: dict[str, Any],
    response_mode: str = "standard",
    include_legacy_fields: bool = False,
) -> dict[str, Any]:
    finalized = dict(response)
    for key in ("verifiedFindings", "unverifiedLeads"):
        if finalized.get(key) == []:
            finalized.pop(key, None)
    finalized = _guided_compact_response_if_needed(tool_name=tool_name, response=finalized)
    if tool_name == "follow_up_research" and "sourcesSuppressed" not in finalized:
        finalized = _apply_follow_up_response_mode(
            finalized,
            response_mode=response_mode,
            include_legacy_fields=include_legacy_fields,
        )
    elif tool_name == "research" and not include_legacy_fields:
        for key in _LEGACY_GUIDED_FIELDS:
            finalized.pop(key, None)
        finalized["legacyFieldsIncluded"] = False
    if "legacyFieldsIncluded" not in finalized:
        finalized["legacyFieldsIncluded"] = any(key in finalized for key in _LEGACY_GUIDED_FIELDS)
    return finalized


async def _guided_contract_fields(
    *,
    query: str,
    intent: str,
    status: str,
    sources: list[dict[str, Any]],
    unverified_leads: list[dict[str, Any]],
    evidence_gaps: list[str],
    coverage_summary: dict[str, Any] | None,
    strategy_metadata: dict[str, Any] | None,
    timeline: dict[str, Any] | None = None,
    pass_modes: list[str] | None = None,
    review_pass_reason: str | None = None,
    answer_text: str = "",
    provider_bundle: Any | None = None,
) -> dict[str, Any]:
    evidence, leads = build_evidence_records(sources=sources, leads=unverified_leads)
    routing_summary = build_routing_decision(
        query=query,
        intent=intent,
        strategy_metadata=strategy_metadata,
        coverage_summary=coverage_summary,
    ).model_dump(by_alias=True)
    result_status = _guided_follow_up_status(status)

    # P0-1 Fix #5: compute the evidence quality profile (same derivation as
    # ``_guided_confidence_signals``) so the answerability ladder and the
    # confidence signals agree. When the profile is ``low`` the ladder must
    # not advertise ``grounded``.
    _verified_on_topic_primary = sum(
        1
        for item in sources
        if item.get("topicalRelevance") == "on_topic" and item.get("verificationStatus") == "verified_primary_source"
    )
    _verified_on_topic = sum(
        1
        for item in sources
        if item.get("topicalRelevance") == "on_topic"
        and item.get("verificationStatus") in {"verified_primary_source", "verified_metadata"}
    )
    if _verified_on_topic_primary > 0 or _verified_on_topic >= 3:
        _evidence_quality_profile = "high"
    elif _verified_on_topic > 0:
        _evidence_quality_profile = "medium"
    else:
        _evidence_quality_profile = "low"

    answerability = classify_answerability(
        status=status,
        evidence=evidence,
        leads=leads,
        evidence_gaps=evidence_gaps,
        answer_text=answer_text,
        evidence_quality_profile=_evidence_quality_profile,
    )

    if provider_bundle is not None and answer_text and answerability == "grounded":
        try:
            validation = await provider_bundle.avalidate_answer_status(
                query=query,
                answer_text=answer_text,
                evidence_count=len(evidence),
            )
            if validation is not None:
                llm_mapped = _LLM_ANSWERABILITY_MAP.get(validation.classification)
                if llm_mapped and llm_mapped != answerability:
                    answerability = llm_mapped
        except Exception:
            logger.debug("LLM answer status validation failed; using deterministic result.")

    metadata_for_routing = strategy_metadata or {}
    subject_card_payload: dict[str, Any] | None = None
    raw_subject_card = metadata_for_routing.get("subjectCard")
    if isinstance(raw_subject_card, dict) and raw_subject_card:
        subject_card_payload = dict(raw_subject_card)
    elif raw_subject_card is not None and hasattr(raw_subject_card, "model_dump"):
        try:
            dumped = raw_subject_card.model_dump(by_alias=True, exclude_none=True)  # type: ignore[union-attr]
        except Exception:  # pragma: no cover - defensive
            dumped = {}
        if dumped:
            subject_card_payload = dumped
    subject_chain_gaps_payload = [
        str(item).strip() for item in metadata_for_routing.get("subjectChainGaps") or [] if str(item).strip()
    ]
    intent_family_payload = str(metadata_for_routing.get("intentFamily") or "").strip() or None
    regulatory_intent_payload = str(metadata_for_routing.get("regulatoryIntent") or "").strip() or None

    return {
        "resultStatus": result_status,
        "answerability": answerability,
        "routingSummary": {
            "intent": routing_summary["intent"],
            "decisionConfidence": routing_summary["confidence"],
            "rationale": routing_summary["rationale"],
            "anchorType": ((routing_summary.get("anchor") or {}).get("anchorType")),
            "anchorValue": ((routing_summary.get("anchor") or {}).get("anchorValue")),
            "querySpecificity": routing_summary.get("querySpecificity"),
            "ambiguityLevel": routing_summary.get("ambiguityLevel"),
            "secondaryIntents": list(routing_summary.get("secondaryIntents") or []),
            "retrievalHypotheses": list(routing_summary.get("retrievalHypotheses") or []),
            "providerPlan": list((routing_summary.get("providerPlan") or {}).get("providers") or []),
            "requiredPrimarySources": list(((routing_summary.get("anchor") or {}).get("requiredPrimarySources") or [])),
            "successCriteria": list(((routing_summary.get("anchor") or {}).get("successCriteria") or [])),
            "passModes": list(pass_modes or []),
            "reviewPassReason": review_pass_reason,
            "providersAttempted": list((coverage_summary or {}).get("providersAttempted") or []),
            "providersMatched": list((coverage_summary or {}).get("providersSucceeded") or []),
            "providersFailed": list((coverage_summary or {}).get("providersFailed") or []),
            "providersNotAttempted": [
                provider
                for provider in list((routing_summary.get("providerPlan") or {}).get("providers") or [])
                if provider not in list((coverage_summary or {}).get("providersAttempted") or [])
            ],
            "whyPartial": evidence_gaps[0] if evidence_gaps and result_status != "succeeded" else None,
            # Phase 4/5 planner classification signals surfaced for agents
            # (additive — existing consumers that only read retrievalHypotheses
            # continue to work unchanged).
            "intentFamily": intent_family_payload,
            "regulatoryIntent": regulatory_intent_payload,
            "subjectCard": subject_card_payload,
            "subjectChainGaps": subject_chain_gaps_payload,
        },
        "coverageSummary": coverage_summary,
        "evidence": evidence,
        "leads": leads,
        "confidenceSignals": _guided_confidence_signals(
            status=result_status,
            sources=sources,
            evidence_gaps=evidence_gaps,
            subject_chain_gaps=subject_chain_gaps_payload,
        ),
        "timeline": timeline,
    }
