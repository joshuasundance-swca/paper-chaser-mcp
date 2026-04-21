"""Phase 7c-1: retrieval hypothesis generation and intent-candidate bookkeeping.

Extracted from :mod:`paper_chaser_mcp.agentic.planner._core` without behavioural
changes. Holds the helpers that translate a :class:`PlannerDecision` plus the
user query into initial :class:`ExpansionCandidate` seeds and that maintain the
working list of :class:`IntentCandidate` rows used during classification.
"""

from __future__ import annotations

from typing import Literal

from ..config import AgenticConfig
from ..models import (
    RETRIEVAL_MODE_MIXED,
    RETRIEVAL_MODE_TARGETED,
    ExpansionCandidate,
    IntentCandidate,
    IntentLabel,
    PlannerDecision,
)
from .normalization import normalize_query
from .specificity import _confidence_rank, _is_definitional_query


def _source_for_intent_candidate(
    intent_source: Literal["explicit", "planner", "heuristic_override", "hybrid_agreement", "fallback_recovery"],
) -> Literal["explicit", "planner", "heuristic", "hybrid", "fallback"]:
    mapping: dict[
        Literal["explicit", "planner", "heuristic_override", "hybrid_agreement", "fallback_recovery"],
        Literal["explicit", "planner", "heuristic", "hybrid", "fallback"],
    ] = {
        "explicit": "explicit",
        "planner": "planner",
        "heuristic_override": "heuristic",
        "hybrid_agreement": "hybrid",
        "fallback_recovery": "fallback",
    }
    return mapping[intent_source]


def _ordered_provider_plan(base_plan: list[str], preferred_order: list[str]) -> list[str]:
    ordered = [provider for provider in preferred_order if provider in base_plan]
    ordered.extend(provider for provider in base_plan if provider not in ordered)
    return ordered


def initial_retrieval_hypotheses(
    *,
    normalized_query: str,
    focus: str | None,
    planner: PlannerDecision,
    config: AgenticConfig,
) -> list[ExpansionCandidate]:
    base_query = normalize_query(" ".join(part for part in [normalized_query, focus or ""] if part))
    if not base_query:
        base_query = normalized_query
    base_plan = list(planner.provider_plan)
    candidates: list[ExpansionCandidate] = [
        ExpansionCandidate(
            variant=base_query,
            source="from_input",
            rationale="Literal user query.",
            providerPlan=base_plan,
        )
    ]
    if planner.intent in {"known_item", "author", "citation", "regulatory"}:
        return candidates
    if planner.first_pass_mode == RETRIEVAL_MODE_TARGETED:
        return candidates

    planner_angles = [str(angle).strip() for angle in planner.search_angles if str(angle).strip()]
    if _is_definitional_query(normalized_query):
        definitional_angles = [f"{base_query} survey", f"{base_query} foundational paper"]
        planner_angles = definitional_angles + [angle for angle in planner_angles if angle not in definitional_angles]
    if not planner_angles:
        return candidates

    max_extra = max(config.max_initial_hypotheses - 1, 0)
    if planner.first_pass_mode == RETRIEVAL_MODE_MIXED:
        planner_angles = planner_angles[: max_extra or 1]
    else:
        planner_angles = planner_angles[:max_extra]

    seen_variants: set[str] = {base_query.lower()}
    preferred_plans = [
        _ordered_provider_plan(base_plan, ["openalex", "semantic_scholar", "core", "arxiv", "scholarapi"]),
        _ordered_provider_plan(base_plan, ["semantic_scholar", "openalex", "scholarapi", "core", "arxiv"]),
        _ordered_provider_plan(base_plan, ["core", "openalex", "semantic_scholar", "arxiv", "scholarapi"]),
        _ordered_provider_plan(base_plan, ["arxiv", "core", "openalex", "semantic_scholar", "scholarapi"]),
    ]
    for index, angle in enumerate(planner_angles):
        lowered = normalize_query(angle).lower()
        if not lowered or lowered in seen_variants:
            continue
        seen_variants.add(lowered)
        candidates.append(
            ExpansionCandidate(
                variant=angle,
                source="hypothesis",
                rationale="Planner-generated retrieval angle.",
                providerPlan=preferred_plans[min(index, len(preferred_plans) - 1)],
            )
        )
    return candidates[: max(config.max_initial_hypotheses, 1)]


def _upsert_intent_candidate(
    *,
    candidates: list[IntentCandidate],
    intent: IntentLabel,
    confidence: Literal["high", "medium", "low"],
    source: Literal["explicit", "planner", "heuristic", "hybrid", "fallback"],
    rationale: str,
) -> None:
    for index, existing in enumerate(candidates):
        if existing.intent != intent:
            continue
        merged_confidence = (
            confidence if _confidence_rank(confidence) >= _confidence_rank(existing.confidence) else existing.confidence
        )
        merged_source = existing.source
        if _confidence_rank(confidence) >= _confidence_rank(existing.confidence):
            merged_source = source
        merged_rationale = existing.rationale
        if rationale and rationale not in merged_rationale:
            merged_rationale = f"{merged_rationale} {rationale}".strip() if merged_rationale else rationale
        candidates[index] = existing.model_copy(
            update={
                "confidence": merged_confidence,
                "source": merged_source,
                "rationale": merged_rationale,
            }
        )
        return
    candidates.append(
        IntentCandidate(
            intent=intent,
            confidence=confidence,
            source=source,
            rationale=rationale,
        )
    )


def _sort_intent_candidates(
    candidates: list[IntentCandidate],
    *,
    preferred_intent: IntentLabel,
) -> list[IntentCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.intent != preferred_intent,
            -_confidence_rank(candidate.confidence),
            candidate.intent,
        ),
    )
