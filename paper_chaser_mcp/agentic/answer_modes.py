"""Follow-up answer-mode classification and evidence-use planning.

This module centralizes the semantics that keep grounded follow-up answers
honest. It exposes:

* ``ANSWER_MODES`` -- the canonical set of follow-up answer modes.
* ``classify_question_mode`` -- deterministic keyword classifier.
* ``build_evidence_use_plan`` -- LLM-signal-aware synthesis-plan builder.
* ``evidence_pool_is_weak`` -- guard that flags pools dominated by weak_match /
  off_topic classifications.

The implementation reuses the LLM relevance classifications produced inside
``ask_result_set`` (via ``provider_bundle.aclassify_relevance_batch``) rather
than making a separate LLM call, so the evidence-sufficiency signals we rely on
carry real model provenance (with ``fallback=True`` when the LLM was
unavailable).
"""

from __future__ import annotations

import os
from typing import Any, Literal

from .models import EvidenceItem, StructuredSourceRecord

ANSWER_MODES: tuple[str, ...] = (
    "metadata",
    "relevance_triage",
    "comparison",
    "mechanism_summary",
    "regulatory_chain",
    "intervention_tradeoff",
    "unknown",
)

# Modes that require synthesis over multiple on-topic sources. A polished answer
# in any of these modes without corroborating non-fallback evidence is exactly
# the failure we are trying to prevent.
SYNTHESIS_MODES: frozenset[str] = frozenset(
    {"comparison", "mechanism_summary", "regulatory_chain", "intervention_tradeoff"}
)

# Modes where deterministic salvage over saved session state is acceptable when
# the LLM is unavailable.
SALVAGEABLE_MODES: frozenset[str] = frozenset({"metadata", "relevance_triage"})

_WEAK_POOL_THRESHOLD_ENV = "PAPER_CHASER_WEAK_EVIDENCE_POOL_THRESHOLD"
_DEFAULT_WEAK_POOL_THRESHOLD = 0.6

# Minimum number of non-fallback on_topic sources required to support a
# synthesis-mode answer.
_SYNTHESIS_MIN_RESPONSIVE = 2


def _weak_pool_threshold() -> float:
    raw = os.environ.get(_WEAK_POOL_THRESHOLD_ENV, "").strip()
    if not raw:
        return _DEFAULT_WEAK_POOL_THRESHOLD
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_WEAK_POOL_THRESHOLD
    if not 0.0 < value <= 1.0:
        return _DEFAULT_WEAK_POOL_THRESHOLD
    return value


def classify_question_mode(question: str, session_metadata: dict[str, Any] | None = None) -> str:
    """Classify a follow-up question into one of ``ANSWER_MODES``.

    Deterministic keyword heuristic. Falls through to ``"unknown"`` when we
    cannot confidently route the question. ``session_metadata`` may carry a
    ``followUpMode`` hint from the planner.
    """
    lowered = (question or "").lower()
    metadata_facets = _metadata_facets(lowered)
    if _looks_like_relevance_triage(lowered):
        return "relevance_triage"
    if metadata_facets:
        return "metadata"
    if any(marker in lowered for marker in ("compare", "versus", " vs ", " vs.", "tradeoff", "trade-off")):
        if "trade" in lowered and "intervent" in lowered:
            return "intervention_tradeoff"
        return "comparison"
    if any(marker in lowered for marker in ("mechanism", "pathway", "causal", "how does", "why does")):
        return "mechanism_summary"
    if any(
        marker in lowered
        for marker in (
            "regulatory history",
            "rulemaking",
            "listing history",
            "timeline",
            "critical habitat",
            "final rule",
            "proposed rule",
        )
    ):
        return "regulatory_chain"
    if any(
        marker in lowered
        for marker in (
            "practical implication",
            "practical implications",
            "intervention tradeoff",
            "policy tradeoff",
            "cost vs",
            "benefit vs",
        )
    ):
        return "intervention_tradeoff"
    metadata_hint = (session_metadata or {}).get("followUpMode") if session_metadata else None
    if isinstance(metadata_hint, str):
        hint = metadata_hint.strip().lower()
        if hint in ANSWER_MODES:
            return hint
        if hint == "claim_check":
            # Historic alias -- treat claim-check prompts as mechanism summaries
            # so they route through synthesis-mode gating.
            return "mechanism_summary"
    return "unknown"


def _looks_like_relevance_triage(lowered_question: str) -> bool:
    markers = (
        "which of these",
        "which papers",
        "which sources",
        "are any of these",
        "which entries",
        "which results",
        "relevant to",
        "off-topic",
        "off topic",
    )
    return any(marker in lowered_question for marker in markers)


def _metadata_facets(lowered_question: str) -> list[str]:
    facets: list[str] = []
    if any(marker in lowered_question for marker in ("author", "authors", "who wrote", "written by")):
        facets.append("authors")
    if any(marker in lowered_question for marker in ("venue", "journal", "publisher", "published in")):
        facets.append("venue")
    if any(marker in lowered_question for marker in ("doi", "identifier")):
        facets.append("identifier")
    if any(marker in lowered_question for marker in ("publication year", "what year", "year published")):
        facets.append("year")
    if any(
        marker in lowered_question for marker in ("what records", "what sources", "which documents", "what documents")
    ):
        facets.append("inventory")
    return facets


def _is_on_topic(source: StructuredSourceRecord) -> bool:
    return getattr(source, "topical_relevance", None) == "on_topic"


def _is_weak_or_off_topic(source: StructuredSourceRecord) -> bool:
    label = getattr(source, "topical_relevance", None)
    return label in {"weak_match", "off_topic"}


def _classification_for(evidence_id: str, llm_relevance: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    if not llm_relevance or not evidence_id:
        return {}
    entry = llm_relevance.get(evidence_id)
    if isinstance(entry, dict):
        return entry
    return {}


def evidence_pool_is_weak(
    source_records: list[StructuredSourceRecord],
    *,
    threshold: float | None = None,
) -> bool:
    """Return True when the saved evidence pool is dominated by weak/off-topic hits.

    The threshold defaults to 60% and can be overridden via
    ``PAPER_CHASER_WEAK_EVIDENCE_POOL_THRESHOLD``.
    """
    if not source_records:
        return False
    effective_threshold = threshold if threshold is not None else _weak_pool_threshold()
    weak = sum(1 for record in source_records if _is_weak_or_off_topic(record))
    return (weak / len(source_records)) > effective_threshold


def build_evidence_use_plan(
    *,
    question: str,
    answer_mode: str,
    evidence: list[EvidenceItem],
    source_records: list[StructuredSourceRecord],
    unsupported_asks: list[str],
    llm_relevance: dict[str, dict[str, Any]] | None = None,
    question_mode: str | None = None,
) -> dict[str, Any]:
    """Build a synthesis plan with evidence-sufficiency signals.

    Produces a dict with the existing ``EvidenceUsePlan`` keys plus additive
    diagnostic fields used by ``ask_result_set``:

    * ``answerMode`` -- one of ``ANSWER_MODES``.
    * ``responsiveEvidenceIds`` -- ids classified as on_topic by the LLM
      relevance pass, excluding entries where the LLM call fell back to a
      deterministic heuristic.
    * ``unsupportedAspects`` -- plain-language gaps (echoed from
      ``unsupported_asks`` plus mode-specific diagnostics).
    * ``sufficient`` -- overall gate: ``True`` only when the mode's evidence
      threshold is satisfied and ``unsupported_asks`` is not large.
    * ``rationale`` -- short string describing why the plan is/isn't sufficient.
    * ``answerSubtype`` / ``directlyResponsiveIds`` /
      ``unsupportedComponents`` / ``retrievalSufficiency`` / ``confidence`` --
      back-compat keys for the existing ``EvidenceUsePlan`` model.
    * ``evidenceQualityProfile`` -- coarse label ``high``/``medium``/``low``.
    * ``synthesisMode`` -- coarse label suitable for confidence signals.
    """
    mode = question_mode or classify_question_mode(question)
    # Treat claim_check (historic answer_mode) as synthesis.
    treat_as_synthesis = mode in SYNTHESIS_MODES or answer_mode in {"claim_check", "comparison"}

    responsive_ids: list[str] = []
    fallback_only_on_topic = 0
    on_topic_count = 0
    for item, source in zip(evidence, source_records, strict=False):
        evidence_id = str(item.evidence_id or item.paper.paper_id or item.paper.canonical_id or "").strip()
        if not evidence_id:
            continue
        on_topic = _is_on_topic(source) and float(item.relevance_score) >= 0.25
        if not on_topic:
            continue
        on_topic_count += 1
        classification = _classification_for(evidence_id, llm_relevance)
        is_fallback = bool(classification.get("fallback"))
        if is_fallback:
            fallback_only_on_topic += 1
            continue
        responsive_ids.append(evidence_id)

    unsupported_components: list[str] = [component for component in (unsupported_asks or []) if str(component).strip()]

    sufficient = True
    retrieval_sufficiency: Literal["sufficient", "thin", "insufficient"] = "sufficient"
    confidence: Literal["high", "medium", "low"] = "medium"
    rationale = "Evidence pool supports the requested synthesis."

    if treat_as_synthesis:
        if len(responsive_ids) >= _SYNTHESIS_MIN_RESPONSIVE and len(unsupported_components) <= 1:
            retrieval_sufficiency = "sufficient"
            confidence = "high"
            rationale = f"{len(responsive_ids)} non-fallback on-topic sources support the requested {mode}."
        elif len(responsive_ids) == 1 and mode != "comparison" and not unsupported_components:
            retrieval_sufficiency = "thin"
            confidence = "low"
            sufficient = False
            rationale = "Only one non-fallback on-topic source is available; synthesis would lean on a single paper."
            unsupported_components.append(
                "Only one directly responsive source is available for this synthesis request."
            )
        else:
            retrieval_sufficiency = "insufficient"
            confidence = "low"
            sufficient = False
            if mode == "comparison":
                rationale = "Comparison requested, but fewer than two non-fallback on-topic sources were retrieved."
                unsupported_components.append("Comparison requires at least two directly responsive sources.")
            else:
                rationale = f"{mode} requested, but the saved evidence does not directly support the synthesis."
                unsupported_components.append("The saved evidence does not directly support the requested synthesis.")
            if fallback_only_on_topic:
                unsupported_components.append(
                    f"{fallback_only_on_topic} on-topic classification(s) relied on a deterministic "
                    "relevance fallback; treat them as leads rather than grounded support."
                )
    else:
        if mode == "metadata":
            retrieval_sufficiency = "sufficient" if source_records else "insufficient"
            confidence = "high" if source_records else "low"
            sufficient = bool(source_records)
            rationale = (
                "Metadata question; answerable from the saved session inventory."
                if source_records
                else "Metadata question with no saved sources to inspect."
            )
        elif mode == "relevance_triage":
            retrieval_sufficiency = "sufficient" if source_records else "insufficient"
            confidence = "medium" if source_records else "low"
            sufficient = bool(source_records)
            rationale = "Relevance triage can run against the saved record even without fresh synthesis."
        elif mode == "unknown":
            # Generic Q&A / synthesis questions we cannot confidently route.
            # Preserve the legacy behaviour: emit a permissive plan so the
            # downstream answer-status machinery (which has its own
            # ``should_abstain`` checks, coverage heuristics, and
            # unsupported-asks upgrades) keeps ownership of the decision.
            if on_topic_count >= 1:
                retrieval_sufficiency = "sufficient"
                confidence = "medium"
                sufficient = True
                rationale = (
                    "Unclassified follow-up with on-topic evidence; deferring to downstream answer-status heuristics."
                )
            else:
                retrieval_sufficiency = "thin"
                confidence = "low"
                sufficient = True
                rationale = (
                    "Unclassified follow-up with no on-topic evidence; "
                    "deferring to downstream answer-status heuristics."
                )

    # De-duplicate while preserving order.
    unsupported_components = list(dict.fromkeys(component for component in unsupported_components if component))

    quality_profile: Literal["high", "medium", "low"]
    if retrieval_sufficiency == "sufficient" and confidence == "high":
        quality_profile = "high"
    elif retrieval_sufficiency == "insufficient":
        quality_profile = "low"
    else:
        quality_profile = "medium"

    synthesis_label = (
        "grounded"
        if sufficient and retrieval_sufficiency == "sufficient"
        else "limited"
        if sufficient
        else "insufficient"
    )

    return {
        "answerMode": mode,
        "answerSubtype": mode,
        "responsiveEvidenceIds": responsive_ids,
        "directlyResponsiveIds": responsive_ids,
        "unsupportedAspects": unsupported_components,
        "unsupportedComponents": unsupported_components,
        "retrievalSufficiency": retrieval_sufficiency,
        "confidence": confidence,
        "sufficient": sufficient,
        "rationale": rationale,
        "evidenceQualityProfile": quality_profile,
        "synthesisMode": synthesis_label,
        "onTopicCount": on_topic_count,
        "fallbackOnlyOnTopic": fallback_only_on_topic,
    }


__all__ = [
    "ANSWER_MODES",
    "SYNTHESIS_MODES",
    "SALVAGEABLE_MODES",
    "classify_question_mode",
    "build_evidence_use_plan",
    "evidence_pool_is_weak",
]
