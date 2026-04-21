"""Phase 7c-1: regulatory intent reconciliation helpers.

Extracted from ``planner/_core.py``. These helpers reconcile the LLM
planner's regulatory subintent emission with independent query-side
signals so downstream consumers don't force literature passes onto
regulation-only asks (or vice versa).
"""

from __future__ import annotations

from typing import cast

from ..models import PlannerDecision, RegulatoryIntentLabel
from .regulatory import (
    _infer_regulatory_subintent,
    detect_literature_intent,
    detect_regulatory_intent,
)

_VALID_REGULATORY_INTENTS: frozenset[str] = frozenset(
    {
        "current_cfr_text",
        "rulemaking_history",
        "species_dossier",
        "guidance_lookup",
        "hybrid_regulatory_plus_literature",
    },
)


def _has_literature_corroboration(
    *,
    planner: PlannerDecision,
    query: str,
    focus: str | None,
) -> bool:
    """Return True when the query-side signals support a hybrid regulatory+literature label.

    The LLM planner occasionally emits ``hybrid_regulatory_plus_literature``
    for regulation-only asks (e.g. "what does EPA require for stormwater
    discharges?"). Trusting that label alone causes the guided workflow to
    tack on an unnecessary literature review pass. This helper checks the
    independent keyword signal on the query, planner-emitted secondary
    intents, and retrieval hypotheses so downstream consumers can demand
    corroboration before honoring the hybrid route.

    Kept in sync with ``dispatch._guided_should_add_review_pass`` so valid
    hybrid labels are not stripped at planner-time only to be accepted at
    dispatch-time (or vice versa).
    """
    if detect_literature_intent(query, focus):
        return True
    for secondary in planner.secondary_intents:
        label = str(secondary).strip().lower()
        if label in {"review", "literature"}:
            return True
    literature_hypothesis_markers = (
        "literature",
        "peer-review",
        "peer review",
        "peer-reviewed",
        "systematic review",
        "meta-analysis",
        "hybrid_policy_science",
    )
    for hypothesis in planner.retrieval_hypotheses:
        lowered = str(hypothesis).lower()
        if any(marker in lowered for marker in literature_hypothesis_markers):
            return True
    return False


def _derive_regulatory_intent(
    *,
    planner: PlannerDecision,
    query: str,
    focus: str | None,
) -> RegulatoryIntentLabel | None:
    """Map LLM planner signals + deterministic cues to a canonical regulatory intent.

    LLM-first: the planner's own ``regulatory_subintent`` (set from its
    ``queryType``/``retrievalHypotheses``/``intent`` fields when available) is
    the preferred source. Deterministic heuristics only fire when the planner
    did not emit a canonical label. When the query is clearly regulatory but
    none of the specific labels match, returns ``"unspecified"``. Returns
    ``None`` for non-regulatory queries.
    """

    existing = planner.regulatory_subintent
    if existing and existing in _VALID_REGULATORY_INTENTS:
        if existing == "hybrid_regulatory_plus_literature" and not _has_literature_corroboration(
            planner=planner, query=query, focus=focus
        ):
            # Fall through to deterministic inference when the LLM emits a
            # hybrid label without any query-side literature cue.
            pass
        else:
            return cast(RegulatoryIntentLabel, existing)

    is_regulatory = planner.intent == "regulatory" or detect_regulatory_intent(query, focus)
    if not is_regulatory:
        return None

    # Consider hybrid literature + regulatory cues from retrieval hypotheses
    # and search angles the LLM already produced.
    llm_signal_text = " ".join(
        str(entry).lower()
        for bucket in (planner.retrieval_hypotheses, planner.search_angles, planner.candidate_concepts)
        for entry in bucket
    )
    if any(
        marker in llm_signal_text
        for marker in (
            "policy and literature",
            "regulatory and literature",
            "scientific and regulatory",
            "hybrid",
        )
    ) and detect_literature_intent(query, focus):
        return "hybrid_regulatory_plus_literature"

    inferred = _infer_regulatory_subintent(query, focus)
    if inferred and inferred in _VALID_REGULATORY_INTENTS:
        return cast(RegulatoryIntentLabel, inferred)
    return "unspecified"
