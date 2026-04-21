"""Query specificity, definitional-query, and ambiguity estimators.

These helpers feed routing decisions (known-item vs broad-concept, how
ambiguous the intent set is, whether the prompt asks for a definition) and are
consumed by the planner orchestrator plus downstream dispatch/expansion code.
"""

from __future__ import annotations

from typing import Literal

from ...citation_repair import looks_like_citation_query
from ..models import IntentCandidate, PlannerQueryType
from .constants import (
    _DEFINITIONAL_PATTERNS,
    QUERYISH_TITLE_BLOCKERS,
)
from .normalization import looks_like_exact_title, normalize_query, query_terms
from .regulatory import _strong_known_item_signal, _strong_regulatory_signal


def _confidence_rank(confidence: Literal["high", "medium", "low"]) -> int:
    return {"high": 3, "medium": 2, "low": 1}[confidence]


def _query_starts_broad(query: str) -> bool:
    lowered = normalize_query(query).lower()
    return lowered.startswith(("what ", "which ", "how ", "compare ", "summarize ", "identify ", "find "))


def _is_definitional_query(query: str) -> bool:
    """Return True when the query asks for a definition, overview, or primer.

    Used to bias retrieval and ranking toward canonical/foundational papers and
    survey articles when the user is seeking conceptual grounding rather than a
    specific result.
    """

    normalized = normalize_query(query or "").lower()
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in _DEFINITIONAL_PATTERNS)


def _looks_broad_concept_query(
    *,
    normalized_query: str,
    focus: str | None,
    year: str | None,
    venue: str | None,
    terms: list[str] | None = None,
) -> bool:
    terms = terms if terms is not None else query_terms(normalized_query)
    has_constraints = bool(focus or year or venue)
    queryish_term_count = sum(term in QUERYISH_TITLE_BLOCKERS for term in terms)
    if _query_starts_broad(normalized_query) and len(terms) >= 6 and not has_constraints:
        return True
    if queryish_term_count >= 3 and len(terms) >= 6:
        return True
    if queryish_term_count >= 2 and len(terms) >= 8 and not has_constraints:
        return True
    return False


def _estimate_query_specificity(
    *,
    normalized_query: str,
    focus: str | None,
    year: str | None,
    venue: str | None,
    planner_query_type: PlannerQueryType | None = None,
    planner_specificity: Literal["high", "medium", "low"] | None = None,
) -> Literal["high", "medium", "low"]:
    """Estimate how specific a query is.

    When the LLM-authored ``planner_query_type`` or ``planner_specificity`` are
    provided we prefer them over raw text heuristics. The title/citation
    regex-based "high" promotion is suppressed whenever the LLM signalled a
    broad-concept query or already chose ``low`` specificity — this avoids the
    "long conceptual question happens to look title-like" false-positive that
    used to force those queries into known-item recovery.
    """
    terms = query_terms(normalized_query)
    broad_concept_signal = _looks_broad_concept_query(
        normalized_query=normalized_query,
        focus=focus,
        year=year,
        venue=venue,
        terms=terms,
    )
    if _strong_known_item_signal(normalized_query) or _strong_regulatory_signal(normalized_query, focus):
        return "high"
    llm_disagrees_with_title_heuristic = planner_query_type == "broad_concept" or planner_specificity == "low"
    if (
        not broad_concept_signal
        and not llm_disagrees_with_title_heuristic
        and (looks_like_exact_title(normalized_query) or looks_like_citation_query(normalized_query))
    ):
        return "high"
    has_constraints = bool(focus or year or venue)
    if broad_concept_signal:
        return "low"
    if planner_specificity is not None:
        if planner_specificity == "low":
            return "low"
        if planner_specificity == "high":
            return "high"
    if has_constraints and len(terms) <= 5:
        return "high"
    if _query_starts_broad(normalized_query) and len(terms) >= 6:
        return "low"
    return "medium"


def _estimate_ambiguity_level(
    *,
    candidates: list[IntentCandidate],
    routing_confidence: Literal["high", "medium", "low"],
    query_specificity: Literal["high", "medium", "low"],
) -> Literal["low", "medium", "high"]:
    if routing_confidence == "low":
        return "high"
    if len(candidates) < 2:
        return "medium" if query_specificity == "low" else "low"
    primary_rank = _confidence_rank(candidates[0].confidence)
    secondary_rank = _confidence_rank(candidates[1].confidence)
    if secondary_rank >= primary_rank:
        return "high"
    if primary_rank - secondary_rank == 1:
        return "high" if query_specificity == "low" else "medium"
    if query_specificity == "low" and secondary_rank >= 1:
        return "medium"
    return "low"


__all__ = [
    "_confidence_rank",
    "_estimate_ambiguity_level",
    "_estimate_query_specificity",
    "_is_definitional_query",
    "_looks_broad_concept_query",
    "_query_starts_broad",
]
