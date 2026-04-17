"""Behavioral regression tests for Workstream E (known-item resolution states).

These tests pin down the new ``knownItemResolutionState`` provenance label and
the tighter known-item gating introduced alongside it. They exercise three
surfaces:

* ``classify_known_item_resolution_state`` in ``citation_repair`` (pure logic).
* ``classify_query`` in ``agentic.planner`` (routing gate / demotion).
* The ``_estimate_query_specificity`` helper (LLM-signal awareness).
"""

from __future__ import annotations

import pytest

from paper_chaser_mcp.agentic.models import PlannerDecision
from paper_chaser_mcp.agentic.planner import (
    _estimate_query_specificity,
    classify_query,
)
from paper_chaser_mcp.citation_repair import classify_known_item_resolution_state


class _StubBundle:
    """Minimal planner bundle that echoes a preset :class:`PlannerDecision`."""

    def __init__(self, decision: PlannerDecision) -> None:
        self._decision = decision

    async def aplan_search(self, **kwargs: object) -> PlannerDecision:
        return self._decision


@pytest.mark.asyncio
async def test_broad_conceptual_query_is_not_force_routed_to_known_item() -> None:
    """Even if a long title-like query trips deterministic heuristics, the LLM
    planner's broad-concept signal should prevent routing into known-item
    recovery when no DOI/arXiv/URL anchor is present."""

    decision = PlannerDecision(
        intent="known_item",
        queryType="broad_concept",
        querySpecificity="low",
        ambiguityLevel="high",
        breadthEstimate=3,
        firstPassMode="mixed",
        candidateConcepts=["urban heat", "environmental justice"],
    )
    _, planner = await classify_query(
        query=(
            "how does disproportionate exposure to urban heat island intensity "
            "across major us cities compare with neighborhood income and tree cover"
        ),
        mode="auto",
        year=None,
        venue=None,
        focus="environmental justice literature",
        provider_bundle=_StubBundle(decision),  # type: ignore[arg-type]
    )

    assert planner.intent == "discovery"
    assert planner.intent_source == "heuristic_override"
    assert "broad conceptual" in planner.intent_rationale.lower()


@pytest.mark.asyncio
async def test_doi_query_still_forces_known_item() -> None:
    """Regression: a DOI query must continue to be routed to known_item even
    when the LLM planner disagrees — identifier signals are the ground truth."""

    decision = PlannerDecision(
        intent="discovery",
        queryType="broad_concept",
        querySpecificity="medium",
        ambiguityLevel="medium",
    )
    _, planner = await classify_query(
        query="10.1038/nature12373 sample paper lookup",
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_StubBundle(decision),  # type: ignore[arg-type]
    )

    assert planner.intent == "known_item"
    assert planner.intent_source in {"heuristic_override", "hybrid_agreement"}


def test_specificity_heuristic_defers_to_llm_broad_concept_signal() -> None:
    """``_estimate_query_specificity`` must not promote citation-like queries to
    'high' specificity when the LLM already labelled them as broad-concept."""

    without_hint = _estimate_query_specificity(
        normalized_query="smith et al 2019 transformer attention baseline",
        focus=None,
        year=None,
        venue=None,
    )
    with_hint = _estimate_query_specificity(
        normalized_query="smith et al 2019 transformer attention baseline",
        focus=None,
        year=None,
        venue=None,
        planner_query_type="broad_concept",
        planner_specificity="low",
    )

    assert without_hint == "high"
    assert with_hint == "low"


def test_resolved_exact_for_identifier_round_trip() -> None:
    state = classify_known_item_resolution_state(
        resolution_confidence="high",
        resolution_strategy="identifier",
        matched_fields=["identifier", "title", "author", "year"],
        conflicting_fields=[],
        title_similarity=0.98,
        year_delta=0,
        author_overlap=2,
        best_score=0.95,
        runner_up_score=0.4,
        candidate_count=1,
    )
    assert state == "resolved_exact"


def test_resolved_probable_for_year_conflict_on_strong_title() -> None:
    """Strong title similarity but a meaningful year disagreement must be
    ``resolved_probable`` rather than ``resolved_exact``."""

    state = classify_known_item_resolution_state(
        resolution_confidence="high",
        resolution_strategy="exact_title",
        matched_fields=["title", "author"],
        conflicting_fields=["year"],
        title_similarity=0.93,
        year_delta=3,
        author_overlap=1,
        best_score=0.78,
        runner_up_score=0.5,
        candidate_count=1,
    )
    assert state == "resolved_probable"

    # Mid-range confidence should also map to probable.
    medium = classify_known_item_resolution_state(
        resolution_confidence="medium",
        resolution_strategy="fuzzy_search",
        matched_fields=["title"],
        conflicting_fields=[],
        title_similarity=0.82,
        year_delta=None,
        author_overlap=0,
        best_score=0.7,
        runner_up_score=0.55,
        candidate_count=3,
    )
    assert medium == "resolved_probable"


def test_needs_disambiguation_for_near_tie_and_no_match() -> None:
    near_tie = classify_known_item_resolution_state(
        resolution_confidence="medium",
        resolution_strategy="fuzzy_search",
        matched_fields=["title"],
        conflicting_fields=[],
        title_similarity=0.8,
        year_delta=None,
        author_overlap=0,
        best_score=0.7,
        runner_up_score=0.68,
        candidate_count=4,
    )
    assert near_tie == "needs_disambiguation"

    no_match = classify_known_item_resolution_state(
        resolution_confidence="low",
        resolution_strategy="none",
        matched_fields=[],
        conflicting_fields=[],
        title_similarity=None,
        year_delta=None,
        author_overlap=None,
        best_score=None,
        runner_up_score=None,
        candidate_count=0,
        has_best_match=False,
    )
    assert no_match == "needs_disambiguation"
