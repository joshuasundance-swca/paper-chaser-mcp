"""Phase 7c-1: tests for the extracted ``planner.hypotheses`` submodule."""

from __future__ import annotations

from paper_chaser_mcp.agentic.config import AgenticConfig
from paper_chaser_mcp.agentic.models import IntentCandidate, PlannerDecision
from paper_chaser_mcp.agentic.planner._core import (
    _ordered_provider_plan,
    _sort_intent_candidates,
    _source_for_intent_candidate,
    _upsert_intent_candidate,
    initial_retrieval_hypotheses,
)
from paper_chaser_mcp.agentic.planner.hypotheses import (
    _ordered_provider_plan as _hypotheses_ordered_provider_plan,
)
from paper_chaser_mcp.agentic.planner.hypotheses import (
    _sort_intent_candidates as _hypotheses_sort_intent_candidates,
)
from paper_chaser_mcp.agentic.planner.hypotheses import (
    _source_for_intent_candidate as _hypotheses_source_for_intent_candidate,
)
from paper_chaser_mcp.agentic.planner.hypotheses import (
    _upsert_intent_candidate as _hypotheses_upsert_intent_candidate,
)
from paper_chaser_mcp.agentic.planner.hypotheses import (
    initial_retrieval_hypotheses as _hypotheses_initial_retrieval_hypotheses,
)


def _make_config(max_initial_hypotheses: int = 3) -> AgenticConfig:
    return AgenticConfig(
        enabled=True,
        provider="deterministic",
        planner_model="deterministic",
        synthesis_model="deterministic",
        embedding_model="deterministic",
        index_backend="memory",
        session_ttl_seconds=1800,
        enable_trace_log=False,
        max_initial_hypotheses=max_initial_hypotheses,
    )


def _make_planner(
    *,
    intent: str = "discovery",
    search_angles: list[str] | None = None,
    provider_plan: list[str] | None = None,
    first_pass_mode: str = "broad",
) -> PlannerDecision:
    return PlannerDecision(
        intent=intent,  # type: ignore[arg-type]
        searchAngles=search_angles or [],
        providerPlan=provider_plan or ["openalex", "semantic_scholar"],  # type: ignore[arg-type]
        firstPassMode=first_pass_mode,  # type: ignore[arg-type]
    )


def test_source_for_intent_candidate_maps_known_sources() -> None:
    assert _source_for_intent_candidate("explicit") == "explicit"
    assert _source_for_intent_candidate("planner") == "planner"
    assert _source_for_intent_candidate("heuristic_override") == "heuristic"
    assert _source_for_intent_candidate("hybrid_agreement") == "hybrid"
    assert _source_for_intent_candidate("fallback_recovery") == "fallback"


def test_ordered_provider_plan_prefers_preferred_order_then_rest() -> None:
    result = _ordered_provider_plan(
        ["arxiv", "openalex", "core"],
        ["openalex", "semantic_scholar", "core"],
    )
    assert result == ["openalex", "core", "arxiv"]


def test_initial_retrieval_hypotheses_targeted_mode_returns_literal_only() -> None:
    planner = _make_planner(
        search_angles=["related angle"],
        first_pass_mode="targeted",
    )
    candidates = initial_retrieval_hypotheses(
        normalized_query="foo",
        focus=None,
        planner=planner,
        config=_make_config(),
    )
    assert len(candidates) == 1
    assert candidates[0].variant == "foo"
    assert candidates[0].source == "from_input"


def test_initial_retrieval_hypotheses_expands_with_planner_angles() -> None:
    planner = _make_planner(
        search_angles=["alpha beta", "gamma delta"],
        first_pass_mode="broad",
    )
    candidates = initial_retrieval_hypotheses(
        normalized_query="research topic",
        focus=None,
        planner=planner,
        config=_make_config(max_initial_hypotheses=3),
    )
    assert candidates[0].source == "from_input"
    assert any(c.source == "hypothesis" for c in candidates)
    assert len(candidates) <= 3


def test_initial_retrieval_hypotheses_known_item_stays_literal() -> None:
    planner = _make_planner(
        intent="known_item",
        search_angles=["x", "y"],
    )
    candidates = initial_retrieval_hypotheses(
        normalized_query="exact paper title",
        focus=None,
        planner=planner,
        config=_make_config(),
    )
    assert len(candidates) == 1


def test_upsert_intent_candidate_inserts_new() -> None:
    candidates: list[IntentCandidate] = []
    _upsert_intent_candidate(
        candidates=candidates,
        intent="discovery",
        confidence="high",
        source="planner",
        rationale="rationale",
    )
    assert len(candidates) == 1
    assert candidates[0].intent == "discovery"
    assert candidates[0].confidence == "high"


def test_upsert_intent_candidate_merges_existing_with_higher_confidence() -> None:
    candidates: list[IntentCandidate] = [
        IntentCandidate(intent="review", confidence="low", source="heuristic", rationale="first"),
    ]
    _upsert_intent_candidate(
        candidates=candidates,
        intent="review",
        confidence="high",
        source="planner",
        rationale="second",
    )
    assert len(candidates) == 1
    assert candidates[0].confidence == "high"
    assert candidates[0].source == "planner"
    assert "second" in candidates[0].rationale


def test_sort_intent_candidates_preferred_first() -> None:
    items = [
        IntentCandidate(intent="review", confidence="high", source="planner", rationale=""),
        IntentCandidate(intent="discovery", confidence="medium", source="planner", rationale=""),
    ]
    sorted_items = _sort_intent_candidates(items, preferred_intent="discovery")
    assert sorted_items[0].intent == "discovery"


def test_core_symbols_identity_matches_hypotheses() -> None:
    assert _source_for_intent_candidate is _hypotheses_source_for_intent_candidate
    assert _ordered_provider_plan is _hypotheses_ordered_provider_plan
    assert initial_retrieval_hypotheses is _hypotheses_initial_retrieval_hypotheses
    assert _upsert_intent_candidate is _hypotheses_upsert_intent_candidate
    assert _sort_intent_candidates is _hypotheses_sort_intent_candidates
