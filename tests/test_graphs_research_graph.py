"""Phase 7b: ``agentic.graphs.research_graph`` submodule (graph expansion helpers)."""

from __future__ import annotations

import pytest

from paper_chaser_mcp.agentic.graphs import _core as core_module
from paper_chaser_mcp.agentic.graphs import research_graph


@pytest.mark.parametrize(
    "name",
    [
        "_filter_graph_frontier",
        "_graph_frontier_scores",
        "_graph_intent_text",
    ],
)
def test_research_graph_exports_helper(name: str) -> None:
    assert getattr(research_graph, name) is not None


def test_research_graph_helpers_match_core_legacy_bindings() -> None:
    assert research_graph._graph_frontier_scores is core_module._graph_frontier_scores
    assert research_graph._graph_intent_text is core_module._graph_intent_text
    assert research_graph._filter_graph_frontier is core_module._filter_graph_frontier


def test_filter_graph_frontier_empty() -> None:
    assert research_graph._filter_graph_frontier([]) == []


def test_filter_graph_frontier_uses_threshold() -> None:
    ranked = [
        ({"paperId": "a"}, 0.9),
        ({"paperId": "b"}, 0.5),
        ({"paperId": "c"}, 0.1),
    ]
    kept = research_graph._filter_graph_frontier(ranked)
    assert ranked[0] in kept
    assert ranked[2] not in kept


def test_graph_intent_text_prefers_normalized_query_from_record() -> None:
    class _Rec:
        metadata = {"strategyMetadata": {"normalizedQuery": "alpha beta"}}
        query = ""

    assert research_graph._graph_intent_text(_Rec(), []) == "alpha beta"


def test_graph_intent_text_falls_back_to_seed_title() -> None:
    assert (
        research_graph._graph_intent_text(None, [{"title": "Paper Title"}])
        == "Paper Title"
    )


def test_facade_reexports_graph_frontier_scores() -> None:
    from paper_chaser_mcp.agentic.graphs import _graph_frontier_scores as facade_fn

    assert facade_fn is research_graph._graph_frontier_scores
