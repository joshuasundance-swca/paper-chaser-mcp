"""Phase 7b: ``agentic.graphs.research_graph`` submodule (graph expansion helpers).

Every private helper is reached via an explicit ``from`` import so the
``tests/test_test_seam_inventory.py`` firewall can see the seam — attribute
access on a module-handle import would silently bypass that guard.
"""

from __future__ import annotations

from paper_chaser_mcp.agentic.graphs import (
    _graph_frontier_scores as facade_graph_frontier_scores,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _filter_graph_frontier as core_filter_graph_frontier,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _graph_frontier_scores as core_graph_frontier_scores,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _graph_intent_text as core_graph_intent_text,
)
from paper_chaser_mcp.agentic.graphs.research_graph import (
    _filter_graph_frontier,
    _graph_frontier_scores,
    _graph_intent_text,
)


def test_research_graph_exports_filter_graph_frontier() -> None:
    assert callable(_filter_graph_frontier)


def test_research_graph_exports_graph_frontier_scores() -> None:
    assert callable(_graph_frontier_scores)


def test_research_graph_exports_graph_intent_text() -> None:
    assert callable(_graph_intent_text)


def test_research_graph_filter_graph_frontier_matches_core() -> None:
    assert _filter_graph_frontier is core_filter_graph_frontier


def test_research_graph_graph_frontier_scores_matches_core() -> None:
    assert _graph_frontier_scores is core_graph_frontier_scores


def test_research_graph_graph_intent_text_matches_core() -> None:
    assert _graph_intent_text is core_graph_intent_text


def test_filter_graph_frontier_empty() -> None:
    assert _filter_graph_frontier([]) == []


def test_filter_graph_frontier_uses_threshold() -> None:
    ranked = [
        ({"paperId": "a"}, 0.9),
        ({"paperId": "b"}, 0.5),
        ({"paperId": "c"}, 0.1),
    ]
    kept = _filter_graph_frontier(ranked)
    assert ranked[0] in kept
    assert ranked[2] not in kept


def test_graph_intent_text_prefers_normalized_query_from_record() -> None:
    class _Rec:
        metadata = {"strategyMetadata": {"normalizedQuery": "alpha beta"}}
        query = ""

    assert _graph_intent_text(_Rec(), []) == "alpha beta"


def test_graph_intent_text_falls_back_to_seed_title() -> None:
    assert _graph_intent_text(None, [{"title": "Paper Title"}]) == "Paper Title"


def test_facade_reexports_graph_frontier_scores() -> None:
    assert facade_graph_frontier_scores is _graph_frontier_scores
