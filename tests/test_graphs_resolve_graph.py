"""Phase 7b: ``agentic.graphs.resolve_graph`` submodule (known-item helpers).

This test pins the direct-import surface of the ``resolve_graph`` submodule
and the identity-equality seam between the submodule-level helper and its
legacy binding on ``_core``. Each private helper is reached via an explicit
``from`` import so the ``tests/test_test_seam_inventory.py`` firewall can see
the seam — attribute access on a module-handle import would silently bypass
that guard (the Phase 6 ``hasattr``-based regression).
"""

from __future__ import annotations

from paper_chaser_mcp.agentic.graphs._core import (
    _anchor_strength_for_resolution as core_anchor_strength_for_resolution,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _known_item_recovery_warning as core_known_item_recovery_warning,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _known_item_resolution_queries as core_known_item_resolution_queries,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _known_item_resolution_state_for_strategy as core_known_item_resolution_state_for_strategy,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _known_item_title_similarity as core_known_item_title_similarity,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _normalization_metadata as core_normalization_metadata,
)
from paper_chaser_mcp.agentic.graphs.resolve_graph import (
    _anchor_strength_for_resolution,
    _known_item_recovery_warning,
    _known_item_resolution_queries,
    _known_item_resolution_state_for_strategy,
    _known_item_title_similarity,
    _normalization_metadata,
)


def test_resolve_graph_exports_anchor_strength_for_resolution() -> None:
    assert callable(_anchor_strength_for_resolution)


def test_resolve_graph_exports_known_item_recovery_warning() -> None:
    assert callable(_known_item_recovery_warning)


def test_resolve_graph_exports_known_item_resolution_queries() -> None:
    assert callable(_known_item_resolution_queries)


def test_resolve_graph_exports_known_item_resolution_state_for_strategy() -> None:
    assert callable(_known_item_resolution_state_for_strategy)


def test_resolve_graph_exports_known_item_title_similarity() -> None:
    assert callable(_known_item_title_similarity)


def test_resolve_graph_exports_normalization_metadata() -> None:
    assert callable(_normalization_metadata)


def test_resolve_graph_anchor_strength_for_resolution_matches_core() -> None:
    assert _anchor_strength_for_resolution is core_anchor_strength_for_resolution


def test_resolve_graph_known_item_recovery_warning_matches_core() -> None:
    assert _known_item_recovery_warning is core_known_item_recovery_warning


def test_resolve_graph_known_item_resolution_queries_matches_core() -> None:
    assert _known_item_resolution_queries is core_known_item_resolution_queries


def test_resolve_graph_known_item_resolution_state_for_strategy_matches_core() -> None:
    assert _known_item_resolution_state_for_strategy is core_known_item_resolution_state_for_strategy


def test_resolve_graph_known_item_title_similarity_matches_core() -> None:
    assert _known_item_title_similarity is core_known_item_title_similarity


def test_resolve_graph_normalization_metadata_matches_core() -> None:
    assert _normalization_metadata is core_normalization_metadata


def test_known_item_title_similarity_basic() -> None:
    assert (
        _known_item_title_similarity(
            "Attention Is All You Need",
            "Attention Is All You Need",
        )
        >= 0.95
    )
    assert _known_item_title_similarity("", "anything") == 0.0


def test_anchor_strength_for_resolution_bands() -> None:
    assert _anchor_strength_for_resolution("citation_resolution") == "high"
    assert _anchor_strength_for_resolution("semantic_title_match") == "medium"
    assert _anchor_strength_for_resolution("other") == "low"


def test_known_item_recovery_warning_strategies() -> None:
    assert "semantic title match" in _known_item_recovery_warning("semantic_title_match")
    assert "OpenAlex autocomplete" in _known_item_recovery_warning("openalex_autocomplete")
    assert "OpenAlex search" in _known_item_recovery_warning("openalex_search")
    # Unknown strategy falls through to the generic title-style recovery message.
    assert "title-style recovery" in _known_item_recovery_warning("other")


def test_normalization_metadata_noop_when_equal() -> None:
    warnings, repaired = _normalization_metadata("same", "same")
    assert warnings == []
    assert repaired == {}


def test_normalization_metadata_reports_change() -> None:
    warnings, repaired = _normalization_metadata("Raw Query", "raw query")
    assert warnings
    assert repaired == {"query": {"from": "Raw Query", "to": "raw query"}}
