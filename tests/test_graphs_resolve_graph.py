"""Phase 7b: ``agentic.graphs.resolve_graph`` submodule (known-item helpers).

This test pins the direct-import surface of the ``resolve_graph`` submodule
and the identity-equality seam between the submodule-level helper and its
legacy binding on ``_core``. Any re-export through the facade must keep
``facade is submodule`` to avoid the Phase 6 ``hasattr``-based seam bug.
"""

from __future__ import annotations

import pytest

from paper_chaser_mcp.agentic.graphs import _core as core_module
from paper_chaser_mcp.agentic.graphs import resolve_graph


@pytest.mark.parametrize(
    "name",
    [
        "_anchor_strength_for_resolution",
        "_known_item_recovery_warning",
        "_known_item_resolution_queries",
        "_known_item_resolution_state_for_strategy",
        "_known_item_title_similarity",
        "_normalization_metadata",
    ],
)
def test_resolve_graph_exports_helper(name: str) -> None:
    helper = getattr(resolve_graph, name)
    assert callable(helper)


def test_resolve_graph_helpers_match_core_legacy_bindings() -> None:
    assert resolve_graph._known_item_title_similarity is core_module._known_item_title_similarity
    assert resolve_graph._known_item_resolution_queries is core_module._known_item_resolution_queries
    assert resolve_graph._normalization_metadata is core_module._normalization_metadata
    assert resolve_graph._anchor_strength_for_resolution is core_module._anchor_strength_for_resolution
    assert (
        resolve_graph._known_item_resolution_state_for_strategy
        is core_module._known_item_resolution_state_for_strategy
    )
    assert resolve_graph._known_item_recovery_warning is core_module._known_item_recovery_warning


def test_known_item_title_similarity_basic() -> None:
    assert resolve_graph._known_item_title_similarity(
        "Attention Is All You Need",
        "Attention Is All You Need",
    ) >= 0.95
    assert (
        resolve_graph._known_item_title_similarity("", "anything") == 0.0
    )


def test_anchor_strength_for_resolution_bands() -> None:
    assert resolve_graph._anchor_strength_for_resolution("citation_resolution") == "high"
    assert resolve_graph._anchor_strength_for_resolution("semantic_title_match") == "medium"
    assert resolve_graph._anchor_strength_for_resolution("other") == "low"


def test_known_item_recovery_warning_strategies() -> None:
    assert "semantic title match" in resolve_graph._known_item_recovery_warning(
        "semantic_title_match"
    )
    assert "OpenAlex autocomplete" in resolve_graph._known_item_recovery_warning(
        "openalex_autocomplete"
    )
    assert "OpenAlex search" in resolve_graph._known_item_recovery_warning(
        "openalex_search"
    )
    # Unknown strategy falls through to the generic title-style recovery message.
    assert "title-style recovery" in resolve_graph._known_item_recovery_warning("other")


def test_normalization_metadata_noop_when_equal() -> None:
    warnings, repaired = resolve_graph._normalization_metadata("same", "same")
    assert warnings == []
    assert repaired == {}


def test_normalization_metadata_reports_change() -> None:
    warnings, repaired = resolve_graph._normalization_metadata("Raw Query", "raw query")
    assert warnings
    assert repaired == {"query": {"from": "Raw Query", "to": "raw query"}}
