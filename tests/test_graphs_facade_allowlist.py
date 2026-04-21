"""Phase 7a amendment: explicit facade allowlist for ``agentic.graphs``.

The Phase 7a rubber-duck review flagged that ``paper_chaser_mcp.agentic.graphs``
previously advertised only ``AgenticRuntime`` and
``TopicalRelevanceClassification`` via ``__all__``, while in-tree tests still
reached private helpers like ``_build_grounded_comparison_answer`` through the
facade. Those imports worked only because ``_core``'s bindings happened to be
re-exported, not because the facade intentionally surfaced them. That kind of
implicit coupling is exactly what Phase 2's dispatch allowlist was introduced
to prevent.

This test freezes the graphs facade the same way
``tests/test_dispatch_facade_allowlist.py`` freezes the dispatch facade:

* every ``_FACADE_EXPORTS`` entry must resolve via ``getattr`` on the package,
* the allowlist must equal a hard-coded expected set (no silent growth),
* private test-only helpers that tests from-import through the facade stay
  reachable.

Any Phase 7b/7c change to the graphs surface must update both this test and
``paper_chaser_mcp/agentic/graphs/__init__.py`` in the same commit.
"""

from __future__ import annotations

import importlib
import types

import pytest

import paper_chaser_mcp.agentic.graphs as graphs_module
from paper_chaser_mcp.agentic.graphs import _FACADE_EXPORTS

# Hard-coded expected allowlist. This is the full, frozen set of names that
# ``paper_chaser_mcp.agentic.graphs`` may expose through the facade. Keep it
# sorted alphabetically and mirrored with the ``from ._core import`` block
# and ``_FACADE_EXPORTS`` tuple in ``paper_chaser_mcp/agentic/graphs/__init__.py``.
EXPECTED_FACADE_EXPORTS: frozenset[str] = frozenset(
    {
        "AgenticRuntime",
        "TopicalRelevanceClassification",
        "_build_grounded_comparison_answer",
        "_classify_topical_relevance_with_provenance",
        "_derive_regulatory_query_flags",
        "_ecos_query_variants",
        "_finalize_theme_label",
        "_graph_frontier_scores",
        "_has_inspectable_sources",
        "_has_on_topic_sources",
        "_is_agency_guidance_query",
        "_is_current_cfr_text_request",
        "_is_opaque_query",
        "_query_requests_regulatory_history",
        "_rank_ecos_variant_hits",
        "_rank_regulatory_documents",
        "_source_record_from_regulatory_document",
    }
)


# Hard public symbol pinned by the public-api-surface guard. Kept as a
# parameter list so a future public addition is a visible diff.
PUBLIC_SYMBOLS: tuple[str, ...] = (
    "AgenticRuntime",
    "TopicalRelevanceClassification",
)


# Test-only private helpers that tests reach via
# ``from paper_chaser_mcp.agentic.graphs import _X``. Sourced from a grep of
# ``tests/`` at Phase 7a amendment time. Any additions/removals here must be
# mirrored in ``_FACADE_EXPORTS`` and in
# ``tests/test_test_seam_inventory.py::KNOWN_TEST_SEAMS``.
TEST_REFERENCED_HELPERS: tuple[str, ...] = (
    "_build_grounded_comparison_answer",
    "_classify_topical_relevance_with_provenance",
    "_derive_regulatory_query_flags",
    "_ecos_query_variants",
    "_finalize_theme_label",
    "_graph_frontier_scores",
    "_has_inspectable_sources",
    "_has_on_topic_sources",
    "_is_agency_guidance_query",
    "_is_current_cfr_text_request",
    "_is_opaque_query",
    "_query_requests_regulatory_history",
    "_rank_ecos_variant_hits",
    "_rank_regulatory_documents",
    "_source_record_from_regulatory_document",
)


def test_facade_exports_is_frozen_tuple() -> None:
    """``_FACADE_EXPORTS`` must be a tuple and match the expected set exactly."""
    assert isinstance(_FACADE_EXPORTS, tuple), "_FACADE_EXPORTS must be a tuple so the allowlist is immutable."
    assert len(_FACADE_EXPORTS) == len(set(_FACADE_EXPORTS)), "_FACADE_EXPORTS must not contain duplicate entries."
    assert set(_FACADE_EXPORTS) == EXPECTED_FACADE_EXPORTS, (
        "paper_chaser_mcp.agentic.graphs._FACADE_EXPORTS drifted from the "
        "frozen expected set. Either update EXPECTED_FACADE_EXPORTS "
        "intentionally (and mirror in KNOWN_TEST_SEAMS) or fix the regression."
    )


def test_facade_exports_length_is_pinned() -> None:
    """The allowlist size is frozen to catch accidental growth in the same diff."""
    assert len(_FACADE_EXPORTS) == len(EXPECTED_FACADE_EXPORTS)


def test_facade_exports_is_sorted_alphabetically() -> None:
    """Keep ``_FACADE_EXPORTS`` sorted alphabetically for review diff discipline."""
    assert list(_FACADE_EXPORTS) == sorted(_FACADE_EXPORTS), (
        "_FACADE_EXPORTS must remain sorted alphabetically so additions land at "
        "the insertion point and reviews stay focused on intent."
    )


def test_all_matches_facade_exports() -> None:
    """``__all__`` must equal the allowlist so ``from ... import *`` stays honest."""
    assert list(graphs_module.__all__) == list(_FACADE_EXPORTS), (
        "graphs.__all__ must mirror _FACADE_EXPORTS exactly so the public "
        "star-import surface and the pinned allowlist cannot diverge."
    )


@pytest.mark.parametrize("symbol", sorted(EXPECTED_FACADE_EXPORTS))
def test_every_allowlisted_name_is_resolvable(symbol: str) -> None:
    """Each ``_FACADE_EXPORTS`` entry must be reachable via ``getattr``."""
    resolved = getattr(graphs_module, symbol)
    assert resolved is not None
    assert symbol in _FACADE_EXPORTS


@pytest.mark.parametrize("symbol", PUBLIC_SYMBOLS)
def test_public_symbols_remain_accessible(symbol: str) -> None:
    """Public symbols pinned by the api-surface guard must stay reachable."""
    getattr(graphs_module, symbol)
    assert symbol in _FACADE_EXPORTS


@pytest.mark.parametrize("symbol", TEST_REFERENCED_HELPERS)
def test_test_referenced_helpers_remain_accessible(symbol: str) -> None:
    """Every helper reached via ``graphs_module.X`` in tests stays reachable."""
    getattr(graphs_module, symbol)
    assert symbol in _FACADE_EXPORTS


def test_facade_matches_allowlist_exactly() -> None:
    """The public attribute surface of the package must equal the allowlist.

    ``dir(paper_chaser_mcp.agentic.graphs)`` — filtered to exclude dunder
    names and submodules imported through normal package mechanics — must
    match ``_FACADE_EXPORTS`` exactly. Any new attribute becomes a visible
    diff in ``paper_chaser_mcp/agentic/graphs/__init__.py``.
    """
    facade_attrs = {
        name
        for name in dir(graphs_module)
        if not name.startswith("__") and not isinstance(getattr(graphs_module, name), types.ModuleType)
    }
    allowlisted = set(_FACADE_EXPORTS) | {"_FACADE_EXPORTS"}
    unexpected = facade_attrs - allowlisted
    assert not unexpected, (
        f"paper_chaser_mcp.agentic.graphs exposes unexpected attributes: "
        f"{sorted(unexpected)}. Either add them to _FACADE_EXPORTS intentionally "
        f"or remove the leak."
    )
    missing = allowlisted - facade_attrs - {"_FACADE_EXPORTS"}
    assert not missing, (
        f"Allowlist declares symbols not reachable on the facade: "
        f"{sorted(missing)}. Either fix the re-export in __init__.py or drop "
        f"the entry from _FACADE_EXPORTS."
    )


def test_public_api_surface_still_exposes_agentic_runtime() -> None:
    """The hard public ``AgenticRuntime`` symbol must remain reachable.

    This duplicates part of ``tests/test_public_api_surface.py`` on purpose:
    it keeps the Phase 7a amendment self-contained and makes it obvious that
    tightening the facade allowlist must not regress the Phase 1 surface pin.
    """
    module = importlib.import_module("paper_chaser_mcp.agentic.graphs")
    assert module.AgenticRuntime is not None
