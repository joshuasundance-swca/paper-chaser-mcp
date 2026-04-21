"""Phase 2 Track C amendment 1: explicit facade allowlist.

The :mod:`paper_chaser_mcp.dispatch` package previously mirrored every
non-dunder attribute from ``paper_chaser_mcp.dispatch._core`` onto the package
namespace via ``for name in dir(_core)``. That loop also copied accidental
internals — ``re``, ``time``, ``logging``, ``typing.Any``, and the imported
argument-model classes — onto the public-facing facade, creating a large and
undisciplined surface. Tests that reached into that surface could become
brittle, and the facade silently grew whenever ``_core`` added an import.

The Phase 2c amendment replaces the mirror loop with an explicit frozen
allowlist. This test pins both halves of the contract:

* accidentally-leaked internals are *not* accessible as
  ``paper_chaser_mcp.dispatch.<name>``,
* the nine hard seam-map private helpers plus the ``dispatch_tool`` public
  symbol are still accessible, and
* every test-only helper currently referenced via ``dispatch_module.X`` in the
  repo remains reachable (no silent regression).
"""

from __future__ import annotations

import importlib

import pytest

import paper_chaser_mcp.dispatch as dispatch_module

SEAM_MAP_PUBLIC_SYMBOLS: tuple[str, ...] = (
    "dispatch_tool",
    "_answer_follow_up_from_session_state",
    "_authoritative_but_weak_source_ids",
    "_cursor_to_offset",
    "_guided_citation_from_paper",
    "_guided_contract_fields",
    "_guided_session_state",
    "_guided_source_record_from_paper",
    "_guided_trust_summary",
)


# Internals that the old mirror loop accidentally surfaced. None of these
# should be reachable from the dispatch package namespace.
LEAKED_INTERNAL_NAMES: tuple[str, ...] = (
    "re",
    "time",
    "logging",
    "Any",
    "Callable",
    "Literal",
    "cast",
    "PackageNotFoundError",
    "package_version",
    "_sys",
    "_ModuleType",
    "_DispatchPackageModule",
    "annotations",
    "SearchProvider",
    "PaperEnrichmentService",
    "CitationFormat",
    "RuntimeSummary",
    "ResearchArgs",
    "TOOL_INPUT_MODELS",
    "ScholarApiError",
    "SerpApiKeyMissingError",
)


# Test-only helpers that tests reach via ``dispatch_module.X``. Sourced from a
# grep of ``tests/`` at the time of the amendment. Any additions/removals
# should be intentional and mirrored in ``paper_chaser_mcp/dispatch/__init__.py``.
TEST_REFERENCED_HELPERS: tuple[str, ...] = (
    "_build_provider_diagnostics_snapshot",
    "_compose_why_classified_weak_match",
    "_direct_read_recommendation_details",
    "_direct_read_recommendations",
    "_evidence_quality_detail",
    "_guided_abstention_details_payload",
    "_guided_best_next_internal_action",
    "_guided_confidence_signals",
    "_guided_deterministic_evidence_gaps",
    "_guided_failure_summary",
    "_guided_machine_failure_payload",
    "_guided_merge_coverage_summaries",
    "_guided_next_actions",
    "_guided_normalize_follow_up_arguments",
    "_guided_normalize_inspect_arguments",
    "_guided_result_meaning",
    "_guided_result_state",
    "_guided_saved_session_topicality",
    "_guided_source_metadata_answers",
    "_guided_source_record_from_structured_source",
    "_guided_sources_from_fr_documents",
    "_guided_summary",
    "_paper_topical_relevance",
    "_synthesis_path",
    "_topical_relevance_from_signals",
    "compute_topical_relevance",
)


def test_facade_has_explicit_allowlist_attribute() -> None:
    """The package must expose a frozen ``_FACADE_EXPORTS`` tuple."""
    assert hasattr(dispatch_module, "_FACADE_EXPORTS"), (
        "paper_chaser_mcp.dispatch must publish an explicit _FACADE_EXPORTS allowlist "
        "so the facade surface is diff-visible and cannot drift silently."
    )
    exports = dispatch_module._FACADE_EXPORTS
    assert isinstance(exports, tuple), "_FACADE_EXPORTS must be a tuple for immutability."
    assert len(exports) == len(set(exports)), "_FACADE_EXPORTS must not contain duplicates."


@pytest.mark.parametrize("symbol", SEAM_MAP_PUBLIC_SYMBOLS)
def test_seam_map_public_symbols_remain_accessible(symbol: str) -> None:
    """The nine seam-map public symbols must still resolve on the facade."""
    getattr(dispatch_module, symbol)  # must not raise
    assert symbol in dispatch_module._FACADE_EXPORTS


@pytest.mark.parametrize("symbol", TEST_REFERENCED_HELPERS)
def test_test_referenced_helpers_remain_accessible(symbol: str) -> None:
    """Every helper reached via dispatch_module.X in tests stays reachable."""
    getattr(dispatch_module, symbol)  # must not raise
    assert symbol in dispatch_module._FACADE_EXPORTS


@pytest.mark.parametrize("symbol", LEAKED_INTERNAL_NAMES)
def test_leaked_internals_are_not_accessible(symbol: str) -> None:
    """Accidental internals from the old mirror loop must not be reachable."""
    # The facade must not re-expose module-level imports like ``re`` or the
    # typing aliases. These were previously accessible only because the
    # ``for name in dir(_core)`` loop copied *everything*.
    assert not hasattr(dispatch_module, symbol), (
        f"paper_chaser_mcp.dispatch.{symbol} is reachable. This is an accidental leak from "
        f"the old mirror loop and must be removed from the explicit _FACADE_EXPORTS allowlist."
    )


def test_facade_matches_allowlist_exactly() -> None:
    """The public-facing attribute surface of the package must equal the allowlist.

    ``dir(paper_chaser_mcp.dispatch)`` — filtered to names the facade itself
    owns (i.e. ignoring dunder attributes and submodules imported through
    normal package mechanics) — must match ``_FACADE_EXPORTS`` exactly. This
    makes any new export a visible diff in ``dispatch/__init__.py``.
    """
    # Submodule names become package attributes whenever Python imports them
    # (either directly or transitively). They are not part of the facade
    # contract, so we exclude them from the comparison. ``_core`` is the
    # canonical submodule; the helper submodules are re-exported through the
    # allowlist. We treat module-type attributes as submodules.
    import types

    facade_attrs = {
        name
        for name in dir(dispatch_module)
        if not name.startswith("__") and not isinstance(getattr(dispatch_module, name), types.ModuleType)
    }
    allowlisted = set(dispatch_module._FACADE_EXPORTS) | {"_FACADE_EXPORTS"}
    unexpected = facade_attrs - allowlisted
    assert not unexpected, (
        f"paper_chaser_mcp.dispatch exposes unexpected attributes: {sorted(unexpected)}. "
        f"Either add them to _FACADE_EXPORTS intentionally or remove the leak."
    )
    missing = allowlisted - facade_attrs - {"_FACADE_EXPORTS"}
    assert not missing, (
        f"Allowlist declares symbols that are not reachable on the facade: {sorted(missing)}. "
        f"Either fix the re-export in __init__.py or drop the entry from _FACADE_EXPORTS."
    )


def test_public_api_surface_test_still_passes_for_dispatch() -> None:
    """The hard public symbol (``dispatch_tool``) from the seam map must exist.

    This duplicates part of ``tests/test_public_api_surface.py`` on purpose:
    it keeps the Phase 2c amendment self-contained and makes it obvious that
    tightening the facade allowlist must not regress the Phase 1 surface pin.
    """
    module = importlib.import_module("paper_chaser_mcp.dispatch")
    assert callable(module.dispatch_tool)
