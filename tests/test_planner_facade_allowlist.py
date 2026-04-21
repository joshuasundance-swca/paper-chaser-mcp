"""Phase 7c-1 amendment: explicit facade allowlist for ``agentic.planner``.

Mirrors ``tests/test_graphs_facade_allowlist.py``. Phase 6 turned the flat
``agentic.planner`` module into a package and re-exported every historical
symbol from the new submodules. Without an explicit allowlist the facade
surface grows silently whenever a submodule gains a new public name, which
is the same implicit-coupling problem the Phase 2 dispatch allowlist and
the Phase 7a graphs allowlist were introduced to prevent.

This test freezes ``paper_chaser_mcp.agentic.planner`` the same way:

* every ``_FACADE_EXPORTS`` entry must resolve via ``getattr`` on the package,
* the allowlist must equal a hard-coded expected set (no silent growth),
* public symbols pinned by ``test_public_api_surface`` stay reachable,
* private test-only helpers that tests from-import through the facade stay
  reachable,
* submodules (``_core``, ``constants``, ``hypotheses``, ...) leak as
  ``types.ModuleType`` via normal package loading and are filtered out.

Any later-phase change to the planner surface must update both this test
and ``paper_chaser_mcp/agentic/planner/__init__.py`` in the same commit.
"""

from __future__ import annotations

import importlib
import types

import pytest

import paper_chaser_mcp.agentic.planner as planner_module
from paper_chaser_mcp.agentic.planner import _FACADE_EXPORTS

# Hard-coded expected allowlist. This is the full, frozen set of names that
# ``paper_chaser_mcp.agentic.planner`` may expose through the facade. Keep it
# sorted alphabetically and mirrored with the ``from .submodule import``
# blocks and ``_FACADE_EXPORTS`` tuple in
# ``paper_chaser_mcp/agentic/planner/__init__.py``.
EXPECTED_FACADE_EXPORTS: frozenset[str] = frozenset(
    {
        "AGENCY_REGULATORY_MARKERS",
        "ARXIV_RE",
        "DOI_RE",
        "FACET_SPLIT_RE",
        "GENERIC_EVIDENCE_WORDS",
        "HYPOTHESIS_QUERY_STOPWORDS",
        "LITERATURE_QUERY_TERMS",
        "QUERYISH_TITLE_BLOCKERS",
        "QUERY_FACET_TOKEN_ALLOWLIST",
        "REGULATORY_QUERY_TERMS",
        "STRONG_REGULATORY_TITLE_BLOCKERS",
        "TITLE_STOPWORDS",
        "VARIANT_DEDUPE_STOPWORDS",
        "_CULTURAL_RESOURCE_MARKERS",
        "_DEFINITIONAL_PATTERNS",
        "_VALID_REGULATORY_INTENTS",
        "_confidence_rank",
        "_derive_regulatory_intent",
        "_detect_cultural_resource_intent",
        "_estimate_ambiguity_level",
        "_estimate_query_specificity",
        "_has_literature_corroboration",
        "_infer_entity_card",
        "_infer_regulatory_subintent",
        "_is_definitional_query",
        "_looks_broad_concept_query",
        "_ordered_provider_plan",
        "_query_starts_broad",
        "_signatures_are_near_duplicates",
        "_sort_intent_candidates",
        "_source_for_intent_candidate",
        "_strong_known_item_signal",
        "_strong_regulatory_signal",
        "_top_evidence_phrases",
        "_upsert_intent_candidate",
        "_variant_signature",
        "classify_query",
        "combine_variants",
        "dedupe_variants",
        "detect_literature_intent",
        "detect_regulatory_intent",
        "grounded_expansion_candidates",
        "initial_retrieval_hypotheses",
        "looks_like_citation_query",
        "looks_like_exact_title",
        "looks_like_near_known_item_query",
        "looks_like_url",
        "normalize_query",
        "query_facets",
        "query_terms",
        "speculative_expansion_candidates",
    }
)


# Public symbols pinned by ``tests/test_public_api_surface.py`` for the
# ``paper_chaser_mcp.agentic.planner`` entry. Kept as a parameter list so a
# future public addition is a visible diff in both places.
PUBLIC_SYMBOLS: tuple[str, ...] = (
    "classify_query",
    "detect_literature_intent",
    "detect_regulatory_intent",
)


# Test-only private helpers that tests reach via
# ``from paper_chaser_mcp.agentic.planner import _X``. Sourced from a grep of
# ``tests/`` at Phase 7c-1 amendment time. Any additions/removals here must
# be mirrored in ``_FACADE_EXPORTS`` and in
# ``tests/test_test_seam_inventory.py::KNOWN_TEST_SEAMS``.
TEST_REFERENCED_HELPERS: tuple[str, ...] = (
    "_CULTURAL_RESOURCE_MARKERS",
    "_DEFINITIONAL_PATTERNS",
    "_detect_cultural_resource_intent",
    "_estimate_ambiguity_level",
    "_estimate_query_specificity",
    "_has_literature_corroboration",
    "_infer_entity_card",
    "_infer_regulatory_subintent",
    "_is_definitional_query",
    "_looks_broad_concept_query",
    "_query_starts_broad",
    "_strong_known_item_signal",
    "_strong_regulatory_signal",
    "_top_evidence_phrases",
)


def test_facade_exports_is_frozen_tuple() -> None:
    """``_FACADE_EXPORTS`` must be a tuple and match the expected set exactly."""
    assert isinstance(_FACADE_EXPORTS, tuple), "_FACADE_EXPORTS must be a tuple so the allowlist is immutable."
    assert len(_FACADE_EXPORTS) == len(set(_FACADE_EXPORTS)), "_FACADE_EXPORTS must not contain duplicate entries."
    assert set(_FACADE_EXPORTS) == EXPECTED_FACADE_EXPORTS, (
        "paper_chaser_mcp.agentic.planner._FACADE_EXPORTS drifted from the "
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
    assert list(planner_module.__all__) == list(_FACADE_EXPORTS), (
        "planner.__all__ must mirror _FACADE_EXPORTS exactly so the public "
        "star-import surface and the pinned allowlist cannot diverge."
    )


@pytest.mark.parametrize("symbol", sorted(EXPECTED_FACADE_EXPORTS))
def test_every_allowlisted_name_is_resolvable(symbol: str) -> None:
    """Each ``_FACADE_EXPORTS`` entry must be reachable via ``getattr``."""
    resolved = getattr(planner_module, symbol)
    assert resolved is not None
    assert symbol in _FACADE_EXPORTS


@pytest.mark.parametrize("symbol", PUBLIC_SYMBOLS)
def test_public_symbols_remain_accessible(symbol: str) -> None:
    """Public symbols pinned by the api-surface guard must stay reachable."""
    getattr(planner_module, symbol)
    assert symbol in _FACADE_EXPORTS


@pytest.mark.parametrize("symbol", TEST_REFERENCED_HELPERS)
def test_test_referenced_helpers_remain_accessible(symbol: str) -> None:
    """Every helper reached via ``planner_module.X`` in tests stays reachable."""
    getattr(planner_module, symbol)
    assert symbol in _FACADE_EXPORTS


def test_facade_matches_allowlist_exactly() -> None:
    """The public attribute surface of the package must equal the allowlist.

    ``dir(paper_chaser_mcp.agentic.planner)`` - filtered to exclude dunder
    names and submodules imported through normal package mechanics - must
    match ``_FACADE_EXPORTS`` exactly. Any new attribute becomes a visible
    diff in ``paper_chaser_mcp/agentic/planner/__init__.py``.
    """
    facade_attrs = {
        name
        for name in dir(planner_module)
        if not name.startswith("__") and not isinstance(getattr(planner_module, name), types.ModuleType)
    }
    allowlisted = set(_FACADE_EXPORTS) | {"_FACADE_EXPORTS"}
    unexpected = facade_attrs - allowlisted
    assert not unexpected, (
        f"paper_chaser_mcp.agentic.planner exposes unexpected attributes: "
        f"{sorted(unexpected)}. Either add them to _FACADE_EXPORTS intentionally "
        f"or remove the leak."
    )
    missing = allowlisted - facade_attrs - {"_FACADE_EXPORTS"}
    assert not missing, (
        f"Allowlist declares symbols not reachable on the facade: "
        f"{sorted(missing)}. Either fix the re-export in __init__.py or drop "
        f"the entry from _FACADE_EXPORTS."
    )


def test_public_api_surface_still_exposes_classify_query() -> None:
    """The hard public ``classify_query`` symbol must remain reachable.

    This duplicates part of ``tests/test_public_api_surface.py`` on purpose:
    it keeps the Phase 7c-1 amendment self-contained and makes it obvious
    that tightening the facade allowlist must not regress the Phase 1
    surface pin.
    """
    module = importlib.import_module("paper_chaser_mcp.agentic.planner")
    assert module.classify_query is not None
