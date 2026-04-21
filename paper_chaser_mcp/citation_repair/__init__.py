"""Citation-repair package facade.

Phase 2 Track B converted ``paper_chaser_mcp/citation_repair.py`` into a package
so its scoring, extraction, and orchestration helpers can be split into focused
submodules without disturbing the public import surface. The original monolith
now lives in :mod:`paper_chaser_mcp.citation_repair._core`; this facade
re-exports a curated, diff-visible set of symbols.

The ``__init__.py`` uses an **explicit** ``_FACADE_EXPORTS`` allowlist (mirroring
the Phase 2 Track C amendment that replaced the ``for name in dir(_core)``
mirror in :mod:`paper_chaser_mcp.dispatch`). This keeps the facade surface
auditable in review diffs and prevents accidental internals (``re``, ``logging``,
``SequenceMatcher`` aliases, etc.) from leaking out of ``_core``.

External callers continue to import from ``paper_chaser_mcp.citation_repair``
directly. The Phase 1 public-api-surface pin lists ``parse_citation``,
``resolve_citation``, ``looks_like_citation_query``, and
``looks_like_paper_identifier`` as the hard public symbols. Test-facing private
helpers remain importable and tracked in ``_FACADE_EXPORTS`` so any intentional
change is visible.
"""

from __future__ import annotations

import sys as _sys
from types import ModuleType as _ModuleType

from . import _core as _core
from ._core import (
    ParsedCitation as ParsedCitation,
)
from ._core import (
    RankedCitationCandidate as RankedCitationCandidate,
)
from ._core import (
    _classify_resolution_confidence as _classify_resolution_confidence,
)
from ._core import (
    _filtered_alternative_candidates as _filtered_alternative_candidates,
)
from ._core import (
    _normalize_identifier_for_openalex as _normalize_identifier_for_openalex,
)
from ._core import (
    _normalize_identifier_for_semantic_scholar as _normalize_identifier_for_semantic_scholar,
)
from ._core import (
    _rank_candidate as _rank_candidate,
)
from ._core import (
    _serialize_citation_response as _serialize_citation_response,
)
from ._core import (
    _sparse_search_queries as _sparse_search_queries,
)
from ._core import (
    _title_similarity as _title_similarity,
)
from ._core import (
    _venue_hint_in_text as _venue_hint_in_text,
)
from ._core import (
    build_match_metadata as build_match_metadata,
)
from ._core import (
    classify_known_item_resolution_state as classify_known_item_resolution_state,
)
from ._core import (
    looks_like_citation_query as looks_like_citation_query,
)
from ._core import (
    looks_like_paper_identifier as looks_like_paper_identifier,
)
from ._core import (
    looks_like_url as looks_like_url,
)
from ._core import (
    normalize_citation_text as normalize_citation_text,
)
from ._core import (
    parse_citation as parse_citation,
)
from ._core import (
    resolve_citation as resolve_citation,
)

# ---------------------------------------------------------------------------
# Explicit facade allowlist.
# ---------------------------------------------------------------------------
#
# This tuple pins the *exact* set of names callers may reach through
# ``paper_chaser_mcp.citation_repair.<name>``. Adding an entry means
# acknowledging that production or test code depends on reaching the symbol
# via the facade; removing an entry is a deliberate surface shrink and must
# be paired with retargeting any consumer to the owning submodule.
#
# The public-api-surface guard (``tests/test_public_api_surface.py``) pins
# ``parse_citation``, ``resolve_citation``, ``looks_like_citation_query``,
# and ``looks_like_paper_identifier`` as the hard public symbols. Everything
# else is a private helper (``_``-prefixed) or a dataclass kept reachable for
# tests and for compatibility with pre-Phase-2 callers.
_FACADE_EXPORTS: tuple[str, ...] = (
    # Hard public API (Phase 1 surface pin).
    "parse_citation",
    "resolve_citation",
    "looks_like_citation_query",
    "looks_like_paper_identifier",
    # Public dataclasses + helpers reached by production + tests.
    "ParsedCitation",
    "RankedCitationCandidate",
    "build_match_metadata",
    "classify_known_item_resolution_state",
    "looks_like_url",
    "normalize_citation_text",
    # Test-facing private helpers pinned by tests/test_test_seam_inventory.py.
    "_classify_resolution_confidence",
    "_filtered_alternative_candidates",
    "_normalize_identifier_for_openalex",
    "_normalize_identifier_for_semantic_scholar",
    "_rank_candidate",
    "_serialize_citation_response",
    "_sparse_search_queries",
    "_title_similarity",
    "_venue_hint_in_text",
)


# ---------------------------------------------------------------------------
# Legacy ``__setattr__`` proxy for monkeypatching.
# ---------------------------------------------------------------------------
#
# Mirrors the Phase 2 Track C dispatch proxy: if a downstream consumer writes
# ``citation_repair_module.some_symbol = fake``, the write is mirrored onto
# ``_core`` so code executing inside ``_core`` sees the patched binding. A
# ``DeprecationWarning`` is raised to nudge callers toward patching the owning
# module directly before later phases relocate helpers into sibling modules.


class _CitationRepairPackageModule(_ModuleType):
    """Module subclass that forwards attribute writes to :mod:`_core`."""

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        if name.startswith("__"):
            return
        if not hasattr(_core, name):
            return
        import warnings

        warnings.warn(
            (
                f"Patching paper_chaser_mcp.citation_repair.{name!r} via the package facade is "
                "deprecated. Patch the owning module directly - e.g. "
                f"``monkeypatch.setattr(paper_chaser_mcp.citation_repair._core, {name!r}, ...)`` "
                "- so later sibling-module extractions remain visible to the code actually "
                "executed."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        setattr(_core, name, value)


_sys.modules[__name__].__class__ = _CitationRepairPackageModule

# Scrub private import aliases from the package namespace so the facade
# reflects only what ``_FACADE_EXPORTS`` advertises.
del _sys, _ModuleType, _CitationRepairPackageModule, annotations


# Public surface advertised by the package. Kept narrow on purpose: the
# Phase 1 public-api-surface guard pins four hard public symbols. Private
# helpers remain importable (tests rely on that) but are intentionally
# omitted from ``__all__`` to discourage new external usage.
__all__ = (
    "parse_citation",
    "resolve_citation",
    "looks_like_citation_query",
    "looks_like_paper_identifier",
)
