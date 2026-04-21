"""Graphs package facade.

Phase 7a converted ``paper_chaser_mcp/agentic/graphs.py`` into a package so its
helpers can be extracted into focused submodules without disturbing the
existing import surface. This facade re-exports the curated, diff-visible set
of symbols that production and test callers actually reach through
``from paper_chaser_mcp.agentic.graphs import X``.

Phase 7a (this branch) is a pure subpackage split with no helper extraction
yet — the low-risk infrastructure lives in ``shared_state.py``,
``source_records.py``, ``hooks.py``, and ``regulatory_routing.py``. The
orchestration surface (``AgenticRuntime`` and its methods) stays in ``_core``
and is slated for Phase 7b/7c.

The Phase 7a rubber-duck review flagged the prior "re-export everything from
``_core``" shape as a MUST-FIX: ``__all__`` only advertised two public
symbols, but many private helpers were reachable purely because ``_core``
happened to bind them. The explicit ``_FACADE_EXPORTS`` allowlist below
mirrors the pattern established by :mod:`paper_chaser_mcp.dispatch` so the
facade surface is diff-visible and cannot drift silently. Any addition or
removal is a deliberate, reviewable change and is frozen by
``tests/test_graphs_facade_allowlist.py``.
"""

from __future__ import annotations

from . import _core as _core
from ._core import (
    AgenticRuntime as AgenticRuntime,
)
from ._core import (
    TopicalRelevanceClassification as TopicalRelevanceClassification,
)
from ._core import (
    _build_grounded_comparison_answer as _build_grounded_comparison_answer,
)
from ._core import (
    _classify_topical_relevance_with_provenance as _classify_topical_relevance_with_provenance,
)
from ._core import (
    _derive_regulatory_query_flags as _derive_regulatory_query_flags,
)
from ._core import (
    _ecos_query_variants as _ecos_query_variants,
)
from ._core import (
    _finalize_theme_label as _finalize_theme_label,
)
from ._core import (
    _graph_frontier_scores as _graph_frontier_scores,
)
from ._core import (
    _has_inspectable_sources as _has_inspectable_sources,
)
from ._core import (
    _has_on_topic_sources as _has_on_topic_sources,
)
from ._core import (
    _is_agency_guidance_query as _is_agency_guidance_query,
)
from ._core import (
    _is_current_cfr_text_request as _is_current_cfr_text_request,
)
from ._core import (
    _is_opaque_query as _is_opaque_query,
)
from ._core import (
    _query_requests_regulatory_history as _query_requests_regulatory_history,
)
from ._core import (
    _rank_ecos_variant_hits as _rank_ecos_variant_hits,
)
from ._core import (
    _rank_regulatory_documents as _rank_regulatory_documents,
)
from ._core import (
    _source_record_from_regulatory_document as _source_record_from_regulatory_document,
)

# ---------------------------------------------------------------------------
# Explicit facade allowlist.
# ---------------------------------------------------------------------------
#
# This tuple pins the *exact* set of names callers may reach through
# ``paper_chaser_mcp.agentic.graphs.<name>``. Adding an entry means
# acknowledging that production or test code depends on reaching the symbol
# via the facade; removing an entry is a deliberate surface shrink and must
# be paired with retargeting any consumer to the owning submodule (``_core``
# today, or a Phase 7b/7c sibling module).
#
# The public-api-surface guard pins ``AgenticRuntime`` as the hard public
# symbol reached through this facade. ``TopicalRelevanceClassification`` is a
# public data class consumed by tests. Everything else is a private helper
# (``_``-prefixed) that the test suite from-imports directly; those entries
# are mirrored in ``tests/test_test_seam_inventory.py::KNOWN_TEST_SEAMS`` so
# the test-only seam surface cannot grow silently. Keeping the ``from ._core
# import`` block above and this tuple in sync is enforced by
# ``tests/test_graphs_facade_allowlist.py``.
#
# Alphabetical ordering is intentional: it makes any diff to this allowlist
# land exactly at the insertion point and keeps reviews focused on intent.
_FACADE_EXPORTS: tuple[str, ...] = (
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
)


__all__ = list(_FACADE_EXPORTS)

# Scrub the ``from __future__`` alias out of the package namespace so the
# facade surface reflects only what ``_FACADE_EXPORTS`` advertises. Otherwise
# ``dir(paper_chaser_mcp.agentic.graphs)`` would expose ``annotations`` as an
# accidental leak, matching the old-surface pattern the dispatch allowlist
# was introduced to eliminate.
del annotations
