"""Dispatch package facade.

Phase 2 converted ``paper_chaser_mcp/dispatch.py`` into a package so its helpers
can be extracted into focused submodules without disturbing the public import
surface. The original monolith now lives in :mod:`paper_chaser_mcp.dispatch._core`;
this facade re-exports a curated, diff-visible set of symbols.

Prior to the Phase 2 Track C amendment, ``__init__.py`` mirrored every
non-dunder attribute from ``_core`` onto the package namespace via
``for name in dir(_core)``. That loop also surfaced accidental internals like
``re``, ``time``, ``logging``, and the imported ``typing`` aliases. The mirror
was replaced with the explicit ``_FACADE_EXPORTS`` allowlist below so the
facade surface is diff-visible and cannot grow silently whenever ``_core`` adds
a new import.

External callers must continue to import from ``paper_chaser_mcp.dispatch``
directly; the Phase 1 public-api-surface pin lists ``dispatch_tool`` as the
only hard public symbol. Test-facing private helpers remain importable and
tracked in ``_FACADE_EXPORTS`` so any intentional change is visible in review
diffs.
"""

from __future__ import annotations

import sys as _sys
from types import ModuleType as _ModuleType

from . import _core as _core
from ._core import (
    DispatchContext as DispatchContext,
)
from ._core import (
    _answer_follow_up_from_session_state as _answer_follow_up_from_session_state,
)
from ._core import (
    _apply_follow_up_response_mode as _apply_follow_up_response_mode,
)
from ._core import (
    _assign_verification_status as _assign_verification_status,
)
from ._core import (
    _authoritative_but_weak_source_ids as _authoritative_but_weak_source_ids,
)
from ._core import (
    _build_provider_diagnostics_snapshot as _build_provider_diagnostics_snapshot,
)
from ._core import (
    _compose_why_classified_weak_match as _compose_why_classified_weak_match,
)
from ._core import (
    _cursor_to_offset as _cursor_to_offset,
)
from ._core import (
    _direct_read_recommendation_details as _direct_read_recommendation_details,
)
from ._core import (
    _direct_read_recommendations as _direct_read_recommendations,
)
from ._core import (
    _evidence_quality_detail as _evidence_quality_detail,
)
from ._core import (
    _guided_abstention_details_payload as _guided_abstention_details_payload,
)
from ._core import (
    _guided_best_next_internal_action as _guided_best_next_internal_action,
)
from ._core import (
    _guided_citation_from_paper as _guided_citation_from_paper,
)
from ._core import (
    _guided_confidence_signals as _guided_confidence_signals,
)
from ._core import (
    _guided_contract_fields as _guided_contract_fields,
)
from ._core import (
    _guided_deterministic_evidence_gaps as _guided_deterministic_evidence_gaps,
)
from ._core import (
    _guided_failure_summary as _guided_failure_summary,
)
from ._core import (
    _guided_finalize_response as _guided_finalize_response,
)
from ._core import (
    _guided_is_mixed_intent_query as _guided_is_mixed_intent_query,
)
from ._core import (
    _guided_machine_failure_payload as _guided_machine_failure_payload,
)
from ._core import (
    _guided_mentions_literature as _guided_mentions_literature,
)
from ._core import (
    _guided_merge_coverage_summaries as _guided_merge_coverage_summaries,
)
from ._core import (
    _guided_next_actions as _guided_next_actions,
)
from ._core import (
    _guided_normalize_follow_up_arguments as _guided_normalize_follow_up_arguments,
)
from ._core import (
    _guided_normalize_inspect_arguments as _guided_normalize_inspect_arguments,
)
from ._core import (
    _guided_result_meaning as _guided_result_meaning,
)
from ._core import (
    _guided_result_state as _guided_result_state,
)
from ._core import (
    _guided_saved_session_topicality as _guided_saved_session_topicality,
)
from ._core import (
    _guided_session_state as _guided_session_state,
)
from ._core import (
    _guided_should_add_review_pass as _guided_should_add_review_pass,
)
from ._core import (
    _guided_source_metadata_answers as _guided_source_metadata_answers,
)
from ._core import (
    _guided_source_record_from_paper as _guided_source_record_from_paper,
)
from ._core import (
    _guided_source_record_from_structured_source as _guided_source_record_from_structured_source,
)
from ._core import (
    _guided_sources_from_fr_documents as _guided_sources_from_fr_documents,
)
from ._core import (
    _guided_summary as _guided_summary,
)
from ._core import (
    _guided_trust_summary as _guided_trust_summary,
)
from ._core import (
    _paper_topical_relevance as _paper_topical_relevance,
)
from ._core import (
    _synthesis_path as _synthesis_path,
)
from ._core import (
    _topical_relevance_from_signals as _topical_relevance_from_signals,
)
from ._core import (
    build_dispatch_context as build_dispatch_context,
)
from ._core import (
    compute_topical_relevance as compute_topical_relevance,
)
from ._core import (
    dispatch_tool as dispatch_tool,
)
from ._core import (
    resolve_citation as resolve_citation,
)

# ---------------------------------------------------------------------------
# Explicit facade allowlist.
# ---------------------------------------------------------------------------
#
# This tuple pins the *exact* set of names callers may reach through
# ``paper_chaser_mcp.dispatch.<name>``. Adding an entry means acknowledging
# that production or test code depends on reaching the symbol via the facade;
# removing an entry is a deliberate surface shrink and must be paired with
# retargeting any consumer to the owning submodule (``_core`` today, or a
# Phase 3+ sibling module).
#
# The public-api-surface guard (``tests/test_public_api_surface.py``) pins
# ``dispatch_tool`` as the only hard public symbol. Everything else below is
# a private helper (``_``-prefixed) kept reachable for tests and for
# compatibility with pre-Phase-2 callers. Keeping the ``from ._core import``
# block above and this tuple in sync is enforced by
# ``tests/test_dispatch_facade_allowlist.py``.
_FACADE_EXPORTS: tuple[str, ...] = (
    "dispatch_tool",
    "DispatchContext",
    "build_dispatch_context",
    "_answer_follow_up_from_session_state",
    "_apply_follow_up_response_mode",
    "_assign_verification_status",
    "_authoritative_but_weak_source_ids",
    "_build_provider_diagnostics_snapshot",
    "_compose_why_classified_weak_match",
    "_cursor_to_offset",
    "_direct_read_recommendation_details",
    "_direct_read_recommendations",
    "_evidence_quality_detail",
    "_guided_abstention_details_payload",
    "_guided_best_next_internal_action",
    "_guided_citation_from_paper",
    "_guided_confidence_signals",
    "_guided_contract_fields",
    "_guided_deterministic_evidence_gaps",
    "_guided_failure_summary",
    "_guided_finalize_response",
    "_guided_is_mixed_intent_query",
    "_guided_machine_failure_payload",
    "_guided_mentions_literature",
    "_guided_merge_coverage_summaries",
    "_guided_next_actions",
    "_guided_normalize_follow_up_arguments",
    "_guided_normalize_inspect_arguments",
    "_guided_result_meaning",
    "_guided_result_state",
    "_guided_saved_session_topicality",
    "_guided_session_state",
    "_guided_should_add_review_pass",
    "_guided_source_metadata_answers",
    "_guided_source_record_from_paper",
    "_guided_source_record_from_structured_source",
    "_guided_sources_from_fr_documents",
    "_guided_summary",
    "_guided_trust_summary",
    "_paper_topical_relevance",
    "_synthesis_path",
    "_topical_relevance_from_signals",
    "compute_topical_relevance",
    "resolve_citation",
)


# ---------------------------------------------------------------------------
# Legacy ``__setattr__`` proxy for monkeypatching.
# ---------------------------------------------------------------------------
#
# Phase 2 Track C amendment 2 retargeted every in-tree ``monkeypatch.setattr``
# call to patch the owning submodule directly (``_core`` today, sibling
# modules in Phase 3). This proxy stays installed for defense in depth: if a
# downstream consumer still writes ``dispatch_module.some_symbol = fake``, the
# write is mirrored onto ``_core`` so code executing inside ``_core`` actually
# sees the patched binding. A ``DeprecationWarning`` is raised so the caller
# knows to retarget before Phase 3 moves the owning module.


class _DispatchPackageModule(_ModuleType):
    """Module subclass that forwards attribute writes to :mod:`_core`.

    Retained as a Phase 2→3 safety net; every *in-tree* patch already targets
    the owning module directly. External callers that still hit the proxy
    receive a ``DeprecationWarning`` identifying the symbol they patched so
    they can retarget before Phase 3 relocates it.
    """

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        if name.startswith("__"):
            return
        if not hasattr(_core, name):
            return
        import warnings

        warnings.warn(
            (
                f"Patching paper_chaser_mcp.dispatch.{name!r} via the package facade is "
                "deprecated. Patch the owning module directly - e.g. "
                f"``monkeypatch.setattr(paper_chaser_mcp.dispatch._core, {name!r}, ...)`` "
                "- so Phase 3 sibling-module extractions remain visible to the code "
                "actually executed."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        setattr(_core, name, value)


_sys.modules[__name__].__class__ = _DispatchPackageModule

# Scrub private import aliases from the package namespace so the facade
# reflects only what ``_FACADE_EXPORTS`` advertises. The class reference is
# held by the module's ``__class__``, so deleting the name is safe.
del _sys, _ModuleType, _DispatchPackageModule, annotations


# Public surface advertised by the package. Kept narrow on purpose: the
# Phase 1 public-api-surface guard pins ``dispatch_tool`` as the single hard
# public symbol. Private helpers remain importable (tests rely on that) but
# are intentionally omitted from ``__all__`` to discourage new external usage.
__all__ = ("dispatch_tool",)
