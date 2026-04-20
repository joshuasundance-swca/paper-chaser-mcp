"""Dispatch package facade.

Phase 2 converted ``paper_chaser_mcp/dispatch.py`` into a package so its helpers
can be extracted into focused submodules without disturbing the public import
surface. The original monolith now lives in :mod:`paper_chaser_mcp.dispatch._core`
and every pre-refactor symbol (public and underscored) is re-exported here so
existing callers keep working unchanged.

External callers must continue to import from ``paper_chaser_mcp.dispatch``
directly; the seam map in ``docs/`` / ``session-state`` lists the hard public
surface that must remain re-exportable (see :data:`__all__`).
"""

from __future__ import annotations

import sys as _sys
from types import ModuleType as _ModuleType

from . import _core
from ._core import *  # noqa: F401, F403

# Explicit re-exports of the nine seam-map public-surface symbols. Listed here
# (rather than relying solely on the namespace copy below) so regressions show
# up as concrete import errors rather than as silent attribute failures.
from ._core import (  # noqa: F401
    _answer_follow_up_from_session_state,
    _apply_follow_up_response_mode,
    _assign_verification_status,
    _authoritative_but_weak_source_ids,
    _build_provider_diagnostics_snapshot,
    _compose_why_classified_weak_match,
    _cursor_to_offset,
    _direct_read_recommendation_details,
    _direct_read_recommendations,
    _evidence_quality_detail,
    _guided_abstention_details_payload,
    _guided_best_next_internal_action,
    _guided_citation_from_paper,
    _guided_confidence_signals,
    _guided_contract_fields,
    _guided_deterministic_evidence_gaps,
    _guided_failure_summary,
    _guided_finalize_response,
    _guided_is_mixed_intent_query,
    _guided_machine_failure_payload,
    _guided_mentions_literature,
    _guided_merge_coverage_summaries,
    _guided_next_actions,
    _guided_normalize_follow_up_arguments,
    _guided_normalize_inspect_arguments,
    _guided_result_meaning,
    _guided_result_state,
    _guided_saved_session_topicality,
    _guided_session_state,
    _guided_should_add_review_pass,
    _guided_source_metadata_answers,
    _guided_source_record_from_paper,
    _guided_source_record_from_structured_source,
    _guided_sources_from_fr_documents,
    _guided_summary,
    _guided_trust_summary,
    _paper_topical_relevance,
    _synthesis_path,
    dispatch_tool,
)

# Mirror every top-level name defined in ``_core`` onto the package namespace.
# This preserves the pre-refactor flat contract where callers could reference
# ``paper_chaser_mcp.dispatch.<any_symbol>``, including private helpers used by
# tests (e.g. ``_guided_sources_from_fr_documents``, ``_paper_topical_relevance``).
_globals = globals()
for _name in dir(_core):
    if _name.startswith("__"):
        continue
    _globals.setdefault(_name, getattr(_core, _name))
del _globals, _name

# Install a ``__setattr__`` proxy so tests that monkeypatch the package
# namespace (e.g. ``monkeypatch.setattr(dispatch_module, "resolve_citation",
# fake)``) also mutate the binding that live helper functions in ``_core`` see.
# Before the Phase 2 package conversion, ``dispatch`` was a single module so
# patches took effect immediately; the shim below preserves that behavior.


class _DispatchPackageModule(_ModuleType):
    """Module subclass that forwards attribute writes to :mod:`_core`."""

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        # Only forward to _core if the attribute already exists there, so we
        # don't accidentally inject unrelated package-scoped state (e.g. the
        # ``_core`` submodule reference itself).
        if not name.startswith("__") and hasattr(_core, name):
            setattr(_core, name, value)


_sys.modules[__name__].__class__ = _DispatchPackageModule

# Public surface advertised by the package. Kept narrow on purpose: the
# Phase 1 public-api-surface guard pins ``dispatch_tool`` as the single hard
# public symbol. Private helpers remain importable (tests rely on that) but
# are intentionally omitted from ``__all__`` to discourage new external usage.
__all__ = ("dispatch_tool",)
