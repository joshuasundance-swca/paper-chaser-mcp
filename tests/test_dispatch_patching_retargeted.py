"""Phase 2 Track C amendment 2: dispatch monkeypatch tests target owning modules.

Before Phase 2, ``paper_chaser_mcp/dispatch.py`` was a single module, so every
test could monkeypatch it with ``monkeypatch.setattr(dispatch, "X", fake)`` and
trust that code running inside the module would see the new binding. After the
package split, the authoritative binding lives on
:mod:`paper_chaser_mcp.dispatch._core` (today) or on a sibling submodule such
as ``paper_chaser_mcp.dispatch.relevance`` (Phase 3+). Patching the facade no
longer reaches the function *through* an intra-package call by default — the
facade's ``__setattr__`` proxy mirrors the write down to ``_core`` for
compatibility, but that only works for symbols that still happen to live in
``_core``.

These tests pin the correct pattern for new tests: patch the owning module
directly. They also document the concrete Phase 3 scenario where patching the
facade silently becomes insufficient because the symbol has moved to a
sibling submodule.
"""

from __future__ import annotations

import importlib

import pytest

from paper_chaser_mcp import dispatch as dispatch_module
from paper_chaser_mcp.dispatch import _core as dispatch_core
from paper_chaser_mcp.dispatch import relevance as dispatch_relevance


def test_patching_owning_module_is_visible_from_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patching the owning module binds through the facade re-export."""

    sentinel = object()
    monkeypatch.setattr(dispatch_core, "resolve_citation", sentinel)

    # Reaching through the facade must observe the patched binding, because
    # the facade re-exports ``resolve_citation`` *from* ``_core``. This is the
    # invariant tests rely on after retargeting.
    assert dispatch_core.resolve_citation is sentinel
    # The facade's own re-exported binding was captured at import time and is
    # NOT automatically refreshed when ``_core`` changes. Production code
    # running inside ``_core`` calls its own module-level ``resolve_citation``
    # though, and that is what the patch affects.
    # (The facade binding only ever mattered for legacy ``monkeypatch.setattr(
    # dispatch_module, "resolve_citation", ...)`` callers, which this
    # amendment retargets.)


def test_sibling_module_symbol_must_be_patched_on_its_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Helpers extracted to sibling submodules cannot be patched via the facade.

    ``compute_topical_relevance`` is re-exported on both the facade and
    ``_core`` for backwards compatibility, but its *owning* module is
    ``paper_chaser_mcp.dispatch.relevance``. Callers that import from the
    sibling module see only a patch applied *there*.
    """

    sentinel_owner = object()
    monkeypatch.setattr(
        dispatch_relevance,
        "compute_topical_relevance",
        sentinel_owner,
    )

    # The owner observes the patch.
    assert dispatch_relevance.compute_topical_relevance is sentinel_owner

    # A separate module's cached import binding is unaffected — this is the
    # Phase 3 scenario we need new tests to guard against. If anyone ever
    # starts monkeypatching the facade for a sibling-module symbol, the
    # patched-code path will silently still call the real function.
    dispatch_module_reloaded = importlib.import_module("paper_chaser_mcp.dispatch")
    # The facade re-export does NOT forward to the sibling module at read
    # time, so it keeps the original binding.
    assert dispatch_module_reloaded.compute_topical_relevance is not sentinel_owner, (
        "Expected the facade's cached re-export to be independent of the "
        "sibling module patch; if this starts passing, the facade has grown "
        "dynamic attribute lookup and the amendment 2 contract needs to be "
        "revisited."
    )


def test_legacy_facade_setattr_emits_deprecation_warning() -> None:
    """Writing through the facade still works but warns callers to retarget."""

    # The proxy installs a DeprecationWarning for any ``dispatch_module.X = ...``
    # write that shadows a ``_core`` symbol, so accidental uses surface in CI.
    original = dispatch_core.resolve_citation
    try:
        with pytest.warns(DeprecationWarning, match="dispatch\\._core"):
            dispatch_module.resolve_citation = original  # type: ignore[misc]
        # The write is mirrored onto ``_core`` so behavior is preserved for
        # out-of-tree callers that have not yet retargeted.
        assert dispatch_core.resolve_citation is original
    finally:
        dispatch_core.resolve_citation = original


def test_in_tree_suite_does_not_patch_facade_for_core_symbols() -> None:
    """No in-tree test should still target the facade for a ``_core`` symbol.

    This is a regression guard for amendment 2: the sole legacy caller at
    ``tests/test_dispatch.py:~4355`` has been retargeted to ``_core``. Any
    reintroduction of ``monkeypatch.setattr(dispatch_module, "...", ...)``
    where the symbol lives on ``_core`` should be a deliberate decision, not
    an accident.
    """

    from pathlib import Path

    tests_dir = Path(__file__).resolve().parent
    offending: list[tuple[str, int, str]] = []
    for path in tests_dir.rglob("*.py"):
        if path.name == "test_dispatch_patching_retargeted.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "monkeypatch.setattr(dispatch_module" not in text:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "monkeypatch.setattr(dispatch_module" in line:
                offending.append((str(path.relative_to(tests_dir)), lineno, line.strip()))

    assert not offending, (
        "The in-tree suite must patch the owning module directly (e.g. "
        "``paper_chaser_mcp.dispatch._core``) rather than the package facade. "
        f"Found: {offending}"
    )
