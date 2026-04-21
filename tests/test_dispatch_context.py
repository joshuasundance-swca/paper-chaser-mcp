"""Tests for :mod:`paper_chaser_mcp.dispatch.context`.

Phase 2 Step 2: introduce a ``DispatchContext`` dataclass at the
:func:`dispatch_tool` entry point so every downstream helper receives a single
dependency bag instead of the 40 individual keyword arguments that the
pre-refactor monolith threaded through ``_dispatch_internal``.

The tests pin:

* the full field list (every dependency ``dispatch_tool`` currently accepts),
* that defaults mirror ``dispatch_tool``'s defaults verbatim,
* immutability (the bag is frozen; rewires should not mutate it mid-flight),
* that :func:`build_dispatch_context` accepts ``dispatch_tool``'s exact kwargs
  and produces an identical instance,
* ``DispatchContext`` has no I/O and is pickleable with ordinary Python values
  (real SDK clients obviously are not, but the dataclass itself must be a
  plain data carrier).
"""

from __future__ import annotations

import dataclasses
import inspect
import pickle
from typing import Any

import pytest

from paper_chaser_mcp.dispatch import dispatch_tool
from paper_chaser_mcp.dispatch.context import (
    DispatchContext,
    build_dispatch_context,
)


def _dispatch_tool_fields() -> dict[str, inspect.Parameter]:
    """Return the keyword-only parameters of :func:`dispatch_tool`.

    ``name`` and ``arguments`` are the per-call inputs and are intentionally
    excluded from the context bag — they change on every dispatch.
    """
    sig = inspect.signature(dispatch_tool)
    return {name: param for name, param in sig.parameters.items() if name not in {"name", "arguments"}}


def test_dispatch_context_covers_every_dispatch_tool_dependency() -> None:
    """DispatchContext must enumerate every kwarg ``dispatch_tool`` accepts."""
    dispatch_params = _dispatch_tool_fields()
    ctx_fields = {f.name for f in dataclasses.fields(DispatchContext)}
    assert ctx_fields == set(dispatch_params.keys()), (
        "DispatchContext fields must match dispatch_tool kwargs exactly. "
        f"missing={set(dispatch_params) - ctx_fields}, "
        f"extra={ctx_fields - set(dispatch_params)}"
    )


def test_dispatch_context_defaults_mirror_dispatch_tool() -> None:
    """Every default on DispatchContext must mirror the dispatch_tool default."""
    dispatch_params = _dispatch_tool_fields()
    ctx_fields_by_name = {f.name: f for f in dataclasses.fields(DispatchContext)}
    for name, param in dispatch_params.items():
        field = ctx_fields_by_name[name]
        if param.default is inspect.Parameter.empty:
            assert field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING, (
                f"{name}: dispatch_tool requires it; DispatchContext must also require it."
            )
        else:
            if field.default_factory is not dataclasses.MISSING:
                assert field.default_factory() == param.default, (
                    f"{name}: default_factory={field.default_factory()!r} vs dispatch_tool default={param.default!r}"
                )
            else:
                assert field.default == param.default, (
                    f"{name}: DispatchContext default={field.default!r} vs dispatch_tool default={param.default!r}"
                )


def _minimal_required_kwargs() -> dict[str, Any]:
    """Smallest set of kwargs that satisfies every required dispatch_tool arg."""
    return {
        "client": None,
        "core_client": None,
        "openalex_client": None,
        "scholarapi_client": None,
        "arxiv_client": None,
        "enable_core": True,
        "enable_semantic_scholar": True,
        "enable_openalex": True,
        "enable_scholarapi": False,
        "enable_arxiv": True,
    }


def test_dispatch_context_instantiates_with_required_fields_only() -> None:
    ctx = DispatchContext(**_minimal_required_kwargs())
    assert ctx.client is None
    assert ctx.enable_core is True
    # Defaulted fields should take their documented defaults.
    assert ctx.transport_mode == "stdio"
    assert ctx.tool_profile == "guided"
    assert ctx.allow_elicitation is True
    assert ctx.guided_escalation_max_passes == 2


def test_dispatch_context_is_frozen() -> None:
    """The bag is immutable — rewires must not mutate ctx mid-dispatch."""
    ctx = DispatchContext(**_minimal_required_kwargs())
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.tool_profile = "expert"  # type: ignore[misc]


def test_build_dispatch_context_accepts_dispatch_tool_kwargs() -> None:
    """build_dispatch_context should be a drop-in factory over dispatch_tool kwargs."""
    kwargs = _minimal_required_kwargs() | {
        "tool_profile": "expert",
        "allow_elicitation": False,
        "guided_escalation_max_passes": 5,
    }
    ctx = build_dispatch_context(**kwargs)
    assert isinstance(ctx, DispatchContext)
    assert ctx.tool_profile == "expert"
    assert ctx.allow_elicitation is False
    assert ctx.guided_escalation_max_passes == 5


def test_build_dispatch_context_rejects_unknown_kwargs() -> None:
    with pytest.raises(TypeError):
        build_dispatch_context(**_minimal_required_kwargs(), totally_bogus_kw=1)


def test_dispatch_context_is_pickleable_with_plain_values() -> None:
    """The dataclass itself must be a pure data carrier (no I/O, pickleable).

    Real SDK client objects won't pickle, but the context struct must — this
    guards against anyone sneaking a ``@dataclass`` ``field(default_factory=...)``
    that binds something stateful (e.g. a lock or an open file) at module import
    time.
    """
    ctx = DispatchContext(**_minimal_required_kwargs())
    restored = pickle.loads(pickle.dumps(ctx))
    assert restored == ctx
    assert isinstance(restored, DispatchContext)


def test_dispatch_context_as_kwargs_roundtrips() -> None:
    """as_kwargs() must produce a dict suitable for re-invoking dispatch_tool."""
    ctx = DispatchContext(**_minimal_required_kwargs())
    as_kwargs = ctx.as_kwargs()
    assert isinstance(as_kwargs, dict)
    # Every DispatchContext field must be present in the kwargs dict so the
    # forwarding call in ``_dispatch_internal`` cannot silently drop a dep.
    expected = {f.name for f in dataclasses.fields(DispatchContext)}
    assert set(as_kwargs.keys()) == expected
    # Round-trip: rebuilding from as_kwargs yields an equal context.
    assert build_dispatch_context(**as_kwargs) == ctx
