"""Phase 2 Track C amendment 3: branch entrypoints accept ``DispatchContext``.

Phase 2 Step 2 introduced :class:`paper_chaser_mcp.dispatch.context.DispatchContext`
as the single dependency bag for ``dispatch_tool`` and its helpers, but the
inline branches inside ``dispatch_tool`` itself still unpack ~40 locals and pass
individual values to everything they call. Phase 3 will relocate these branches
into sibling submodules, and that is only safe if the branch functions already
take a ``DispatchContext`` rather than re-deriving the same state on every move.

This amendment extracts four branch entrypoints — ``research``,
``follow_up_research``, ``search_papers_smart``, ``ask_result_set`` — as module-
level async functions on ``paper_chaser_mcp.dispatch._core``:

* ``_dispatch_research(ctx: DispatchContext, arguments: dict[str, Any])``
* ``_dispatch_follow_up_research(ctx, arguments)``
* ``_dispatch_search_papers_smart(ctx, arguments)``
* ``_dispatch_ask_result_set(ctx, arguments)``

The tests here pin the signature shape (``ctx`` first, positional-or-keyword,
typed as ``DispatchContext``) so a future refactor cannot silently regress the
contract. The actual branch logic is validated end-to-end by the existing
characterization tests, which must stay green without
``PAPER_CHASER_CHAR_REGEN=1``.
"""

from __future__ import annotations

import inspect

import pytest

from paper_chaser_mcp.dispatch import _core as dispatch_core
from paper_chaser_mcp.dispatch.context import DispatchContext

BRANCH_ENTRYPOINTS = (
    "_dispatch_research",
    "_dispatch_follow_up_research",
    "_dispatch_search_papers_smart",
    "_dispatch_ask_result_set",
)


@pytest.mark.parametrize("name", BRANCH_ENTRYPOINTS)
def test_branch_entrypoint_exists(name: str) -> None:
    """Each branch entrypoint must exist as a module-level async function."""

    assert hasattr(dispatch_core, name), (
        f"paper_chaser_mcp.dispatch._core must define {name!r} as a module-level "
        f"entrypoint so Phase 3 can relocate the branch into a sibling submodule "
        f"without rewiring every caller."
    )
    func = getattr(dispatch_core, name)
    assert inspect.iscoroutinefunction(func), f"{name} must be an async function; dispatch_tool awaits the result."


@pytest.mark.parametrize("name", BRANCH_ENTRYPOINTS)
def test_branch_entrypoint_first_parameter_is_ctx(name: str) -> None:
    """The first parameter must be ``ctx: DispatchContext``."""

    func = getattr(dispatch_core, name)
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    assert params, f"{name} must accept at least one parameter"
    first = params[0]
    assert first.name == "ctx", (
        f"{name}'s first parameter must be named 'ctx' (got {first.name!r}). "
        f"This is the shared dependency-bag convention pinned by Phase 2 Step 2."
    )
    assert first.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ), f"{name}'s ctx parameter must be positional-or-keyword, got {first.kind!r}"


@pytest.mark.parametrize("name", BRANCH_ENTRYPOINTS)
def test_branch_entrypoint_ctx_is_dispatch_context(name: str) -> None:
    """The ``ctx`` parameter must be typed as ``DispatchContext``.

    This pin keeps Phase 3 honest: once branches move to sibling modules they
    must continue to accept the same frozen dependency bag rather than
    reintroducing per-branch ad-hoc kwargs.
    """

    func = getattr(dispatch_core, name)
    sig = inspect.signature(func)
    ctx_param = sig.parameters["ctx"]
    annotation = ctx_param.annotation
    # Annotations may be stringified under ``from __future__ import annotations``;
    # accept both the real class and its string name.
    assert annotation in (DispatchContext, "DispatchContext"), (
        f"{name}'s ctx parameter must be typed as DispatchContext, got "
        f"{annotation!r}. Use `ctx: DispatchContext` in the source to keep "
        f"Phase 3 moves mechanical."
    )


@pytest.mark.parametrize("name", BRANCH_ENTRYPOINTS)
def test_branch_entrypoint_accepts_arguments_dict(name: str) -> None:
    """The entrypoint must also accept an ``arguments`` mapping parameter.

    ``dispatch_tool`` normalizes the incoming tool arguments once and then
    forwards the validated dict to the branch. Keeping the parameter name
    stable (``arguments``) lets Phase 3 grep for call sites safely.
    """

    func = getattr(dispatch_core, name)
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    assert len(params) >= 2, f"{name} must accept at least (ctx, arguments); got {[p.name for p in params]}"
    assert params[1].name == "arguments", (
        f"{name}'s second parameter must be named 'arguments' (got {params[1].name!r}) to keep call sites greppable."
    )
