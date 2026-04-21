"""Paper Chaser MCP package entrypoint.

Phase 7c-1 deferred the ``main`` import to attribute-access time via
``__getattr__``. Importing ``paper_chaser_mcp`` (or any of its light
submodules such as ``paper_chaser_mcp.agentic.planner``) no longer eagerly
loads ``server.py`` and its heavy transitive dependencies (fastmcp, LLM
provider bundles, etc.). Calling ``paper_chaser_mcp.main()`` or
``from paper_chaser_mcp import main`` still works unchanged because the
lazy loader resolves ``main`` on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from .server import main as main

__version__ = "0.2.2"


__all__ = ["main"]


def __getattr__(name: str) -> Any:
    """Lazily surface ``main`` without forcing ``server`` to load."""

    if name == "main":
        from .server import main as _main

        return _main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
