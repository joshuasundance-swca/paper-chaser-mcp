"""Additive smart research layer for Paper Chaser MCP.

Phase 7c-1 deferred the ``resolve_provider_bundle`` import from module
import time to attribute-access time via ``__getattr__``. The heavy LLM
provider backends (``provider_langchain`` / ``provider_openai``) are only
needed when a caller actually resolves a provider bundle; importing
``paper_chaser_mcp.agentic`` (or any of its light submodules such as
``planner`` and ``ranking``) no longer transitively drags them in. This
is what breaks the latent ``planner -> providers -> provider_base ->
planner`` cycle into a clean tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import AgenticConfig
from .graphs import AgenticRuntime
from .workspace import WorkspaceRegistry

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from .providers import resolve_provider_bundle as resolve_provider_bundle

__all__ = [
    "AgenticConfig",
    "AgenticRuntime",
    "WorkspaceRegistry",
    "resolve_provider_bundle",
]


def __getattr__(name: str) -> Any:
    """Lazily surface ``resolve_provider_bundle`` without forcing provider imports."""

    if name == "resolve_provider_bundle":
        from .providers import resolve_provider_bundle as _resolve_provider_bundle

        return _resolve_provider_bundle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include ``__getattr__``-surfaced names in ``dir(paper_chaser_mcp.agentic)``.

    PEP 562 notes that module-level ``__getattr__`` hides lazy names from
    ``dir()`` unless a sibling ``__dir__`` also exposes them. Phase 7c-1
    review flagged this as a discoverability gap; without this hook,
    ``resolve_provider_bundle`` would be invisible to ``dir`` / Sphinx /
    IDE autocomplete, even though it is part of the public surface pinned
    in ``__all__``.
    """

    return sorted(set(globals()) | set(__all__))
