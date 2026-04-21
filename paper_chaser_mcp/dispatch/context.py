"""Single dependency bag for :func:`paper_chaser_mcp.dispatch.dispatch_tool`.

Phase 2 Step 2 refactor: before this module existed, ``dispatch_tool`` threaded
~40 keyword arguments through an inner ``_dispatch_internal`` forwarder by
spelling every argument twice (once in the outer signature, once in the
inner call). Phases 3–5 extract branches of ``dispatch_tool`` into sibling
modules and need a single stable object to pass around instead of rebuilding
that 40-kwarg list in every new helper.

``DispatchContext`` is that object. It is:

* a frozen dataclass — rewires cannot mutate it mid-flight and tests can rely
  on value equality,
* a pure data carrier — no I/O, no methods that reach out to the network or
  disk, and pickleable when populated with plain Python values,
* authoritative — every dependency ``dispatch_tool`` accepts appears here.
  The public API surface guard and ``tests/test_dispatch_context.py`` pin
  this in both directions so neither side can drift.

The dataclass deliberately holds client objects, registries, and runtime
flags as ``Any`` / narrow unions to avoid a large typing migration during the
refactor. Future phases can tighten these types once the surrounding code
has been split into focused submodules.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from ..enrichment import PaperEnrichmentService
from ..models.tools import SearchProvider


@dataclasses.dataclass(frozen=True)
class DispatchContext:
    """Frozen dependency bag for :func:`dispatch_tool` and its helpers."""

    # Required dependencies — every dispatch invocation must supply these.
    client: Any
    core_client: Any
    openalex_client: Any
    scholarapi_client: Any
    arxiv_client: Any
    enable_core: bool
    enable_semantic_scholar: bool
    enable_openalex: bool
    enable_scholarapi: bool
    enable_arxiv: bool

    # Optional clients / toggles — defaults must mirror dispatch_tool exactly.
    serpapi_client: Any = None
    enable_serpapi: bool = False
    crossref_client: Any = None
    unpaywall_client: Any = None
    ecos_client: Any = None
    federal_register_client: Any = None
    govinfo_client: Any = None
    enable_crossref: bool = True
    enable_unpaywall: bool = True
    enable_ecos: bool = True
    enable_federal_register: bool = True
    enable_govinfo_cfr: bool = True

    # Enrichment / runtime wiring.
    enrichment_service: PaperEnrichmentService | None = None
    provider_order: list[SearchProvider] | None = None
    provider_registry: Any = None
    workspace_registry: Any = None
    agentic_runtime: Any = None

    # Transport / profile.
    transport_mode: str = "stdio"
    tool_profile: str = "guided"
    hide_disabled_tools: bool = False
    session_ttl_seconds: int | None = None
    embeddings_enabled: bool | None = None

    # Guided-surface policy knobs.
    guided_research_latency_profile: str = "deep"
    guided_follow_up_latency_profile: str = "deep"
    guided_allow_paid_providers: bool = True
    guided_escalation_enabled: bool = True
    guided_escalation_max_passes: int = 2
    guided_escalation_allow_paid_providers: bool = True

    # Per-call context object (MCP ``Context`` etc.) and elicitation policy.
    ctx: Any = None
    allow_elicitation: bool = True

    def as_kwargs(self) -> dict[str, Any]:
        """Return a shallow dict of every field, suitable for ``**kwargs``.

        This is the canonical way for internal forwarders (today:
        ``_dispatch_internal`` inside :func:`dispatch_tool`) to re-invoke
        ``dispatch_tool`` without re-spelling every dependency. It is a
        *shallow* copy — mutable fields like ``provider_order`` are
        intentionally shared by reference because the frozen dataclass does
        not deep-copy them on construction either.
        """
        return {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}


def build_dispatch_context(**kwargs: Any) -> DispatchContext:
    """Factory that builds a :class:`DispatchContext` from dispatch_tool kwargs.

    Exposed as a plain function (not a classmethod) so test code and future
    extractions can construct contexts without importing the dataclass
    directly. Unknown kwargs raise :class:`TypeError` via the dataclass
    constructor, which is the pre-refactor behavior (``dispatch_tool`` also
    rejects unknown kwargs via Python's normal keyword-argument handling).
    """
    return DispatchContext(**kwargs)


__all__ = ("DispatchContext", "build_dispatch_context")
