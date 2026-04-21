"""Smart-search orchestration extracted in Phase 7c-2 from ``graphs/_core``.

Phase 7c-2 relocates the ``search_papers_smart`` orchestration body, its
smart-specific pure helpers, and the LangGraph ``StateGraph`` compilation
helper out of :mod:`paper_chaser_mcp.agentic.graphs._core` so the multiplexing
entry point and its LLM recovery routing live in a dedicated module.
``AgenticRuntime.search_papers_smart`` remains on the class as a thin delegate
(Pattern B — see the Phase 7c-2 prompt); the body is the module-level
``run_search_papers_smart`` coroutine. ``_maybe_compile_graphs`` is extracted
as a module-level function that takes ``runtime`` as a parameter (Pattern A).

LangGraph optional-dependency stubs (``START``, ``END``, ``StateGraph``,
``InMemorySaver``) are imported from :mod:`shared_state` rather than
``_core``; Phase 7a moved the single-source try/except into
``shared_state.py``, so importing from there preserves identity without
creating a back-edge to ``_core`` that would cycle during module loading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


__all__: list[str] = []
