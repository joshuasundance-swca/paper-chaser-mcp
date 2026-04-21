"""Smart-profile dispatch submodules extracted from ``dispatch/_core.py``.

Phase 4 of the dispatch refactor relocates the smart-profile tool entrypoints
(``search_papers_smart``, ``ask_result_set``, ``map_research_landscape``,
``expand_research_graph``) into focused submodules under this package.

Submodules are leaf-only: each exposes a single ``_dispatch_<tool>`` coroutine
following the ctx-first calling convention pinned in Phase 2 Track C.
"""

from __future__ import annotations
