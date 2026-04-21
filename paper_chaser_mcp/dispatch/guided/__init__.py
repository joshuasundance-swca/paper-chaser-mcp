"""Guided dispatch submodules extracted from ``dispatch/_core.py``.

Phase 3 of the dispatch refactor relocates the guided orchestration helpers
into focused submodules under this package. The parent ``_core.py`` re-imports
each extracted symbol so existing external callers (including the Phase 2
facade allowlist in :mod:`paper_chaser_mcp.dispatch`) continue to work.

Submodules are kept leaf-first in the dependency order declared by the
Phase 3 plan so circular imports stay avoidable.
"""

from __future__ import annotations
