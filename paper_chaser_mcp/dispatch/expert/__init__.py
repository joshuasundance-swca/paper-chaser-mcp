"""Expert-profile dispatch submodules extracted from ``dispatch/_core.py``.

Phase 4 of the dispatch refactor relocates the expert-profile tool entrypoints
(regulatory primary-sources, raw provider-family tools, provider-specific
endpoints) into focused submodules under this package.

Submodules are leaf-only: each exposes one or more ``_dispatch_<tool>``
coroutines following the ctx-first calling convention pinned in Phase 2
Track C.
"""

from __future__ import annotations
