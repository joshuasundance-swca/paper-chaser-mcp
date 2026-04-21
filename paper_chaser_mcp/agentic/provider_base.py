"""Compatibility shim that re-exports provider base contracts.

The real implementation moved to ``paper_chaser_mcp.agentic.providers.base``
during Phase 8b-1 of the modularization refactor. This module is preserved as
a thin pass-through so that existing ``from .provider_base import ...``
imports (both package-internal and in the test suite) keep working without a
behavior change.
"""

from __future__ import annotations

from .providers.base import (
    DeterministicProviderBundle,
    ModelProviderBundle,
    classify_relevance_without_llm,
    relevance_paper_identifier,
)
from .providers.base.classification import (
    _fallback_query_facets as _fallback_query_facets,  # noqa: PLC0414
)
from .providers.base.classification import (
    _fallback_query_terms as _fallback_query_terms,  # noqa: PLC0414
)
from .providers.base.identifiers import (
    _default_selected_evidence_ids as _default_selected_evidence_ids,  # noqa: PLC0414
)

__all__ = [
    "DeterministicProviderBundle",
    "ModelProviderBundle",
    "classify_relevance_without_llm",
    "relevance_paper_identifier",
]
