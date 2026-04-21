"""Provider base subpackage: bundle contracts + deterministic fallback helpers."""

from __future__ import annotations

from .bundle import DeterministicProviderBundle, ModelProviderBundle
from .classification import (
    _fallback_query_facets,
    _fallback_query_terms,
    classify_relevance_without_llm,
)
from .identifiers import _default_selected_evidence_ids, relevance_paper_identifier

__all__ = [
    "DeterministicProviderBundle",
    "ModelProviderBundle",
    "_default_selected_evidence_ids",
    "_fallback_query_facets",
    "_fallback_query_terms",
    "classify_relevance_without_llm",
    "relevance_paper_identifier",
]
