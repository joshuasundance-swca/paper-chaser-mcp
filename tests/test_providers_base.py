"""Phase 8b-1 identity tests for the providers/base subpackage extraction.

These tests assert that the public and private symbols reachable via the
``paper_chaser_mcp.agentic.provider_base`` shim are the *same object* as
the ones defined inside the new ``paper_chaser_mcp.agentic.providers.base``
subpackage. That protects downstream callers against any future drift
where someone accidentally rewrites the shim to return a duplicated copy
of a class or function.
"""

from __future__ import annotations

from paper_chaser_mcp.agentic import provider_base as _shim
from paper_chaser_mcp.agentic.providers import base as _base
from paper_chaser_mcp.agentic.providers.base import (
    bundle as _bundle,
)
from paper_chaser_mcp.agentic.providers.base import (
    classification as _classification,
)
from paper_chaser_mcp.agentic.providers.base import (
    identifiers as _identifiers,
)


def test_model_provider_bundle_lives_in_bundle_module() -> None:
    assert _base.ModelProviderBundle.__module__ == ("paper_chaser_mcp.agentic.providers.base.bundle")
    assert _bundle.ModelProviderBundle is _base.ModelProviderBundle


def test_deterministic_provider_bundle_lives_in_bundle_module() -> None:
    assert _base.DeterministicProviderBundle.__module__ == ("paper_chaser_mcp.agentic.providers.base.bundle")
    assert _bundle.DeterministicProviderBundle is _base.DeterministicProviderBundle


def test_classify_relevance_without_llm_lives_in_classification_module() -> None:
    assert _base.classify_relevance_without_llm.__module__ == ("paper_chaser_mcp.agentic.providers.base.classification")
    assert _classification.classify_relevance_without_llm is _base.classify_relevance_without_llm


def test_relevance_paper_identifier_lives_in_identifiers_module() -> None:
    assert _base.relevance_paper_identifier.__module__ == ("paper_chaser_mcp.agentic.providers.base.identifiers")
    assert _identifiers.relevance_paper_identifier is _base.relevance_paper_identifier


def test_shim_public_symbols_are_the_same_objects() -> None:
    assert _shim.ModelProviderBundle is _base.ModelProviderBundle
    assert _shim.DeterministicProviderBundle is _base.DeterministicProviderBundle
    assert _shim.classify_relevance_without_llm is _base.classify_relevance_without_llm
    assert _shim.relevance_paper_identifier is _base.relevance_paper_identifier


def test_shim_private_symbols_are_the_same_objects() -> None:
    assert _shim._fallback_query_facets is _classification._fallback_query_facets
    assert _shim._fallback_query_terms is _classification._fallback_query_terms
    assert _shim._default_selected_evidence_ids is _identifiers._default_selected_evidence_ids
