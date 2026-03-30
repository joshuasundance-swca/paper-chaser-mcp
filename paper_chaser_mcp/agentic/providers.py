"""Compatibility facade for smart-layer provider bundles."""

from __future__ import annotations

from ..provider_runtime import (
    ProviderDiagnosticsRegistry,
    execute_provider_call,
    execute_provider_call_sync,
)
from .config import AgenticConfig
from .provider_base import DeterministicProviderBundle, ModelProviderBundle
from .provider_helpers import (
    COMMON_QUERY_WORDS,
    _coerce_langchain_structured_response,
    _cosine_similarity,
    _extract_seed_identifiers,
    _lexical_similarity,
    _normalize_confidence_label,
    _normalized_embedding_text,
    _PlannerConstraintsSchema,
    _PlannerResponseSchema,
    _tokenize,
    _top_terms,
)
from .provider_langchain import (
    AnthropicProviderBundle,
    GoogleProviderBundle,
    LangChainChatProviderBundle,
)
from .provider_openai import AzureOpenAIProviderBundle, OpenAIProviderBundle

__all__ = [
    "COMMON_QUERY_WORDS",
    "AnthropicProviderBundle",
    "AzureOpenAIProviderBundle",
    "DeterministicProviderBundle",
    "GoogleProviderBundle",
    "LangChainChatProviderBundle",
    "ModelProviderBundle",
    "OpenAIProviderBundle",
    "execute_provider_call",
    "execute_provider_call_sync",
    "resolve_provider_bundle",
    "_coerce_langchain_structured_response",
    "_cosine_similarity",
    "_extract_seed_identifiers",
    "_lexical_similarity",
    "_normalize_confidence_label",
    "_normalized_embedding_text",
    "_PlannerConstraintsSchema",
    "_PlannerResponseSchema",
    "_tokenize",
    "_top_terms",
]


def resolve_provider_bundle(
    config: AgenticConfig,
    *,
    openai_api_key: str | None,
    azure_openai_api_key: str | None = None,
    azure_openai_endpoint: str | None = None,
    azure_openai_api_version: str | None = None,
    azure_openai_planner_deployment: str | None = None,
    azure_openai_synthesis_deployment: str | None = None,
    anthropic_api_key: str | None = None,
    google_api_key: str | None = None,
    provider_registry: ProviderDiagnosticsRegistry | None = None,
) -> ModelProviderBundle:
    """Resolve the configured provider bundle with deterministic fallback."""
    if config.provider == "deterministic":
        return DeterministicProviderBundle(config)
    if config.provider == "azure-openai":
        return AzureOpenAIProviderBundle(
            config,
            azure_openai_api_key,
            azure_openai_endpoint,
            azure_openai_api_version,
            azure_planner_deployment=azure_openai_planner_deployment,
            azure_synthesis_deployment=azure_openai_synthesis_deployment,
            provider_registry=provider_registry,
        )
    if config.provider == "anthropic":
        return AnthropicProviderBundle(
            config,
            anthropic_api_key,
            provider_registry=provider_registry,
        )
    if config.provider == "google":
        return GoogleProviderBundle(
            config,
            google_api_key,
            provider_registry=provider_registry,
        )
    return OpenAIProviderBundle(
        config,
        openai_api_key,
        provider_registry=provider_registry,
    )
