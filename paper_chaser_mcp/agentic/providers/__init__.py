"""Compatibility facade for smart-layer provider bundles."""

from __future__ import annotations

from ...provider_runtime import (
    ProviderDiagnosticsRegistry,
    execute_provider_call,
    execute_provider_call_sync,
)
from ..config import AgenticConfig
from ..provider_base import DeterministicProviderBundle, ModelProviderBundle
from ..provider_helpers import (
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
from ..provider_langchain import (
    AnthropicProviderBundle,
    GoogleProviderBundle,
    HuggingFaceProviderBundle,
    LangChainChatProviderBundle,
    MistralProviderBundle,
    NvidiaProviderBundle,
    OpenRouterProviderBundle,
)
from ..provider_openai import AzureOpenAIProviderBundle, OpenAIProviderBundle

__all__ = [
    "COMMON_QUERY_WORDS",
    "AnthropicProviderBundle",
    "AzureOpenAIProviderBundle",
    "DeterministicProviderBundle",
    "GoogleProviderBundle",
    "HuggingFaceProviderBundle",
    "LangChainChatProviderBundle",
    "MistralProviderBundle",
    "ModelProviderBundle",
    "NvidiaProviderBundle",
    "OpenRouterProviderBundle",
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
    nvidia_api_key: str | None = None,
    nvidia_nim_base_url: str | None = None,
    google_api_key: str | None = None,
    mistral_api_key: str | None = None,
    huggingface_api_key: str | None = None,
    huggingface_base_url: str = "https://router.huggingface.co/v1",
    openrouter_api_key: str | None = None,
    openrouter_base_url: str = "https://openrouter.ai/api/v1",
    openrouter_http_referer: str | None = None,
    openrouter_title: str | None = None,
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
    if config.provider == "nvidia":
        return NvidiaProviderBundle(
            config,
            nvidia_api_key,
            base_url=nvidia_nim_base_url,
            provider_registry=provider_registry,
        )
    if config.provider == "google":
        return GoogleProviderBundle(
            config,
            google_api_key,
            provider_registry=provider_registry,
        )
    if config.provider == "mistral":
        return MistralProviderBundle(
            config,
            mistral_api_key,
            provider_registry=provider_registry,
        )
    if config.provider == "huggingface":
        return HuggingFaceProviderBundle(
            config,
            huggingface_api_key,
            base_url=huggingface_base_url,
            provider_registry=provider_registry,
        )
    if config.provider == "openrouter":
        return OpenRouterProviderBundle(
            config,
            openrouter_api_key,
            base_url=openrouter_base_url,
            http_referer=openrouter_http_referer,
            title=openrouter_title,
            provider_registry=provider_registry,
        )
    return OpenAIProviderBundle(
        config,
        openai_api_key,
        provider_registry=provider_registry,
    )
