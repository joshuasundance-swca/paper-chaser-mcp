"""Compatibility shim that re-exports the LangChain provider bundles.

The real implementation moved to
:mod:`paper_chaser_mcp.agentic.providers.langchain` during Phase 8d of the
modularization refactor. This module is preserved as a thin pass-through
so existing ``from .provider_langchain import ...`` imports (both
package-internal and in the test suite) keep working without a behavior
change.
"""

from __future__ import annotations

from .providers.langchain import (  # noqa: F401
    AnthropicProviderBundle,
    GoogleProviderBundle,
    HuggingFaceProviderBundle,
    LangChainChatProviderBundle,
    MistralProviderBundle,
    NvidiaProviderBundle,
    OpenRouterProviderBundle,
)

__all__ = [
    "AnthropicProviderBundle",
    "GoogleProviderBundle",
    "HuggingFaceProviderBundle",
    "LangChainChatProviderBundle",
    "MistralProviderBundle",
    "NvidiaProviderBundle",
    "OpenRouterProviderBundle",
]
