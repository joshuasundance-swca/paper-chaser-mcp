"""Compatibility shim that re-exports the OpenAI provider bundles.

The real implementation moved to
:mod:`paper_chaser_mcp.agentic.providers.openai` during Phase 8c of the
modularization refactor. This module is preserved as a thin pass-through
so existing ``from .provider_openai import ...`` imports (both
package-internal and in the test suite) keep working without a behavior
change.
"""

from __future__ import annotations

from .providers.openai import (  # noqa: F401
    AzureOpenAIProviderBundle,
    OpenAIProviderBundle,
)

__all__ = [
    "AzureOpenAIProviderBundle",
    "OpenAIProviderBundle",
]
