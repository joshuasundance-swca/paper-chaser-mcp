"""Identity checks pinning the providers/openai subpackage layout.

These tests guard the Phase 8c extraction: OpenAIProviderBundle lives
in ``providers/openai/bundle.py``, AzureOpenAIProviderBundle lives in
``providers/openai/azure.py``, and ``agentic/provider_openai.py`` remains
a thin re-export shim for the original import path.
"""

from paper_chaser_mcp.agentic import provider_openai as _shim
from paper_chaser_mcp.agentic.providers import openai as _openai_pkg
from paper_chaser_mcp.agentic.providers.openai import azure as _azure
from paper_chaser_mcp.agentic.providers.openai import bundle as _bundle


def test_bundle_module_owns_openai_provider() -> None:
    value = getattr(_bundle, "OpenAIProviderBundle")
    assert getattr(_openai_pkg, "OpenAIProviderBundle") is value
    assert getattr(_shim, "OpenAIProviderBundle") is value
    assert value.__module__ == "paper_chaser_mcp.agentic.providers.openai.bundle"


def test_azure_module_owns_azure_openai_provider() -> None:
    value = getattr(_azure, "AzureOpenAIProviderBundle")
    assert getattr(_openai_pkg, "AzureOpenAIProviderBundle") is value
    assert getattr(_shim, "AzureOpenAIProviderBundle") is value
    assert value.__module__ == "paper_chaser_mcp.agentic.providers.openai.azure"


def test_shim_public_surface_matches_subpackage() -> None:
    assert _shim.OpenAIProviderBundle is _openai_pkg.OpenAIProviderBundle
    assert _shim.AzureOpenAIProviderBundle is _openai_pkg.AzureOpenAIProviderBundle
