"""Identity checks pinning the providers/langchain subpackage layout.

These tests guard the Phase 8d extraction: LangChainChatProviderBundle
lives in ``providers/langchain/bundle.py``, the six concrete adapter
classes (Anthropic, Google, NVIDIA, Mistral, HuggingFace, OpenRouter)
live in ``providers/langchain/adapters.py``, and
``agentic/provider_langchain.py`` remains a thin re-export shim for the
original import path.
"""

from paper_chaser_mcp.agentic import provider_langchain as _shim
from paper_chaser_mcp.agentic.providers import langchain as _langchain_pkg
from paper_chaser_mcp.agentic.providers.langchain import adapters as _adapters
from paper_chaser_mcp.agentic.providers.langchain import bundle as _bundle

_ADAPTER_NAMES = (
    "AnthropicProviderBundle",
    "GoogleProviderBundle",
    "HuggingFaceProviderBundle",
    "MistralProviderBundle",
    "NvidiaProviderBundle",
    "OpenRouterProviderBundle",
)


def test_bundle_module_owns_langchain_chat_bundle() -> None:
    value = getattr(_bundle, "LangChainChatProviderBundle")
    assert getattr(_langchain_pkg, "LangChainChatProviderBundle") is value
    assert getattr(_shim, "LangChainChatProviderBundle") is value
    assert value.__module__ == "paper_chaser_mcp.agentic.providers.langchain.bundle"


def test_adapters_module_owns_concrete_provider_bundles() -> None:
    for name in _ADAPTER_NAMES:
        value = getattr(_adapters, name)
        assert getattr(_langchain_pkg, name) is value, name
        assert getattr(_shim, name) is value, name
        assert value.__module__ == "paper_chaser_mcp.agentic.providers.langchain.adapters", name


def test_shim_public_surface_matches_subpackage() -> None:
    for name in ("LangChainChatProviderBundle", *_ADAPTER_NAMES):
        assert getattr(_shim, name) is getattr(_langchain_pkg, name), name
