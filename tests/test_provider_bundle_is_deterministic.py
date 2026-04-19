"""Provenance contract: only DeterministicProviderBundle is ``is_deterministic``.

Real LLM-backed bundles subclass :class:`DeterministicProviderBundle` so they
inherit its offline fallback methods, but the ``is_deterministic`` property
must report ``False`` for them. Otherwise provenance consumers (e.g.
``subject_grounding.resolve_subject_card`` via ``planner.classify_query``)
silently stamp ``source="deterministic_fallback"`` on real LLM planner calls.

See Finding 1 of the Phase 6 rubber-duck review.
"""

from __future__ import annotations

from paper_chaser_mcp.agentic.config import AgenticConfig
from paper_chaser_mcp.agentic.provider_base import DeterministicProviderBundle
from paper_chaser_mcp.agentic.provider_langchain import (
    AnthropicProviderBundle,
    GoogleProviderBundle,
    HuggingFaceProviderBundle,
    LangChainChatProviderBundle,
    MistralProviderBundle,
    NvidiaProviderBundle,
    OpenRouterProviderBundle,
)
from paper_chaser_mcp.agentic.provider_openai import (
    AzureOpenAIProviderBundle,
    OpenAIProviderBundle,
)


def _config(provider: str = "openai") -> AgenticConfig:
    return AgenticConfig(
        enabled=True,
        provider=provider,
        planner_model="gpt-5.4-mini",
        synthesis_model="gpt-5.4",
        embedding_model="text-embedding-3-large",
        index_backend="memory",
        session_ttl_seconds=1800,
        enable_trace_log=False,
    )


def test_deterministic_bundle_reports_deterministic() -> None:
    bundle = DeterministicProviderBundle(_config())
    assert bundle.is_deterministic is True


def test_openai_bundle_reports_non_deterministic() -> None:
    bundle = OpenAIProviderBundle(_config(), api_key="sk-test")
    assert bundle.is_deterministic is False


def test_azure_openai_bundle_reports_non_deterministic() -> None:
    bundle = AzureOpenAIProviderBundle(
        _config("azure-openai"),
        api_key="az-test",
        azure_endpoint="https://example.openai.azure.com",
        api_version="2024-02-15-preview",
    )
    assert bundle.is_deterministic is False


def test_langchain_chat_bundle_reports_non_deterministic() -> None:
    bundle = LangChainChatProviderBundle(_config(), provider_name="langchain-chat", api_key="lc-test")
    assert bundle.is_deterministic is False


def test_langchain_subclasses_report_non_deterministic() -> None:
    for cls in (
        AnthropicProviderBundle,
        GoogleProviderBundle,
        MistralProviderBundle,
        NvidiaProviderBundle,
        OpenRouterProviderBundle,
        HuggingFaceProviderBundle,
    ):
        bundle = cls(_config(), api_key="stub-key")
        assert bundle.is_deterministic is False, (
            f"{cls.__name__} must override is_deterministic to False so "
            "provenance is not silently stamped as deterministic_fallback"
        )
