"""Public API surface guard for paper_chaser_mcp.

Phase 1 surface-pinning test. Pins the hard public symbols enumerated in
``docs/refactor-seam-maps/public-api-surface.md`` (a copy lives in the
session-state seam map). Phases 2-11 extractions must preserve every symbol
below: if a refactor has to move a symbol, the owning module must re-export it.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

import pytest

# Hard public contract copied from the Phase 1 seam map
# (public-api-surface.md). Each tuple lists the symbol names cross-referenced
# outside the owning module in production code. Test-only and private (``_``)
# names are intentionally excluded.
PUBLIC_API: dict[str, tuple[str, ...]] = {
    "paper_chaser_mcp.dispatch": ("dispatch_tool",),
    "paper_chaser_mcp.server": (
        "main",
        "build_http_app",
        "app",
        "http_app",
        "settings",
    ),
    "paper_chaser_mcp.tools": (
        "get_tool_definitions",
        "TOOL_DESCRIPTIONS",
        "OPAQUE_CURSOR_CONTRACT",
    ),
    "paper_chaser_mcp.agentic.graphs": ("AgenticRuntime",),
    "paper_chaser_mcp.agentic.planner": (
        "detect_literature_intent",
        "detect_regulatory_intent",
        "classify_query",
        "normalize_query",
        "query_terms",
        "query_facets",
        "looks_like_exact_title",
    ),
    "paper_chaser_mcp.agentic.provider_openai": (
        "OpenAIProviderBundle",
        "AzureOpenAIProviderBundle",
    ),
    "paper_chaser_mcp.agentic.provider_langchain": (
        "LangChainChatProviderBundle",
        "AnthropicProviderBundle",
        "GoogleProviderBundle",
        "MistralProviderBundle",
        "HuggingFaceProviderBundle",
        "NvidiaProviderBundle",
        "OpenRouterProviderBundle",
    ),
    "paper_chaser_mcp.agentic.provider_helpers": (
        "generate_evidence_gaps_without_llm",
        "AnswerStatusValidation",
        "COMMON_QUERY_WORDS",
    ),
    "paper_chaser_mcp.agentic.provider_base": (
        "DeterministicProviderBundle",
        "ModelProviderBundle",
        "classify_relevance_without_llm",
    ),
    "paper_chaser_mcp.agentic.providers": (
        "resolve_provider_bundle",
        "execute_provider_call",
        "execute_provider_call_sync",
    ),
    "paper_chaser_mcp.agentic.workspace": ("WorkspaceRegistry",),
    "paper_chaser_mcp.agentic.answer_modes": (
        "ANSWER_MODES",
        "SYNTHESIS_MODES",
        "LLMModeClassifier",
        "AsyncLLMModeClassifier",
    ),
    "paper_chaser_mcp.agentic.models": (
        "PlannerDecision",
        "StructuredSourceRecord",
        "EvidenceItem",
        "ExpansionCandidate",
        "SubjectCard",
    ),
    "paper_chaser_mcp.citation_repair": (
        "parse_citation",
        "resolve_citation",
        "looks_like_citation_query",
        "looks_like_paper_identifier",
    ),
    "paper_chaser_mcp.eval_curation": ("maybe_capture_eval_candidate",),
    "paper_chaser_mcp.eval_canary": ("run_eval_canary",),
    "paper_chaser_mcp.search": ("search_papers_with_fallback",),
    "paper_chaser_mcp.search_executor": (
        "SearchExecutor",
        "SearchClientBundle",
        "ProviderSearchRequest",
        "arxiv_result",
        "core_result",
        "semantic_result",
        "serpapi_result",
    ),
    "paper_chaser_mcp.enrichment": (
        "PaperEnrichmentService",
        "attach_enrichments_to_paper_payload",
        "hydrate_paper_for_enrichment",
    ),
    "paper_chaser_mcp.provider_runtime": (
        "ProviderDiagnosticsRegistry",
        "execute_provider_call",
        "ProviderOutcomeEnvelope",
        "provider_attempt_reason",
        "provider_status_to_attempt_status",
    ),
    "paper_chaser_mcp.settings": (
        "AppSettings",
        "_env_bool",
        "ToolProfile",
        "AgenticProvider",
    ),
    "paper_chaser_mcp.compat": (
        "build_clarification",
        "augment_tool_result",
    ),
}

# Flat list of (module, symbol) pairs for parametrization.
PUBLIC_API_PAIRS: list[tuple[str, str]] = [
    (module, symbol) for module, symbols in PUBLIC_API.items() for symbol in symbols
]

# The 5 guided-profile tools enumerated in the README / golden paths.
GUIDED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "research",
        "follow_up_research",
        "resolve_reference",
        "inspect_source",
        "get_runtime_status",
    }
)


@pytest.mark.parametrize(
    ("module_name", "symbol_name"),
    PUBLIC_API_PAIRS,
    ids=[f"{m}::{s}" for m, s in PUBLIC_API_PAIRS],
)
def test_public_symbol_is_importable(module_name: str, symbol_name: str) -> None:
    """Every hard-public symbol must be importable from its owning module.

    The private ``_env_bool`` entry on :mod:`paper_chaser_mcp.settings` is
    deliberately kept in this list because it is re-imported by
    :mod:`paper_chaser_mcp.server` as part of the hard public contract today;
    Phase 2+ facade work may promote it to a public name.
    """
    module = importlib.import_module(module_name)
    # getattr-raises-AttributeError is the failure mode we want to catch.
    value: Any = getattr(module, symbol_name)

    # If the symbol looks like a class or function, ensure it is at least
    # introspectable (callable or a class object). Constants, type aliases,
    # and ``frozenset``/``tuple`` values are allowed through unchanged.
    if inspect.isclass(value):
        # Classes must be constructible (either concrete or an ABC); we do not
        # instantiate because many take required config. Confirming it is a
        # type is enough.
        assert isinstance(value, type)
    elif inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value):
        assert callable(value)


def test_public_api_pair_count_matches_surface_map() -> None:
    """Pin the size of the hard public surface so drift is visible in diffs."""
    # Phase-1 baseline: 22 modules / 73 symbols. Update intentionally if the
    # surface map legitimately changes; Phase 12 should tighten or widen both.
    assert len(PUBLIC_API) == 22
    assert len(PUBLIC_API_PAIRS) == 73


def test_guided_profile_exposes_five_tools() -> None:
    """The guided tool profile must expose exactly the 5 guided tools."""
    from paper_chaser_mcp.tools import get_tool_definitions

    guided = get_tool_definitions(tool_profile="guided")
    names = {tool.name for tool in guided}
    assert names == GUIDED_TOOL_NAMES, (
        f"Guided profile drifted. Expected {sorted(GUIDED_TOOL_NAMES)}, got {sorted(names)}"
    )


def test_expert_profile_is_superset_of_guided() -> None:
    """The expert profile must expose every guided tool plus more."""
    from paper_chaser_mcp.tools import get_tool_definitions

    expert = get_tool_definitions(tool_profile="expert")
    expert_names = {tool.name for tool in expert}
    assert GUIDED_TOOL_NAMES.issubset(expert_names), (
        f"Expert profile is missing guided tools: {sorted(GUIDED_TOOL_NAMES - expert_names)}"
    )
    assert len(expert_names) > len(GUIDED_TOOL_NAMES), "Expert profile must be a strict superset of the guided profile."


def test_tool_specs_iter_matches_get_tool_definitions() -> None:
    """The low-level tool-spec iterator must agree with the published schema.

    Pins the seam between :mod:`paper_chaser_mcp.tool_specs` and
    :mod:`paper_chaser_mcp.tools` so Phase 2+ facade work cannot accidentally
    drop or rename tools.
    """
    from paper_chaser_mcp.tool_specs import iter_visible_tool_specs
    from paper_chaser_mcp.tools import get_tool_definitions

    for profile in ("guided", "expert"):
        specs = iter_visible_tool_specs(
            tool_profile=profile,  # type: ignore[arg-type]
            hide_disabled_tools=False,
        )
        defs = get_tool_definitions(tool_profile=profile)  # type: ignore[arg-type]
        assert [spec.name for spec in specs] == [tool.name for tool in defs], (
            f"Tool spec / definition ordering drifted for profile {profile!r}."
        )
