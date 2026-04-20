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


# ---------------------------------------------------------------------------
# Signature pins
# ---------------------------------------------------------------------------
#
# Phase 1 pin: lock the parameter *shape* (ordered names + kind + whether a
# default exists) for every hard-public function that Phases 6-10 might
# refactor in place. We deliberately ignore type annotations because Phase 2+
# facade work will normalize those; any rename, reorder, kind change, or
# default add/remove surfaces as a failure.
#
# Default values are only pinned when they are simple literals (``None``,
# bool, int, str, or empty tuple) because richer defaults (e.g. dataclass
# instances, dict constructors) are expected to evolve. Other defaults are
# encoded as the sentinel ``_DEFAULT_OPAQUE`` - their *presence* is still
# pinned but their value is not.

_DEFAULT_MISSING = "__NO_DEFAULT__"
_DEFAULT_OPAQUE = "__OPAQUE_DEFAULT__"


def _default_token(param: inspect.Parameter) -> str:
    """Return a stable string representation of the parameter default."""
    if param.default is inspect.Parameter.empty:
        return _DEFAULT_MISSING
    value = param.default
    if value is None or isinstance(value, (bool, int, str)):
        return f"literal::{value!r}"
    if isinstance(value, tuple) and len(value) == 0:
        return "literal::()"
    return _DEFAULT_OPAQUE


def _signature_shape(fn: Any) -> tuple[tuple[str, str, str], ...]:
    """Compute a cosmetic-change-resilient shape for ``fn``'s signature.

    The shape is an ordered tuple of ``(name, kind, default_token)``
    triples. Kinds are kept as enum names (e.g. ``KEYWORD_ONLY``) so a
    positional→keyword-only migration shows up as a signature change.
    """
    sig = inspect.signature(fn)
    return tuple((p.name, p.kind.name, _default_token(p)) for p in sig.parameters.values())


# Expected shapes frozen at Phase 1 baseline. Read from the production
# modules at the time this test was authored. Update intentionally when a
# refactor legitimately changes the contract (and explain the change in the
# commit message).
EXPECTED_SIGNATURE_SHAPES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "paper_chaser_mcp.dispatch:dispatch_tool": (
        ("name", "POSITIONAL_OR_KEYWORD", _DEFAULT_MISSING),
        ("arguments", "POSITIONAL_OR_KEYWORD", _DEFAULT_MISSING),
        ("client", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("core_client", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("openalex_client", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("scholarapi_client", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("arxiv_client", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("enable_core", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("enable_semantic_scholar", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("enable_openalex", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("enable_scholarapi", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("enable_arxiv", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("serpapi_client", "KEYWORD_ONLY", "literal::None"),
        ("enable_serpapi", "KEYWORD_ONLY", "literal::False"),
        ("crossref_client", "KEYWORD_ONLY", "literal::None"),
        ("unpaywall_client", "KEYWORD_ONLY", "literal::None"),
        ("ecos_client", "KEYWORD_ONLY", "literal::None"),
        ("federal_register_client", "KEYWORD_ONLY", "literal::None"),
        ("govinfo_client", "KEYWORD_ONLY", "literal::None"),
        ("enable_crossref", "KEYWORD_ONLY", "literal::True"),
        ("enable_unpaywall", "KEYWORD_ONLY", "literal::True"),
        ("enable_ecos", "KEYWORD_ONLY", "literal::True"),
        ("enable_federal_register", "KEYWORD_ONLY", "literal::True"),
        ("enable_govinfo_cfr", "KEYWORD_ONLY", "literal::True"),
        ("enrichment_service", "KEYWORD_ONLY", "literal::None"),
        ("provider_order", "KEYWORD_ONLY", "literal::None"),
        ("provider_registry", "KEYWORD_ONLY", "literal::None"),
        ("workspace_registry", "KEYWORD_ONLY", "literal::None"),
        ("agentic_runtime", "KEYWORD_ONLY", "literal::None"),
        ("transport_mode", "KEYWORD_ONLY", "literal::'stdio'"),
        ("tool_profile", "KEYWORD_ONLY", "literal::'guided'"),
        ("hide_disabled_tools", "KEYWORD_ONLY", "literal::False"),
        ("session_ttl_seconds", "KEYWORD_ONLY", "literal::None"),
        ("embeddings_enabled", "KEYWORD_ONLY", "literal::None"),
        ("guided_research_latency_profile", "KEYWORD_ONLY", "literal::'deep'"),
        ("guided_follow_up_latency_profile", "KEYWORD_ONLY", "literal::'deep'"),
        ("guided_allow_paid_providers", "KEYWORD_ONLY", "literal::True"),
        ("guided_escalation_enabled", "KEYWORD_ONLY", "literal::True"),
        ("guided_escalation_max_passes", "KEYWORD_ONLY", "literal::2"),
        ("guided_escalation_allow_paid_providers", "KEYWORD_ONLY", "literal::True"),
        ("ctx", "KEYWORD_ONLY", "literal::None"),
        ("allow_elicitation", "KEYWORD_ONLY", "literal::True"),
    ),
    "paper_chaser_mcp.server:main": (),
    "paper_chaser_mcp.server:build_http_app": (
        ("path", "KEYWORD_ONLY", "literal::None"),
        ("transport", "KEYWORD_ONLY", "literal::None"),
        ("middleware", "KEYWORD_ONLY", "literal::None"),
    ),
    "paper_chaser_mcp.tools:get_tool_definitions": (
        ("tool_profile", "KEYWORD_ONLY", "literal::'guided'"),
        ("hide_disabled_tools", "KEYWORD_ONLY", "literal::False"),
        ("enabled_flags", "KEYWORD_ONLY", "literal::None"),
    ),
    "paper_chaser_mcp.tool_specs:iter_visible_tool_specs": (
        ("tool_profile", "KEYWORD_ONLY", "literal::'expert'"),
        ("hide_disabled_tools", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("enabled_flags", "KEYWORD_ONLY", "literal::None"),
    ),
    "paper_chaser_mcp.agentic.planner:classify_query": (
        ("query", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("mode", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("year", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("venue", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("focus", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("provider_bundle", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("request_outcomes", "KEYWORD_ONLY", "literal::None"),
        ("request_id", "KEYWORD_ONLY", "literal::None"),
    ),
    "paper_chaser_mcp.agentic.providers:resolve_provider_bundle": (
        ("config", "POSITIONAL_OR_KEYWORD", _DEFAULT_MISSING),
        ("openai_api_key", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("azure_openai_api_key", "KEYWORD_ONLY", "literal::None"),
        ("azure_openai_endpoint", "KEYWORD_ONLY", "literal::None"),
        ("azure_openai_api_version", "KEYWORD_ONLY", "literal::None"),
        ("azure_openai_planner_deployment", "KEYWORD_ONLY", "literal::None"),
        ("azure_openai_synthesis_deployment", "KEYWORD_ONLY", "literal::None"),
        ("anthropic_api_key", "KEYWORD_ONLY", "literal::None"),
        ("nvidia_api_key", "KEYWORD_ONLY", "literal::None"),
        ("nvidia_nim_base_url", "KEYWORD_ONLY", "literal::None"),
        ("google_api_key", "KEYWORD_ONLY", "literal::None"),
        ("mistral_api_key", "KEYWORD_ONLY", "literal::None"),
        ("huggingface_api_key", "KEYWORD_ONLY", "literal::None"),
        ("huggingface_base_url", "KEYWORD_ONLY", "literal::'https://router.huggingface.co/v1'"),
        ("openrouter_api_key", "KEYWORD_ONLY", "literal::None"),
        ("openrouter_base_url", "KEYWORD_ONLY", "literal::'https://openrouter.ai/api/v1'"),
        ("openrouter_http_referer", "KEYWORD_ONLY", "literal::None"),
        ("openrouter_title", "KEYWORD_ONLY", "literal::None"),
        ("provider_registry", "KEYWORD_ONLY", "literal::None"),
    ),
    "paper_chaser_mcp.provider_runtime:execute_provider_call": (
        ("provider", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("endpoint", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("operation", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("registry", "KEYWORD_ONLY", "literal::None"),
        ("policy", "KEYWORD_ONLY", "literal::None"),
        ("budget", "KEYWORD_ONLY", "literal::None"),
        ("request_outcomes", "KEYWORD_ONLY", "literal::None"),
        ("request_id", "KEYWORD_ONLY", "literal::None"),
        ("is_empty", "KEYWORD_ONLY", "literal::None"),
        ("metadata_extractor", "KEYWORD_ONLY", "literal::None"),
        ("propagate_exceptions", "KEYWORD_ONLY", "literal::()"),
    ),
    "paper_chaser_mcp.citation_repair:parse_citation": (
        ("citation", "POSITIONAL_OR_KEYWORD", _DEFAULT_MISSING),
        ("title_hint", "KEYWORD_ONLY", "literal::None"),
        ("author_hint", "KEYWORD_ONLY", "literal::None"),
        ("year_hint", "KEYWORD_ONLY", "literal::None"),
        ("venue_hint", "KEYWORD_ONLY", "literal::None"),
        ("doi_hint", "KEYWORD_ONLY", "literal::None"),
    ),
    "paper_chaser_mcp.citation_repair:resolve_citation": (
        ("citation", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("max_candidates", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("client", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("enable_core", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("enable_semantic_scholar", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("enable_openalex", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("enable_arxiv", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("enable_serpapi", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("core_client", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("openalex_client", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("arxiv_client", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("serpapi_client", "KEYWORD_ONLY", _DEFAULT_MISSING),
        ("title_hint", "KEYWORD_ONLY", "literal::None"),
        ("author_hint", "KEYWORD_ONLY", "literal::None"),
        ("year_hint", "KEYWORD_ONLY", "literal::None"),
        ("venue_hint", "KEYWORD_ONLY", "literal::None"),
        ("doi_hint", "KEYWORD_ONLY", "literal::None"),
        ("include_enrichment", "KEYWORD_ONLY", "literal::False"),
        ("enrichment_service", "KEYWORD_ONLY", "literal::None"),
    ),
}


# Methods the graphs.md seam map lists as the public contract of
# ``AgenticRuntime``. Phases 6-10 may legitimately refactor internals, but
# every symbol below must remain callable on the class. ``__init__`` is
# included because the seam map explicitly pins it.
AGENTIC_RUNTIME_PUBLIC_METHODS: frozenset[str] = frozenset(
    {
        "__init__",
        "search_papers_smart",
        "ask_result_set",
        "map_research_landscape",
        "expand_research_graph",
        "smart_provider_diagnostics",
        "aclose",
    }
)


@pytest.mark.parametrize(
    ("function_path", "expected_shape"),
    list(EXPECTED_SIGNATURE_SHAPES.items()),
    ids=list(EXPECTED_SIGNATURE_SHAPES.keys()),
)
def test_public_function_signatures(
    function_path: str,
    expected_shape: tuple[tuple[str, str, str], ...],
) -> None:
    """Pin parameter names, kinds, and literal defaults for public functions.

    Type annotations are intentionally *not* pinned (they churn without
    breaking callers). Any rename, reorder, kind migration, or default
    add/remove will show up as a failure. Update the expected shape
    deliberately when a refactor legitimately changes the contract.
    """
    module_name, attr = function_path.split(":")
    module = importlib.import_module(module_name)
    fn = getattr(module, attr)
    actual = _signature_shape(fn)
    assert actual == expected_shape, (
        f"Signature of {function_path} drifted.\nExpected: {expected_shape}\nActual:   {actual}"
    )


def test_agentic_runtime_public_methods() -> None:
    """Every method listed in the graphs.md seam map must exist on AgenticRuntime."""
    from paper_chaser_mcp.agentic.graphs import AgenticRuntime

    missing = [name for name in AGENTIC_RUNTIME_PUBLIC_METHODS if not hasattr(AgenticRuntime, name)]
    assert not missing, f"AgenticRuntime is missing public methods pinned by the Phase 1 seam map: {sorted(missing)}"
    for name in AGENTIC_RUNTIME_PUBLIC_METHODS:
        attr = getattr(AgenticRuntime, name)
        assert callable(attr), f"AgenticRuntime.{name} is no longer callable"


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
