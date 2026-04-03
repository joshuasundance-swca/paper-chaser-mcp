"""Authoritative tool metadata, visibility, and result policy registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

from pydantic import BaseModel

from ..models import TOOL_INPUT_MODELS
from ..tool_specs.descriptions import TOOL_DESCRIPTIONS

ResourceEmitter = Literal[
    "search_session",
    "paper_like",
    "author_like",
    "data_items",
    "smart_results",
    "citation_candidates",
]
ToolHandlerKind = Literal["custom", "search", "agentic", "lookup", "utility"]


@dataclass(frozen=True)
class SearchSessionPolicy:
    """Whether a tool should persist a reusable result set."""

    persist_result_set: bool = False
    query_hint_arg: str | None = None


@dataclass(frozen=True)
class ToolResultPolicy:
    """Agent-facing metadata policy for a tool response."""

    search_session: SearchSessionPolicy = field(default_factory=SearchSessionPolicy)
    hint_profile: str | None = None
    clarification_profile: str | None = None
    resource_emitters: tuple[ResourceEmitter, ...] = ()


@dataclass(frozen=True)
class ToolExecutionSpec:
    """Declarative execution classification for tool routing."""

    kind: ToolHandlerKind


@dataclass(frozen=True)
class ToolSpec:
    """Published metadata and result policy for one tool."""

    name: str
    input_model: type[BaseModel]
    description: str
    tags: tuple[str, ...]
    profiles: tuple[Literal["guided", "expert"], ...]
    execution: ToolExecutionSpec
    result_policy: ToolResultPolicy


def _default_tags(name: str) -> tuple[str, ...]:
    provider_tags: dict[str, tuple[str, ...]] = {
        "research": ("guided", "research"),
        "follow_up_research": ("guided", "follow-up"),
        "resolve_reference": ("guided", "reference-resolution"),
        "inspect_source": ("guided", "source-inspection"),
        "get_runtime_status": ("guided", "runtime-status"),
        "search_papers": ("search", "brokered"),
        "search_papers_smart": ("search", "smart", "agentic"),
        "resolve_citation": ("known-item", "citation-repair", "recovery"),
        "search_papers_core": ("search", "provider-specific", "provider:core"),
        "search_papers_semantic_scholar": (
            "search",
            "provider-specific",
            "provider:semantic_scholar",
        ),
        "search_papers_serpapi": (
            "search",
            "provider-specific",
            "provider:serpapi_google_scholar",
        ),
        "search_papers_scholarapi": (
            "search",
            "provider-specific",
            "provider:scholarapi",
        ),
        "search_papers_arxiv": ("search", "provider-specific", "provider:arxiv"),
        "search_papers_openalex": ("search", "provider-specific", "provider:openalex"),
        "list_papers_scholarapi": (
            "search",
            "provider-specific",
            "provider:scholarapi",
        ),
        "paper_autocomplete_openalex": (
            "search",
            "provider-specific",
            "provider:openalex",
        ),
        "search_papers_openalex_bulk": (
            "search",
            "provider-specific",
            "provider:openalex",
        ),
        "search_entities_openalex": (
            "search",
            "provider-specific",
            "provider:openalex",
        ),
        "search_papers_openalex_by_entity": (
            "search",
            "provider-specific",
            "provider:openalex",
        ),
        "search_papers_serpapi_cited_by": (
            "search",
            "provider-specific",
            "provider:serpapi_google_scholar",
        ),
        "search_papers_serpapi_versions": (
            "search",
            "provider-specific",
            "provider:serpapi_google_scholar",
        ),
        "get_author_profile_serpapi": (
            "author",
            "provider-specific",
            "provider:serpapi_google_scholar",
        ),
        "get_author_articles_serpapi": (
            "author",
            "provider-specific",
            "provider:serpapi_google_scholar",
        ),
        "get_serpapi_account_status": (
            "provider-specific",
            "provider:serpapi_google_scholar",
        ),
        "get_paper_metadata_crossref": (
            "paper",
            "provider-specific",
            "provider:crossref",
        ),
        "get_paper_open_access_unpaywall": (
            "paper",
            "provider-specific",
            "provider:unpaywall",
        ),
        "search_species_ecos": (
            "search",
            "species",
            "provider-specific",
            "provider:ecos",
        ),
        "get_species_profile_ecos": ("species", "provider-specific", "provider:ecos"),
        "list_species_documents_ecos": (
            "species",
            "documents",
            "provider-specific",
            "provider:ecos",
        ),
        "get_document_text_ecos": (
            "documents",
            "provider-specific",
            "provider:ecos",
        ),
        "search_federal_register": (
            "search",
            "regulations",
            "provider-specific",
            "provider:federal_register",
        ),
        "get_federal_register_document": (
            "regulations",
            "provider-specific",
            "provider:govinfo",
        ),
        "get_cfr_text": (
            "regulations",
            "provider-specific",
            "provider:govinfo",
        ),
        "enrich_paper": ("paper", "enrichment"),
        "get_provider_diagnostics": ("diagnostics", "provider-health"),
        "ask_result_set": ("smart", "grounded-answer"),
        "map_research_landscape": ("smart", "landscape"),
        "expand_research_graph": ("smart", "graph"),
    }
    if name in provider_tags:
        return provider_tags[name]
    if name.startswith("search_"):
        return ("search",)
    if name.startswith("get_paper_"):
        return ("paper",)
    if name.startswith("get_author_") or name == "search_authors":
        return ("author",)
    if name.startswith("batch_"):
        return ("batch",)
    return ("paper-chaser",)


def _default_execution_kind(name: str) -> ToolHandlerKind:
    if name in {
        "research",
        "follow_up_research",
        "resolve_reference",
        "inspect_source",
        "get_runtime_status",
        "search_papers",
        "search_papers_core",
        "search_papers_semantic_scholar",
        "search_papers_serpapi",
        "search_papers_scholarapi",
        "search_papers_arxiv",
        "search_papers_openalex",
        "search_papers_openalex_bulk",
        "list_papers_scholarapi",
        "search_papers_bulk",
        "search_papers_match",
        "resolve_citation",
        "search_snippets",
        "search_authors",
        "search_authors_openalex",
        "search_entities_openalex",
        "search_papers_openalex_by_entity",
        "search_federal_register",
    }:
        return "search"
    if name in {
        "search_papers_smart",
        "ask_result_set",
        "map_research_landscape",
        "expand_research_graph",
    }:
        return "agentic"
    if name.startswith("get_") or name.startswith("batch_"):
        return "lookup"
    return "utility"


def _default_search_session_policy(name: str) -> SearchSessionPolicy:
    if name in {
        "research",
        "search_papers",
        "search_papers_core",
        "search_papers_semantic_scholar",
        "search_papers_serpapi",
        "search_papers_scholarapi",
        "search_papers_arxiv",
        "search_papers_openalex",
        "search_papers_openalex_bulk",
        "list_papers_scholarapi",
        "search_papers_bulk",
        "get_paper_citations",
        "get_paper_citations_openalex",
        "get_paper_references",
        "get_paper_references_openalex",
        "get_paper_authors",
        "search_authors",
        "search_authors_openalex",
        "get_author_papers",
        "get_author_papers_openalex",
        "resolve_citation",
        "search_papers_smart",
        "search_federal_register",
    }:
        return SearchSessionPolicy(
            persist_result_set=True,
            query_hint_arg=(
                "citation" if name == "resolve_citation" else ("reference" if name == "resolve_reference" else "query")
            ),
        )
    return SearchSessionPolicy()


def _default_hint_profile(name: str) -> str | None:
    if name == "research":
        return "research"
    if name == "follow_up_research":
        return "follow_up_research"
    if name == "resolve_reference":
        return "resolve_reference"
    if name == "inspect_source":
        return "inspect_source"
    if name == "get_runtime_status":
        return "get_runtime_status"
    if name in {
        "search_papers",
        "search_papers_core",
        "search_papers_semantic_scholar",
        "search_papers_serpapi",
        "search_papers_scholarapi",
        "search_papers_arxiv",
        "search_papers_openalex",
        "search_papers_openalex_bulk",
        "list_papers_scholarapi",
        "search_papers_bulk",
    }:
        return "paper_search"
    if name == "resolve_citation":
        return "resolve_citation"
    if name in {"search_papers_match", "get_paper_details"}:
        return "paper_anchor"
    if name in {
        "get_paper_metadata_crossref",
        "get_paper_open_access_unpaywall",
        "enrich_paper",
    }:
        return "paper_enrichment"
    if name in {
        "get_paper_citations",
        "get_paper_citations_openalex",
        "get_paper_references",
        "get_paper_references_openalex",
        "expand_research_graph",
    }:
        return "paper_expansion"
    if name == "search_authors":
        return "author_search"
    if name == "search_authors_openalex":
        return "author_search_openalex"
    if name == "get_author_papers":
        return "author_papers"
    if name == "get_author_papers_openalex":
        return "author_papers_openalex"
    if name == "ask_result_set":
        return "ask_result_set"
    if name == "map_research_landscape":
        return "map_research_landscape"
    if name == "search_snippets":
        return "search_snippets"
    if name == "search_species_ecos":
        return "search_species_ecos"
    if name == "get_species_profile_ecos":
        return "get_species_profile_ecos"
    if name == "list_species_documents_ecos":
        return "list_species_documents_ecos"
    if name == "get_document_text_ecos":
        return "get_document_text_ecos"
    if name == "search_federal_register":
        return "search_federal_register"
    if name == "get_federal_register_document":
        return "get_federal_register_document"
    if name == "get_cfr_text":
        return "get_cfr_text"
    return None


def _default_clarification_profile(name: str) -> str | None:
    if name in {
        "research",
        "resolve_reference",
        "search_papers",
        "search_papers_match",
        "resolve_citation",
        "search_authors",
    }:
        return name
    return None


def _default_resource_emitters(name: str) -> tuple[ResourceEmitter, ...]:
    emitters: list[ResourceEmitter] = []
    if _default_search_session_policy(name).persist_result_set:
        emitters.append("search_session")
    if name in {
        "research",
        "search_papers",
        "search_papers_core",
        "search_papers_semantic_scholar",
        "search_papers_serpapi",
        "search_papers_scholarapi",
        "search_papers_arxiv",
        "search_papers_openalex",
        "search_papers_openalex_bulk",
        "list_papers_scholarapi",
        "search_papers_bulk",
        "search_authors",
        "search_authors_openalex",
        "get_author_papers",
        "get_author_papers_openalex",
        "search_federal_register",
    }:
        emitters.extend(("data_items",))
    if name in {
        "inspect_source",
        "search_papers_match",
        "get_paper_details",
        "get_paper_text_scholarapi",
        "get_paper_texts_scholarapi",
        "get_paper_pdf_scholarapi",
        "get_paper_details_openalex",
        "get_paper_citations",
        "get_paper_citations_openalex",
        "get_paper_references",
        "get_paper_references_openalex",
        "get_paper_authors",
        "get_paper_metadata_crossref",
        "get_paper_open_access_unpaywall",
        "enrich_paper",
    }:
        emitters.append("paper_like")
    if name in {"get_author_info", "get_author_info_openalex"}:
        emitters.append("author_like")
    if name in {"search_papers_smart", "ask_result_set", "map_research_landscape"}:
        emitters.append("smart_results")
    if name == "resolve_citation":
        emitters.append("citation_candidates")
    return tuple(dict.fromkeys(emitters))


def _default_profiles(name: str) -> tuple[Literal["guided", "expert"], ...]:
    guided_tools = {
        "research",
        "follow_up_research",
        "resolve_reference",
        "inspect_source",
        "get_runtime_status",
    }
    if name in guided_tools:
        return ("guided", "expert")
    return ("expert",)


def _build_result_policy(name: str) -> ToolResultPolicy:
    return ToolResultPolicy(
        search_session=_default_search_session_policy(name),
        hint_profile=_default_hint_profile(name),
        clarification_profile=_default_clarification_profile(name),
        resource_emitters=_default_resource_emitters(name),
    )


def _build_spec(name: str, input_model: type) -> ToolSpec:
    return ToolSpec(
        name=name,
        input_model=input_model,
        description=TOOL_DESCRIPTIONS[name],
        tags=_default_tags(name),
        profiles=_default_profiles(name),
        execution=ToolExecutionSpec(kind=_default_execution_kind(name)),
        result_policy=_build_result_policy(name),
    )


TOOL_SPECS: dict[str, ToolSpec] = {
    name: _build_spec(name, input_model) for name, input_model in TOOL_INPUT_MODELS.items()
}

_PROVIDER_TAG_ENABLE_FLAG: dict[str, str] = {
    "provider:core": "enable_core",
    "provider:semantic_scholar": "enable_semantic_scholar",
    "provider:openalex": "enable_openalex",
    "provider:scholarapi": "enable_scholarapi",
    "provider:serpapi_google_scholar": "enable_serpapi",
    "provider:crossref": "enable_crossref",
    "provider:unpaywall": "enable_unpaywall",
    "provider:ecos": "enable_ecos",
    "provider:federal_register": "enable_federal_register",
    "provider:arxiv": "enable_arxiv",
}

_AGENTIC_TOOL_NAMES = {
    "search_papers_smart",
    "ask_result_set",
    "map_research_landscape",
    "expand_research_graph",
}

_SEMANTIC_SCHOLAR_GENERIC_TOOL_NAMES = {
    "search_papers_bulk",
    "search_papers_match",
    "paper_autocomplete",
    "get_paper_details",
    "get_paper_citations",
    "get_paper_references",
    "get_paper_authors",
    "get_author_info",
    "get_author_papers",
    "search_authors",
    "batch_get_authors",
    "search_snippets",
    "get_paper_recommendations",
    "get_paper_recommendations_post",
    "batch_get_papers",
    "get_paper_citation_formats",
}


def tool_is_visible(
    name: str,
    *,
    tool_profile: Literal["guided", "expert"] = "expert",
    hide_disabled_tools: bool,
    enabled_flags: Mapping[str, bool] | None = None,
) -> bool:
    """Return whether one tool should be advertised for the current settings."""

    spec = get_tool_spec(name)
    if tool_profile not in spec.profiles:
        return False

    if not hide_disabled_tools:
        return True

    flags = enabled_flags or {}
    if name in _AGENTIC_TOOL_NAMES:
        return flags.get("enable_agentic", False)
    if name in _SEMANTIC_SCHOLAR_GENERIC_TOOL_NAMES:
        return flags.get("enable_semantic_scholar", True)
    if name == "search_papers":
        return any(
            (
                flags.get("enable_semantic_scholar", True),
                flags.get("enable_arxiv", True),
                flags.get("enable_core", True),
                flags.get("enable_serpapi", True),
                flags.get("enable_scholarapi", True),
            )
        )
    if name == "resolve_citation":
        return any(
            (
                flags.get("enable_semantic_scholar", True),
                flags.get("enable_arxiv", True),
                flags.get("enable_core", True),
                flags.get("enable_serpapi", True),
                flags.get("enable_openalex", True),
            )
        )
    if name == "get_federal_register_document":
        return flags.get("enable_federal_register", False) or flags.get("govinfo_available", False)
    if name == "get_cfr_text":
        return flags.get("enable_govinfo_cfr", False)

    for tag in spec.tags:
        flag_name = _PROVIDER_TAG_ENABLE_FLAG.get(tag)
        if flag_name is not None:
            return flags.get(flag_name, True)
    return True


def iter_visible_tool_specs(
    *,
    tool_profile: Literal["guided", "expert"] = "expert",
    hide_disabled_tools: bool,
    enabled_flags: Mapping[str, bool] | None = None,
) -> tuple[ToolSpec, ...]:
    """Return tool specs filtered by visibility settings."""

    return tuple(
        spec
        for spec in iter_tool_specs()
        if tool_is_visible(
            spec.name,
            tool_profile=tool_profile,
            hide_disabled_tools=hide_disabled_tools,
            enabled_flags=enabled_flags,
        )
    )


def iter_tool_specs() -> tuple[ToolSpec, ...]:
    """Return tool specs in published order."""
    return tuple(TOOL_SPECS.values())


def get_tool_spec(name: str) -> ToolSpec:
    """Return one tool spec by name."""
    return TOOL_SPECS[name]


def tool_tags(name: str) -> tuple[str, ...]:
    """Return published tags for one tool."""
    return get_tool_spec(name).tags


__all__ = [
    "SearchSessionPolicy",
    "TOOL_SPECS",
    "ToolExecutionSpec",
    "ToolResultPolicy",
    "ToolSpec",
    "get_tool_spec",
    "iter_tool_specs",
    "iter_visible_tool_specs",
    "tool_is_visible",
    "tool_tags",
]
