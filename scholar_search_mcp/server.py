"""FastMCP-backed public server surface for Scholar Search."""

import json
import logging
from contextlib import asynccontextmanager
from inspect import Parameter, Signature
from typing import Any, Literal, cast

from fastmcp import Context, FastMCP
from fastmcp.server.middleware.timing import TimingMiddleware
from mcp.types import TextContent, Tool, ToolAnnotations
from pydantic import Field
from pydantic.fields import PydanticUndefined

from .agentic import (
    AgenticConfig,
    AgenticRuntime,
    WorkspaceRegistry,
    resolve_provider_bundle,
)
from .clients import (
    ArxivClient,
    CoreApiClient,
    CrossrefClient,
    EcosClient,
    OpenAlexClient,
    SemanticScholarClient,
    UnpaywallClient,
)
from .clients.serpapi import SerpApiScholarClient
from .compat import sanitize_published_schema
from .constants import (
    API_BASE_URL,
    ARXIV_API_BASE,
    ARXIV_NS,
    ATOM_NS,
    CORE_API_BASE,
    DEFAULT_AUTHOR_FIELDS,
    DEFAULT_PAPER_FIELDS,
    MAX_429_RETRIES,
    OPENSEARCH_NS,
    RECOMMENDATIONS_BASE_URL,
    SEMANTIC_SCHOLAR_MIN_INTERVAL,
)
from .dispatch import dispatch_tool
from .enrichment import PaperEnrichmentService
from .models import TOOL_INPUT_MODELS, dump_jsonable
from .parsing import _arxiv_id_from_url, _text
from .provider_runtime import ProviderDiagnosticsRegistry
from .runtime import run_server
from .search import _core_response_to_merged, _merge_search_results
from .settings import AppSettings, _env_bool
from .tools import TOOL_DESCRIPTIONS
from .transport import asyncio, httpx, maybe_close_async_resource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scholar-search-mcp")

SERVER_INSTRUCTIONS = """
Decision tree for tool selection:

1. CONCEPT-LEVEL DISCOVERY / REVIEW → search_papers_smart
   (returns searchSessionId, strategyMetadata, resourceUris, and agentHints;
   use latencyProfile=fast for smoke tests, balanced for default work, and deep
   for controlled multi-provider expansion)
2. QUICK RAW DISCOVERY → search_papers
   (brokered, single page, returns brokerMetadata plus agentHints/resourceUris)
3. EXHAUSTIVE / MULTI-PAGE → search_papers_bulk
   (cursor-paginated, up to 1 000 returned/call; read retrievalNote because
   default bulk ordering is not relevance-ranked)
4. CITATION REPAIR / ALMOST-RIGHT REFERENCES → resolve_citation
5. KNOWN ITEM (messy title) → search_papers_match
6. KNOWN ITEM (DOI / arXiv / URL) → get_paper_details
7. PAPER ENRICHMENT / OA CHECK → get_paper_metadata_crossref,
   get_paper_open_access_unpaywall, or enrich_paper after you already have a
   concrete paper, DOI, or DOI-bearing identifier
8. GROUNDED FOLLOW-UP → ask_result_set or map_research_landscape using searchSessionId
9. CITATION EXPANSION → get_paper_citations (cited-by) or get_paper_references (refs)
10. AUTHOR PIVOT → search_authors → get_author_info → get_author_papers
11. PHRASE / QUOTE RECOVERY → search_snippets (last resort)
12. OPENALEX-SPECIFIC PATHS → use the *_openalex tools when you explicitly need
   OpenAlex-native DOI/ID lookup, OpenAlex cursor paging, author pivots, or
   source/institution/topic pivots via search_entities_openalex and
   search_papers_openalex_by_entity
13. SERPAPI RECOVERY PATHS → use search_papers_serpapi_cited_by,
   search_papers_serpapi_versions, get_author_profile_serpapi,
   get_author_articles_serpapi, or get_serpapi_account_status only when
   SCHOLAR_SEARCH_ENABLE_SERPAPI=true and the workflow justifies paid recall recovery
14. ECOS SPECIES DOSSIERS → search_species_ecos → get_species_profile_ecos →
   list_species_documents_ecos → get_document_text_ecos for species pages,
   regulatory documents, and recovery PDFs from the U.S. Fish and Wildlife
   Service ECOS system
15. PROVIDER HEALTH / DEBUGGING → get_provider_diagnostics

After search_papers: read brokerMetadata.nextStepHint for the recommended next move.
After search_papers_smart: reuse searchSessionId for ask_result_set,
map_research_landscape, or expand_research_graph, and inspect acceptedExpansions,
rejectedExpansions, speculativeExpansions, providersUsed, driftWarnings,
latencyProfile, providerBudgetApplied, and providerOutcomes. Set
includeEnrichment=true only when you want Crossref and Unpaywall metadata on the
final smart-ranked hits; enrichment is post-ranking only and never changes
retrieval or provider ordering.
Primary read tools now also return agentHints, clarification, resourceUris, and,
when they produce reusable result sets, searchSessionId.
For known-item flows, includeEnrichment=true on search_papers_match,
get_paper_details, or resolve_citation adds Crossref and Unpaywall metadata only
after the base paper resolution succeeds.
For Semantic Scholar expansion tools, prefer paper.recommendedExpansionId when
present. If paper.expansionIdStatus is not_portable, do not retry with brokered
paperId/sourceId/canonicalId values; resolve the paper through DOI or a
Semantic Scholar-native lookup first.
If search_papers_match returns no match, or if the user has a broken
bibliography line, partial reference, or almost-right citation, prefer
resolve_citation before guessing. A no-match can still mean the item is a
dissertation, software release, report, or other output outside the indexed
paper surface.
For common-name author lookup, add affiliation, coauthor, venue, or topic clues
before expanding into get_author_info/get_author_papers.
To steer the broker: use preferredProvider (try-first) or providerOrder (full override).
Provider names: semantic_scholar, arxiv, core, serpapi / serpapi_google_scholar.
Provider-specific search inputs: search_papers_core, search_papers_serpapi, and
search_papers_arxiv only accept query/limit/year; search_papers_semantic_scholar
supports the wider Semantic Scholar filter set. OpenAlex is available through
explicit *_openalex tools instead of the broker because its citation, author,
and pagination semantics differ from Semantic Scholar.
Continuation rule: search_papers_bulk is the closest continuation path only for
Semantic Scholar-style retrieval; from CORE, arXiv, or SerpApi results it is a
Semantic Scholar pivot rather than another page from the same provider.
Even on Semantic Scholar paths, default bulk ordering is NOT relevance-ranked;
it is not 'page 2' of search_papers. Read retrievalNote in each bulk response,
or pass sort='citationCount:desc' for citation-ranked bulk traversal.
For small targeted pages, prefer search_papers or search_papers_semantic_scholar;
Semantic Scholar's bulk endpoint may ignore small limits internally.
For agentic UX review loops, run a small smoke baseline first, then widen into
OpenAlex, snippet recovery, paper-to-author pivots, or a feature-specific probe
only when the workflow goal calls for broader coverage. Capture any defects as
reproduction-ready issues that can guide code changes and documentation updates.

Pagination rule: treat pagination.nextCursor as opaque — pass it back exactly as
returned, do not derive, edit, or fabricate it, and do not reuse it across a
different tool or query flow.
""".strip()

AGENT_WORKFLOW_GUIDE = """
# Scholar Search agent workflow guide

## Quick decision tree

- **Concept-level discovery or literature review**: `search_papers_smart` →
  reuse `searchSessionId` with `ask_result_set`, `map_research_landscape`,
  or `expand_research_graph`. Use `latencyProfile=fast` for smoke baselines,
  `balanced` for the default path, and `deep` when controlled multi-provider
  expansion is worth the extra latency.
- **Quick literature discovery**: `search_papers` → inspect
  `brokerMetadata.nextStepHint`, `agentHints`, and `resourceUris` to decide
  whether to broaden, narrow, paginate, or pivot.
- **Exhaustive / multi-page retrieval**: `search_papers_bulk` with cursor loop until
  `pagination.hasMore` is false; read `retrievalNote` because default bulk
  ordering is not relevance-ranked.
- **Small targeted Semantic Scholar page**: `search_papers_semantic_scholar` (or
  `search_papers` if brokered discovery is fine) instead of bulk retrieval.
- **Citation repair / incomplete references**: `resolve_citation`
- **Known-item lookup (messy title)**: `search_papers_match`
- **Known-item lookup (DOI / arXiv / URL / S2 ID)**: `get_paper_details`
- **Post-resolution paper enrichment**: `get_paper_metadata_crossref`,
  `get_paper_open_access_unpaywall`, or `enrich_paper` after you already have a
  paper or DOI. For known-item and smart flows, opt in with
  `includeEnrichment=true` when you want additive metadata without changing the
  retrieval path.
- **Citation chasing (cited-by expansion)**: `get_paper_citations`
- **Citation chasing (backward references)**: `get_paper_references`
- **Author-centric workflows**: `search_authors` → `get_author_info` →
  `get_author_papers`; pivot to `get_paper_authors` if starting from a paper.
- **Common-name author disambiguation**: add affiliation, coauthor, venue, or
  topic clues to `search_authors`, then confirm identity with
  `get_author_info`/`get_author_papers`.
- **Cross-provider ID portability**: for Semantic Scholar expansion tools prefer
  `paper.recommendedExpansionId` when it is present. If
  `paper.expansionIdStatus` is `not_portable`, brokered `paperId`, `sourceId`,
  and `canonicalId` values are still provider-specific and must be resolved
  through DOI or a Semantic Scholar-native lookup first.
- **Outside-paper outputs**: dissertations, software releases, reports, and
  other grey literature may fall outside the indexed paper surface even when a
  title is real; treat a structured no-match from `search_papers_match` as a
  signal to verify externally.
- **Almost-right citations or broken bibliography lines**: prefer
  `resolve_citation` before bouncing between title match, snippets, and broad
  search. The resolver returns confidence, alternatives, conflicts, and the
  fastest next clue to add.
- **Quote or snippet validation**: `search_snippets` — special-purpose recovery
  tool only when title/keyword search is weak; provider 4xx/5xx errors degrade
  to empty results with retry guidance.
- **Citation export**: `get_paper_citation_formats` — pass
  `result_id=paper.scholarResultId` (not `paper.sourceId`) from any
  `serpapi_google_scholar` result to get MLA, APA, BibTeX, etc.
- **OpenAlex-specific workflows**: use `search_papers_openalex` for one explicit
  OpenAlex page, `search_papers_openalex_bulk` for cursor-paginated OpenAlex
  traversal, `get_paper_details_openalex` for OpenAlex ID/DOI lookup, and the
  OpenAlex citation/author tools when you want OpenAlex-native semantics. Use
  `paper_autocomplete_openalex`, `search_entities_openalex`, and
  `search_papers_openalex_by_entity` for title hints plus venue, affiliation,
  or topic pivots.
- **SerpApi recovery workflows**: keep SerpApi off the default broad path and use
  `search_papers_serpapi_cited_by`, `search_papers_serpapi_versions`,
  `get_author_profile_serpapi`, `get_author_articles_serpapi`, and
  `get_serpapi_account_status` only when Google Scholar recall recovery or quota
  inspection is explicitly needed.
- **Provider/runtime debugging**: `get_provider_diagnostics`
- **ECOS species dossiers**: `search_species_ecos` -> `get_species_profile_ecos`
  -> `list_species_documents_ecos` -> `get_document_text_ecos`
- **Grounded follow-up over a saved result set**: `ask_result_set` for QA,
  claim checks, or comparisons; `map_research_landscape` for themes, gaps, and
  disagreements; `expand_research_graph` for a compact citation or author graph.

## Provider steering

Set `preferredProvider` on `search_papers` to try one provider first while keeping
the fallback chain. Set `providerOrder` to override the full broker chain for one
call. Use `search_papers_core`, `search_papers_semantic_scholar`,
`search_papers_serpapi`, or `search_papers_arxiv` for single-source searches.

## Provider-specific tool contracts

- `search_papers_core`, `search_papers_serpapi`, and `search_papers_arxiv`
    expose only `query`, `limit`, and `year`.
- `search_papers_openalex` exposes one explicit OpenAlex page, while
    `search_papers_openalex_bulk` exposes OpenAlex cursor pagination.
- `search_papers_semantic_scholar` exposes the wider Semantic Scholar-compatible
    filter set.

## Continuation vs pivot

- `search_papers_bulk` is the closest continuation path when the task is already
    aligned with Semantic Scholar retrieval semantics.
- Even on Semantic Scholar paths, default bulk ordering is NOT relevance-ranked;
    it is not "page 2" of `search_papers`. Read `retrievalNote` and use
    `sort='citationCount:desc'` for citation-ranked bulk traversal.
- If `search_papers` returned CORE, arXiv, or SerpApi results, `search_papers_bulk`
    is a Semantic Scholar pivot, not another page from the same provider.
- Venue-filtered Semantic Scholar searches can also broaden when moved to bulk
    retrieval.
- For small targeted pages, prefer `search_papers` or
    `search_papers_semantic_scholar`; the upstream bulk endpoint may ignore small
    `limit` values internally.

## Pagination contract

For every paginated tool: treat `pagination.nextCursor` as opaque, pass it back
exactly as returned, and do not derive, edit, fabricate, or cross-reuse it.

## Agent-facing metadata

- Primary read tools now return `agentHints`, `clarification`, and
  `resourceUris`.
- Discovery and expansion tools that create reusable result sets also return
  `searchSessionId`.
- The resources surfaced from tool outputs are `paper://{paper_id}`,
  `author://{author_id}`, `search://{searchSessionId}`, and
  `trail://paper/{paper_id}?direction=citations|references`.

## Agentic UX review loop

- Start with a smoke baseline across discovery, known-item lookup, pagination,
  and author workflows so regressions in the core paths stay visible.
- If the task is a broader UX review, add deeper probes such as
  `get_paper_references`, `get_paper_authors`, `search_snippets`, or the
  explicit `*_openalex` tools.
- If the task is a feature-specific probe, keep the baseline short and spend the
  remaining effort on the supplied feature or UX hypothesis.
- When you find a concrete defect, capture the exact tool calls, expected vs
  actual behavior, and whether the likely fix belongs in code, docs, or both.
""".strip()

__all__ = [
    "API_BASE_URL",
    "ARXIV_API_BASE",
    "ARXIV_NS",
    "ATOM_NS",
    "CORE_API_BASE",
    "DEFAULT_AUTHOR_FIELDS",
    "DEFAULT_PAPER_FIELDS",
    "MAX_429_RETRIES",
    "OPENSEARCH_NS",
    "RECOMMENDATIONS_BASE_URL",
    "SEMANTIC_SCHOLAR_MIN_INTERVAL",
    "SemanticScholarClient",
    "CoreApiClient",
    "CrossrefClient",
    "OpenAlexClient",
    "ArxivClient",
    "SerpApiScholarClient",
    "UnpaywallClient",
    "_arxiv_id_from_url",
    "_text",
    "_core_response_to_merged",
    "_merge_search_results",
    "_env_bool",
    "asyncio",
    "httpx",
    "app",
    "http_app",
    "build_http_app",
    "settings",
    "agentic_config",
    "api_key",
    "openai_api_key",
    "core_api_key",
    "serpapi_api_key",
    "openalex_api_key",
    "openalex_mailto",
    "crossref_mailto",
    "unpaywall_email",
    "enable_core",
    "enable_semantic_scholar",
    "enable_openalex",
    "enable_arxiv",
    "enable_serpapi",
    "enable_crossref",
    "enable_unpaywall",
    "enable_ecos",
    "client",
    "core_client",
    "crossref_client",
    "ecos_client",
    "openalex_client",
    "arxiv_client",
    "serpapi_client",
    "unpaywall_client",
    "provider_registry",
    "enrichment_service",
    "workspace_registry",
    "provider_bundle",
    "agentic_runtime",
    "list_tools",
    "call_tool",
    "main",
]


def _format_tool_display_name(name: str) -> str:
    return name.replace("_", " ").title()


def _tool_tags(name: str) -> set[str]:
    provider_tags = {
        "search_papers": {"search", "brokered"},
        "search_papers_smart": {"search", "smart", "agentic"},
        "resolve_citation": {"known-item", "citation-repair", "recovery"},
        "search_papers_core": {"search", "provider-specific", "provider:core"},
        "search_papers_semantic_scholar": {
            "search",
            "provider-specific",
            "provider:semantic_scholar",
        },
        "search_papers_serpapi": {
            "search",
            "provider-specific",
            "provider:serpapi_google_scholar",
        },
        "search_papers_arxiv": {"search", "provider-specific", "provider:arxiv"},
        "search_papers_openalex": {
            "search",
            "provider-specific",
            "provider:openalex",
        },
        "paper_autocomplete_openalex": {
            "search",
            "provider-specific",
            "provider:openalex",
        },
        "search_papers_openalex_bulk": {
            "search",
            "provider-specific",
            "provider:openalex",
        },
        "search_entities_openalex": {
            "search",
            "provider-specific",
            "provider:openalex",
        },
        "search_papers_openalex_by_entity": {
            "search",
            "provider-specific",
            "provider:openalex",
        },
        "search_papers_serpapi_cited_by": {
            "search",
            "provider-specific",
            "provider:serpapi_google_scholar",
        },
        "search_papers_serpapi_versions": {
            "search",
            "provider-specific",
            "provider:serpapi_google_scholar",
        },
        "get_author_profile_serpapi": {
            "author",
            "provider-specific",
            "provider:serpapi_google_scholar",
        },
        "get_author_articles_serpapi": {
            "author",
            "provider-specific",
            "provider:serpapi_google_scholar",
        },
        "get_serpapi_account_status": {
            "provider-specific",
            "provider:serpapi_google_scholar",
        },
        "get_paper_metadata_crossref": {
            "paper",
            "provider-specific",
            "provider:crossref",
        },
        "get_paper_open_access_unpaywall": {
            "paper",
            "provider-specific",
            "provider:unpaywall",
        },
        "search_species_ecos": {
            "search",
            "species",
            "provider-specific",
            "provider:ecos",
        },
        "get_species_profile_ecos": {
            "species",
            "provider-specific",
            "provider:ecos",
        },
        "list_species_documents_ecos": {
            "species",
            "documents",
            "provider-specific",
            "provider:ecos",
        },
        "get_document_text_ecos": {
            "documents",
            "provider-specific",
            "provider:ecos",
        },
        "enrich_paper": {"paper", "enrichment"},
        "get_provider_diagnostics": {"diagnostics", "provider-health"},
        "ask_result_set": {"smart", "grounded-answer"},
        "map_research_landscape": {"smart", "landscape"},
        "expand_research_graph": {"smart", "graph"},
    }
    if name in provider_tags:
        return provider_tags[name]
    if name.startswith("search_"):
        return {"search"}
    if name.startswith("get_paper_"):
        return {"paper"}
    if name.startswith("get_author_") or name == "search_authors":
        return {"author"}
    if name.startswith("batch_"):
        return {"batch"}
    return {"scholar-search"}


def _parameter_name(field_name: str, alias: str | None) -> str:
    return alias or field_name


def _parameter_default(model_field: Any) -> Any:
    if model_field.is_required():
        return Parameter.empty
    default = model_field.default
    if default is PydanticUndefined:
        default = None
    return Field(default=default, description=model_field.description)


def _build_signature(model: Any) -> tuple[Signature, dict[str, Any]]:
    parameters: list[Parameter] = []
    annotations: dict[str, Any] = {"return": dict[str, Any]}
    for field_name, model_field in model.model_fields.items():
        parameter_name = _parameter_name(field_name, model_field.alias)
        annotations[parameter_name] = model_field.annotation
        parameters.append(
            Parameter(
                parameter_name,
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=model_field.annotation,
                default=_parameter_default(model_field),
            )
        )
    annotations["ctx"] = Context
    parameters.append(
        Parameter(
            "ctx",
            Parameter.POSITIONAL_OR_KEYWORD,
            annotation=Context,
            default=None,
        )
    )
    return Signature(parameters=parameters), annotations


def _register_tool(tool_name: str) -> None:
    signature, annotations = _build_signature(TOOL_INPUT_MODELS[tool_name])

    async def _tool_impl(**kwargs: Any) -> dict[str, Any]:
        ctx = kwargs.pop("ctx", None)
        return await _execute_tool(tool_name, kwargs, ctx=ctx)

    _tool_impl.__name__ = tool_name
    _tool_impl.__doc__ = TOOL_DESCRIPTIONS[tool_name]
    setattr(_tool_impl, "__signature__", signature)
    _tool_impl.__annotations__ = annotations

    app.tool(
        name=tool_name,
        title=_format_tool_display_name(tool_name),
        description=TOOL_DESCRIPTIONS[tool_name],
        tags=_tool_tags(tool_name),
        annotations=ToolAnnotations(
            title=_format_tool_display_name(tool_name),
            readOnlyHint=True,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )(_tool_impl)
    _sanitize_registered_tool_schema(tool_name)


def _sanitize_registered_tool_schema(tool_name: str) -> None:
    tool = cast(Any, app.local_provider._components[f"tool:{tool_name}@"])
    tool.parameters = sanitize_published_schema(tool.parameters)


async def _execute_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    ctx: Context | None = None,
) -> dict[str, Any]:
    return await dispatch_tool(
        name,
        arguments,
        client=client,
        core_client=core_client,
        openalex_client=openalex_client,
        arxiv_client=arxiv_client,
        enable_core=enable_core,
        enable_semantic_scholar=enable_semantic_scholar,
        enable_openalex=enable_openalex,
        enable_arxiv=enable_arxiv,
        serpapi_client=serpapi_client,
        enable_serpapi=enable_serpapi,
        crossref_client=crossref_client,
        unpaywall_client=unpaywall_client,
        ecos_client=ecos_client,
        enable_crossref=enable_crossref,
        enable_unpaywall=enable_unpaywall,
        enable_ecos=enable_ecos,
        enrichment_service=enrichment_service,
        provider_order=provider_order,
        provider_registry=provider_registry,
        workspace_registry=workspace_registry,
        agentic_runtime=agentic_runtime,
        ctx=ctx,
    )


settings = AppSettings.from_env()
agentic_config = AgenticConfig.from_settings(settings)
api_key = settings.semantic_scholar_api_key
openai_api_key = settings.openai_api_key
core_api_key = settings.core_api_key
openalex_api_key = settings.openalex_api_key
openalex_mailto = settings.openalex_mailto
serpapi_api_key = settings.serpapi_api_key
crossref_mailto = settings.crossref_mailto
unpaywall_email = settings.unpaywall_email
ecos_base_url = settings.ecos_base_url
enable_core = settings.enable_core
enable_semantic_scholar = settings.enable_semantic_scholar
enable_openalex = settings.enable_openalex
enable_arxiv = settings.enable_arxiv
enable_serpapi = settings.enable_serpapi
enable_crossref = settings.enable_crossref
enable_unpaywall = settings.enable_unpaywall
enable_ecos = settings.enable_ecos
provider_order = list(settings.provider_order)
provider_registry = ProviderDiagnosticsRegistry()
client = SemanticScholarClient(api_key=api_key)
core_client = CoreApiClient(api_key=core_api_key)
crossref_client = CrossrefClient(
    mailto=crossref_mailto,
    timeout=settings.crossref_timeout_seconds,
)
ecos_client = EcosClient(
    base_url=ecos_base_url,
    timeout=settings.ecos_timeout_seconds,
    document_timeout=settings.ecos_document_timeout_seconds,
    max_document_size_mb=settings.ecos_max_document_size_mb,
    provider_registry=provider_registry,
)
openalex_client = OpenAlexClient(api_key=openalex_api_key, mailto=openalex_mailto)
arxiv_client = ArxivClient()
serpapi_client = SerpApiScholarClient(api_key=serpapi_api_key)
unpaywall_client = UnpaywallClient(
    email=unpaywall_email,
    timeout=settings.unpaywall_timeout_seconds,
)
enrichment_service = PaperEnrichmentService(
    crossref_client=crossref_client,
    unpaywall_client=unpaywall_client,
    enable_crossref=enable_crossref,
    enable_unpaywall=enable_unpaywall,
    provider_registry=provider_registry,
)
provider_bundle = resolve_provider_bundle(
    agentic_config,
    openai_api_key=openai_api_key,
    provider_registry=provider_registry,
)
workspace_registry = WorkspaceRegistry(
    ttl_seconds=settings.session_ttl_seconds,
    enable_trace_log=settings.enable_agentic_trace_log,
    index_backend=settings.agentic_index_backend,
    similarity_fn=provider_bundle.similarity,
    async_batched_similarity_fn=provider_bundle.abatched_similarity,
    async_embed_query_fn=(
        None if settings.disable_embeddings else provider_bundle.aembed_query
    ),
    async_embed_texts_fn=(
        None if settings.disable_embeddings else provider_bundle.aembed_texts
    ),
    embed_query_fn=(
        None if settings.disable_embeddings else provider_bundle.embed_query
    ),
    embed_texts_fn=(
        None if settings.disable_embeddings else provider_bundle.embed_texts
    ),
)
agentic_runtime = AgenticRuntime(
    config=agentic_config,
    provider_bundle=provider_bundle,
    workspace_registry=workspace_registry,
    client=client,
    core_client=core_client,
    openalex_client=openalex_client,
    arxiv_client=arxiv_client,
    serpapi_client=serpapi_client,
    enable_core=enable_core,
    enable_semantic_scholar=enable_semantic_scholar,
    enable_openalex=enable_openalex,
    enable_arxiv=enable_arxiv,
    enable_serpapi=enable_serpapi,
    provider_registry=provider_registry,
    enrichment_service=enrichment_service,
)


@asynccontextmanager
async def _server_lifespan(_: FastMCP):
    try:
        yield
    finally:
        await agentic_runtime.aclose()
        await maybe_close_async_resource(client)
        await maybe_close_async_resource(core_client)
        await maybe_close_async_resource(openalex_client)
        await maybe_close_async_resource(arxiv_client)
        await maybe_close_async_resource(serpapi_client)
        await maybe_close_async_resource(crossref_client)
        await maybe_close_async_resource(unpaywall_client)
        await maybe_close_async_resource(ecos_client)


app = FastMCP(
    "scholar-search",
    instructions=SERVER_INSTRUCTIONS,
    lifespan=_server_lifespan,
    strict_input_validation=True,
)
app.add_middleware(TimingMiddleware(logger=logger))

for _tool_name in TOOL_INPUT_MODELS:
    _register_tool(_tool_name)


def _resource_text(payload: dict[str, Any]) -> str:
    return json.dumps(dump_jsonable(payload), ensure_ascii=False, indent=2)


def _paper_resource_payload(paper: dict[str, Any]) -> dict[str, Any]:
    authors = ", ".join(
        author.get("name", "")
        for author in (paper.get("authors") or [])
        if isinstance(author, dict) and author.get("name")
    )
    paper_identifier = paper.get("paperId") or paper.get("canonicalId") or "unknown"
    markdown_lines = [
        f"# {paper.get('title') or paper.get('paperId') or 'Paper'}",
        "",
        f"- Paper ID: `{paper_identifier}`",
    ]
    if paper.get("year"):
        markdown_lines.append(f"- Year: {paper['year']}")
    if paper.get("venue"):
        markdown_lines.append(f"- Venue: {paper['venue']}")
    if authors:
        markdown_lines.append(f"- Authors: {authors}")
    if paper.get("abstract"):
        markdown_lines.extend(["", "## Abstract", "", str(paper["abstract"])])
    return {"markdown": "\n".join(markdown_lines), "data": paper}


def _author_resource_payload(author: dict[str, Any]) -> dict[str, Any]:
    markdown_lines = [
        f"# {author.get('name') or author.get('authorId') or 'Author'}",
        "",
        f"- Author ID: `{author.get('authorId') or 'unknown'}`",
    ]
    affiliations = author.get("affiliations") or []
    if affiliations:
        markdown_lines.append(f"- Affiliations: {', '.join(affiliations)}")
    return {"markdown": "\n".join(markdown_lines), "data": author}


@app.resource(
    "guide://scholar-search/agent-workflows",
    title="Scholar Search agent workflows",
    description="How to choose the right scholar-search tools and pagination flow.",
)
def agent_workflows() -> str:
    """Return a compact workflow guide for agents."""
    return AGENT_WORKFLOW_GUIDE


@app.resource(
    "paper://{paper_id}",
    title="Paper resource",
    description="Compact cached or fetched paper payload plus markdown summary.",
    mime_type="application/json",
)
async def paper_resource(paper_id: str) -> str:
    cached = workspace_registry.render_paper_resource(paper_id)
    if cached is not None:
        return _resource_text(cached)
    last_error: Exception | None = None
    for fetch in (
        lambda: client.get_paper_details(paper_id),
        lambda: openalex_client.get_paper_details(paper_id),
    ):
        try:
            paper = await fetch()
            return _resource_text(_paper_resource_payload(paper))
        except Exception as exc:
            last_error = exc
            logger.debug("Paper resource fetch failed for %r: %s", paper_id, exc)
    raise ValueError(
        f"Could not resolve paper resource for {paper_id!r}."
    ) from last_error


@app.resource(
    "author://{author_id}",
    title="Author resource",
    description="Compact cached or fetched author payload plus markdown summary.",
    mime_type="application/json",
)
async def author_resource(author_id: str) -> str:
    cached = workspace_registry.render_author_resource(author_id)
    if cached is not None:
        return _resource_text(cached)
    last_error: Exception | None = None
    for fetch in (
        lambda: client.get_author_info(author_id),
        lambda: openalex_client.get_author_info(author_id),
    ):
        try:
            author = await fetch()
            return _resource_text(_author_resource_payload(author))
        except Exception as exc:
            last_error = exc
            logger.debug("Author resource fetch failed for %r: %s", author_id, exc)
    raise ValueError(
        f"Could not resolve author resource for {author_id!r}."
    ) from last_error


@app.resource(
    "search://{search_session_id}",
    title="Search session resource",
    description="Saved result-set handle surfaced from tool outputs.",
    mime_type="application/json",
)
def search_session_resource(search_session_id: str) -> str:
    return _resource_text(workspace_registry.render_search_resource(search_session_id))


@app.resource(
    "trail://paper/{paper_id}?direction={direction}",
    title="Paper trail resource",
    description=(
        "Citation or reference trail for a paper, preferably discovered "
        "through tool outputs."
    ),
    mime_type="application/json",
)
async def paper_trail_resource(
    paper_id: str,
    direction: Literal["citations", "references"],
) -> str:
    cached_trail = workspace_registry.find_trail(paper_id=paper_id, direction=direction)
    if cached_trail is not None:
        return _resource_text(
            workspace_registry.render_search_resource(cached_trail.search_session_id)
        )
    payload = await (
        client.get_paper_citations(
            paper_id=paper_id, limit=25, fields=None, offset=None
        )
        if direction == "citations"
        else client.get_paper_references(
            paper_id=paper_id,
            limit=25,
            fields=None,
            offset=None,
        )
    )
    title = "Citations" if direction == "citations" else "References"
    summary = {
        "markdown": (
            f"# {title} trail for `{paper_id}`\n\n"
            f"- Direction: {direction}\n"
            f"- Results: {len((payload or {}).get('data') or [])}"
        ),
        "data": payload,
    }
    return _resource_text(summary)


@app.prompt(
    name="plan_scholar_search",
    title="Plan Scholar Search",
    description="Generate a tool-first plan for a literature search task.",
)
def plan_scholar_search(
    topic: str,
    goal: str = "find relevant papers, follow citations, and summarize next steps",
    mode: Literal["smoke", "comprehensive", "feature_probe"] = "smoke",
    focus_prompt: str | None = None,
) -> str:
    """Create a reusable research workflow prompt for clients."""
    mode_guidance = {
        "smoke": (
            "Run a smoke-style review that stays focused on the primary golden "
            "paths: quick discovery, known-item lookup, pagination, author pivot, "
            "and optional citation export."
        ),
        "comprehensive": (
            "Run a comprehensive UX review: cover the smoke baseline first, then "
            "add deeper probes for references, paper-to-author pivots, snippet "
            "recovery, and explicit OpenAlex workflows."
        ),
        "feature_probe": (
            "Run a feature-probe review: keep a short smoke baseline, then spend "
            "most of the effort on the requested feature or UX hypothesis and the "
            "tool paths that exercise it."
        ),
    }
    focus_text = (
        f" Focus prompt: {focus_prompt}."
        if focus_prompt
        else " No extra focus prompt was supplied."
    )
    return (
        f"You are planning a scholar-search workflow about '{topic}'. Goal: {goal}. "
        f"Mode: {mode}. {mode_guidance[mode]}{focus_text} "
        "If the task is concept-level discovery, literature review, or a grounded "
        "follow-up over a reusable result set, prefer search_papers_smart first "
        "and reuse searchSessionId with ask_result_set, map_research_landscape, "
        "or expand_research_graph. Fall back to raw tools when you need provider-"
        "specific control, pagination, or exact low-level semantics. "
        "Start with search_papers for quick literature discovery, then read "
        "brokerMetadata.nextStepHint to decide whether to broaden, narrow, paginate, "
        "pivot providers, or pivot into authors. "
        "Treat search_papers_bulk as the closest continuation path only when the "
        "workflow is already aligned with Semantic Scholar retrieval semantics. "
        "Even then, default bulk ordering is NOT relevance-ranked, so it is not "
        "'page 2' of search_papers; read retrievalNote in each bulk response, or "
        "pass sort='citationCount:desc' for citation-ranked bulk traversal. If "
        "results came from CORE, arXiv, or SerpApi, bulk retrieval is a Semantic "
        "Scholar pivot rather than another page from the same provider. "
        "If the task is exhaustive retrieval, first N results, or multi-page "
        "collection, use search_papers_bulk. For small targeted pages, prefer "
        "search_papers or search_papers_semantic_scholar because the upstream "
        "bulk endpoint may ignore small limit values internally. "
        "If the task explicitly needs OpenAlex-native DOI/ID lookup, OpenAlex "
        "cursor pagination, or OpenAlex author/citation semantics, use the "
        "dedicated *_openalex tools instead of the default broker. "
        "If the task is citation repair, broken bibliography recovery, or "
        "almost-right reference correction, use resolve_citation first. If the "
        "task is known-item lookup, use search_papers_match for messy titles "
        "and get_paper_details for DOI, arXiv ID, URL, or canonical IDs. Treat a "
        "structured no-match from search_papers_match as a hint that the item may "
        "be a dissertation, software release, report, or other output outside the "
        "indexed paper surface. Once you have a stable paper anchor, use "
        "get_paper_metadata_crossref, get_paper_open_access_unpaywall, or "
        "enrich_paper for additive metadata and OA/PDF discovery. Known-item "
        "tools and search_papers_smart also expose includeEnrichment=true when "
        "you want post-resolution enrichment without changing ranking. "
        "If the task starts from a known paper, use get_paper_citations for cited-by "
        "expansion and get_paper_references for backward references, and explain "
        "that direction clearly. "
        "For author-centric workflows use search_authors, get_author_info, and "
        "get_author_papers. For common names, add affiliation, coauthor, venue, "
        "or topic clues before confirming the best candidate. For Semantic "
        "Scholar expansion tools prefer paper.recommendedExpansionId when it is "
        "present. If paper.expansionIdStatus is not_portable, do not retry with "
        "brokered paperId/sourceId/canonicalId values; resolve the paper "
        "through DOI or a Semantic Scholar-native lookup first. "
        "Use search_snippets only as a special-purpose recovery tool when quote or "
        "phrase search is needed and title/keyword search is weak; if the provider "
        "rejects that query, expect an empty degraded response rather than a raw "
        "4xx/5xx. "
        "Use preferredProvider/providerOrder or provider-specific search_papers_* "
        "tools only when source choice matters. Remember that search_papers_core, "
        "search_papers_serpapi, and search_papers_arxiv only support query, limit, "
        "and year, while search_papers_semantic_scholar supports the wider filter set. "
        "If you uncover a defect or confusing UX, summarize the exact tool calls, "
        "expected vs actual behavior, and whether the best follow-up is a code "
        "change, a documentation update, or both so the result can turn into an "
        "actionable issue for a GitHub Copilot coding agent. "
        "Treat pagination.nextCursor as opaque: reuse it exactly as returned, do "
        "not edit or fabricate it, and keep it scoped to the tool/query flow that "
        "produced it."
    )


@app.prompt(
    name="plan_smart_scholar_search",
    title="Plan Smart Scholar Search",
    description=(
        "Generate a smart-tool-first research plan for concept-level discovery."
    ),
)
def plan_smart_scholar_search(
    topic: str,
    goal: str = (
        "map the literature, answer grounded follow-up questions, and "
        "identify the best next actions"
    ),
    mode: Literal[
        "discovery", "review", "known_item", "author", "citation"
    ] = "discovery",
) -> str:
    return (
        f"You are planning a smart scholar-search workflow about '{topic}'. "
        f"Goal: {goal}. Mode: {mode}. Start with search_papers_smart for "
        "concept-level discovery or "
        "known-item resolution. For broken citations or almost-right "
        "references, prefer resolve_citation before broader discovery. Reuse "
        "searchSessionId across ask_result_set, "
        "map_research_landscape, and expand_research_graph. If the smart workflow "
        "cannot stay grounded, drop to raw tools: search_papers, search_papers_bulk, "
        "get_paper_details, resolve_citation, get_paper_citations, "
        "get_paper_references, search_authors, and get_author_papers."
        " When you already have the right paper and want richer metadata or OA "
        "signals, use includeEnrichment=true on the smart or known-item path, or "
        "call enrich_paper explicitly."
    )


@app.prompt(
    name="triage_literature",
    title="Triage Literature",
    description="Turn a research topic into a compact triage workflow.",
)
def triage_literature(
    topic: str,
    goal: str = "identify core themes, strongest anchors, and the next best tool call",
) -> str:
    return (
        f"Triage literature for '{topic}'. Goal: {goal}. Start with "
        "search_papers_smart. Inspect strategyMetadata, "
        "acceptedExpansions, rejectedExpansions, resourceUris, and "
        "agentHints. Save the searchSessionId, then ask one grounded question with "
        "ask_result_set and one clustering question with map_research_landscape. "
        "If one hit becomes a strong anchor, optionally enrich it with Crossref "
        "and Unpaywall before the next citation or QA step."
    )


@app.prompt(
    name="plan_citation_chase",
    title="Plan Citation Chase",
    description="Generate a citation-expansion workflow from a paper anchor.",
)
def plan_citation_chase(
    paper_id: str,
    direction: Literal["citations", "references"] = "citations",
    goal: str = "find the most influential neighboring work and preserve provenance",
) -> str:
    return (
        f"Plan a citation chase from paper '{paper_id}' in the "
        f"'{direction}' direction. Goal: {goal}. Prefer expand_research_graph "
        "for a compact frontier. If you need "
        "provider-native control or pagination, use get_paper_citations or "
        "get_paper_references directly and treat pagination.nextCursor as opaque."
    )


@app.prompt(
    name="refine_query",
    title="Refine Query",
    description="Generate a bounded query-refinement workflow for the current topic.",
)
def refine_query(
    query: str,
    weakness: str = "results are too broad, too narrow, or too noisy",
) -> str:
    return (
        f"Refine the query '{query}'. Problem signal: {weakness}. "
        "Try search_papers_smart first and inspect acceptedExpansions, "
        "rejectedExpansions, speculativeExpansions, and driftWarnings. "
        "If needed, add year/venue/focus constraints or fall back to raw search_papers "
        "with providerOrder or preferredProvider."
    )


http_app_transport = cast(
    Literal["http", "streamable-http", "sse"],
    settings.transport if settings.transport != "stdio" else "streamable-http",
)


def build_http_app(
    *,
    path: str | None = None,
    transport: Literal["http", "streamable-http", "sse"] | None = None,
    middleware: list[Any] | None = None,
) -> Any:
    """Build an ASGI app for local/dev HTTP use or custom deployment hardening."""
    return app.http_app(
        path=path or settings.http_path,
        transport=transport or http_app_transport,
        middleware=middleware,
    )


http_app = build_http_app()


async def list_tools() -> list[Tool]:
    """Compatibility helper returning the registered tool schemas."""
    return [
        cast(Tool, tool.to_mcp_tool() if hasattr(tool, "to_mcp_tool") else tool)
        for tool in await app.list_tools(run_middleware=False)
    ]


async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Compatibility helper preserving the historic JSON-in-text test contract."""
    result = await _execute_tool(name, arguments)
    return [
        TextContent(
            type="text",
            text=json.dumps(dump_jsonable(result), ensure_ascii=False, indent=2),
        )
    ]


def main() -> None:
    """Run the MCP server."""
    run_server(app=app, logger=logger, settings=settings)


if __name__ == "__main__":
    main()
