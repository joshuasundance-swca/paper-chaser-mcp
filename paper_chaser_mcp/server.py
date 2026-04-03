"""FastMCP-backed public server surface for Paper Chaser."""

import json
import logging
from contextlib import asynccontextmanager
from inspect import Parameter, Signature
from typing import Any, Literal, cast

from fastmcp import Context, FastMCP
from fastmcp.server.middleware.timing import TimingMiddleware
from mcp.types import TextContent, Tool, ToolAnnotations
from pydantic import Field
from pydantic_core import PydanticUndefined

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
    FederalRegisterClient,
    GovInfoClient,
    OpenAlexClient,
    ScholarApiClient,
    SemanticScholarClient,
    UnpaywallClient,
)
from .clients.serpapi import SerpApiScholarClient
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
from .models import dump_jsonable
from .parsing import _arxiv_id_from_url, _text
from .provider_runtime import ProviderDiagnosticsRegistry
from .renderers.resources import (
    render_author_resource_payload,
    render_paper_resource_payload,
)
from .runtime import run_server
from .search import _core_response_to_merged, _merge_search_results
from .settings import AppSettings, _env_bool
from .tool_schema import sanitize_published_schema
from .tool_specs import get_tool_spec, iter_visible_tool_specs
from .transport import asyncio, httpx, maybe_close_async_resource

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("paper-chaser-mcp")

SERVER_INSTRUCTIONS = """
Decision tree for tool selection:

1. CONCEPT-LEVEL DISCOVERY / REVIEW → search_papers_smart
    (returns searchSessionId, strategyMetadata, trust-graded sections such as
    verifiedFindings/likelyUnverified/evidenceGaps/structuredSources, plus
    resourceUris and agentHints;
   use latencyProfile=fast for smoke tests, balanced for default work, and deep
   for controlled multi-provider expansion)
2. QUICK RAW DISCOVERY → search_papers
   (brokered, single page, returns brokerMetadata plus agentHints/resourceUris)
3. EXHAUSTIVE / MULTI-PAGE → search_papers_bulk
   (cursor-paginated, up to 1 000 returned/call; read retrievalNote because
   default bulk ordering is not relevance-ranked)
4. CITATION REPAIR / ALMOST-RIGHT REFERENCES → resolve_citation
5. KNOWN ITEM (messy title) → search_papers_match
   (takes only a query string — the title text — not separate author/year/venue
   fields; use resolve_citation for multi-field bibliographic references)
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
13. SCHOLARAPI FULL-TEXT PATHS → use search_papers_scholarapi,
    list_papers_scholarapi, get_paper_text_scholarapi,
    get_paper_texts_scholarapi, or get_paper_pdf_scholarapi when the workflow
    explicitly needs ScholarAPI-ranked discovery, indexed-at monitoring,
    accessible full text, or binary PDF retrieval
14. SERPAPI RECOVERY PATHS → use search_papers_serpapi_cited_by,
   search_papers_serpapi_versions, get_author_profile_serpapi,
   get_author_articles_serpapi, or get_serpapi_account_status only when
   PAPER_CHASER_ENABLE_SERPAPI=true and the workflow justifies paid recall recovery
15. ECOS SPECIES DOSSIERS → search_species_ecos → get_species_profile_ecos →
   list_species_documents_ecos → get_document_text_ecos for species pages,
   regulatory documents, and recovery PDFs from the U.S. Fish and Wildlife
   Service ECOS system
16. REGULATORY PRIMARY SOURCES → search_federal_register for discovery,
    get_federal_register_document for one notice or rule, and get_cfr_text for
    authoritative CFR part/section text. NOTE: Biological opinions, Section 7
    consultation records, and incidental take permits live in ECOS, not the
    Federal Register — use the ECOS species dossier chain for those.
17. PROVIDER HEALTH / DEBUGGING → get_provider_diagnostics

After search_papers: read brokerMetadata.nextStepHint for the recommended next move.
After search_papers_smart: reuse searchSessionId for ask_result_set,
map_research_landscape, or expand_research_graph, and inspect verifiedFindings,
likelyUnverified, evidenceGaps, structuredSources, coverageSummary,
failureSummary, acceptedExpansions, rejectedExpansions, speculativeExpansions,
providersUsed, driftWarnings, latencyProfile, providerBudgetApplied, and
providerOutcomes. Set
includeEnrichment=true only when you want Crossref, Unpaywall, and OpenAlex metadata on the
final smart-ranked hits; enrichment is post-ranking only and never changes
retrieval or provider ordering. When ScholarAPI is enabled, smart retrieval may
also include it explicitly, and providerBudget.maxScholarApiCalls can cap that
paid path.
When the query is clearly regulatory or species-history oriented, search_papers_smart can also route into
ECOS/Federal Register/CFR retrieval first and return a regulatoryTimeline instead of paper-centric ranking.
Primary read tools now also return agentHints, clarification, resourceUris, and,
when they produce reusable result sets, searchSessionId.
For known-item flows, includeEnrichment=true on search_papers_match,
get_paper_details, or resolve_citation adds Crossref, Unpaywall, and OpenAlex metadata only
after the base paper resolution succeeds.
For Semantic Scholar expansion tools, prefer paper.recommendedExpansionId when
present. If paper.expansionIdStatus is not_portable, do not retry with brokered
paperId/sourceId/canonicalId values; resolve the paper through DOI or a
Semantic Scholar-native lookup first.
If search_papers_match returns no match, or if the user has a broken
bibliography line, partial reference, or almost-right citation, prefer
resolve_citation before guessing. A no-match can still mean the item is a
dissertation, software release, report, or other output outside the indexed
paper surface. If the citation is clearly regulatory (for example a Federal
Register or CFR reference), switch to search_federal_register,
get_federal_register_document, or get_cfr_text instead of forcing a paper
lookup.
For common-name author lookup, add affiliation, coauthor, venue, or topic clues
before expanding into get_author_info/get_author_papers.
To steer the broker: use preferredProvider (try-first) or providerOrder (full override).
Provider names: semantic_scholar, arxiv, core, scholarapi, serpapi / serpapi_google_scholar.
Provider-specific search inputs: search_papers_core, search_papers_serpapi, and
search_papers_arxiv only accept query/limit/year; search_papers_semantic_scholar
supports the wider Semantic Scholar filter set; search_papers_scholarapi and
list_papers_scholarapi expose ScholarAPI-specific cursor/date/full-text filters.
OpenAlex is available through explicit *_openalex tools instead of the broker because
its citation, author, and pagination semantics differ from Semantic Scholar.
Continuation rule: search_papers_bulk is the closest continuation path only for
Semantic Scholar-style retrieval; from CORE, arXiv, ScholarAPI, or SerpApi results it is a
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

GUIDED_SERVER_INSTRUCTIONS = """
Default guided workflow:

1. RESEARCH -> research
   Use this for topic discovery, literature review, known-item recovery, citation repair,
   and regulatory or species-history requests when you want one trust-graded answer.
2. FOLLOW UP -> follow_up_research
   Reuse searchSessionId from research to ask one grounded question. The tool abstains
   when the saved evidence is too weak or off-topic.
3. RESOLVE ONE REFERENCE -> resolve_reference
   Use this for citations, DOI strings, arXiv IDs, URLs, title fragments, and regulatory references.
4. INSPECT ONE SOURCE -> inspect_source
   Pass searchSessionId plus sourceId from research to inspect provenance, trust state, and direct-read next steps.

Use get_runtime_status when behavior looks different across environments and you need the active runtime truth.
The guided surface is intentionally opinionated: it prefers trust-graded evidence, explicit abstention, and direct next
actions over raw provider control.
""".strip()

AGENT_WORKFLOW_GUIDE = """
# Paper Chaser agent workflow guide

## Default guided path

- Start with `research` for topic discovery, literature review, known-item recovery,
  citation repair, and regulatory or species-history requests.
- Save the returned `searchSessionId`. It is the anchor for `follow_up_research`
  and `inspect_source`.
- Use `follow_up_research` for one grounded question over the saved evidence.
  It is supposed to abstain when the evidence is weak, off-topic, or incomplete.
- Use `resolve_reference` when the user already has a citation, DOI, arXiv ID,
  URL, title fragment, or regulatory reference and wants the safest next anchor.
- Use `inspect_source` with `searchSessionId` plus `sourceId` to inspect
  provenance, trust state, access status, and direct-read next steps.
- Use `get_runtime_status` when behavior differs across environments and you need
  the active runtime truth without digging through low-level diagnostics.

## Guided output contract

- `research` returns `intent`, `status`, `summary`, `findings`, `sources`,
  `candidateLeads`, `trustSummary`, `coverage`, `failure`, `nextActions`, and `clarification`.
- `status` is one of `succeeded`, `partial`, `needs_disambiguation`,
  `abstained`, or `failed`.
- Treat `findings` as the compact trust-graded claims list and `sources` as the
  canonical source records for inspection and citation.
- Treat `candidateLeads` as auditable but not yet trusted evidence. It is the
  place for weak, filtered, or off-topic leads that should not be promoted into
  verified findings.
- If the tool abstains or asks for clarification, do not smooth that over with
  your own synthesis. Ask a narrower question or inspect the returned sources.

## Expert fallback

- Use the expert surface only when you truly need raw provider control,
  pagination semantics, or provider-native payloads.
- Expert discovery tools include `search_papers`, `search_papers_bulk`,
  `search_papers_smart`, `map_research_landscape`, and `expand_research_graph`.
- Expert primary-source tools include `search_federal_register`,
  `get_federal_register_document`, `get_cfr_text`, `search_species_ecos`,
  `get_species_profile_ecos`, and `list_species_documents_ecos`.
- Expert runtime debugging lives under `get_provider_diagnostics`.

## Safety habits

- Prefer guided tools unless a concrete expert-only need is present.
- Reuse `searchSessionId` instead of rephrasing the same question into multiple
  raw tools.
- Treat `pagination.nextCursor` as opaque whenever you are on an expert paginated
  path.
- Capture defects with the exact tool call, what the user expected, what the
  tool returned, and whether the fix belongs in code, docs, or both.
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
    "ScholarApiClient",
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
    "scholarapi_api_key",
    "openalex_api_key",
    "openalex_mailto",
    "crossref_mailto",
    "unpaywall_email",
    "enable_core",
    "enable_semantic_scholar",
    "enable_openalex",
    "enable_arxiv",
    "enable_serpapi",
    "enable_scholarapi",
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
    "scholarapi_client",
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
    return set(get_tool_spec(name).tags)


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
    tool_spec = get_tool_spec(tool_name)
    signature, annotations = _build_signature(tool_spec.input_model)

    async def _tool_impl(**kwargs: Any) -> dict[str, Any]:
        ctx = kwargs.pop("ctx", None)
        return await _execute_tool(tool_name, kwargs, ctx=ctx)

    _tool_impl.__name__ = tool_name
    _tool_impl.__doc__ = tool_spec.description
    setattr(_tool_impl, "__signature__", signature)
    _tool_impl.__annotations__ = annotations

    app.tool(
        name=tool_name,
        title=_format_tool_display_name(tool_name),
        description=tool_spec.description,
        tags=set(tool_spec.tags),
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
        scholarapi_client=scholarapi_client,
        arxiv_client=arxiv_client,
        enable_core=enable_core,
        enable_semantic_scholar=enable_semantic_scholar,
        enable_openalex=enable_openalex,
        enable_scholarapi=enable_scholarapi,
        enable_arxiv=enable_arxiv,
        serpapi_client=serpapi_client,
        enable_serpapi=enable_serpapi,
        crossref_client=crossref_client,
        unpaywall_client=unpaywall_client,
        ecos_client=ecos_client,
        federal_register_client=federal_register_client,
        govinfo_client=govinfo_client,
        enable_crossref=enable_crossref,
        enable_unpaywall=enable_unpaywall,
        enable_ecos=enable_ecos,
        enable_federal_register=enable_federal_register,
        enable_govinfo_cfr=enable_govinfo_cfr,
        enrichment_service=enrichment_service,
        provider_order=provider_order,
        provider_registry=provider_registry,
        workspace_registry=workspace_registry,
        agentic_runtime=agentic_runtime,
        transport_mode=settings.transport,
        tool_profile=settings.tool_profile,
        hide_disabled_tools=settings.hide_disabled_tools,
        session_ttl_seconds=settings.session_ttl_seconds,
        embeddings_enabled=not settings.disable_embeddings,
        ctx=ctx,
    )


settings = AppSettings.from_env()
for warning in settings.runtime_warnings():
    logger.warning("Runtime configuration warning: %s", warning)
agentic_config = AgenticConfig.from_settings(settings)
api_key = settings.semantic_scholar_api_key
openai_api_key = settings.openai_api_key
nvidia_api_key = settings.nvidia_api_key
nvidia_nim_base_url = settings.nvidia_nim_base_url
azure_openai_api_key = settings.azure_openai_api_key
azure_openai_endpoint = settings.azure_openai_endpoint
azure_openai_api_version = settings.azure_openai_api_version
azure_openai_planner_deployment = settings.azure_openai_planner_deployment
azure_openai_synthesis_deployment = settings.azure_openai_synthesis_deployment
anthropic_api_key = settings.anthropic_api_key
google_api_key = settings.google_api_key
mistral_api_key = settings.mistral_api_key
huggingface_api_key = settings.huggingface_api_key
huggingface_base_url = settings.huggingface_base_url
core_api_key = settings.core_api_key
openalex_api_key = settings.openalex_api_key
openalex_mailto = settings.openalex_mailto
serpapi_api_key = settings.serpapi_api_key
scholarapi_api_key = settings.scholarapi_api_key
govinfo_api_key = settings.govinfo_api_key
crossref_mailto = settings.crossref_mailto
unpaywall_email = settings.unpaywall_email
ecos_base_url = settings.ecos_base_url
enable_core = settings.enable_core
enable_semantic_scholar = settings.enable_semantic_scholar
enable_openalex = settings.enable_openalex
enable_arxiv = settings.enable_arxiv
enable_serpapi = settings.enable_serpapi
enable_scholarapi = settings.enable_scholarapi
enable_crossref = settings.enable_crossref
enable_unpaywall = settings.enable_unpaywall
enable_ecos = settings.enable_ecos
enable_federal_register = settings.enable_federal_register
enable_govinfo_cfr = settings.enable_govinfo_cfr
hide_disabled_tools = settings.hide_disabled_tools
provider_order = list(settings.provider_order)
provider_registry = ProviderDiagnosticsRegistry()
client = SemanticScholarClient(api_key=api_key)
core_client = CoreApiClient(api_key=core_api_key)
federal_register_client = FederalRegisterClient(timeout=settings.federal_register_timeout_seconds)
govinfo_client = GovInfoClient(
    api_key=govinfo_api_key,
    timeout=settings.govinfo_timeout_seconds,
    document_timeout=settings.govinfo_document_timeout_seconds,
    max_document_size_mb=settings.govinfo_max_document_size_mb,
)
crossref_client = CrossrefClient(
    mailto=crossref_mailto,
    timeout=settings.crossref_timeout_seconds,
)
ecos_client = EcosClient(
    base_url=ecos_base_url,
    timeout=settings.ecos_timeout_seconds,
    document_timeout=settings.ecos_document_timeout_seconds,
    document_conversion_timeout=settings.ecos_document_conversion_timeout_seconds,
    max_document_size_mb=settings.ecos_max_document_size_mb,
    verify_tls=settings.ecos_verify_tls,
    ca_bundle=settings.ecos_ca_bundle,
    provider_registry=provider_registry,
)
openalex_client = OpenAlexClient(api_key=openalex_api_key, mailto=openalex_mailto)
scholarapi_client = ScholarApiClient(api_key=scholarapi_api_key)
arxiv_client = ArxivClient()
serpapi_client = SerpApiScholarClient(api_key=serpapi_api_key)
unpaywall_client = UnpaywallClient(
    email=unpaywall_email,
    timeout=settings.unpaywall_timeout_seconds,
)
enrichment_service = PaperEnrichmentService(
    crossref_client=crossref_client,
    unpaywall_client=unpaywall_client,
    openalex_client=openalex_client,
    enable_crossref=enable_crossref,
    enable_unpaywall=enable_unpaywall,
    enable_openalex=enable_openalex,
    provider_registry=provider_registry,
)
provider_bundle = resolve_provider_bundle(
    agentic_config,
    openai_api_key=openai_api_key,
    azure_openai_api_key=azure_openai_api_key,
    azure_openai_endpoint=azure_openai_endpoint,
    azure_openai_api_version=azure_openai_api_version,
    azure_openai_planner_deployment=azure_openai_planner_deployment,
    azure_openai_synthesis_deployment=azure_openai_synthesis_deployment,
    anthropic_api_key=anthropic_api_key,
    nvidia_api_key=nvidia_api_key,
    nvidia_nim_base_url=nvidia_nim_base_url,
    google_api_key=google_api_key,
    mistral_api_key=mistral_api_key,
    huggingface_api_key=huggingface_api_key,
    huggingface_base_url=huggingface_base_url,
    provider_registry=provider_registry,
)
workspace_registry = WorkspaceRegistry(
    ttl_seconds=settings.session_ttl_seconds,
    enable_trace_log=settings.enable_agentic_trace_log,
    index_backend=settings.agentic_index_backend,
    similarity_fn=provider_bundle.similarity,
    async_batched_similarity_fn=provider_bundle.abatched_similarity,
    async_embed_query_fn=(None if not provider_bundle.supports_embeddings() else provider_bundle.aembed_query),
    async_embed_texts_fn=(None if not provider_bundle.supports_embeddings() else provider_bundle.aembed_texts),
    embed_query_fn=(None if not provider_bundle.supports_embeddings() else provider_bundle.embed_query),
    embed_texts_fn=(None if not provider_bundle.supports_embeddings() else provider_bundle.embed_texts),
)
agentic_runtime = AgenticRuntime(
    config=agentic_config,
    provider_bundle=provider_bundle,
    workspace_registry=workspace_registry,
    client=client,
    core_client=core_client,
    openalex_client=openalex_client,
    scholarapi_client=scholarapi_client,
    arxiv_client=arxiv_client,
    serpapi_client=serpapi_client,
    ecos_client=ecos_client,
    federal_register_client=federal_register_client,
    govinfo_client=govinfo_client,
    enable_core=enable_core,
    enable_semantic_scholar=enable_semantic_scholar,
    enable_openalex=enable_openalex,
    enable_scholarapi=enable_scholarapi,
    enable_arxiv=enable_arxiv,
    enable_serpapi=enable_serpapi,
    enable_ecos=enable_ecos,
    enable_federal_register=enable_federal_register,
    enable_govinfo_cfr=enable_govinfo_cfr,
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
        await maybe_close_async_resource(scholarapi_client)
        await maybe_close_async_resource(arxiv_client)
        await maybe_close_async_resource(serpapi_client)
        await maybe_close_async_resource(crossref_client)
        await maybe_close_async_resource(unpaywall_client)
        await maybe_close_async_resource(ecos_client)
        await maybe_close_async_resource(federal_register_client)
        await maybe_close_async_resource(govinfo_client)


app = FastMCP(
    "paper-chaser",
    instructions=(GUIDED_SERVER_INSTRUCTIONS if settings.tool_profile == "guided" else SERVER_INSTRUCTIONS),
    lifespan=_server_lifespan,
    strict_input_validation=True,
)
app.add_middleware(TimingMiddleware(logger=logger))

_enabled_tool_flags = {
    "enable_core": enable_core,
    "enable_semantic_scholar": enable_semantic_scholar,
    "enable_arxiv": enable_arxiv,
    "enable_openalex": enable_openalex,
    "enable_serpapi": enable_serpapi,
    "enable_scholarapi": enable_scholarapi,
    "enable_crossref": enable_crossref,
    "enable_unpaywall": enable_unpaywall,
    "enable_ecos": enable_ecos,
    "enable_federal_register": enable_federal_register,
    "enable_govinfo_cfr": enable_govinfo_cfr,
    "enable_agentic": settings.enable_agentic,
    "govinfo_available": bool(govinfo_api_key),
}

for _tool_spec in iter_visible_tool_specs(
    tool_profile=settings.tool_profile,
    hide_disabled_tools=hide_disabled_tools,
    enabled_flags=_enabled_tool_flags,
):
    _register_tool(_tool_spec.name)


def _resource_text(payload: dict[str, Any]) -> str:
    return json.dumps(dump_jsonable(payload), ensure_ascii=False, indent=2)


def _paper_resource_payload(paper: dict[str, Any]) -> dict[str, Any]:
    return render_paper_resource_payload(paper)


def _author_resource_payload(author: dict[str, Any]) -> dict[str, Any]:
    return render_author_resource_payload(author)


@app.resource(
    "guide://paper-chaser/agent-workflows",
    title="Paper Chaser agent workflows",
    description="How to choose the right Paper Chaser tools and pagination flow.",
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
    raise ValueError(f"Could not resolve paper resource for {paper_id!r}.") from last_error


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
    raise ValueError(f"Could not resolve author resource for {author_id!r}.") from last_error


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
    description=("Citation or reference trail for a paper, preferably discovered through tool outputs."),
    mime_type="application/json",
)
async def paper_trail_resource(
    paper_id: str,
    direction: Literal["citations", "references"],
) -> str:
    cached_trail = workspace_registry.find_trail(paper_id=paper_id, direction=direction)
    if cached_trail is not None:
        return _resource_text(workspace_registry.render_search_resource(cached_trail.search_session_id))
    payload = await (
        client.get_paper_citations(paper_id=paper_id, limit=25, fields=None, offset=None)
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
    name="plan_paper_chaser_search",
    title="Plan Paper Chaser",
    description="Generate a tool-first plan for a literature search task.",
)
def plan_paper_chaser_search(
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
    focus_text = f" Focus prompt: {focus_prompt}." if focus_prompt else " No extra focus prompt was supplied."
    return (
        f"You are planning a Paper Chaser workflow about '{topic}'. Goal: {goal}. "
        f"Mode: {mode}. {mode_guidance[mode]}{focus_text} "
        "Default to the guided surface. Start with research for discovery, known-item "
        "recovery, citation repair, and regulatory routing. If the user already has a "
        "citation-like string, use resolve_reference first. Reuse searchSessionId with "
        "follow_up_research for one grounded question and inspect_source for provenance. "
        "Treat abstentions and clarification requests as real outputs, not failures to hide. "
        "Only fall back to the expert surface when you need provider-specific control, "
        "pagination, or provider-native payloads. On the expert surface, search_papers is "
        "the quick brokered path, search_papers_bulk is the exhaustive Semantic Scholar-style "
        "path, and search_papers_smart/map_research_landscape/expand_research_graph are the "
        "deeper agentic tools. If the task explicitly needs OpenAlex-native DOI/ID lookup, "
        "OpenAlex cursor pagination, or OpenAlex author/citation semantics, use the dedicated "
        "*_openalex tools instead of the default broker. For exact paper follow-through, use "
        "get_paper_details, get_paper_citations, get_paper_references, search_authors, "
        "get_author_info, and get_author_papers as needed. "
        "For regulatory work, prefer the guided path first; if you need exact primary-source "
        "control, pivot into search_federal_register, get_federal_register_document, get_cfr_text, "
        "or the ECOS tools. "
        "If you uncover a defect or confusing UX, summarize the exact tool calls, "
        "expected vs actual behavior, and whether the best follow-up is a code "
        "change, a documentation update, or both so the result can turn into an "
        "actionable issue for a GitHub Copilot coding agent. "
        "Treat pagination.nextCursor as opaque: reuse it exactly as returned, do "
        "not edit or fabricate it, and keep it scoped to the tool/query flow that "
        "produced it."
    )


@app.prompt(
    name="plan_smart_paper_chaser_search",
    title="Plan Smart Paper Chaser",
    description=("Generate a smart-tool-first research plan for concept-level discovery."),
)
def plan_smart_paper_chaser_search(
    topic: str,
    goal: str = ("map the literature, answer grounded follow-up questions, and identify the best next actions"),
    mode: Literal["discovery", "review", "known_item", "author", "citation"] = "discovery",
) -> str:
    return (
        f"You are planning a smart Paper Chaser workflow about '{topic}'. "
        f"Goal: {goal}. Mode: {mode}. Start with research unless you have a concrete "
        "reason to force the expert smart surface. If you do need expert smart behavior, "
        "use search_papers_smart for concept-level discovery and reuse searchSessionId across "
        "ask_result_set, map_research_landscape, and expand_research_graph. For broken citations "
        "or almost-right references, prefer resolve_reference on the guided path before broader "
        "discovery. If the smart workflow cannot stay grounded, drop back to research, "
        "inspect_source, or the raw expert tools such as search_papers, search_papers_bulk, "
        "get_paper_details, get_paper_citations, get_paper_references, search_authors, and "
        "get_author_papers. When you already have the right paper and want richer metadata or OA "
        "signals, use the enrichment tools after resolution."
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
        "research. Inspect status, findings, sources, trustSummary, coverage, failure, "
        "and clarification. Save the searchSessionId, then ask one grounded question with "
        "follow_up_research. If one hit becomes a strong anchor, use inspect_source for "
        "provenance and only then pivot into expert enrichment or citation tools."
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
        "Try research first and inspect status, trustSummary, coverage, failure, and "
        "clarification. If the guided path abstains, add a concrete anchor such as a year, "
        "venue, DOI, species name, agency, or title fragment. If you still need lower-level "
        "control after that, fall back to expert search_papers or search_papers_smart."
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
        transport=cast(Literal["http", "streamable-http", "sse"], transport or http_app_transport),
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
