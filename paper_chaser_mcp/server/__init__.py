"""FastMCP-backed public server surface for Paper Chaser."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Literal, cast

from fastmcp import Context, FastMCP
from fastmcp.server.middleware.timing import TimingMiddleware
from mcp.types import TextContent, Tool, ToolAnnotations

from ..agentic import (
    AgenticConfig,
    AgenticRuntime,
    WorkspaceRegistry,
    resolve_provider_bundle,
)
from ..clients import (
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
from ..clients.serpapi import SerpApiScholarClient
from ..constants import (
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
from ..dispatch import dispatch_tool
from ..enrichment import PaperEnrichmentService
from ..eval_curation import maybe_capture_eval_candidate
from ..models import dump_jsonable
from ..parsing import _arxiv_id_from_url, _text
from ..provider_runtime import ProviderDiagnosticsRegistry
from ..runtime import run_server
from ..search import _core_response_to_merged, _merge_search_results
from ..settings import AppSettings, _env_bool
from ..tool_specs import get_tool_spec, iter_visible_tool_specs
from ..transport import asyncio, httpx, maybe_close_async_resource
from .instructions import (
    AGENT_WORKFLOW_GUIDE,
    GUIDED_SERVER_INSTRUCTIONS,
    SERVER_INSTRUCTIONS,
)
from .prompts import register_prompts
from .registration import (
    _build_signature,
    _format_tool_display_name,
    _parameter_default,
    _parameter_name,
    _sanitize_registered_tool_schema,
    _tool_tags,
)
from .resources import (
    _author_resource_payload,
    _paper_resource_payload,
    _resource_text,
    register_resources,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("paper-chaser-mcp")


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
    _sanitize_registered_tool_schema(app, tool_name)


async def _execute_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    ctx: Context | None = None,
) -> dict[str, Any]:
    _initialize_runtime()
    started = time.perf_counter()
    result = await dispatch_tool(
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
        guided_research_latency_profile=settings.guided_research_latency_profile,
        guided_follow_up_latency_profile=settings.guided_follow_up_latency_profile,
        guided_allow_paid_providers=settings.guided_allow_paid_providers,
        guided_escalation_enabled=settings.guided_escalation_enabled,
        guided_escalation_max_passes=settings.guided_escalation_max_passes,
        guided_escalation_allow_paid_providers=settings.guided_escalation_allow_paid_providers,
        ctx=ctx,
    )
    duration_ms = int((time.perf_counter() - started) * 1000)
    maybe_capture_eval_candidate(
        workspace_registry=workspace_registry,
        tool_name=name,
        arguments=arguments,
        result=result,
        run_id=os.environ.get("PAPER_CHASER_EVAL_RUN_ID"),
        batch_id=os.environ.get("PAPER_CHASER_EVAL_BATCH_ID"),
        duration_ms=duration_ms,
    )
    return result


settings = AppSettings()
agentic_config = AgenticConfig.from_settings(settings)
api_key = settings.semantic_scholar_api_key
openai_api_key = settings.openai_api_key
openrouter_api_key = settings.openrouter_api_key
openrouter_base_url = settings.openrouter_base_url
openrouter_http_referer = settings.openrouter_http_referer
openrouter_title = settings.openrouter_title
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
client: SemanticScholarClient | Any | None = None
core_client: CoreApiClient | Any | None = None
federal_register_client: FederalRegisterClient | Any | None = None
govinfo_client: GovInfoClient | Any | None = None
crossref_client: CrossrefClient | Any | None = None
ecos_client: EcosClient | Any | None = None
openalex_client: OpenAlexClient | Any | None = None
scholarapi_client: ScholarApiClient | Any | None = None
arxiv_client: ArxivClient | Any | None = None
serpapi_client: SerpApiScholarClient | Any | None = None
unpaywall_client: UnpaywallClient | Any | None = None
enrichment_service: PaperEnrichmentService | Any | None = None
provider_bundle: Any | None = None
workspace_registry: WorkspaceRegistry | Any | None = None
agentic_runtime: AgenticRuntime | Any | None = None
_runtime_initialized = False


@asynccontextmanager
async def _server_lifespan(_: FastMCP):
    _initialize_runtime()
    try:
        yield
    finally:
        if agentic_runtime is not None:
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
    instructions=GUIDED_SERVER_INSTRUCTIONS,
    lifespan=_server_lifespan,
    strict_input_validation=True,
)
app.add_middleware(TimingMiddleware(logger=logger))


def _enabled_tool_flags() -> dict[str, bool]:
    return {
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


def _configure_registered_tools() -> None:
    for tool_spec in iter_visible_tool_specs(
        tool_profile=settings.tool_profile,
        hide_disabled_tools=hide_disabled_tools,
        enabled_flags=_enabled_tool_flags(),
    ):
        if f"tool:{tool_spec.name}@" not in app.local_provider._components:
            _register_tool(tool_spec.name)


def _initialize_runtime() -> None:
    global settings, agentic_config, api_key, openai_api_key, openrouter_api_key, openrouter_base_url
    global openrouter_http_referer, openrouter_title, nvidia_api_key, nvidia_nim_base_url
    global azure_openai_api_key, azure_openai_endpoint, azure_openai_api_version
    global azure_openai_planner_deployment, azure_openai_synthesis_deployment, anthropic_api_key
    global google_api_key, mistral_api_key, huggingface_api_key, huggingface_base_url
    global core_api_key, openalex_api_key, openalex_mailto, serpapi_api_key, scholarapi_api_key
    global govinfo_api_key, crossref_mailto, unpaywall_email, ecos_base_url
    global enable_core, enable_semantic_scholar, enable_openalex, enable_arxiv, enable_serpapi
    global enable_scholarapi, enable_crossref, enable_unpaywall, enable_ecos
    global enable_federal_register, enable_govinfo_cfr, hide_disabled_tools, provider_order
    global provider_registry, client, core_client, federal_register_client, govinfo_client
    global crossref_client, ecos_client, openalex_client, scholarapi_client, arxiv_client
    global serpapi_client, unpaywall_client, enrichment_service, provider_bundle
    global workspace_registry, agentic_runtime, http_app_transport, http_app, _runtime_initialized

    if _runtime_initialized:
        return

    runtime_settings = AppSettings.from_env()
    for warning in runtime_settings.runtime_warnings():
        logger.warning("Runtime configuration warning: %s", warning)

    settings = runtime_settings
    agentic_config = AgenticConfig.from_settings(settings)
    api_key = settings.semantic_scholar_api_key
    openai_api_key = settings.openai_api_key
    openrouter_api_key = settings.openrouter_api_key
    openrouter_base_url = settings.openrouter_base_url
    openrouter_http_referer = settings.openrouter_http_referer
    openrouter_title = settings.openrouter_title
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

    if provider_registry is None:
        provider_registry = ProviderDiagnosticsRegistry()
    if client is None:
        client = SemanticScholarClient(api_key=api_key)
    if core_client is None:
        core_client = CoreApiClient(api_key=core_api_key)
    if federal_register_client is None:
        federal_register_client = FederalRegisterClient(timeout=settings.federal_register_timeout_seconds)
    if govinfo_client is None:
        govinfo_client = GovInfoClient(
            api_key=govinfo_api_key,
            timeout=settings.govinfo_timeout_seconds,
            document_timeout=settings.govinfo_document_timeout_seconds,
            max_document_size_mb=settings.govinfo_max_document_size_mb,
        )
    if crossref_client is None:
        crossref_client = CrossrefClient(
            mailto=crossref_mailto,
            timeout=settings.crossref_timeout_seconds,
        )
    if ecos_client is None:
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
    if openalex_client is None:
        openalex_client = OpenAlexClient(api_key=openalex_api_key, mailto=openalex_mailto)
    if scholarapi_client is None:
        scholarapi_client = ScholarApiClient(api_key=scholarapi_api_key)
    if arxiv_client is None:
        arxiv_client = ArxivClient()
    if serpapi_client is None:
        serpapi_client = SerpApiScholarClient(api_key=serpapi_api_key)
    if unpaywall_client is None:
        unpaywall_client = UnpaywallClient(
            email=unpaywall_email,
            timeout=settings.unpaywall_timeout_seconds,
        )
    if enrichment_service is None:
        enrichment_service = PaperEnrichmentService(
            crossref_client=crossref_client,
            unpaywall_client=unpaywall_client,
            openalex_client=openalex_client,
            enable_crossref=enable_crossref,
            enable_unpaywall=enable_unpaywall,
            enable_openalex=enable_openalex,
            provider_registry=provider_registry,
        )
    if provider_bundle is None:
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
            openrouter_api_key=openrouter_api_key,
            openrouter_base_url=openrouter_base_url,
            openrouter_http_referer=openrouter_http_referer,
            openrouter_title=openrouter_title,
            provider_registry=provider_registry,
        )
    if workspace_registry is None:
        workspace_registry = WorkspaceRegistry(
            ttl_seconds=settings.session_ttl_seconds,
            enable_trace_log=settings.enable_agentic_trace_log,
            eval_trace_path=(settings.eval_trace_path if settings.enable_eval_trace_capture else None),
            index_backend=settings.agentic_index_backend,
            similarity_fn=provider_bundle.similarity,
            async_batched_similarity_fn=provider_bundle.abatched_similarity,
            async_embed_query_fn=(None if not provider_bundle.supports_embeddings() else provider_bundle.aembed_query),
            async_embed_texts_fn=(None if not provider_bundle.supports_embeddings() else provider_bundle.aembed_texts),
            embed_query_fn=(None if not provider_bundle.supports_embeddings() else provider_bundle.embed_query),
            embed_texts_fn=(None if not provider_bundle.supports_embeddings() else provider_bundle.embed_texts),
        )
    if agentic_runtime is None:
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

    _configure_registered_tools()
    app.instructions = GUIDED_SERVER_INSTRUCTIONS if settings.tool_profile == "guided" else SERVER_INSTRUCTIONS
    http_app_transport = cast(
        Literal["http", "streamable-http", "sse"],
        settings.transport if settings.transport != "stdio" else "streamable-http",
    )
    http_app = app.http_app(
        path=settings.http_path,
        transport=http_app_transport,
        middleware=None,
    )
    _runtime_initialized = True


_configure_registered_tools()


def _require_workspace_registry() -> WorkspaceRegistry | Any:
    _initialize_runtime()
    assert workspace_registry is not None
    return workspace_registry


def _require_semantic_client() -> SemanticScholarClient | Any:
    _initialize_runtime()
    assert client is not None
    return client


def _require_openalex_client() -> OpenAlexClient | Any:
    _initialize_runtime()
    assert openalex_client is not None
    return openalex_client


_resources_registered = register_resources(
    app,
    require_workspace_registry=_require_workspace_registry,
    require_semantic_client=_require_semantic_client,
    require_openalex_client=_require_openalex_client,
    agent_workflow_guide=AGENT_WORKFLOW_GUIDE,
    logger=logger,
)
agent_workflows = _resources_registered["agent_workflows"]
paper_resource = _resources_registered["paper_resource"]
author_resource = _resources_registered["author_resource"]
search_session_resource = _resources_registered["search_session_resource"]
paper_trail_resource = _resources_registered["paper_trail_resource"]

_prompts_registered = register_prompts(app)
plan_paper_chaser_search = _prompts_registered["plan_paper_chaser_search"]
plan_smart_paper_chaser_search = _prompts_registered["plan_smart_paper_chaser_search"]
triage_literature = _prompts_registered["triage_literature"]
plan_citation_chase = _prompts_registered["plan_citation_chase"]
refine_query = _prompts_registered["refine_query"]


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
    _initialize_runtime()
    return app.http_app(
        path=path or settings.http_path,
        transport=cast(Literal["http", "streamable-http", "sse"], transport or http_app_transport),
        middleware=middleware,
    )


http_app = app.http_app(
    path=settings.http_path,
    transport=http_app_transport,
    middleware=None,
)


async def list_tools() -> list[Tool]:
    """Compatibility helper returning the registered tool schemas."""
    _initialize_runtime()
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
    _initialize_runtime()
    run_server(app=app, logger=logger, settings=settings)


if __name__ == "__main__":
    main()
