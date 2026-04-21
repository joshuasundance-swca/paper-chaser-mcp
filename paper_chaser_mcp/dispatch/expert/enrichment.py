"""Expert enrichment + generic fallback dispatch entrypoints.

Handles:

* ``get_paper_metadata_crossref``
* ``get_paper_open_access_unpaywall``
* ``enrich_paper``
* ``PROVIDER_SEARCH_TOOLS`` fallthrough (provider-scoped ``search_papers_*``)
* ``NON_SEARCH_TOOL_HANDLERS`` generic fallback
"""

from __future__ import annotations

from typing import Any, cast

from ...models import TOOL_INPUT_MODELS, dump_jsonable
from ...models.tools import BasicSearchPapersArgs
from ...search import search_papers_with_fallback
from ...utils.cursor import OFFSET_TOOLS, compute_context_hash
from ..context import DispatchContext
from ..paging import _encode_next_cursor
from ..snippet_fallback import _maybe_fallback_snippet_search


def _finalize(
    tool_name: str,
    arguments: dict[str, Any],
    result: Any,
    ctx: DispatchContext,
) -> dict[str, Any]:
    from .._core import _finalize_tool_result

    return _finalize_tool_result(
        tool_name,
        arguments,
        result,
        workspace_registry=ctx.workspace_registry,
    )


async def _dispatch_get_paper_metadata_crossref(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    name = "get_paper_metadata_crossref"
    if ctx.enrichment_service is None:
        raise ValueError("get_paper_metadata_crossref requires Crossref enrichment, which is disabled.")
    crossref_result = await ctx.enrichment_service.get_crossref_metadata(
        **TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
    )
    return _finalize(name, arguments, dump_jsonable(crossref_result), ctx)


async def _dispatch_get_paper_open_access_unpaywall(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    name = "get_paper_open_access_unpaywall"
    if ctx.enrichment_service is None:
        raise ValueError("get_paper_open_access_unpaywall requires Unpaywall enrichment, which is disabled.")
    unpaywall_result = await ctx.enrichment_service.get_unpaywall_open_access(
        **TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
    )
    return _finalize(name, arguments, dump_jsonable(unpaywall_result), ctx)


async def _dispatch_enrich_paper(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    name = "enrich_paper"
    if ctx.enrichment_service is None:
        raise ValueError("enrich_paper requires the enrichment service, which is disabled.")
    enrichment_result = await ctx.enrichment_service.enrich_paper(
        **TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
    )
    return _finalize(name, arguments, dump_jsonable(enrichment_result), ctx)


async def _dispatch_provider_search_tool(
    ctx: DispatchContext, arguments: dict[str, Any], *, name: str
) -> dict[str, Any]:
    """Fallthrough for provider-scoped ``search_papers_*`` tools."""
    from .._core import PROVIDER_SEARCH_TOOLS

    provider_arguments = cast(
        BasicSearchPapersArgs,
        TOOL_INPUT_MODELS[name].model_validate(arguments),
    )
    provider_search_result = await search_papers_with_fallback(
        query=provider_arguments.query,
        limit=provider_arguments.limit,
        year=provider_arguments.year,
        fields=getattr(provider_arguments, "fields", None),
        venue=getattr(provider_arguments, "venue", None),
        publication_date_or_year=getattr(provider_arguments, "publication_date_or_year", None),
        fields_of_study=getattr(provider_arguments, "fields_of_study", None),
        publication_types=getattr(provider_arguments, "publication_types", None),
        open_access_pdf=getattr(provider_arguments, "open_access_pdf", None),
        min_citation_count=getattr(provider_arguments, "min_citation_count", None),
        enable_core=ctx.enable_core,
        enable_semantic_scholar=ctx.enable_semantic_scholar,
        enable_arxiv=ctx.enable_arxiv,
        enable_serpapi=ctx.enable_serpapi,
        enable_scholarapi=ctx.enable_scholarapi,
        provider_order=[PROVIDER_SEARCH_TOOLS[name]],
        core_client=ctx.core_client,
        semantic_client=ctx.client,
        arxiv_client=ctx.arxiv_client,
        serpapi_client=ctx.serpapi_client,
        scholarapi_client=ctx.scholarapi_client,
        provider_registry=ctx.provider_registry,
        allow_default_hedging=False,
    )
    return _finalize(name, arguments, provider_search_result, ctx)


async def _dispatch_non_search_tool(ctx: DispatchContext, arguments: dict[str, Any], *, name: str) -> dict[str, Any]:
    """Generic fallback for tools registered in ``NON_SEARCH_TOOL_HANDLERS``."""
    from .._core import NON_SEARCH_TOOL_HANDLERS, _maybe_elicit_and_retry

    try:
        method_name, build_args = NON_SEARCH_TOOL_HANDLERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown tool: {name}") from exc

    args_dict = TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
    ctx_hash = compute_context_hash(name, args_dict) if name in OFFSET_TOOLS else None
    method = getattr(ctx.client, method_name)
    result = await method(**build_args(args_dict))
    serialized = dump_jsonable(result)
    if name == "search_snippets" and isinstance(serialized, dict):
        serialized = await _maybe_fallback_snippet_search(
            serialized=serialized,
            args_dict=args_dict,
            client=ctx.client,
        )
    if name in OFFSET_TOOLS:
        serialized = _encode_next_cursor(serialized, name, context_hash=ctx_hash)
    elicited = await _maybe_elicit_and_retry(
        tool_name=name,
        arguments=arguments,
        result=serialized,
        client=ctx.client,
        core_client=ctx.core_client,
        openalex_client=ctx.openalex_client,
        arxiv_client=ctx.arxiv_client,
        serpapi_client=ctx.serpapi_client,
        scholarapi_client=ctx.scholarapi_client,
        crossref_client=ctx.crossref_client,
        unpaywall_client=ctx.unpaywall_client,
        ecos_client=ctx.ecos_client,
        federal_register_client=ctx.federal_register_client,
        govinfo_client=ctx.govinfo_client,
        enable_core=ctx.enable_core,
        enable_semantic_scholar=ctx.enable_semantic_scholar,
        enable_openalex=ctx.enable_openalex,
        enable_arxiv=ctx.enable_arxiv,
        enable_serpapi=ctx.enable_serpapi,
        enable_scholarapi=ctx.enable_scholarapi,
        enable_crossref=ctx.enable_crossref,
        enable_unpaywall=ctx.enable_unpaywall,
        enable_ecos=ctx.enable_ecos,
        enable_federal_register=ctx.enable_federal_register,
        enable_govinfo_cfr=ctx.enable_govinfo_cfr,
        provider_order=ctx.provider_order,
        provider_registry=ctx.provider_registry,
        workspace_registry=ctx.workspace_registry,
        enrichment_service=ctx.enrichment_service,
        agentic_runtime=ctx.agentic_runtime,
        ctx=ctx.ctx,
        allow_elicitation=ctx.allow_elicitation,
    )
    if elicited is not None:
        return elicited
    return _finalize(name, arguments, serialized, ctx)
