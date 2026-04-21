"""Expert raw Semantic Scholar / multi-provider dispatch entrypoints.

Phase 4 extracted these branches from ``_core.py``:

* ``search_papers``
* ``search_papers_match``
* ``get_paper_details``
* ``resolve_citation``
* ``search_papers_bulk``

Every entrypoint is a ``ctx``-first async function: ``(ctx: DispatchContext,
arguments: dict[str, Any]) -> dict[str, Any]``.  Helpers that live in
``_core.py`` (``_maybe_elicit_and_retry``, ``_finalize_tool_result``,
``_cursor_to_bulk_token``, ``_encode_next_bulk_cursor``) are imported lazily
at call time to avoid circular imports.
"""

from __future__ import annotations

from typing import Any, cast

from ...citation_repair import resolve_citation as _resolve_citation_default
from ...enrichment import (
    attach_enrichments_to_paper_payload,
    hydrate_paper_for_enrichment,
)
from ...models import TOOL_INPUT_MODELS, dump_jsonable
from ...models.tools import (
    PaperLookupArgs,
    PaperMatchArgs,
    ResolveCitationArgs,
    SearchPapersArgs,
)
from ...search import search_papers_with_fallback
from ...utils.cursor import compute_context_hash
from ..context import DispatchContext


def _finalize(
    tool_name: str,
    arguments: dict[str, Any],
    result: Any,
    ctx: DispatchContext,
) -> dict[str, Any]:
    """Proxy ``_finalize_tool_result`` through a lazy import."""
    from .._core import _finalize_tool_result

    return _finalize_tool_result(
        tool_name,
        arguments,
        result,
        workspace_registry=ctx.workspace_registry,
    )


async def _resolve_via_core(**kwargs: Any) -> Any:
    """Dispatch to the ``resolve_citation`` name on ``dispatch._core``.

    Tests monkeypatch ``paper_chaser_mcp.dispatch._core.resolve_citation``; this
    indirection ensures those patches still take effect after extraction.
    """
    from .. import _core as _dispatch_core

    resolve_fn = getattr(_dispatch_core, "resolve_citation", _resolve_citation_default)
    return await resolve_fn(**kwargs)


async def _maybe_elicit(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result: Any,
    ctx: DispatchContext,
) -> dict[str, Any] | None:
    """Proxy ``_maybe_elicit_and_retry`` through a lazy import."""
    from .._core import _maybe_elicit_and_retry

    return await _maybe_elicit_and_retry(
        tool_name=tool_name,
        arguments=arguments,
        result=result,
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


async def _dispatch_search_papers(
    ctx: DispatchContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Entrypoint for the ``search_papers`` tool."""
    name = "search_papers"
    search_args = cast(
        SearchPapersArgs,
        TOOL_INPUT_MODELS[name].model_validate(arguments),
    )
    search_result = await search_papers_with_fallback(
        query=search_args.query,
        limit=search_args.limit,
        year=search_args.year,
        fields=search_args.fields,
        venue=search_args.venue,
        publication_date_or_year=search_args.publication_date_or_year,
        fields_of_study=search_args.fields_of_study,
        publication_types=search_args.publication_types,
        open_access_pdf=search_args.open_access_pdf,
        min_citation_count=search_args.min_citation_count,
        enable_core=ctx.enable_core,
        enable_semantic_scholar=ctx.enable_semantic_scholar,
        enable_arxiv=ctx.enable_arxiv,
        enable_serpapi=ctx.enable_serpapi,
        enable_scholarapi=ctx.enable_scholarapi,
        preferred_provider=search_args.preferred_provider,
        provider_order=search_args.provider_order or ctx.provider_order,
        core_client=ctx.core_client,
        semantic_client=ctx.client,
        arxiv_client=ctx.arxiv_client,
        serpapi_client=ctx.serpapi_client,
        scholarapi_client=ctx.scholarapi_client,
        provider_registry=ctx.provider_registry,
        allow_default_hedging=(search_args.preferred_provider is None and search_args.provider_order is None),
    )
    elicited = await _maybe_elicit(
        tool_name=name,
        arguments=arguments,
        result=search_result,
        ctx=ctx,
    )
    if elicited is not None:
        return elicited
    return _finalize(name, arguments, search_result, ctx)


async def _dispatch_search_papers_match(
    ctx: DispatchContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Entrypoint for the ``search_papers_match`` tool."""
    name = "search_papers_match"
    match_args = cast(
        PaperMatchArgs,
        TOOL_INPUT_MODELS[name].model_validate(arguments),
    )
    serialized = dump_jsonable(
        await ctx.client.search_papers_match(
            query=match_args.query,
            fields=match_args.fields,
            openalex_client=ctx.openalex_client,
            enable_openalex=ctx.enable_openalex,
            crossref_client=ctx.crossref_client,
            enable_crossref=ctx.enable_crossref,
        )
    )
    if (
        match_args.include_enrichment
        and isinstance(serialized, dict)
        and serialized.get("matchFound", True) is not False
        and ctx.enrichment_service is not None
    ):
        enrichment_source = await hydrate_paper_for_enrichment(
            serialized,
            detail_client=ctx.client,
        )
        enriched_payload = await ctx.enrichment_service.enrich_paper_payload(
            enrichment_source,
            query=serialized.get("title") or match_args.query,
        )
        serialized = attach_enrichments_to_paper_payload(
            serialized,
            enriched_paper=enriched_payload,
        )
    elicited = await _maybe_elicit(
        tool_name=name,
        arguments=arguments,
        result=serialized,
        ctx=ctx,
    )
    if elicited is not None:
        return elicited
    return _finalize(name, arguments, serialized, ctx)


async def _dispatch_get_paper_details(
    ctx: DispatchContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Entrypoint for the ``get_paper_details`` tool."""
    name = "get_paper_details"
    paper_lookup_args = cast(
        PaperLookupArgs,
        TOOL_INPUT_MODELS[name].model_validate(arguments),
    )
    serialized = dump_jsonable(
        await ctx.client.get_paper_details(
            paper_id=paper_lookup_args.paper_id,
            fields=paper_lookup_args.fields,
        )
    )
    if paper_lookup_args.include_enrichment and isinstance(serialized, dict) and ctx.enrichment_service is not None:
        enrichment_source = await hydrate_paper_for_enrichment(
            serialized,
            detail_client=ctx.client,
        )
        enriched_payload = await ctx.enrichment_service.enrich_paper_payload(
            enrichment_source,
            query=serialized.get("title"),
        )
        serialized = attach_enrichments_to_paper_payload(
            serialized,
            enriched_paper=enriched_payload,
        )
    return _finalize(name, arguments, serialized, ctx)


async def _dispatch_resolve_citation(
    ctx: DispatchContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Entrypoint for the ``resolve_citation`` tool."""
    name = "resolve_citation"
    citation_args = cast(
        ResolveCitationArgs,
        TOOL_INPUT_MODELS[name].model_validate(arguments),
    )
    result = await _resolve_via_core(
        citation=citation_args.citation,
        max_candidates=citation_args.max_candidates,
        client=ctx.client,
        enable_core=ctx.enable_core,
        enable_semantic_scholar=ctx.enable_semantic_scholar,
        enable_openalex=ctx.enable_openalex,
        enable_arxiv=ctx.enable_arxiv,
        enable_serpapi=ctx.enable_serpapi,
        core_client=ctx.core_client,
        openalex_client=ctx.openalex_client,
        arxiv_client=ctx.arxiv_client,
        serpapi_client=ctx.serpapi_client,
        title_hint=citation_args.title_hint,
        author_hint=citation_args.author_hint,
        year_hint=citation_args.year_hint,
        venue_hint=citation_args.venue_hint,
        doi_hint=citation_args.doi_hint,
        include_enrichment=citation_args.include_enrichment,
        enrichment_service=ctx.enrichment_service,
    )
    return _finalize(name, arguments, result, ctx)


async def _dispatch_search_papers_bulk(
    ctx: DispatchContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Entrypoint for the ``search_papers_bulk`` tool."""
    from .._core import _cursor_to_bulk_token, _encode_next_bulk_cursor

    name = "search_papers_bulk"
    validated_payload = TOOL_INPUT_MODELS[name].model_validate(arguments)
    args_dict = validated_payload.model_dump(by_alias=False)
    ctx_hash = compute_context_hash(name, args_dict)
    method = getattr(ctx.client, "search_papers_bulk")
    result = await method(
        query=args_dict["query"],
        fields=args_dict.get("fields"),
        token=_cursor_to_bulk_token(
            args_dict.get("cursor"),
            tool=name,
            context_hash=ctx_hash,
        ),
        sort=args_dict.get("sort"),
        limit=args_dict.get("limit", 100),
        year=args_dict.get("year"),
        publication_date_or_year=args_dict.get("publication_date_or_year"),
        fields_of_study=args_dict.get("fields_of_study"),
        publication_types=args_dict.get("publication_types"),
        open_access_pdf=args_dict.get("open_access_pdf"),
        min_citation_count=args_dict.get("min_citation_count"),
    )
    serialized = dump_jsonable(result)
    serialized = _encode_next_bulk_cursor(serialized, name, context_hash=ctx_hash)
    sort_param = args_dict.get("sort")
    if sort_param:
        retrieval_note = (
            f"Results are sorted by '{sort_param}' (Semantic Scholar "
            "/paper/search/bulk). Bulk retrieval is exhaustive corpus "
            "collection — results are ordered by the specified sort, not "
            "by relevance to the query. This is a different contract from "
            "search_papers. Use pagination.nextCursor to continue."
        )
    else:
        retrieval_note = (
            "ORDERING: search_papers_bulk uses exhaustive corpus traversal "
            "with an internal ordering that is NOT relevance-ranked. This is "
            "NOT 'page 2' of search_papers — the ranking semantics differ and "
            "results may appear unrelated to the discovery page. For "
            "relevance-ranked results use search_papers or "
            "search_papers_semantic_scholar. For citation-ranked bulk "
            "retrieval pass sort='citationCount:desc'. Use "
            "pagination.nextCursor to continue this bulk stream."
        )
    serialized.setdefault("retrievalNote", retrieval_note)
    return _finalize(name, arguments, serialized, ctx)
