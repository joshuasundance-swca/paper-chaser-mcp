"""Expert SerpApi-specific dispatch entrypoints."""

from __future__ import annotations

from typing import Any, cast

from ...clients.serpapi import SerpApiKeyMissingError
from ...models import TOOL_INPUT_MODELS, CitationFormatsResponse, dump_jsonable
from ...models.common import CitationFormat, ExportLink
from ...models.tools import GetCitationFormatsArgs
from ...utils.cursor import compute_context_hash
from ..context import DispatchContext
from ..paging import _cursor_to_offset, _encode_next_cursor


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


def _require_serpapi(ctx: DispatchContext, tool_name: str) -> None:
    if not ctx.enable_serpapi:
        raise ValueError(
            f"{tool_name} requires SerpApi, which is not enabled. "
            "Set PAPER_CHASER_ENABLE_SERPAPI=true and provide SERPAPI_API_KEY."
        )


async def _dispatch_search_papers_serpapi_cited_by(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    name = "search_papers_serpapi_cited_by"
    if not ctx.enable_serpapi:
        raise ValueError(
            "search_papers_serpapi_cited_by requires SerpApi, "
            "which is not enabled. "
            "Set PAPER_CHASER_ENABLE_SERPAPI=true and provide SERPAPI_API_KEY."
        )
    args_dict = TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
    ctx_hash = compute_context_hash(name, args_dict)
    result = await ctx.serpapi_client.search_cited_by(
        cites_id=args_dict["cites_id"],
        query=args_dict.get("query"),
        limit=args_dict.get("limit", 10),
        start=_cursor_to_offset(
            args_dict.get("cursor"),
            name,
            context_hash=ctx_hash,
            expected_provider="serpapi_google_scholar",
        )
        or 0,
        year=args_dict.get("year"),
    )
    serialized = _encode_next_cursor(
        dump_jsonable(result),
        name,
        context_hash=ctx_hash,
        provider="serpapi_google_scholar",
    )
    return _finalize(name, arguments, serialized, ctx)


async def _dispatch_search_papers_serpapi_versions(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    name = "search_papers_serpapi_versions"
    if not ctx.enable_serpapi:
        raise ValueError(
            "search_papers_serpapi_versions requires SerpApi, "
            "which is not enabled. "
            "Set PAPER_CHASER_ENABLE_SERPAPI=true and provide SERPAPI_API_KEY."
        )
    args_dict = TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
    ctx_hash = compute_context_hash(name, args_dict)
    result = await ctx.serpapi_client.search_versions(
        cluster_id=args_dict["cluster_id"],
        limit=args_dict.get("limit", 10),
        start=_cursor_to_offset(
            args_dict.get("cursor"),
            name,
            context_hash=ctx_hash,
            expected_provider="serpapi_google_scholar",
        )
        or 0,
    )
    serialized = _encode_next_cursor(
        dump_jsonable(result),
        name,
        context_hash=ctx_hash,
        provider="serpapi_google_scholar",
    )
    return _finalize(name, arguments, serialized, ctx)


async def _dispatch_get_author_profile_serpapi(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    name = "get_author_profile_serpapi"
    _require_serpapi(ctx, name)
    args_dict = TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
    result = await ctx.serpapi_client.get_author_profile(author_id=args_dict["author_id"])
    return _finalize(name, arguments, result, ctx)


async def _dispatch_get_author_articles_serpapi(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    name = "get_author_articles_serpapi"
    _require_serpapi(ctx, name)
    args_dict = TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
    ctx_hash = compute_context_hash(name, args_dict)
    result = await ctx.serpapi_client.get_author_articles(
        author_id=args_dict["author_id"],
        limit=args_dict.get("limit", 10),
        start=_cursor_to_offset(
            args_dict.get("cursor"),
            name,
            context_hash=ctx_hash,
            expected_provider="serpapi_google_scholar",
        )
        or 0,
        sort=args_dict.get("sort"),
    )
    serialized = _encode_next_cursor(
        dump_jsonable(result),
        name,
        context_hash=ctx_hash,
        provider="serpapi_google_scholar",
    )
    return _finalize(name, arguments, serialized, ctx)


async def _dispatch_get_serpapi_account_status(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    name = "get_serpapi_account_status"
    _require_serpapi(ctx, name)
    result = await ctx.serpapi_client.get_account_status()
    return _finalize(name, arguments, result, ctx)


async def _dispatch_get_paper_citation_formats(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    name = "get_paper_citation_formats"
    validated_cf = cast(
        GetCitationFormatsArgs,
        TOOL_INPUT_MODELS[name].model_validate(arguments),
    )
    if not ctx.enable_serpapi:
        raise ValueError(
            "get_paper_citation_formats requires SerpApi, which is not enabled. "
            "Set PAPER_CHASER_ENABLE_SERPAPI=true and provide SERPAPI_API_KEY "
            "to use this tool. SerpApi is a paid service — see "
            "https://serpapi.com for details."
        )
    if ctx.serpapi_client is None:
        raise ValueError("SerpApi client is not available. Set PAPER_CHASER_ENABLE_SERPAPI=true and SERPAPI_API_KEY.")
    try:
        raw = await ctx.serpapi_client.get_citation_formats(result_id=validated_cf.result_id)
    except SerpApiKeyMissingError:
        raise
    raw_citations = raw.get("citations") or []
    raw_links = raw.get("links") or []
    citation_response = CitationFormatsResponse(
        result_id=validated_cf.result_id,
        citations=[
            CitationFormat(
                title=str(c.get("title") or ""),
                snippet=str(c.get("snippet") or ""),
            )
            for c in raw_citations
            if isinstance(c, dict)
        ],
        export_links=[
            ExportLink(
                name=str(lnk.get("name") or ""),
                link=str(lnk.get("link") or ""),
            )
            for lnk in raw_links
            if isinstance(lnk, dict)
        ],
    )
    return _finalize(name, arguments, dump_jsonable(citation_response), ctx)
