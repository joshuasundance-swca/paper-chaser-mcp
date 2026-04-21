"""Expert OpenAlex-specific dispatch entrypoints."""

from __future__ import annotations

from typing import Any

from ...models import TOOL_INPUT_MODELS, dump_jsonable
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


def _require_openalex(ctx: DispatchContext, tool_name: str) -> None:
    if not ctx.enable_openalex:
        raise ValueError(
            f"{tool_name} requires OpenAlex, which is disabled. Set PAPER_CHASER_ENABLE_OPENALEX=true to use this tool."
        )


async def _dispatch_paper_autocomplete_openalex(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    name = "paper_autocomplete_openalex"
    _require_openalex(ctx, name)
    args_dict = TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
    result = await ctx.openalex_client.paper_autocomplete(
        query=args_dict["query"],
        limit=args_dict.get("limit", 10),
    )
    return _finalize(name, arguments, result, ctx)


async def _dispatch_search_papers_openalex(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    name = "search_papers_openalex"
    _require_openalex(ctx, name)
    args_dict = TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
    result = await ctx.openalex_client.search(
        query=args_dict["query"],
        limit=args_dict.get("limit", 10),
        year=args_dict.get("year"),
    )
    return _finalize(name, arguments, result, ctx)


async def _dispatch_search_entities_openalex(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from .._core import _cursor_to_bulk_token, _encode_next_bulk_cursor

    name = "search_entities_openalex"
    _require_openalex(ctx, name)
    args_dict = TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
    ctx_hash = compute_context_hash(name, args_dict)
    result = await ctx.openalex_client.search_entities(
        entity_type=args_dict["entity_type"],
        query=args_dict["query"],
        limit=args_dict.get("limit", 10),
        cursor=_cursor_to_bulk_token(
            args_dict.get("cursor"),
            tool=name,
            context_hash=ctx_hash,
            expected_provider="openalex",
        ),
    )
    serialized = _encode_next_bulk_cursor(dump_jsonable(result), name, context_hash=ctx_hash, provider="openalex")
    return _finalize(name, arguments, serialized, ctx)


async def _dispatch_search_papers_openalex_by_entity(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from .._core import _cursor_to_bulk_token, _encode_next_bulk_cursor

    name = "search_papers_openalex_by_entity"
    if not ctx.enable_openalex:
        raise ValueError(
            "search_papers_openalex_by_entity requires OpenAlex, "
            "which is disabled. "
            "Set PAPER_CHASER_ENABLE_OPENALEX=true to use this tool."
        )
    args_dict = TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
    ctx_hash = compute_context_hash(name, args_dict)
    result = await ctx.openalex_client.search_works_by_entity(
        entity_type=args_dict["entity_type"],
        entity_id=args_dict["entity_id"],
        limit=args_dict.get("limit", 100),
        cursor=_cursor_to_bulk_token(
            args_dict.get("cursor"),
            tool=name,
            context_hash=ctx_hash,
            expected_provider="openalex",
        ),
        year=args_dict.get("year"),
    )
    serialized = _encode_next_bulk_cursor(dump_jsonable(result), name, context_hash=ctx_hash, provider="openalex")
    return _finalize(name, arguments, serialized, ctx)


async def _dispatch_search_papers_openalex_bulk(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from .._core import _cursor_to_bulk_token, _encode_next_bulk_cursor

    name = "search_papers_openalex_bulk"
    _require_openalex(ctx, name)
    args_dict = TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
    ctx_hash = compute_context_hash(name, args_dict)
    result = await ctx.openalex_client.search_bulk(
        query=args_dict["query"],
        limit=args_dict.get("limit", 100),
        cursor=_cursor_to_bulk_token(
            args_dict.get("cursor"),
            tool=name,
            context_hash=ctx_hash,
            expected_provider="openalex",
        ),
        year=args_dict.get("year"),
    )
    serialized = _encode_next_bulk_cursor(dump_jsonable(result), name, context_hash=ctx_hash, provider="openalex")
    return _finalize(name, arguments, serialized, ctx)


async def _dispatch_get_paper_details_openalex(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    name = "get_paper_details_openalex"
    args_dict = TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
    result = await ctx.openalex_client.get_paper_details(paper_id=args_dict["paper_id"])
    return _finalize(name, arguments, result, ctx)


async def _dispatch_get_paper_citations_openalex(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from .._core import _cursor_to_bulk_token, _encode_next_bulk_cursor

    name = "get_paper_citations_openalex"
    args_dict = TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
    ctx_hash = compute_context_hash(name, args_dict)
    result = await ctx.openalex_client.get_paper_citations(
        paper_id=args_dict["paper_id"],
        limit=args_dict.get("limit", 100),
        cursor=_cursor_to_bulk_token(
            args_dict.get("cursor"),
            tool=name,
            context_hash=ctx_hash,
            expected_provider="openalex",
        ),
    )
    serialized = _encode_next_bulk_cursor(dump_jsonable(result), name, context_hash=ctx_hash, provider="openalex")
    return _finalize(name, arguments, serialized, ctx)


async def _dispatch_get_paper_references_openalex(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    name = "get_paper_references_openalex"
    args_dict = TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
    ctx_hash = compute_context_hash(name, args_dict)
    result = await ctx.openalex_client.get_paper_references(
        paper_id=args_dict["paper_id"],
        limit=args_dict.get("limit", 100),
        offset=_cursor_to_offset(
            args_dict.get("cursor"),
            name,
            context_hash=ctx_hash,
            expected_provider="openalex",
        )
        or 0,
    )
    serialized = _encode_next_cursor(dump_jsonable(result), name, context_hash=ctx_hash, provider="openalex")
    return _finalize(name, arguments, serialized, ctx)


async def _dispatch_search_authors_openalex(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from .._core import _cursor_to_bulk_token, _encode_next_bulk_cursor

    name = "search_authors_openalex"
    args_dict = TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
    ctx_hash = compute_context_hash(name, args_dict)
    result = await ctx.openalex_client.search_authors(
        query=args_dict["query"],
        limit=args_dict.get("limit", 10),
        cursor=_cursor_to_bulk_token(
            args_dict.get("cursor"),
            tool=name,
            context_hash=ctx_hash,
            expected_provider="openalex",
        ),
    )
    serialized = _encode_next_bulk_cursor(dump_jsonable(result), name, context_hash=ctx_hash, provider="openalex")
    return _finalize(name, arguments, serialized, ctx)


async def _dispatch_get_author_info_openalex(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    name = "get_author_info_openalex"
    args_dict = TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
    result = await ctx.openalex_client.get_author_info(author_id=args_dict["author_id"])
    return _finalize(name, arguments, result, ctx)


async def _dispatch_get_author_papers_openalex(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from .._core import _cursor_to_bulk_token, _encode_next_bulk_cursor

    name = "get_author_papers_openalex"
    args_dict = TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
    ctx_hash = compute_context_hash(name, args_dict)
    result = await ctx.openalex_client.get_author_papers(
        author_id=args_dict["author_id"],
        limit=args_dict.get("limit", 100),
        cursor=_cursor_to_bulk_token(
            args_dict.get("cursor"),
            tool=name,
            context_hash=ctx_hash,
            expected_provider="openalex",
        ),
        year=args_dict.get("year"),
    )
    serialized = _encode_next_bulk_cursor(dump_jsonable(result), name, context_hash=ctx_hash, provider="openalex")
    return _finalize(name, arguments, serialized, ctx)
