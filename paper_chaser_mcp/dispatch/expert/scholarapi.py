"""Expert ScholarAPI-specific dispatch entrypoints."""

from __future__ import annotations

from typing import Any

from ...models import TOOL_INPUT_MODELS, dump_jsonable
from ...utils.cursor import compute_context_hash
from ..context import DispatchContext


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


def _require_scholarapi(ctx: DispatchContext, tool_name: str) -> None:
    if not ctx.enable_scholarapi:
        raise ValueError(
            f"{tool_name} requires ScholarAPI, which is not enabled. "
            "Set PAPER_CHASER_ENABLE_SCHOLARAPI=true and provide SCHOLARAPI_API_KEY."
        )


async def _dispatch_search_papers_scholarapi(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from .._core import (
        _call_explicit_scholarapi_tool,
        _cursor_to_bulk_token,
        _encode_next_bulk_cursor,
    )

    name = "search_papers_scholarapi"
    _require_scholarapi(ctx, name)
    args_dict = TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
    ctx_hash = compute_context_hash(name, args_dict)
    result = await _call_explicit_scholarapi_tool(
        operation=lambda: ctx.scholarapi_client.search(
            query=args_dict["query"],
            limit=args_dict.get("limit", 10),
            cursor=_cursor_to_bulk_token(
                args_dict.get("cursor"),
                tool=name,
                context_hash=ctx_hash,
                expected_provider="scholarapi",
            ),
            indexed_after=args_dict.get("indexed_after"),
            indexed_before=args_dict.get("indexed_before"),
            published_after=args_dict.get("published_after"),
            published_before=args_dict.get("published_before"),
            has_text=args_dict.get("has_text"),
            has_pdf=args_dict.get("has_pdf"),
        ),
        endpoint="search",
        provider_registry=ctx.provider_registry,
        request_id=f"tool-{name}",
    )
    serialized = _encode_next_bulk_cursor(dump_jsonable(result), name, context_hash=ctx_hash, provider="scholarapi")
    return _finalize(name, arguments, serialized, ctx)


async def _dispatch_list_papers_scholarapi(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from .._core import (
        SCHOLARAPI_LIST_RETRIEVAL_NOTE,
        _call_explicit_scholarapi_tool,
        _cursor_to_bulk_token,
        _encode_next_bulk_cursor,
    )

    name = "list_papers_scholarapi"
    _require_scholarapi(ctx, name)
    args_dict = TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
    ctx_hash = compute_context_hash(name, args_dict)
    result = await _call_explicit_scholarapi_tool(
        operation=lambda: ctx.scholarapi_client.list_papers(
            query=args_dict.get("query"),
            limit=args_dict.get("limit", 100),
            cursor=_cursor_to_bulk_token(
                args_dict.get("cursor"),
                tool=name,
                context_hash=ctx_hash,
                expected_provider="scholarapi",
            ),
            indexed_after=args_dict.get("indexed_after"),
            indexed_before=args_dict.get("indexed_before"),
            published_after=args_dict.get("published_after"),
            published_before=args_dict.get("published_before"),
            has_text=args_dict.get("has_text"),
            has_pdf=args_dict.get("has_pdf"),
        ),
        endpoint="list",
        provider_registry=ctx.provider_registry,
        request_id=f"tool-{name}",
    )
    serialized = dump_jsonable(result)
    serialized.setdefault("retrievalNote", SCHOLARAPI_LIST_RETRIEVAL_NOTE)
    serialized = _encode_next_bulk_cursor(serialized, name, context_hash=ctx_hash, provider="scholarapi")
    return _finalize(name, arguments, serialized, ctx)


async def _dispatch_get_paper_text_scholarapi(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from .._core import _call_explicit_scholarapi_tool

    name = "get_paper_text_scholarapi"
    _require_scholarapi(ctx, name)
    args_dict = TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
    result = await _call_explicit_scholarapi_tool(
        operation=lambda: ctx.scholarapi_client.get_text(paper_id=args_dict["paper_id"]),
        endpoint="text",
        provider_registry=ctx.provider_registry,
        request_id=f"tool-{name}",
    )
    return _finalize(name, arguments, result, ctx)


async def _dispatch_get_paper_texts_scholarapi(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from .._core import _call_explicit_scholarapi_tool

    name = "get_paper_texts_scholarapi"
    _require_scholarapi(ctx, name)
    args_dict = TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
    result = await _call_explicit_scholarapi_tool(
        operation=lambda: ctx.scholarapi_client.get_texts(paper_ids=args_dict["paper_ids"]),
        endpoint="texts",
        provider_registry=ctx.provider_registry,
        request_id=f"tool-{name}",
    )
    return _finalize(name, arguments, result, ctx)


async def _dispatch_get_paper_pdf_scholarapi(ctx: DispatchContext, arguments: dict[str, Any]) -> dict[str, Any]:
    from .._core import _call_explicit_scholarapi_tool

    name = "get_paper_pdf_scholarapi"
    _require_scholarapi(ctx, name)
    args_dict = TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
    result = await _call_explicit_scholarapi_tool(
        operation=lambda: ctx.scholarapi_client.get_pdf(paper_id=args_dict["paper_id"]),
        endpoint="pdf",
        provider_registry=ctx.provider_registry,
        request_id=f"tool-{name}",
    )
    return _finalize(name, arguments, result, ctx)
