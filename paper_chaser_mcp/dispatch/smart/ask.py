"""Ctx-first entrypoint for the ``ask_result_set`` tool (Phase 4)."""

from __future__ import annotations

from typing import Any, cast

from ...models import TOOL_INPUT_MODELS
from ...models.tools import AskResultSetArgs
from ..context import DispatchContext


async def _dispatch_ask_result_set(
    ctx: DispatchContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Entrypoint for the ``ask_result_set`` tool.

    Returns ``FEATURE_NOT_CONFIGURED`` when the agentic runtime is missing,
    otherwise forwards the validated arguments to
    ``ctx.agentic_runtime.ask_result_set``.
    """
    ask_args = cast(
        AskResultSetArgs,
        TOOL_INPUT_MODELS["ask_result_set"].model_validate(arguments),
    )
    if ctx.agentic_runtime is None:
        return {
            "error": "FEATURE_NOT_CONFIGURED",
            "message": "ask_result_set requires the agentic runtime to be enabled.",
            "fallbackTools": [
                "search_papers",
                "get_paper_details",
                "get_paper_citations",
            ],
        }
    return await ctx.agentic_runtime.ask_result_set(
        search_session_id=ask_args.search_session_id,
        question=ask_args.question,
        top_k=ask_args.top_k,
        answer_mode=ask_args.answer_mode,
        latency_profile=ask_args.latency_profile,
        ctx=ctx.ctx,
    )
