"""Ctx-first entrypoint for the ``search_papers_smart`` tool (Phase 4)."""

from __future__ import annotations

from typing import Any, cast

from ...models import TOOL_INPUT_MODELS
from ...models.tools import SmartSearchPapersArgs
from ..context import DispatchContext


async def _dispatch_search_papers_smart(
    ctx: DispatchContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Entrypoint for the ``search_papers_smart`` tool.

    Returns ``FEATURE_NOT_CONFIGURED`` when the agentic runtime is missing,
    otherwise forwards the validated arguments to
    ``ctx.agentic_runtime.search_papers_smart``.
    """
    smart_args = cast(
        SmartSearchPapersArgs,
        TOOL_INPUT_MODELS["search_papers_smart"].model_validate(arguments),
    )
    if ctx.agentic_runtime is None:
        return {
            "error": "FEATURE_NOT_CONFIGURED",
            "message": ("search_papers_smart is not available because the agentic runtime was not initialized."),
            "fallbackTools": [
                "search_papers",
                "search_papers_bulk",
                "search_papers_match",
            ],
        }
    return await ctx.agentic_runtime.search_papers_smart(
        query=smart_args.query,
        limit=smart_args.limit,
        search_session_id=smart_args.search_session_id,
        mode=smart_args.mode,
        year=smart_args.year,
        venue=smart_args.venue,
        focus=smart_args.focus,
        latency_profile=smart_args.latency_profile,
        provider_budget=(
            smart_args.provider_budget.model_dump(by_alias=False) if smart_args.provider_budget is not None else None
        ),
        include_enrichment=smart_args.include_enrichment,
        ctx=ctx.ctx,
    )
