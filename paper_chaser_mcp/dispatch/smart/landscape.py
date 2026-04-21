"""Ctx-first entrypoint for the ``map_research_landscape`` tool (Phase 4)."""

from __future__ import annotations

from typing import Any, cast

from ...models import TOOL_INPUT_MODELS
from ...models.tools import MapResearchLandscapeArgs
from ..context import DispatchContext


async def _dispatch_map_research_landscape(
    ctx: DispatchContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Entrypoint for the ``map_research_landscape`` smart tool.

    Returns ``FEATURE_NOT_CONFIGURED`` when the agentic runtime is missing,
    otherwise forwards the validated arguments to
    ``ctx.agentic_runtime.map_research_landscape``.
    """
    landscape_args = cast(
        MapResearchLandscapeArgs,
        TOOL_INPUT_MODELS["map_research_landscape"].model_validate(arguments),
    )
    if ctx.agentic_runtime is None:
        return {
            "error": "FEATURE_NOT_CONFIGURED",
            "message": ("map_research_landscape requires the agentic runtime to be enabled."),
            "fallbackTools": [
                "search_papers",
                "search_papers_bulk",
                "get_paper_citations",
            ],
        }
    return await ctx.agentic_runtime.map_research_landscape(
        search_session_id=landscape_args.search_session_id,
        max_themes=landscape_args.max_themes,
        latency_profile=landscape_args.latency_profile,
        ctx=ctx.ctx,
    )
