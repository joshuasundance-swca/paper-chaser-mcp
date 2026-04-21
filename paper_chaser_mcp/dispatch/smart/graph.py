"""Ctx-first entrypoint for the ``expand_research_graph`` tool (Phase 4)."""

from __future__ import annotations

from typing import Any, cast

from ...models import TOOL_INPUT_MODELS
from ...models.tools import ExpandResearchGraphArgs
from ..context import DispatchContext


async def _dispatch_expand_research_graph(
    ctx: DispatchContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Entrypoint for the ``expand_research_graph`` smart tool.

    Returns ``FEATURE_NOT_CONFIGURED`` when the agentic runtime is missing,
    otherwise forwards the validated arguments to
    ``ctx.agentic_runtime.expand_research_graph``.
    """
    graph_args = cast(
        ExpandResearchGraphArgs,
        TOOL_INPUT_MODELS["expand_research_graph"].model_validate(arguments),
    )
    if ctx.agentic_runtime is None:
        return {
            "error": "FEATURE_NOT_CONFIGURED",
            "message": ("expand_research_graph requires the agentic runtime to be enabled."),
            "fallbackTools": [
                "get_paper_citations",
                "get_paper_references",
                "get_paper_authors",
            ],
        }
    return await ctx.agentic_runtime.expand_research_graph(
        seed_paper_ids=graph_args.seed_paper_ids,
        seed_search_session_id=graph_args.seed_search_session_id,
        direction=graph_args.direction,
        hops=graph_args.hops,
        per_seed_limit=graph_args.per_seed_limit,
        latency_profile=graph_args.latency_profile,
        ctx=ctx.ctx,
    )
