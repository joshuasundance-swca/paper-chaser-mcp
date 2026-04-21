"""Phase 4 TDD tests for ``dispatch/smart/graph.py``."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from paper_chaser_mcp.dispatch.context import DispatchContext
from paper_chaser_mcp.dispatch.smart.graph import _dispatch_expand_research_graph


def _make_ctx(**overrides: Any) -> DispatchContext:
    base: dict[str, Any] = {
        "client": None,
        "core_client": None,
        "openalex_client": None,
        "scholarapi_client": None,
        "arxiv_client": None,
        "enable_core": False,
        "enable_semantic_scholar": False,
        "enable_openalex": False,
        "enable_scholarapi": False,
        "enable_arxiv": False,
    }
    base.update(overrides)
    return DispatchContext(**base)


@pytest.mark.asyncio
async def test_graph_returns_feature_not_configured_when_runtime_missing() -> None:
    ctx = _make_ctx(agentic_runtime=None)
    result = await _dispatch_expand_research_graph(ctx, {"seedPaperIds": ["p1"], "direction": "citations"})
    assert result["error"] == "FEATURE_NOT_CONFIGURED"
    assert "get_paper_citations" in result["fallbackTools"]


@pytest.mark.asyncio
async def test_graph_delegates_to_runtime_when_configured() -> None:
    captured: dict[str, Any] = {}

    @dataclasses.dataclass
    class FakeRuntime:
        async def expand_research_graph(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {"nodes": []}

    ctx = _make_ctx(agentic_runtime=FakeRuntime(), ctx="MCPCTX")
    result = await _dispatch_expand_research_graph(
        ctx,
        {
            "seedPaperIds": ["p1", "p2"],
            "direction": "citations",
            "hops": 1,
        },
    )
    assert result == {"nodes": []}
    assert captured["seed_paper_ids"] == ["p1", "p2"]
    assert captured["direction"] == "citations"
    assert captured["ctx"] == "MCPCTX"
