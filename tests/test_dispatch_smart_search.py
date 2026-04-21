"""Phase 4 TDD tests for ``dispatch/smart/search.py``."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from paper_chaser_mcp.dispatch.context import DispatchContext
from paper_chaser_mcp.dispatch.smart.search import _dispatch_search_papers_smart


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
async def test_returns_feature_not_configured_when_runtime_missing() -> None:
    ctx = _make_ctx(agentic_runtime=None)
    result = await _dispatch_search_papers_smart(ctx, {"query": "x", "limit": 1})
    assert result["error"] == "FEATURE_NOT_CONFIGURED"
    assert "search_papers" in result["fallbackTools"]


@pytest.mark.asyncio
async def test_delegates_to_runtime_when_configured() -> None:
    captured: dict[str, Any] = {}

    @dataclasses.dataclass
    class FakeRuntime:
        async def search_papers_smart(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {"ok": True}

    ctx = _make_ctx(agentic_runtime=FakeRuntime(), ctx="MCPCTX")
    result = await _dispatch_search_papers_smart(ctx, {"query": "bees", "limit": 3})
    assert result == {"ok": True}
    assert captured["query"] == "bees"
    assert captured["limit"] == 3
    assert captured["ctx"] == "MCPCTX"
