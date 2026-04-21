"""Phase 4 TDD tests for ``dispatch/smart/ask.py``."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from paper_chaser_mcp.dispatch.context import DispatchContext
from paper_chaser_mcp.dispatch.smart.ask import _dispatch_ask_result_set


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
async def test_ask_returns_feature_not_configured_when_runtime_missing() -> None:
    ctx = _make_ctx(agentic_runtime=None)
    result = await _dispatch_ask_result_set(ctx, {"searchSessionId": "s1", "question": "why?"})
    assert result["error"] == "FEATURE_NOT_CONFIGURED"
    assert "get_paper_details" in result["fallbackTools"]


@pytest.mark.asyncio
async def test_ask_delegates_to_runtime_when_configured() -> None:
    captured: dict[str, Any] = {}

    @dataclasses.dataclass
    class FakeRuntime:
        async def ask_result_set(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {"answer": "42"}

    ctx = _make_ctx(agentic_runtime=FakeRuntime(), ctx="MCPCTX")
    result = await _dispatch_ask_result_set(ctx, {"searchSessionId": "ssn_123", "question": "why?"})
    assert result == {"answer": "42"}
    assert captured["search_session_id"] == "ssn_123"
    assert captured["question"] == "why?"
    assert captured["ctx"] == "MCPCTX"
