"""Phase 4 TDD tests for ``dispatch/expert/openalex.py``.

Peer expert submodules carry 93-100% line coverage; this module targets the
gaps the rubber-duck review flagged in ``openalex.py`` — the disabled-provider
branch of :func:`_require_openalex`, the direct route through
:func:`_dispatch_search_papers_openalex`, the inlined disabled-provider guard
in :func:`_dispatch_search_papers_openalex_by_entity`, and the direct route
through :func:`_dispatch_get_paper_citations_openalex` which is not exercised
by the existing server-level routing tests.
"""

from __future__ import annotations

from typing import Any

import pytest

from paper_chaser_mcp.dispatch.context import DispatchContext
from paper_chaser_mcp.dispatch.expert.openalex import (
    _dispatch_get_paper_citations_openalex,
    _dispatch_search_papers_openalex,
    _dispatch_search_papers_openalex_by_entity,
    _require_openalex,
)
from tests.helpers import RecordingOpenAlexClient


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


def test_require_openalex_raises_when_disabled() -> None:
    ctx = _make_ctx(enable_openalex=False)
    with pytest.raises(ValueError, match="OpenAlex, which is disabled"):
        _require_openalex(ctx, "search_papers_openalex")


def test_require_openalex_allows_when_enabled() -> None:
    ctx = _make_ctx(enable_openalex=True, openalex_client=RecordingOpenAlexClient())
    _require_openalex(ctx, "search_papers_openalex")


@pytest.mark.asyncio
async def test_dispatch_search_papers_openalex_raises_when_disabled() -> None:
    ctx = _make_ctx(enable_openalex=False, openalex_client=None)
    with pytest.raises(ValueError, match="OpenAlex, which is disabled"):
        await _dispatch_search_papers_openalex(ctx, {"query": "transformers"})


@pytest.mark.asyncio
async def test_dispatch_search_papers_openalex_delegates() -> None:
    client = RecordingOpenAlexClient()
    ctx = _make_ctx(enable_openalex=True, openalex_client=client)
    result = await _dispatch_search_papers_openalex(
        ctx,
        {"query": "transformers", "limit": 5, "year": "2024"},
    )
    assert isinstance(result, dict)
    assert client.calls[0][0] == "search"
    assert client.calls[0][1] == {"query": "transformers", "limit": 5, "year": "2024"}


@pytest.mark.asyncio
async def test_dispatch_search_papers_openalex_by_entity_raises_when_disabled() -> None:
    ctx = _make_ctx(enable_openalex=False, openalex_client=None)
    with pytest.raises(ValueError, match="OpenAlex, which is disabled"):
        await _dispatch_search_papers_openalex_by_entity(
            ctx,
            {"entityType": "source", "entityId": "S1"},
        )


@pytest.mark.asyncio
async def test_dispatch_get_paper_citations_openalex_delegates() -> None:
    client = RecordingOpenAlexClient()
    ctx = _make_ctx(enable_openalex=True, openalex_client=client)
    result = await _dispatch_get_paper_citations_openalex(ctx, {"paper_id": "W1"})
    assert isinstance(result, dict)
    assert client.calls[0][0] == "get_paper_citations"
    assert client.calls[0][1]["paper_id"] == "W1"
    assert client.calls[0][1]["limit"] == 100
    assert client.calls[0][1]["cursor"] is None
