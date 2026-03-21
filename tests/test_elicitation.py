from __future__ import annotations

from types import SimpleNamespace

import pytest

from scholar_search_mcp.agentic import WorkspaceRegistry
from scholar_search_mcp.dispatch import dispatch_tool
from tests.helpers import RecordingOpenAlexClient, RecordingSemanticClient


class FakeElicitationContext:
    def __init__(self, *, action: str, data: str | None = None) -> None:
        self.action = action
        self.data = data
        self.elicit_calls: list[tuple[str, object]] = []

    def client_supports_extension(self, extension_id: str) -> bool:
        return extension_id == "elicitation"

    async def elicit(self, message: str, response_type: object = None) -> object:
        self.elicit_calls.append((message, response_type))
        if self.action == "accept":
            return SimpleNamespace(action="accept", data=self.data)
        return SimpleNamespace(action=self.action)


def _workspace_registry() -> WorkspaceRegistry:
    return WorkspaceRegistry(ttl_seconds=1800, enable_trace_log=False)


@pytest.mark.asyncio
async def test_search_papers_elicitation_refines_low_specificity_query() -> None:
    semantic = RecordingSemanticClient()
    ctx = FakeElicitationContext(action="accept", data="method focus")

    result = await dispatch_tool(
        "search_papers",
        {"query": "AI", "limit": 5},
        client=semantic,
        core_client=object(),
        openalex_client=RecordingOpenAlexClient(),
        arxiv_client=object(),
        enable_core=False,
        enable_semantic_scholar=True,
        enable_openalex=True,
        enable_arxiv=False,
        serpapi_client=None,
        enable_serpapi=False,
        provider_order=None,
        workspace_registry=_workspace_registry(),
        agentic_runtime=None,
        ctx=ctx,
    )

    semantic_search_calls = [
        kwargs for name, kwargs in semantic.calls if name == "search_papers"
    ]
    assert len(semantic_search_calls) == 2
    assert semantic_search_calls[-1]["query"] == "AI method focus"
    assert ctx.elicit_calls
    assert result["searchSessionId"]
    assert "clarification" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["decline", "cancel"])
async def test_search_authors_decline_or_cancel_falls_back_to_clarification(
    action: str,
) -> None:
    semantic = RecordingSemanticClient()
    ctx = FakeElicitationContext(action=action)

    async def ambiguous_search_authors(**kwargs):
        semantic.calls.append(("search_authors", kwargs))
        return {
            "total": 2,
            "offset": 0,
            "data": [
                {"authorId": "a-1", "name": "John Smith"},
                {"authorId": "a-2", "name": "John Smith"},
            ],
        }

    semantic.search_authors = ambiguous_search_authors  # type: ignore[method-assign]

    result = await dispatch_tool(
        "search_authors",
        {"query": "John Smith", "limit": 5},
        client=semantic,
        core_client=object(),
        openalex_client=RecordingOpenAlexClient(),
        arxiv_client=object(),
        enable_core=False,
        enable_semantic_scholar=True,
        enable_openalex=True,
        enable_arxiv=False,
        serpapi_client=None,
        enable_serpapi=False,
        provider_order=None,
        workspace_registry=_workspace_registry(),
        agentic_runtime=None,
        ctx=ctx,
    )

    author_search_calls = [
        kwargs for name, kwargs in semantic.calls if name == "search_authors"
    ]
    assert len(author_search_calls) == 1
    assert ctx.elicit_calls
    assert result["clarification"]["reason"] == "ambiguous_author_identity"
    assert result["agentHints"]["nextToolCandidates"] == [
        "get_author_info",
        "get_author_papers",
    ]


@pytest.mark.asyncio
async def test_search_papers_match_can_resolve_identifier_from_elicitation() -> None:
    semantic = RecordingSemanticClient()
    ctx = FakeElicitationContext(action="accept", data="arXiv:1706.03762")

    async def no_match(**kwargs):
        semantic.calls.append(("search_papers_match", kwargs))
        return {
            "matchFound": False,
            "matchStrategy": "no_match",
            "message": "No close title match was found.",
        }

    semantic.search_papers_match = no_match  # type: ignore[method-assign]

    result = await dispatch_tool(
        "search_papers_match",
        {"query": "attention is all you need"},
        client=semantic,
        core_client=object(),
        openalex_client=RecordingOpenAlexClient(),
        arxiv_client=object(),
        enable_core=False,
        enable_semantic_scholar=True,
        enable_openalex=True,
        enable_arxiv=False,
        serpapi_client=None,
        enable_serpapi=False,
        provider_order=None,
        workspace_registry=_workspace_registry(),
        agentic_runtime=None,
        ctx=ctx,
    )

    assert ctx.elicit_calls
    assert result["paperId"] == "arXiv:1706.03762"
    assert result["matchFound"] is True
    assert result["matchStrategy"] == "elicited_identifier"
    assert result["agentHints"]["nextToolCandidates"] == [
        "get_paper_citations",
        "get_paper_references",
        "get_paper_authors",
        "expand_research_graph",
    ]
