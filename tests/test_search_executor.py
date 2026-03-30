from __future__ import annotations

import pytest

from paper_chaser_mcp.provider_runtime import ProviderDiagnosticsRegistry
from tests.helpers import RecordingOpenAlexClient, RecordingScholarApiClient, RecordingSemanticClient


def test_search_executor_reports_missing_serpapi_client_as_skipped() -> None:
    from paper_chaser_mcp.search_executor import SearchClientBundle, SearchExecutor

    executor = SearchExecutor()
    attempt = executor.disabled_attempt(
        "serpapi_google_scholar",
        enabled={
            "core": False,
            "semantic_scholar": False,
            "openalex": False,
            "arxiv": False,
            "serpapi_google_scholar": True,
        },
        clients=SearchClientBundle(),
    )

    assert attempt is not None
    assert attempt.provider == "serpapi_google_scholar"
    assert attempt.status == "skipped"
    assert "no SerpApi client" in str(attempt.reason)


def test_search_executor_reports_missing_scholarapi_client_as_skipped() -> None:
    from paper_chaser_mcp.search_executor import SearchClientBundle, SearchExecutor

    executor = SearchExecutor()
    attempt = executor.disabled_attempt(
        "scholarapi",
        enabled={
            "core": False,
            "semantic_scholar": False,
            "openalex": False,
            "arxiv": False,
            "serpapi_google_scholar": False,
            "scholarapi": True,
        },
        clients=SearchClientBundle(),
    )

    assert attempt is not None
    assert attempt.provider == "scholarapi"
    assert attempt.status == "skipped"
    assert "no ScholarAPI client" in str(attempt.reason)


@pytest.mark.asyncio
async def test_search_executor_parallel_search_shapes_semantic_and_openalex_results() -> None:
    from paper_chaser_mcp.search_executor import (
        ProviderSearchRequest,
        SearchClientBundle,
        SearchExecutor,
    )

    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    outcomes: list[dict[str, object]] = []
    executor = SearchExecutor()

    results = await executor.execute_parallel_requests(
        provider_requests=[
            (
                "semantic_scholar",
                ProviderSearchRequest(
                    query="transformers for ecology",
                    limit=2,
                    fields=["paperId", "title"],
                    year="2024",
                ),
            ),
            (
                "openalex",
                ProviderSearchRequest(
                    query="transformers for ecology",
                    limit=3,
                    year="2024",
                ),
            ),
        ],
        clients=SearchClientBundle(
            semantic_client=semantic,
            openalex_client=openalex,
        ),
        provider_registry=ProviderDiagnosticsRegistry(),
        request_outcomes=outcomes,
        request_id="search-executor-test",
    )

    assert [result.provider for result in results] == ["semantic_scholar", "openalex"]
    assert results[0].response is not None
    assert results[0].response.data[0].paper_id == "semantic-1"
    assert results[1].response is not None
    assert results[1].response.data[0].paper_id == "W1"
    assert len(outcomes) == 2
    assert {entry["provider"] for entry in outcomes} == {
        "semantic_scholar",
        "openalex",
    }


@pytest.mark.asyncio
async def test_search_executor_parallel_search_shapes_scholarapi_results() -> None:
    from paper_chaser_mcp.search_executor import ProviderSearchRequest, SearchClientBundle, SearchExecutor

    scholarapi = RecordingScholarApiClient()
    outcomes: list[dict[str, object]] = []
    executor = SearchExecutor()

    results = await executor.execute_parallel_requests(
        provider_requests=[
            (
                "scholarapi",
                ProviderSearchRequest(
                    query="graphene full text",
                    limit=2,
                    year="2024",
                ),
            ),
        ],
        clients=SearchClientBundle(
            scholarapi_client=scholarapi,
        ),
        provider_registry=ProviderDiagnosticsRegistry(),
        request_outcomes=outcomes,
        request_id="search-executor-scholarapi-test",
    )

    assert [result.provider for result in results] == ["scholarapi"]
    assert results[0].response is not None
    assert results[0].response.data[0].paper_id == "ScholarAPI:sa-1"
    assert len(outcomes) == 1
    assert outcomes[0]["provider"] == "scholarapi"
