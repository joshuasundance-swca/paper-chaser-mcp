import json
from typing import Any

import pytest

from scholar_search_mcp import server
from tests.helpers import RecordingSemanticClient, _payload


@pytest.mark.asyncio
async def test_call_tool_search_papers_returns_empty_when_all_channels_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "enable_core", False)
    monkeypatch.setattr(server, "enable_semantic_scholar", False)
    monkeypatch.setattr(server, "enable_arxiv", False)

    response = await server.call_tool("search_papers", {"query": "transformers"})

    assert len(response) == 1
    assert '"total": 0' in response[0].text
    assert '"data": []' in response[0].text


@pytest.mark.asyncio
async def test_search_papers_falls_back_from_core_to_semantic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingCoreClient:
        async def search(self, **kwargs) -> dict:
            raise RuntimeError("core unavailable")

    class SemanticClient:
        async def search_papers(self, **kwargs) -> dict:
            return {
                "total": 1,
                "offset": 0,
                "data": [{"paperId": "s2-1", "title": "Fallback paper"}],
            }

    monkeypatch.setattr(server, "enable_core", True)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_arxiv", True)
    monkeypatch.setattr(server, "core_client", FailingCoreClient())
    monkeypatch.setattr(server, "client", SemanticClient())

    response = await server.call_tool("search_papers", {"query": "fallback"})
    payload = json.loads(response[0].text)

    assert payload["total"] == 1
    assert payload["data"][0]["paperId"] == "s2-1"
    assert payload["data"][0]["source"] == "semantic_scholar"


@pytest.mark.asyncio
async def test_search_papers_falls_back_to_arxiv_when_other_sources_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingCoreClient:
        async def search(self, **kwargs) -> dict:
            raise RuntimeError("core unavailable")

    class FailingSemanticClient:
        async def search_papers(self, **kwargs) -> dict:
            raise RuntimeError("semantic unavailable")

    class ArxivClient:
        async def search(self, **kwargs) -> dict:
            return {
                "totalResults": 1,
                "entries": [
                    {
                        "paperId": "arxiv-1",
                        "title": "arXiv fallback paper",
                        "url": "https://arxiv.org/abs/arxiv-1",
                        "source": "arxiv",
                    }
                ],
            }

    monkeypatch.setattr(server, "enable_core", True)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_arxiv", True)
    monkeypatch.setattr(server, "core_client", FailingCoreClient())
    monkeypatch.setattr(server, "client", FailingSemanticClient())
    monkeypatch.setattr(server, "arxiv_client", ArxivClient())

    response = await server.call_tool("search_papers", {"query": "fallback"})
    payload = json.loads(response[0].text)

    assert payload["total"] == 1
    assert payload["data"][0]["paperId"] == "arxiv-1"
    assert payload["data"][0]["source"] == "arxiv"


@pytest.mark.asyncio
async def test_search_papers_exposes_new_filter_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = RecordingSemanticClient()
    monkeypatch.setattr(server, "enable_core", False)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_arxiv", False)
    monkeypatch.setattr(server, "client", fake_client)

    await server.call_tool(
        "search_papers",
        {
            "query": "ml",
            "publicationDateOrYear": "2020:2023",
            "fieldsOfStudy": "Computer Science",
            "minCitationCount": 5,
        },
    )

    assert len(fake_client.calls) == 1
    method, kwargs = fake_client.calls[0]
    assert method == "search_papers"
    assert kwargs["publication_date_or_year"] == "2020:2023"
    assert kwargs["fields_of_study"] == "Computer Science"
    assert kwargs["min_citation_count"] == 5


@pytest.mark.asyncio
async def test_search_papers_preferred_provider_runs_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SpyCoreClient:
        def __init__(self) -> None:
            self.called = False

        async def search(self, **kwargs) -> dict:
            self.called = True
            return {
                "total": 1,
                "entries": [
                    {
                        "paperId": "core-1",
                        "title": "CORE paper",
                        "url": "https://example.com/core-1",
                    }
                ],
            }

    class SemanticClient:
        async def search_papers(self, **kwargs) -> dict:
            return {
                "total": 1,
                "offset": 0,
                "data": [{"paperId": "s2-1", "title": "Semantic first"}],
            }

    spy_core = SpyCoreClient()
    monkeypatch.setattr(server, "enable_core", True)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_arxiv", True)
    monkeypatch.setattr(server, "core_client", spy_core)
    monkeypatch.setattr(server, "client", SemanticClient())

    payload = _payload(
        await server.call_tool(
            "search_papers",
            {
                "query": "fallback",
                "preferredProvider": "semantic_scholar",
            },
        )
    )

    assert payload["data"][0]["paperId"] == "s2-1"
    assert payload["brokerMetadata"]["providerUsed"] == "semantic_scholar"
    assert payload["brokerMetadata"]["attemptedProviders"][0] == {
        "provider": "semantic_scholar",
        "status": "returned_results",
        "reason": None,
    }
    assert payload["brokerMetadata"]["attemptedProviders"][1]["provider"] == "core"
    assert payload["brokerMetadata"]["attemptedProviders"][1]["status"] == "skipped"
    assert spy_core.called is False


@pytest.mark.asyncio
async def test_search_papers_provider_order_can_override_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SpyCoreClient:
        def __init__(self) -> None:
            self.called = False

        async def search(self, **kwargs) -> dict:
            self.called = True
            return {
                "total": 1,
                "entries": [
                    {
                        "paperId": "core-1",
                        "title": "CORE paper",
                        "url": "https://example.com/core-1",
                    }
                ],
            }

    class ArxivClient:
        async def search(self, **kwargs) -> dict:
            return {
                "totalResults": 1,
                "entries": [
                    {
                        "paperId": "arxiv-1",
                        "title": "arXiv only",
                        "url": "https://arxiv.org/abs/arxiv-1",
                        "source": "arxiv",
                    }
                ],
            }

    spy_core = SpyCoreClient()
    monkeypatch.setattr(server, "enable_core", True)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_arxiv", True)
    monkeypatch.setattr(server, "core_client", spy_core)
    monkeypatch.setattr(server, "arxiv_client", ArxivClient())

    payload = _payload(
        await server.call_tool(
            "search_papers",
            {"query": "fallback", "providerOrder": ["arxiv"]},
        )
    )

    assert payload["data"][0]["paperId"] == "arxiv-1"
    assert payload["brokerMetadata"]["providerUsed"] == "arxiv"
    assert payload["brokerMetadata"]["attemptedProviders"] == [
        {"provider": "arxiv", "status": "returned_results", "reason": None}
    ]
    assert spy_core.called is False


@pytest.mark.asyncio
async def test_provider_specific_search_tool_does_not_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CoreClient:
        async def search(self, **kwargs) -> dict:
            return {"total": 0, "entries": []}

    class SemanticClient:
        def __init__(self) -> None:
            self.called = False

        async def search_papers(self, **kwargs) -> dict:
            self.called = True
            return {
                "total": 1,
                "offset": 0,
                "data": [{"paperId": "s2-1"}],
            }

    semantic_client = SemanticClient()
    monkeypatch.setattr(server, "enable_core", True)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "core_client", CoreClient())
    monkeypatch.setattr(server, "client", semantic_client)

    payload = _payload(await server.call_tool("search_papers_core", {"query": "test"}))

    assert payload["total"] == 0
    assert payload["brokerMetadata"]["providerUsed"] == "none"
    assert payload["brokerMetadata"]["attemptedProviders"] == [
        {"provider": "core", "status": "returned_no_results", "reason": None}
    ]
    assert semantic_client.called is False


@pytest.mark.asyncio
async def test_search_papers_invalid_provider_name_has_clear_error() -> None:
    with pytest.raises(Exception, match="Unsupported provider 'bogus'"):
        await server.call_tool(
            "search_papers",
            {"query": "fallback", "preferredProvider": "bogus"},
        )


@pytest.mark.asyncio
async def test_search_papers_skips_core_when_ss_only_filter_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CORE must be bypassed whenever a Semantic Scholar-only filter is used.

    Regression guard for the issue where CORE would be called first and return
    page-1 un-filtered results despite the caller requesting a Semantic Scholar-
    specific filter.
    """

    class SpyCoreClient:
        def __init__(self) -> None:
            self.called = False

        async def search(self, **kwargs) -> dict:
            self.called = True
            return {
                "total": 1,
                "entries": [
                    {
                        "paperId": "core-1",
                        "title": "Core result",
                        "url": "https://example.com",
                    }
                ],
            }

    class SuccessSemanticClient:
        async def search_papers(self, **kwargs) -> dict:
            return {
                "total": 1,
                "offset": 0,
                "data": [{"paperId": "s2-result"}],
            }

    spy_core = SpyCoreClient()

    monkeypatch.setattr(server, "enable_core", True)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_arxiv", False)
    monkeypatch.setattr(server, "core_client", spy_core)
    monkeypatch.setattr(server, "client", SuccessSemanticClient())

    # publicationDateOrYear is an SS-only filter; CORE must be skipped.
    response = await server.call_tool(
        "search_papers",
        {"query": "neural nets", "publicationDateOrYear": "2020:2024"},
    )
    payload = json.loads(response[0].text)

    assert not spy_core.called, "CORE should have been skipped for SS-only filter"
    assert payload["data"][0]["paperId"] == "s2-result"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra_filter",
    [
        {"publicationDateOrYear": "2020:2023"},
        {"fieldsOfStudy": "Computer Science"},
        {"publicationTypes": "JournalArticle"},
        {"openAccessPdf": True},
        {"minCitationCount": 10},
    ],
)
async def test_search_papers_ss_filters_always_bypass_core(
    monkeypatch: pytest.MonkeyPatch,
    extra_filter: dict,
) -> None:
    """Each individual SS-only filter independently causes CORE to be bypassed."""

    class SpyCoreClient:
        def __init__(self) -> None:
            self.called = False

        async def search(self, **kwargs) -> dict:
            self.called = True
            return {"total": 0, "entries": []}

    class SemanticClient:
        async def search_papers(self, **kwargs) -> dict:
            return {"total": 0, "offset": 0, "data": []}

    spy_core = SpyCoreClient()
    monkeypatch.setattr(server, "enable_core", True)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_arxiv", False)
    monkeypatch.setattr(server, "core_client", spy_core)
    monkeypatch.setattr(server, "client", SemanticClient())

    await server.call_tool("search_papers", {"query": "test", **extra_filter})

    assert not spy_core.called, (
        f"CORE should be skipped when filter {extra_filter} is set"
    )


@pytest.mark.asyncio
async def test_search_papers_response_has_no_pagination_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """search_papers is non-paginated; response must not contain a pagination key."""

    class CoreClient:
        async def search(self, **kwargs) -> dict:
            return {
                "total": 2,
                "entries": [
                    {"paperId": "c1", "title": "Paper", "url": "https://x.com/1"},
                ],
            }

    monkeypatch.setattr(server, "enable_core", True)
    monkeypatch.setattr(server, "enable_semantic_scholar", False)
    monkeypatch.setattr(server, "enable_arxiv", False)
    monkeypatch.setattr(server, "core_client", CoreClient())

    response = await server.call_tool("search_papers", {"query": "test"})
    payload = json.loads(response[0].text)

    assert "pagination" not in payload
    assert set(payload.keys()) == {"total", "offset", "data", "brokerMetadata"}
    broker_meta = payload["brokerMetadata"]
    assert broker_meta["mode"] == "brokered_single_page"
    assert broker_meta["providerUsed"] == "core"
    assert broker_meta["continuationSupported"] is False


@pytest.mark.asyncio
async def test_search_papers_broker_metadata_present_for_semantic_scholar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """search_papers response includes brokerMetadata when Semantic Scholar is used."""

    class SemanticClient:
        async def search_papers(self, **kwargs) -> dict:
            return {
                "total": 1,
                "offset": 0,
                "data": [{"paperId": "s2-1", "title": "Test"}],
            }

    monkeypatch.setattr(server, "enable_core", False)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_arxiv", False)
    monkeypatch.setattr(server, "client", SemanticClient())

    response = await server.call_tool("search_papers", {"query": "test"})
    payload = json.loads(response[0].text)

    assert "brokerMetadata" in payload
    broker_meta = payload["brokerMetadata"]
    assert broker_meta["mode"] == "brokered_single_page"
    assert broker_meta["providerUsed"] == "semantic_scholar"
    assert broker_meta["continuationSupported"] is False


@pytest.mark.asyncio
async def test_search_papers_broker_metadata_present_for_arxiv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """search_papers response includes brokerMetadata when arXiv is used."""

    class ArxivClient:
        async def search(self, **kwargs) -> dict:
            return {
                "totalResults": 1,
                "entries": [
                    {
                        "paperId": "arxiv-1",
                        "title": "arXiv paper",
                        "url": "https://arxiv.org/abs/arxiv-1",
                        "source": "arxiv",
                    }
                ],
            }

    monkeypatch.setattr(server, "enable_core", False)
    monkeypatch.setattr(server, "enable_semantic_scholar", False)
    monkeypatch.setattr(server, "enable_arxiv", True)
    monkeypatch.setattr(server, "arxiv_client", ArxivClient())

    response = await server.call_tool("search_papers", {"query": "test"})
    payload = json.loads(response[0].text)

    assert "brokerMetadata" in payload
    broker_meta = payload["brokerMetadata"]
    assert broker_meta["mode"] == "brokered_single_page"
    assert broker_meta["providerUsed"] == "arxiv"
    assert broker_meta["continuationSupported"] is False


@pytest.mark.asyncio
async def test_search_papers_broker_metadata_present_when_no_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """search_papers response includes brokerMetadata even when all channels
    disabled."""

    monkeypatch.setattr(server, "enable_core", False)
    monkeypatch.setattr(server, "enable_semantic_scholar", False)
    monkeypatch.setattr(server, "enable_arxiv", False)

    response = await server.call_tool("search_papers", {"query": "test"})
    payload = json.loads(response[0].text)

    assert "brokerMetadata" in payload
    broker_meta = payload["brokerMetadata"]
    assert broker_meta["mode"] == "brokered_single_page"
    assert broker_meta["continuationSupported"] is False


@pytest.mark.asyncio
async def test_search_papers_broker_metadata_continuation_always_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """continuationSupported must always be False in search_papers brokerMetadata."""

    class CoreClient:
        async def search(self, **kwargs) -> dict:
            return {
                "total": 1,
                "entries": [{"paperId": "c1", "title": "Core paper"}],
            }

    monkeypatch.setattr(server, "enable_core", True)
    monkeypatch.setattr(server, "enable_semantic_scholar", False)
    monkeypatch.setattr(server, "enable_arxiv", False)
    monkeypatch.setattr(server, "core_client", CoreClient())

    response = await server.call_tool("search_papers", {"query": "test"})
    payload = json.loads(response[0].text)

    assert payload["brokerMetadata"]["continuationSupported"] is False


def test_ss_paper_has_provenance_fields_with_doi() -> None:
    """Semantic Scholar papers with a DOI must prefer the DOI as canonicalId."""
    from scholar_search_mcp.models import Paper
    from scholar_search_mcp.search import _enrich_ss_paper

    paper = Paper.model_validate(
        {
            "paperId": "649def34f8be52c8b66281af98ae884c09aef38b",
            "title": "Attention Is All You Need",
            "externalIds": {
                "DOI": "10.5555/3295222.3295349",
                "ArXiv": "1706.03762",
            },
        }
    )
    enriched = _enrich_ss_paper(paper)

    assert enriched.source == "semantic_scholar"
    assert enriched.source_id == "649def34f8be52c8b66281af98ae884c09aef38b"
    assert enriched.canonical_id == "10.5555/3295222.3295349"
    assert enriched.recommended_expansion_id == "10.5555/3295222.3295349"
    assert enriched.expansion_id_status == "portable"


def test_ss_paper_canonical_id_falls_back_to_paper_id_without_doi() -> None:
    """SS papers without a DOI must fall back to paperId as canonicalId."""
    from scholar_search_mcp.models import Paper
    from scholar_search_mcp.search import _enrich_ss_paper

    paper = Paper.model_validate(
        {
            "paperId": "aabbcc1234",
            "title": "No DOI Paper",
            "externalIds": {"ArXiv": "2001.00001"},
        }
    )
    enriched = _enrich_ss_paper(paper)

    assert enriched.source == "semantic_scholar"
    assert enriched.source_id == "aabbcc1234"
    assert enriched.canonical_id == "aabbcc1234"
    assert enriched.recommended_expansion_id == "aabbcc1234"
    assert enriched.expansion_id_status == "portable"


def test_ss_paper_canonical_id_uses_arxiv_id_when_no_doi_or_paper_id() -> None:
    """When paperId is absent, arXiv ID is used as canonicalId."""
    from scholar_search_mcp.models import Paper
    from scholar_search_mcp.search import _enrich_ss_paper

    paper = Paper.model_validate(
        {
            "title": "ArXiv Only Paper",
            "externalIds": {"ArXiv": "2111.99999"},
        }
    )
    enriched = _enrich_ss_paper(paper)

    assert enriched.source == "semantic_scholar"
    assert enriched.source_id is None
    assert enriched.canonical_id == "2111.99999"
    assert enriched.recommended_expansion_id == "2111.99999"
    assert enriched.expansion_id_status == "portable"


@pytest.mark.asyncio
async def test_search_papers_ss_results_include_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """search_papers results from Semantic Scholar must include provenance fields."""

    class ProvenanceFakeClient:
        async def search_papers(self, **kwargs: Any) -> dict:
            return {
                "total": 1,
                "offset": 0,
                "data": [
                    {
                        "paperId": "ss-paper-id",
                        "title": "SS Provenance Test",
                        "externalIds": {"DOI": "10.9999/test"},
                    }
                ],
            }

    monkeypatch.setattr(server, "enable_core", False)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_arxiv", False)
    monkeypatch.setattr(server, "client", ProvenanceFakeClient())

    response = await server.call_tool("search_papers", {"query": "provenance test"})
    payload = json.loads(response[0].text)
    paper = payload["data"][0]

    assert paper["source"] == "semantic_scholar"
    assert paper["sourceId"] == "ss-paper-id"
    assert paper["canonicalId"] == "10.9999/test"
    assert paper["recommendedExpansionId"] == "10.9999/test"
    assert paper["expansionIdStatus"] == "portable"


@pytest.mark.asyncio
async def test_search_papers_core_result_without_doi_marks_expansion_id_not_portable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CoreClient:
        async def search(self, **kwargs: Any) -> dict:
            return {
                "total": 1,
                "entries": [
                    {
                        "paperId": "core-1",
                        "title": "CORE no DOI",
                        "source": "core",
                        "sourceId": "core-1",
                        "canonicalId": "core-1",
                        "expansionIdStatus": "not_portable",
                    }
                ],
            }

    monkeypatch.setattr(server, "enable_core", True)
    monkeypatch.setattr(server, "enable_semantic_scholar", False)
    monkeypatch.setattr(server, "enable_arxiv", False)
    monkeypatch.setattr(server, "enable_serpapi", False)
    monkeypatch.setattr(server, "core_client", CoreClient())

    response = await server.call_tool("search_papers", {"query": "core no doi"})
    payload = json.loads(response[0].text)
    paper = payload["data"][0]

    assert paper["canonicalId"] == "core-1"
    assert paper["expansionIdStatus"] == "not_portable"
    assert paper.get("recommendedExpansionId") is None
    assert "recommendedExpansionId" not in paper


@pytest.mark.asyncio
async def test_search_papers_broker_metadata_reports_attempts_and_filter_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic = RecordingSemanticClient()

    class EmptyCoreClient:
        async def search(self, **kwargs) -> dict[str, Any]:
            return {"total": 0, "entries": []}

    class EmptyArxivClient:
        async def search(self, **kwargs) -> dict[str, Any]:
            return {"totalResults": 0, "entries": []}

    monkeypatch.setattr(server, "core_client", EmptyCoreClient())
    monkeypatch.setattr(server, "client", semantic)
    monkeypatch.setattr(server, "arxiv_client", EmptyArxivClient())
    monkeypatch.setattr(server, "enable_core", True)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_arxiv", True)
    monkeypatch.setattr(server, "enable_serpapi", False)

    response = await server.call_tool(
        "search_papers",
        {
            "query": "transformers",
            "publicationDateOrYear": "2020:2024",
        },
    )
    payload = _payload(response)
    broker_meta = payload["brokerMetadata"]

    assert broker_meta["providerUsed"] == "semantic_scholar"
    assert broker_meta["semanticScholarOnlyFilters"] == ["publicationDateOrYear"]
    assert broker_meta["recommendedPaginationTool"] == "search_papers_bulk"
    assert broker_meta["attemptedProviders"][0]["provider"] == "core"
    assert broker_meta["attemptedProviders"][0]["status"] == "skipped"
    assert (
        "Semantic Scholar-only filters"
        in broker_meta["attemptedProviders"][0]["reason"]
    )
    assert broker_meta["attemptedProviders"][1]["provider"] == "semantic_scholar"
    assert broker_meta["attemptedProviders"][1]["status"] == "returned_results"


# ---------------------------------------------------------------------------
# resultQuality and bulkSearchIsProviderPivot metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_broker_metadata_result_quality_strong_for_semantic_scholar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resultQuality must be 'strong' when Semantic Scholar supplies relevant results.

    The mock paper title deliberately contains the query token so that the
    relevance check does not downgrade the quality to 'low_relevance'.
    """

    class SemanticClient:
        async def search_papers(self, **kwargs: Any) -> dict:
            return {
                "total": 1,
                "offset": 0,
                "data": [{"paperId": "ss-1", "title": "Transformers in deep learning"}],
            }

    monkeypatch.setattr(server, "enable_core", False)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_arxiv", False)
    monkeypatch.setattr(server, "client", SemanticClient())

    payload = _payload(
        await server.call_tool("search_papers", {"query": "transformers"})
    )
    broker_meta = payload["brokerMetadata"]

    assert broker_meta["resultQuality"] == "strong"
    assert broker_meta["bulkSearchIsProviderPivot"] is False


@pytest.mark.asyncio
async def test_broker_metadata_result_quality_low_relevance_for_nonsense_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resultQuality must be 'low_relevance' when Semantic Scholar returns results
    that don't contain distinctive query tokens.

    This is the primary regression test for the issue where a nonsense query such
    as 'asdkfjhasdkjfh research paper nonsense' returned results marked as 'strong'
    even though the gibberish token did not appear in any result title or abstract.
    """

    class SemanticClient:
        async def search_papers(self, **kwargs: Any) -> dict:
            # Simulate Semantic Scholar matching only the generic words in the
            # query and ignoring the distinctive gibberish token entirely.
            return {
                "total": 2,
                "offset": 0,
                "data": [
                    {
                        "paperId": "ss-1",
                        "title": "Relationship Between Science and Nonsense",
                        "abstract": "A study of nonsense.",
                    },
                    {
                        "paperId": "ss-2",
                        "title": "New Approaches to the Circle of Nonsense",
                        "abstract": None,
                    },
                ],
            }

    monkeypatch.setattr(server, "enable_core", False)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_arxiv", False)
    monkeypatch.setattr(server, "client", SemanticClient())

    payload = _payload(
        await server.call_tool(
            "search_papers",
            {"query": "asdkfjhasdkjfh research paper nonsense"},
        )
    )
    broker_meta = payload["brokerMetadata"]

    # Provider is still Semantic Scholar — just quality is downgraded.
    assert broker_meta["providerUsed"] == "semantic_scholar"
    assert broker_meta["resultQuality"] == "low_relevance"
    assert broker_meta["bulkSearchIsProviderPivot"] is False
    # The hint must clearly warn that results are weak and not to trust them.
    hint = broker_meta["nextStepHint"]
    assert "low_relevance" in hint
    assert "weak" in hint.lower() or "irrelevant" in hint.lower()


@pytest.mark.asyncio
async def test_broker_metadata_result_quality_low_relevance_with_both_nonsense_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two gibberish tokens in the query both absent from results → low_relevance."""

    class SemanticClient:
        async def search_papers(self, **kwargs: Any) -> dict:
            return {
                "total": 1,
                "offset": 0,
                "data": [
                    {
                        "paperId": "ss-1",
                        "title": "A paper about nothing",
                        "abstract": "Some general content.",
                    }
                ],
            }

    monkeypatch.setattr(server, "enable_core", False)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_arxiv", False)
    monkeypatch.setattr(server, "client", SemanticClient())

    payload = _payload(
        await server.call_tool(
            "search_papers",
            {"query": "asdkfjhasdkjfh qzxqzxqzx research paper"},
        )
    )
    broker_meta = payload["brokerMetadata"]

    assert broker_meta["providerUsed"] == "semantic_scholar"
    assert broker_meta["resultQuality"] == "low_relevance"


@pytest.mark.asyncio
async def test_broker_metadata_result_quality_low_relevance_uses_abstract_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a distinctive token appears in the abstract (not title) the result
    is still considered relevant and quality stays 'strong'."""

    class SemanticClient:
        async def search_papers(self, **kwargs: Any) -> dict:
            return {
                "total": 1,
                "offset": 0,
                "data": [
                    {
                        "paperId": "ss-1",
                        "title": "Advances in neural network training",
                        # The distinctive token 'transformers' appears in abstract
                        "abstract": "We study transformers and related architectures.",
                    }
                ],
            }

    monkeypatch.setattr(server, "enable_core", False)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_arxiv", False)
    monkeypatch.setattr(server, "client", SemanticClient())

    payload = _payload(
        await server.call_tool("search_papers", {"query": "transformers"})
    )
    broker_meta = payload["brokerMetadata"]

    assert broker_meta["providerUsed"] == "semantic_scholar"
    assert broker_meta["resultQuality"] == "strong"



@pytest.mark.asyncio
async def test_broker_metadata_result_quality_lexical_for_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resultQuality must be 'lexical' when CORE supplies results; a nonsense query
    that returns CORE results should expose the weak-match signal so agents do not
    treat keyword overlap as topical relevance."""

    class CoreClient:
        async def search(self, **kwargs: Any) -> dict:
            # Simulate CORE returning lexical hits for a nonsense query
            return {
                "total": 2,
                "entries": [
                    {"paperId": "c-1", "title": "Paper about nonsense research"},
                    {"paperId": "c-2", "title": "Another nonsense paper"},
                ],
            }

    monkeypatch.setattr(server, "enable_core", True)
    monkeypatch.setattr(server, "enable_semantic_scholar", False)
    monkeypatch.setattr(server, "enable_arxiv", False)
    monkeypatch.setattr(server, "enable_serpapi", False)
    monkeypatch.setattr(server, "core_client", CoreClient())

    # Use a nonsense-style query to mirror the live UX audit scenario
    payload = _payload(
        await server.call_tool(
            "search_papers", {"query": "asdkfjhasdkjfh research paper nonsense"}
        )
    )
    broker_meta = payload["brokerMetadata"]

    assert broker_meta["providerUsed"] == "core"
    assert broker_meta["resultQuality"] == "lexical"
    assert broker_meta["bulkSearchIsProviderPivot"] is True
    # The hint must warn about weak matches
    assert "lexical" in broker_meta["nextStepHint"].lower()
    # The hint must describe bulk as a pivot away from CORE
    assert "provider pivot" in broker_meta["nextStepHint"]
    assert "CORE" in broker_meta["nextStepHint"]


@pytest.mark.asyncio
async def test_broker_metadata_result_quality_lexical_for_arxiv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resultQuality must be 'lexical' when arXiv supplies results."""

    class ArxivClient:
        async def search(self, **kwargs: Any) -> dict:
            return {
                "totalResults": 1,
                "entries": [
                    {
                        "paperId": "arxiv-1",
                        "title": "arXiv paper",
                        "url": "https://arxiv.org/abs/arxiv-1",
                        "source": "arxiv",
                    }
                ],
            }

    monkeypatch.setattr(server, "enable_core", False)
    monkeypatch.setattr(server, "enable_semantic_scholar", False)
    monkeypatch.setattr(server, "enable_arxiv", True)
    monkeypatch.setattr(server, "enable_serpapi", False)
    monkeypatch.setattr(server, "arxiv_client", ArxivClient())

    payload = _payload(await server.call_tool("search_papers", {"query": "quantum"}))
    broker_meta = payload["brokerMetadata"]

    assert broker_meta["providerUsed"] == "arxiv"
    assert broker_meta["resultQuality"] == "lexical"
    assert broker_meta["bulkSearchIsProviderPivot"] is True
    assert "provider pivot" in broker_meta["nextStepHint"]
    assert "arXiv" in broker_meta["nextStepHint"]


@pytest.mark.asyncio
async def test_broker_metadata_result_quality_unknown_for_serpapi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resultQuality must be 'unknown' when SerpApi supplies results."""

    class FakeSerpApiClient:
        async def search(self, **kwargs: Any) -> list:
            return [
                {
                    "paperId": None,
                    "title": "Scholar paper",
                    "source": "serpapi_google_scholar",
                    "scholarResultId": "sr-123",
                }
            ]

    monkeypatch.setattr(server, "enable_core", False)
    monkeypatch.setattr(server, "enable_semantic_scholar", False)
    monkeypatch.setattr(server, "enable_arxiv", False)
    monkeypatch.setattr(server, "enable_serpapi", True)
    monkeypatch.setattr(server, "serpapi_client", FakeSerpApiClient())

    payload = _payload(await server.call_tool("search_papers", {"query": "ml"}))
    broker_meta = payload["brokerMetadata"]

    assert broker_meta["providerUsed"] == "serpapi_google_scholar"
    assert broker_meta["resultQuality"] == "unknown"
    assert broker_meta["bulkSearchIsProviderPivot"] is True


@pytest.mark.asyncio
async def test_broker_metadata_result_quality_unknown_for_no_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resultQuality must be 'unknown' (providerUsed='none') when no provider
    returns results."""

    monkeypatch.setattr(server, "enable_core", False)
    monkeypatch.setattr(server, "enable_semantic_scholar", False)
    monkeypatch.setattr(server, "enable_arxiv", False)
    monkeypatch.setattr(server, "enable_serpapi", False)

    payload = _payload(
        await server.call_tool("search_papers", {"query": "asdkfjhasdkjfh"})
    )
    broker_meta = payload["brokerMetadata"]

    assert broker_meta["providerUsed"] == "none"
    assert broker_meta["resultQuality"] == "unknown"
    # bulk pivot flag defaults to True for the no-result path too
    assert broker_meta["bulkSearchIsProviderPivot"] is True


def test_result_quality_helper_covers_all_providers() -> None:
    """_result_quality must map every expected provider string without raising."""
    from scholar_search_mcp.search import _result_quality

    assert _result_quality("semantic_scholar") == "strong"
    assert _result_quality("core") == "lexical"
    assert _result_quality("arxiv") == "lexical"
    assert _result_quality("serpapi_google_scholar") == "unknown"
    assert _result_quality("none") == "unknown"


def test_broker_metadata_fields_serialized() -> None:
    """resultQuality and bulkSearchIsProviderPivot must appear in serialized output."""
    from scholar_search_mcp.models.common import SearchResponse
    from scholar_search_mcp.search import _dump_search_response, _metadata

    meta = _metadata(
        provider_used="core",
        attempts=[],
        ss_only_filters=[],
    )
    response = SearchResponse(total=1, offset=0, data=[], broker_metadata=meta)
    serialized = _dump_search_response(response)

    broker = serialized["brokerMetadata"]
    assert broker["resultQuality"] == "lexical"
    assert broker["bulkSearchIsProviderPivot"] is True

    # Also verify Semantic Scholar path
    meta_ss = _metadata(
        provider_used="semantic_scholar",
        attempts=[],
        ss_only_filters=[],
    )
    response_ss = SearchResponse(total=1, offset=0, data=[], broker_metadata=meta_ss)
    serialized_ss = _dump_search_response(response_ss)
    broker_ss = serialized_ss["brokerMetadata"]
    assert broker_ss["resultQuality"] == "strong"
    assert broker_ss["bulkSearchIsProviderPivot"] is False
