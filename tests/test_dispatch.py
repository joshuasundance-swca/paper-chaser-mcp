import pytest

import scholar_search_mcp
import scholar_search_mcp.__main__ as server_main
import scholar_search_mcp.cli as cli
from scholar_search_mcp import server
from scholar_search_mcp.enrichment import PaperEnrichmentService
from scholar_search_mcp.utils.cursor import decode_bulk_cursor, decode_cursor
from tests.helpers import (
    RecordingCrossrefClient,
    RecordingOpenAlexClient,
    RecordingSemanticClient,
    RecordingUnpaywallClient,
    _payload,
)


def _assert_additive_metadata(
    payload: dict,
    *,
    expect_search_session_id: bool = False,
) -> None:
    assert "agentHints" in payload
    assert "resourceUris" in payload
    if expect_search_session_id:
        assert payload.get("searchSessionId")


def _install_recording_enrichment(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[RecordingCrossrefClient, RecordingUnpaywallClient]:
    crossref = RecordingCrossrefClient()
    unpaywall = RecordingUnpaywallClient()
    service = PaperEnrichmentService(
        crossref_client=crossref,
        unpaywall_client=unpaywall,
        enable_crossref=True,
        enable_unpaywall=True,
        provider_registry=server.provider_registry,
    )
    monkeypatch.setattr(server, "enable_crossref", True)
    monkeypatch.setattr(server, "enable_unpaywall", True)
    monkeypatch.setattr(server, "crossref_client", crossref)
    monkeypatch.setattr(server, "unpaywall_client", unpaywall)
    monkeypatch.setattr(server, "enrichment_service", service)
    return crossref, unpaywall


@pytest.mark.asyncio
async def test_list_tools_returns_expected_public_contract() -> None:
    tools = await server.list_tools()

    assert len(tools) == 46
    tool_map = {tool.name: tool for tool in tools}
    assert set(tool_map) == {
        "search_papers",
        "search_papers_core",
        "search_papers_semantic_scholar",
        "search_papers_serpapi",
        "search_papers_arxiv",
        "search_papers_openalex",
        "search_papers_openalex_bulk",
        "search_papers_bulk",
        "search_papers_match",
        "resolve_citation",
        "paper_autocomplete",
        "paper_autocomplete_openalex",
        "get_paper_details",
        "get_paper_metadata_crossref",
        "get_paper_open_access_unpaywall",
        "enrich_paper",
        "get_paper_details_openalex",
        "get_paper_citations",
        "get_paper_citations_openalex",
        "get_paper_references",
        "get_paper_references_openalex",
        "get_paper_authors",
        "get_author_info",
        "get_author_info_openalex",
        "get_author_papers",
        "search_authors",
        "search_authors_openalex",
        "search_entities_openalex",
        "search_papers_openalex_by_entity",
        "get_author_papers_openalex",
        "batch_get_authors",
        "search_snippets",
        "get_paper_recommendations",
        "get_paper_recommendations_post",
        "batch_get_papers",
        "get_paper_citation_formats",
        "search_papers_serpapi_cited_by",
        "search_papers_serpapi_versions",
        "get_author_profile_serpapi",
        "get_author_articles_serpapi",
        "get_serpapi_account_status",
        "get_provider_diagnostics",
        "search_papers_smart",
        "ask_result_set",
        "map_research_landscape",
        "expand_research_graph",
    }
    assert tool_map["search_papers"].inputSchema["required"] == ["query"]
    assert set(tool_map["search_papers"].inputSchema["properties"]) == {
        "query",
        "limit",
        "fields",
        "year",
        "venue",
        "preferredProvider",
        "providerOrder",
        "publicationDateOrYear",
        "fieldsOfStudy",
        "publicationTypes",
        "openAccessPdf",
        "minCitationCount",
    }
    assert set(tool_map["search_papers_core"].inputSchema["properties"]) == {
        "query",
        "limit",
        "year",
    }
    assert set(
        tool_map["search_papers_semantic_scholar"].inputSchema["properties"]
    ) == {
        "query",
        "limit",
        "fields",
        "year",
        "venue",
        "publicationDateOrYear",
        "fieldsOfStudy",
        "publicationTypes",
        "openAccessPdf",
        "minCitationCount",
    }
    assert set(tool_map["search_papers_serpapi"].inputSchema["properties"]) == {
        "query",
        "limit",
        "year",
    }
    assert set(tool_map["search_papers_arxiv"].inputSchema["properties"]) == {
        "query",
        "limit",
        "year",
    }
    assert set(tool_map["search_papers_openalex"].inputSchema["properties"]) == {
        "query",
        "limit",
        "year",
    }
    assert set(tool_map["paper_autocomplete_openalex"].inputSchema["properties"]) == {
        "query",
        "limit",
    }
    assert set(tool_map["search_papers_openalex_bulk"].inputSchema["properties"]) == {
        "query",
        "limit",
        "year",
        "cursor",
    }
    assert set(tool_map["search_entities_openalex"].inputSchema["properties"]) == {
        "query",
        "entityType",
        "limit",
        "cursor",
    }
    assert set(
        tool_map["search_papers_openalex_by_entity"].inputSchema["properties"]
    ) == {
        "entityType",
        "entityId",
        "limit",
        "cursor",
        "year",
    }
    assert set(tool_map["resolve_citation"].inputSchema["properties"]) == {
        "citation",
        "maxCandidates",
        "titleHint",
        "authorHint",
        "yearHint",
        "venueHint",
        "doiHint",
        "includeEnrichment",
    }
    assert set(tool_map["get_paper_details"].inputSchema["properties"]) == {
        "paper_id",
        "fields",
        "includeEnrichment",
    }
    assert set(tool_map["get_paper_metadata_crossref"].inputSchema["properties"]) == {
        "paper_id",
        "doi",
        "query",
    }
    assert set(
        tool_map["get_paper_open_access_unpaywall"].inputSchema["properties"]
    ) == {
        "paper_id",
        "doi",
    }
    assert set(tool_map["enrich_paper"].inputSchema["properties"]) == {
        "paper_id",
        "doi",
        "query",
    }
    search_tags = tool_map["search_papers"].meta or {}
    semantic_tags = tool_map["search_papers_semantic_scholar"].meta or {}
    openalex_tags = tool_map["search_papers_openalex"].meta or {}
    citation_tags = tool_map["resolve_citation"].meta or {}
    assert set(search_tags["fastmcp"]["tags"]) == {
        "search",
        "brokered",
    }
    assert set(semantic_tags["fastmcp"]["tags"]) == {
        "search",
        "provider-specific",
        "provider:semantic_scholar",
    }
    assert set(openalex_tags["fastmcp"]["tags"]) == {
        "search",
        "provider-specific",
        "provider:openalex",
    }
    assert set(citation_tags["fastmcp"]["tags"]) == {
        "citation-repair",
        "known-item",
        "recovery",
    }
    assert set(tool_map["search_papers_smart"].inputSchema["properties"]) == {
        "query",
        "limit",
        "searchSessionId",
        "mode",
        "year",
        "venue",
        "focus",
        "latencyProfile",
        "providerBudget",
        "includeEnrichment",
    }
    assert set(tool_map["ask_result_set"].inputSchema["properties"]) == {
        "searchSessionId",
        "question",
        "topK",
        "answerMode",
        "latencyProfile",
    }
    assert set(tool_map["map_research_landscape"].inputSchema["properties"]) == {
        "searchSessionId",
        "maxThemes",
        "latencyProfile",
    }
    assert set(tool_map["expand_research_graph"].inputSchema["properties"]) == {
        "seedPaperIds",
        "seedSearchSessionId",
        "direction",
        "hops",
        "perSeedLimit",
        "latencyProfile",
    }
    assert tool_map["batch_get_papers"].inputSchema["required"] == ["paper_ids"]
    assert tool_map["batch_get_authors"].inputSchema["required"] == ["author_ids"]
    post_rec_schema = tool_map["get_paper_recommendations_post"].inputSchema
    assert "positivePaperIds" in post_rec_schema["required"]
    citation_schema = tool_map["get_paper_citation_formats"].inputSchema
    assert citation_schema["required"] == ["result_id"]
    assert set(tool_map["get_provider_diagnostics"].inputSchema["properties"]) == {
        "includeRecentOutcomes"
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected_call", "expected_payload"),
    [
        (
            "get_paper_details",
            {"paper_id": "paper-1", "fields": ["title"]},
            ("get_paper_details", {"paper_id": "paper-1", "fields": ["title"]}),
            {"paperId": "paper-1"},
        ),
        (
            "get_paper_citations",
            {"paper_id": "paper-2"},
            (
                "get_paper_citations",
                {"paper_id": "paper-2", "limit": 100, "fields": None, "offset": None},
            ),
            {"data": [{"paperId": "paper-2"}]},
        ),
        (
            "get_paper_references",
            {"paper_id": "paper-3", "limit": 12, "fields": ["authors"]},
            (
                "get_paper_references",
                {
                    "paper_id": "paper-3",
                    "limit": 12,
                    "fields": ["authors"],
                    "offset": None,
                },
            ),
            {"data": [{"paperId": "paper-3"}]},
        ),
        (
            "get_author_info",
            {"author_id": "author-1"},
            ("get_author_info", {"author_id": "author-1", "fields": None}),
            {"authorId": "author-1"},
        ),
        (
            "get_author_papers",
            {"author_id": "author-2", "limit": 25},
            (
                "get_author_papers",
                {
                    "author_id": "author-2",
                    "limit": 25,
                    "fields": None,
                    "offset": None,
                    "publication_date_or_year": None,
                },
            ),
            {"data": [{"authorId": "author-2"}]},
        ),
        (
            "get_paper_recommendations",
            {"paper_id": "paper-4"},
            (
                "get_recommendations",
                {"paper_id": "paper-4", "limit": 10, "fields": None},
            ),
            {"recommendedPapers": [{"paperId": "paper-4"}]},
        ),
        (
            "batch_get_papers",
            {"paper_ids": ["p1", "p2"], "fields": ["title"]},
            (
                "batch_get_papers",
                {"paper_ids": ["p1", "p2"], "fields": ["title"]},
            ),
            [{"paperId": "p1"}, {"paperId": "p2"}],
        ),
    ],
)
async def test_call_tool_routes_non_search_tools(
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
    arguments: dict,
    expected_call: tuple[str, dict],
    expected_payload: dict | list,
) -> None:
    fake_client = RecordingSemanticClient()
    monkeypatch.setattr(server, "client", fake_client)

    payload = _payload(await server.call_tool(tool_name, arguments))

    assert fake_client.calls == [expected_call]
    if isinstance(expected_payload, list):
        assert payload == expected_payload
        return
    for key, value in expected_payload.items():
        assert payload[key] == value
    _assert_additive_metadata(
        payload,
        expect_search_session_id=tool_name
        in {"get_paper_citations", "get_paper_references", "get_author_papers"},
    )


@pytest.mark.asyncio
async def test_explicit_enrichment_tools_route_through_recording_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crossref, unpaywall = _install_recording_enrichment(monkeypatch)

    crossref_payload = _payload(
        await server.call_tool(
            "get_paper_metadata_crossref",
            {"doi": "https://doi.org/10.1234/seed-doi"},
        )
    )
    unpaywall_payload = _payload(
        await server.call_tool(
            "get_paper_open_access_unpaywall",
            {"doi": "doi:10.1234/seed-doi"},
        )
    )
    merged_payload = _payload(
        await server.call_tool(
            "enrich_paper",
            {"query": "Crossref Query Paper"},
        )
    )

    assert crossref.calls[0] == ("get_work", {"doi": "10.1234/seed-doi"})
    assert crossref_payload["found"] is True
    assert crossref_payload["resolvedDoi"] == "10.1234/seed-doi"
    assert crossref_payload["work"]["publisher"] == "Crossref Publisher"
    _assert_additive_metadata(crossref_payload)

    assert unpaywall.calls[0] == ("get_open_access", {"doi": "10.1234/seed-doi"})
    assert unpaywall_payload["found"] is True
    assert unpaywall_payload["isOa"] is True
    assert unpaywall_payload["pdfUrl"] == "https://oa.example/10.1234/seed-doi.pdf"
    _assert_additive_metadata(unpaywall_payload)

    assert ("search_work", {"query": "Crossref Query Paper"}) in crossref.calls
    assert (
        "get_open_access",
        {"doi": "10.1234/crossref-query"},
    ) in unpaywall.calls
    assert merged_payload["doiResolution"]["resolvedDoi"] == "10.1234/crossref-query"
    assert merged_payload["enrichments"]["crossref"]["doi"] == "10.1234/crossref-query"
    assert merged_payload["enrichments"]["unpaywall"]["isOa"] is True
    _assert_additive_metadata(merged_payload)


@pytest.mark.asyncio
async def test_search_papers_match_include_enrichment_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = RecordingSemanticClient()
    crossref, unpaywall = _install_recording_enrichment(monkeypatch)
    monkeypatch.setattr(server, "client", fake_client)

    baseline = _payload(
        await server.call_tool(
            "search_papers_match",
            {"query": "Best match"},
        )
    )
    enriched = _payload(
        await server.call_tool(
            "search_papers_match",
            {"query": "Best match", "includeEnrichment": True},
        )
    )

    assert "enrichments" not in baseline
    assert crossref.calls == [("search_work", {"query": "Best match"})]
    assert unpaywall.calls == [("get_open_access", {"doi": "10.1234/crossref-query"})]
    assert enriched["enrichments"]["crossref"]["doi"] == "10.1234/crossref-query"
    assert enriched["enrichments"]["unpaywall"]["bestOaUrl"].endswith(
        "10.1234/crossref-query"
    )
    assert fake_client.calls == [
        ("search_papers_match", {"query": "Best match", "fields": None}),
        ("search_papers_match", {"query": "Best match", "fields": None}),
    ]


@pytest.mark.asyncio
async def test_get_paper_details_include_enrichment_uses_resolved_doi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = RecordingSemanticClient()

    async def fake_get_paper_details(**kwargs: dict) -> dict:
        fake_client.calls.append(("get_paper_details", kwargs))
        return {
            "paperId": kwargs["paper_id"],
            "title": "Detailed paper",
            "canonicalId": "doi:10.5678/details-paper",
        }

    fake_client.get_paper_details = fake_get_paper_details  # type: ignore[method-assign]
    crossref, unpaywall = _install_recording_enrichment(monkeypatch)
    monkeypatch.setattr(server, "client", fake_client)

    payload = _payload(
        await server.call_tool(
            "get_paper_details",
            {"paper_id": "doi:10.5678/details-paper", "includeEnrichment": True},
        )
    )

    assert fake_client.calls == [
        (
            "get_paper_details",
            {
                "paper_id": "doi:10.5678/details-paper",
                "fields": None,
            },
        )
    ]
    assert crossref.calls == [("get_work", {"doi": "10.5678/details-paper"})]
    assert unpaywall.calls == [("get_open_access", {"doi": "10.5678/details-paper"})]
    assert payload["enrichments"]["crossref"]["publisher"] == "Crossref Publisher"
    assert payload["enrichments"]["unpaywall"]["license"] == "cc-by"
    _assert_additive_metadata(payload)


@pytest.mark.asyncio
async def test_search_papers_forwards_clamped_semantic_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = RecordingSemanticClient()

    monkeypatch.setattr(server, "enable_core", False)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_arxiv", False)
    monkeypatch.setattr(server, "client", fake_client)

    payload = _payload(
        await server.call_tool(
            "search_papers",
            {
                "query": "graph neural networks",
                "limit": 999,
                "fields": ["title", "year"],
                "year": "2022",
                "venue": ["NeurIPS"],
            },
        )
    )

    assert fake_client.calls == [
        (
            "search_papers",
            {
                "query": "graph neural networks",
                "limit": 100,
                "fields": ["title", "year"],
                "year": "2022",
                "venue": ["NeurIPS"],
                "publication_date_or_year": None,
                "fields_of_study": None,
                "publication_types": None,
                "open_access_pdf": None,
                "min_citation_count": None,
            },
        )
    ]
    assert payload["data"][0]["source"] == "semantic_scholar"


@pytest.mark.asyncio
async def test_openalex_tool_routes_and_wraps_provider_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openalex = RecordingOpenAlexClient()
    monkeypatch.setattr(server, "openalex_client", openalex)
    monkeypatch.setattr(server, "enable_openalex", True)

    bulk_payload = _payload(
        await server.call_tool(
            "search_papers_openalex_bulk",
            {"query": "transformers", "year": "2024"},
        )
    )
    bulk_cursor = decode_bulk_cursor(bulk_payload["pagination"]["nextCursor"])

    author_payload = _payload(
        await server.call_tool(
            "search_authors_openalex",
            {"query": "Yoshua Bengio"},
        )
    )
    author_cursor = decode_bulk_cursor(author_payload["pagination"]["nextCursor"])

    refs_payload = _payload(
        await server.call_tool(
            "get_paper_references_openalex",
            {"paper_id": "W1"},
        )
    )
    refs_cursor = decode_cursor(refs_payload["pagination"]["nextCursor"])

    assert openalex.calls[0] == (
        "search_bulk",
        {"query": "transformers", "limit": 100, "cursor": None, "year": "2024"},
    )
    assert bulk_cursor.provider == "openalex"
    assert bulk_cursor.tool == "search_papers_openalex_bulk"
    assert bulk_cursor.token == "oa-next"

    assert openalex.calls[1] == (
        "search_authors",
        {"query": "Yoshua Bengio", "limit": 10, "cursor": None},
    )
    assert author_cursor.provider == "openalex"
    assert author_cursor.tool == "search_authors_openalex"
    assert author_cursor.token == "oa-authors"

    assert openalex.calls[2] == (
        "get_paper_references",
        {"paper_id": "W1", "limit": 100, "offset": 0},
    )
    assert refs_cursor.provider == "openalex"
    assert refs_cursor.tool == "get_paper_references_openalex"
    assert refs_cursor.offset == 25


@pytest.mark.asyncio
async def test_openalex_detail_and_author_tools_route_to_openalex_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openalex = RecordingOpenAlexClient()
    monkeypatch.setattr(server, "openalex_client", openalex)
    monkeypatch.setattr(server, "enable_openalex", True)

    paper = _payload(
        await server.call_tool("get_paper_details_openalex", {"paper_id": "W42"})
    )
    author = _payload(
        await server.call_tool("get_author_info_openalex", {"author_id": "A42"})
    )
    author_papers = _payload(
        await server.call_tool(
            "get_author_papers_openalex",
            {"author_id": "A42", "year": "2023"},
        )
    )

    assert paper["paperId"] == "W42"
    assert author["authorId"] == "A42"
    assert openalex.calls == [
        ("get_paper_details", {"paper_id": "W42"}),
        ("get_author_info", {"author_id": "A42"}),
        (
            "get_author_papers",
            {"author_id": "A42", "limit": 100, "cursor": None, "year": "2023"},
        ),
    ]
    assert decode_bulk_cursor(author_papers["pagination"]["nextCursor"]).provider == (
        "openalex"
    )


@pytest.mark.asyncio
async def test_new_openalex_tools_route_and_wrap_bulk_cursors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openalex = RecordingOpenAlexClient()
    monkeypatch.setattr(server, "openalex_client", openalex)
    monkeypatch.setattr(server, "enable_openalex", True)

    autocomplete = _payload(
        await server.call_tool(
            "paper_autocomplete_openalex",
            {"query": "transformer"},
        )
    )
    entities = _payload(
        await server.call_tool(
            "search_entities_openalex",
            {"query": "neurips", "entityType": "source"},
        )
    )
    entity_papers = _payload(
        await server.call_tool(
            "search_papers_openalex_by_entity",
            {"entityType": "source", "entityId": "S1"},
        )
    )

    assert autocomplete["matches"][0]["source"] == "openalex"
    assert openalex.calls[0] == (
        "paper_autocomplete",
        {"query": "transformer", "limit": 10},
    )
    assert openalex.calls[1] == (
        "search_entities",
        {"entity_type": "source", "query": "neurips", "limit": 10, "cursor": None},
    )
    assert decode_bulk_cursor(entities["pagination"]["nextCursor"]).provider == (
        "openalex"
    )
    assert openalex.calls[2] == (
        "search_works_by_entity",
        {
            "entity_type": "source",
            "entity_id": "S1",
            "limit": 100,
            "cursor": None,
            "year": None,
        },
    )
    assert decode_bulk_cursor(entity_papers["pagination"]["nextCursor"]).provider == (
        "openalex"
    )


@pytest.mark.asyncio
async def test_new_serpapi_tools_and_provider_diagnostics_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RecordingSerpApiClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def search_cited_by(self, **kwargs) -> dict:
            self.calls.append(("search_cited_by", kwargs))
            return {
                "provider": "serpapi_google_scholar",
                "total": 1,
                "offset": 0,
                "data": [
                    {
                        "paperId": "serp-cites-1",
                        "source": "serpapi_google_scholar",
                    }
                ],
                "pagination": {"hasMore": True, "nextCursor": "10"},
            }

        async def search_versions(self, **kwargs) -> dict:
            self.calls.append(("search_versions", kwargs))
            return {
                "provider": "serpapi_google_scholar",
                "total": 1,
                "offset": 0,
                "data": [
                    {
                        "paperId": "serp-version-1",
                        "source": "serpapi_google_scholar",
                    }
                ],
                "pagination": {"hasMore": True, "nextCursor": "10"},
            }

        async def get_author_profile(self, **kwargs) -> dict:
            self.calls.append(("get_author_profile", kwargs))
            return {
                "provider": "serpapi_google_scholar",
                "authorId": kwargs["author_id"],
                "name": "Scholar Author",
            }

        async def get_author_articles(self, **kwargs) -> dict:
            self.calls.append(("get_author_articles", kwargs))
            return {
                "provider": "serpapi_google_scholar",
                "authorId": kwargs["author_id"],
                "total": 1,
                "offset": kwargs.get("start", 0),
                "data": [
                    {
                        "paperId": "serp-author-1",
                        "source": "serpapi_google_scholar",
                    }
                ],
                "pagination": {"hasMore": True, "nextCursor": "10"},
            }

        async def get_account_status(self) -> dict:
            self.calls.append(("get_account_status", {}))
            return {
                "provider": "serpapi_google_scholar",
                "plan_searches_left": 42,
            }

    monkeypatch.setattr(server, "enable_serpapi", True)
    monkeypatch.setattr(server, "serpapi_client", RecordingSerpApiClient())

    cited_by = _payload(
        await server.call_tool(
            "search_papers_serpapi_cited_by",
            {"citesId": "cites-1"},
        )
    )
    versions = _payload(
        await server.call_tool(
            "search_papers_serpapi_versions",
            {"clusterId": "cluster-1"},
        )
    )
    author = _payload(
        await server.call_tool(
            "get_author_profile_serpapi",
            {"authorId": "author-1"},
        )
    )
    articles = _payload(
        await server.call_tool(
            "get_author_articles_serpapi",
            {"authorId": "author-1"},
        )
    )
    account = _payload(await server.call_tool("get_serpapi_account_status", {}))
    diagnostics = _payload(await server.call_tool("get_provider_diagnostics", {}))

    assert cited_by["data"][0]["paperId"] == "serp-cites-1"
    assert decode_cursor(cited_by["pagination"]["nextCursor"]).provider == (
        "serpapi_google_scholar"
    )
    assert versions["data"][0]["paperId"] == "serp-version-1"
    assert decode_cursor(versions["pagination"]["nextCursor"]).provider == (
        "serpapi_google_scholar"
    )
    assert author["authorId"] == "author-1"
    assert articles["authorId"] == "author-1"
    assert decode_cursor(articles["pagination"]["nextCursor"]).provider == (
        "serpapi_google_scholar"
    )
    assert account["provider"] == "serpapi_google_scholar"
    assert any(item["provider"] == "openai" for item in diagnostics["providers"])


@pytest.mark.asyncio
async def test_provider_diagnostics_surface_crossref_and_unpaywall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_recording_enrichment(monkeypatch)

    await server.call_tool("get_paper_metadata_crossref", {"doi": "10.1234/diag"})
    await server.call_tool(
        "get_paper_open_access_unpaywall",
        {"doi": "10.1234/diag"},
    )
    diagnostics = _payload(await server.call_tool("get_provider_diagnostics", {}))
    provider_map = {
        item["provider"]: item
        for item in diagnostics["providers"]
        if isinstance(item, dict)
    }

    assert provider_map["crossref"]["enabled"] is True
    assert provider_map["unpaywall"]["enabled"] is True
    assert provider_map["crossref"]["lastEndpoint"] == "get_work"
    assert provider_map["unpaywall"]["lastEndpoint"] == "get_open_access"
    assert provider_map["crossref"]["recentOutcomes"]
    assert provider_map["unpaywall"]["recentOutcomes"]


def test_package_import_and_module_entrypoints_keep_expected_targets() -> None:
    """Package import exposes server.main while python -m routes through cli.main."""
    assert scholar_search_mcp.main is server.main
    assert server_main.main is cli.main


def test_cli_default_entrypoint_runs_server(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(cli, "_run_server", lambda: calls.append("server"))

    cli.main([])

    assert calls == ["server"]


def test_cli_deployment_http_subcommand_runs_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        cli,
        "_run_deployment_http",
        lambda: calls.append("deployment-http"),
    )

    cli.main(["deployment-http"])

    assert calls == ["deployment-http"]


@pytest.mark.asyncio
async def test_call_tool_raises_for_unknown_tool() -> None:
    with pytest.raises(ValueError, match="Unknown tool"):
        await server.call_tool("unknown_tool", {})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected_method", "check_payload"),
    [
        (
            "search_papers_bulk",
            {"query": "neural networks", "limit": 50},
            "search_papers_bulk",
            lambda p: p["data"][0]["paperId"] == "bulk-1",
        ),
        (
            "search_papers_match",
            {"query": "attention is all you need"},
            "search_papers_match",
            lambda p: p["paperId"] == "match-1",
        ),
        (
            "resolve_citation",
            {"citation": "Attention Is All You Need"},
            "search_papers_match",
            lambda p: bool(p["bestMatch"]["paper"]["paperId"]),
        ),
        (
            "paper_autocomplete",
            {"query": "transformer"},
            "paper_autocomplete",
            lambda p: "matches" in p,
        ),
        (
            "get_paper_authors",
            {"paper_id": "paper-5"},
            "get_paper_authors",
            lambda p: p["total"] == 1,
        ),
        (
            "search_authors",
            {"query": "Yoshua Bengio"},
            "search_authors",
            lambda p: p["total"] == 1,
        ),
        (
            "batch_get_authors",
            {"author_ids": ["a1", "a2"]},
            "batch_get_authors",
            lambda p: len(p) == 2,
        ),
        (
            "search_snippets",
            {"query": "deep learning"},
            "search_snippets",
            lambda p: len(p["data"]) == 1,
        ),
        (
            "get_paper_recommendations_post",
            {"positivePaperIds": ["p1", "p2"]},
            "get_recommendations_post",
            lambda p: p["recommendedPapers"][0]["paperId"] == "rec-post-1",
        ),
    ],
)
async def test_call_tool_routes_new_tools(
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str,
    arguments: dict,
    expected_method: str,
    check_payload,
) -> None:
    fake_client = RecordingSemanticClient()
    monkeypatch.setattr(server, "client", fake_client)

    payload = _payload(await server.call_tool(tool_name, arguments))

    assert any(method == expected_method for method, _ in fake_client.calls), (
        f"Expected {expected_method!r} to have been called; got {fake_client.calls}"
    )
    assert check_payload(payload)
    if isinstance(payload, dict):
        _assert_additive_metadata(
            payload,
            expect_search_session_id=tool_name
            in {
                "search_papers_bulk",
                "get_paper_authors",
                "search_authors",
                "resolve_citation",
            },
        )


@pytest.mark.asyncio
async def test_smart_tools_return_structured_feature_errors_when_disabled() -> None:
    payload = _payload(
        await server.call_tool("search_papers_smart", {"query": "transformers"})
    )

    assert payload["error"] == "FEATURE_NOT_CONFIGURED"
    assert "fallbackTools" in payload
    assert payload["agentHints"]["nextToolCandidates"]
