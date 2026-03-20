import pytest

import scholar_search_mcp
import scholar_search_mcp.__main__ as server_main
import scholar_search_mcp.cli as cli
from scholar_search_mcp import server
from scholar_search_mcp.utils.cursor import decode_bulk_cursor, decode_cursor
from tests.helpers import RecordingOpenAlexClient, RecordingSemanticClient, _payload


@pytest.mark.asyncio
async def test_list_tools_returns_expected_public_contract() -> None:
    tools = await server.list_tools()

    assert len(tools) == 29
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
        "paper_autocomplete",
        "get_paper_details",
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
        "get_author_papers_openalex",
        "batch_get_authors",
        "search_snippets",
        "get_paper_recommendations",
        "get_paper_recommendations_post",
        "batch_get_papers",
        "get_paper_citation_formats",
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
    assert set(tool_map["search_papers_openalex_bulk"].inputSchema["properties"]) == {
        "query",
        "limit",
        "year",
        "cursor",
    }
    search_tags = tool_map["search_papers"].meta or {}
    semantic_tags = tool_map["search_papers_semantic_scholar"].meta or {}
    openalex_tags = tool_map["search_papers_openalex"].meta or {}
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
    assert tool_map["batch_get_papers"].inputSchema["required"] == ["paper_ids"]
    assert tool_map["batch_get_authors"].inputSchema["required"] == ["author_ids"]
    post_rec_schema = tool_map["get_paper_recommendations_post"].inputSchema
    assert "positivePaperIds" in post_rec_schema["required"]
    citation_schema = tool_map["get_paper_citation_formats"].inputSchema
    assert citation_schema["required"] == ["result_id"]


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
    assert payload == expected_payload


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


def test_package_entrypoints_stay_aligned() -> None:
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
