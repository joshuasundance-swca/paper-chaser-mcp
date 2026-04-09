from typing import Any

import pytest

import paper_chaser_mcp
import paper_chaser_mcp.__main__ as server_main
import paper_chaser_mcp.cli as cli
import paper_chaser_mcp.clients.serpapi.client as serpapi_client_module
import paper_chaser_mcp.dispatch as dispatch_module
from paper_chaser_mcp import server
from paper_chaser_mcp.clients.serpapi import SerpApiScholarClient
from paper_chaser_mcp.enrichment import PaperEnrichmentService
from paper_chaser_mcp.tool_specs import iter_visible_tool_specs
from paper_chaser_mcp.utils.cursor import decode_bulk_cursor, decode_cursor
from tests.helpers import (
    DummyResponse,
    DummySerpApiAsyncClient,
    RecordingCrossrefClient,
    RecordingEcosClient,
    RecordingOpenAlexClient,
    RecordingScholarApiClient,
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
) -> tuple[RecordingCrossrefClient, RecordingUnpaywallClient, RecordingOpenAlexClient]:
    crossref = RecordingCrossrefClient()
    unpaywall = RecordingUnpaywallClient()
    openalex = RecordingOpenAlexClient()
    service = PaperEnrichmentService(
        crossref_client=crossref,
        unpaywall_client=unpaywall,
        openalex_client=openalex,
        enable_crossref=True,
        enable_unpaywall=True,
        enable_openalex=True,
        provider_registry=server.provider_registry,
    )
    monkeypatch.setattr(server, "enable_crossref", True)
    monkeypatch.setattr(server, "enable_unpaywall", True)
    monkeypatch.setattr(server, "enable_openalex", True)
    monkeypatch.setattr(server, "crossref_client", crossref)
    monkeypatch.setattr(server, "unpaywall_client", unpaywall)
    monkeypatch.setattr(server, "openalex_client", openalex)
    monkeypatch.setattr(server, "enrichment_service", service)
    return crossref, unpaywall, openalex


@pytest.mark.asyncio
async def test_list_tools_returns_expected_public_contract() -> None:
    tools = await server.list_tools()

    assert len(tools) == 5
    tool_map = {tool.name: tool for tool in tools}
    assert set(tool_map) == {
        "research",
        "follow_up_research",
        "resolve_reference",
        "inspect_source",
        "get_runtime_status",
    }
    assert tool_map["research"].inputSchema["required"] == ["query"]
    assert set(tool_map["research"].inputSchema["properties"]) == {
        "query",
        "limit",
        "year",
        "venue",
        "focus",
    }
    assert tool_map["follow_up_research"].inputSchema["required"] == ["question"]
    assert set(tool_map["follow_up_research"].inputSchema["properties"]) == {
        "searchSessionId",
        "question",
    }
    assert tool_map["resolve_reference"].inputSchema["required"] == ["reference"]
    assert set(tool_map["resolve_reference"].inputSchema["properties"]) == {"reference"}
    assert tool_map["inspect_source"].inputSchema["required"] == ["sourceId"]
    assert set(tool_map["inspect_source"].inputSchema["properties"]) == {
        "searchSessionId",
        "sourceId",
    }
    assert set(tool_map["get_runtime_status"].inputSchema["properties"]) == set()
    research_tags = tool_map["research"].meta or {}
    follow_up_tags = tool_map["follow_up_research"].meta or {}
    resolve_tags = tool_map["resolve_reference"].meta or {}
    inspect_tags = tool_map["inspect_source"].meta or {}
    runtime_tags = tool_map["get_runtime_status"].meta or {}
    assert set(research_tags["fastmcp"]["tags"]) == {"guided", "research"}
    assert set(follow_up_tags["fastmcp"]["tags"]) == {"guided", "follow-up"}
    assert set(resolve_tags["fastmcp"]["tags"]) == {"guided", "reference-resolution"}
    assert set(inspect_tags["fastmcp"]["tags"]) == {"guided", "source-inspection"}
    assert set(runtime_tags["fastmcp"]["tags"]) == {"guided", "runtime-status"}


@pytest.mark.asyncio
async def test_guided_research_uses_server_owned_deep_profile_and_ignores_client_latency_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_calls: list[dict[str, Any]] = []

    class _FakeRuntime:
        async def search_papers_smart(self, **kwargs: Any) -> dict[str, Any]:
            captured_calls.append(kwargs)
            return {
                "searchSessionId": "ssn-guided-quality-default",
                "strategyMetadata": {"intent": "review", "latencyProfile": kwargs.get("latencyProfile")},
                "structuredSources": [
                    {
                        "sourceId": "paper-1",
                        "title": "Quality-first guided retrieval",
                        "provider": "semantic_scholar",
                        "sourceType": "scholarly_article",
                        "verificationStatus": "verified_metadata",
                        "accessStatus": "abstract_only",
                        "topicalRelevance": "on_topic",
                        "confidence": "high",
                        "isPrimarySource": False,
                    }
                ],
                "candidateLeads": [],
                "evidenceGaps": [],
                "coverageSummary": {"searchMode": "literature_review"},
                "failureSummary": None,
                "clarification": None,
            }

    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime())

    payload = _payload(
        await server.call_tool(
            "research",
            {
                "query": "quality-first guided retrieval defaults",
                "latencyProfile": "fast",
            },
        )
    )

    assert captured_calls
    assert (captured_calls[0].get("latencyProfile") or captured_calls[0].get("latency_profile")) == "deep"
    assert payload["status"] in {"succeeded", "partial"}


@pytest.mark.asyncio
async def test_guided_research_surfaces_evidence_contract_and_routing_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        async def search_papers_smart(self, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            return {
                "searchSessionId": "ssn-guided-evidence-contract",
                "strategyMetadata": {
                    "intent": "regulatory",
                    "anchorType": "cfr_citation",
                    "anchoredSubject": "Northern long-eared bat",
                    "routingConfidence": "high",
                    "providerPlan": ["govinfo", "federal_register", "ecos"],
                    "providersUsed": ["govinfo", "federal_register"],
                },
                "structuredSources": [
                    {
                        "sourceId": "50 CFR 17.11",
                        "title": "50 CFR 17.11 Endangered and threatened wildlife",
                        "provider": "govinfo",
                        "sourceType": "primary_regulatory",
                        "verificationStatus": "verified_primary_source",
                        "accessStatus": "full_text_verified",
                        "topicalRelevance": "on_topic",
                        "confidence": "high",
                        "isPrimarySource": True,
                        "canonicalUrl": "https://www.govinfo.gov/example/cfr",
                        "citation": {
                            "title": "50 CFR 17.11",
                            "url": "https://www.govinfo.gov/example/cfr",
                            "sourceType": "primary_regulatory",
                        },
                        "date": "2024-10-01",
                    }
                ],
                "candidateLeads": [
                    {
                        "sourceId": "2024-99999",
                        "title": "Unrelated Federal Register notice",
                        "provider": "federal_register",
                        "sourceType": "primary_regulatory",
                        "verificationStatus": "verified_metadata",
                        "accessStatus": "access_unverified",
                        "topicalRelevance": "off_topic",
                        "confidence": "medium",
                        "isPrimarySource": True,
                        "canonicalUrl": "https://www.federalregister.gov/example",
                        "date": "2024-02-01",
                        "note": "Filtered because it did not match the anchored subject.",
                    }
                ],
                "evidenceGaps": ["ECOS did not return a species dossier match."],
                "coverageSummary": {
                    "providersAttempted": ["govinfo", "federal_register", "ecos"],
                    "providersSucceeded": ["govinfo", "federal_register"],
                    "providersZeroResults": ["ecos"],
                    "searchMode": "regulatory_primary_source",
                },
                "failureSummary": None,
                "clarification": None,
                "resultStatus": "succeeded",
            }

    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime())

    payload = _payload(
        await server.call_tool(
            "research",
            {"query": "50 CFR 17.11 northern long-eared bat current text"},
        )
    )

    assert payload["resultStatus"] == "succeeded"
    assert payload["answerability"] == "grounded"
    assert payload["routingSummary"]["intent"] == "regulatory"
    assert payload["routingSummary"]["anchorType"] == "cfr_citation"
    assert payload["routingSummary"]["anchorValue"] == "Northern long-eared bat"
    assert payload["routingSummary"]["providerPlan"] == ["govinfo", "federal_register", "ecos"]
    assert payload["coverageSummary"]["searchMode"] == "regulatory_primary_source"
    assert [item["evidenceId"] for item in payload["evidence"]] == ["50 CFR 17.11"]
    assert payload["evidence"][0]["whyIncluded"]
    assert payload["leads"][0]["evidenceId"] == "2024-99999"
    assert payload["leads"][0]["whyNotVerified"]
    assert not any(item["evidenceId"] == "2024-99999" for item in payload["evidence"])


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
        expect_search_session_id=tool_name in {"get_paper_citations", "get_paper_references", "get_author_papers"},
    )


@pytest.mark.asyncio
async def test_explicit_enrichment_tools_route_through_recording_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crossref, unpaywall, openalex = _install_recording_enrichment(monkeypatch)

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

    assert crossref.calls == [("get_work", {"doi": "10.1234/seed-doi"})]
    assert unpaywall.calls == [("get_open_access", {"doi": "10.1234/seed-doi"})]
    assert openalex.calls == []
    assert merged_payload["doiResolution"]["resolvedDoi"] is None
    assert merged_payload["crossref"]["found"] is False
    assert merged_payload["unpaywall"]["found"] is False
    assert merged_payload["openalex"]["found"] is False
    assert merged_payload.get("enrichments") is None
    _assert_additive_metadata(merged_payload)


def test_visible_tool_specs_hide_disabled_provider_families_when_enabled() -> None:
    visible = {
        spec.name
        for spec in iter_visible_tool_specs(
            tool_profile="expert",
            hide_disabled_tools=True,
            enabled_flags={
                "enable_core": False,
                "enable_semantic_scholar": False,
                "enable_arxiv": False,
                "enable_openalex": False,
                "enable_serpapi": False,
                "enable_scholarapi": False,
                "enable_crossref": False,
                "enable_unpaywall": False,
                "enable_ecos": False,
                "enable_federal_register": False,
                "enable_govinfo_cfr": False,
                "enable_agentic": False,
                "govinfo_available": False,
            },
        )
    }

    assert "enrich_paper" in visible
    assert "get_provider_diagnostics" in visible
    assert "search_papers" not in visible
    assert "search_papers_core" not in visible
    assert "search_papers_bulk" not in visible
    assert "search_papers_match" not in visible
    assert "paper_autocomplete" not in visible
    assert "search_papers_arxiv" not in visible
    assert "search_papers_openalex" not in visible
    assert "search_papers_scholarapi" not in visible
    assert "search_papers_serpapi" not in visible
    assert "resolve_citation" not in visible
    assert "get_paper_details" not in visible
    assert "get_paper_citations" not in visible
    assert "get_paper_references" not in visible
    assert "get_paper_authors" not in visible
    assert "get_author_info" not in visible
    assert "get_author_papers" not in visible
    assert "search_authors" not in visible
    assert "batch_get_authors" not in visible
    assert "search_snippets" not in visible
    assert "get_paper_recommendations" not in visible
    assert "get_paper_recommendations_post" not in visible
    assert "batch_get_papers" not in visible
    assert "get_paper_citation_formats" not in visible
    assert "get_paper_metadata_crossref" not in visible
    assert "get_paper_open_access_unpaywall" not in visible
    assert "search_species_ecos" not in visible
    assert "search_federal_register" not in visible
    assert "get_federal_register_document" not in visible
    assert "get_cfr_text" not in visible
    assert "search_papers_smart" not in visible


def test_visible_tool_specs_keep_govinfo_document_when_available() -> None:
    visible = {
        spec.name
        for spec in iter_visible_tool_specs(
            tool_profile="expert",
            hide_disabled_tools=True,
            enabled_flags={
                "enable_federal_register": False,
                "enable_govinfo_cfr": False,
                "enable_agentic": False,
                "govinfo_available": True,
            },
        )
    }

    assert "get_federal_register_document" in visible


def test_visible_tool_specs_keep_search_and_repair_when_any_provider_available() -> None:
    visible = {
        spec.name
        for spec in iter_visible_tool_specs(
            tool_profile="expert",
            hide_disabled_tools=True,
            enabled_flags={
                "enable_semantic_scholar": False,
                "enable_arxiv": False,
                "enable_core": False,
                "enable_serpapi": False,
                "enable_scholarapi": True,
                "enable_openalex": True,
            },
        )
    }

    assert "search_papers" in visible
    assert "resolve_citation" in visible


def test_visible_tool_specs_guided_profile_exposes_only_guided_tools() -> None:
    visible = {
        spec.name
        for spec in iter_visible_tool_specs(
            tool_profile="guided",
            hide_disabled_tools=True,
            enabled_flags={
                "enable_core": False,
                "enable_semantic_scholar": False,
                "enable_arxiv": False,
                "enable_openalex": False,
                "enable_serpapi": False,
                "enable_scholarapi": False,
                "enable_crossref": False,
                "enable_unpaywall": False,
                "enable_ecos": False,
                "enable_federal_register": False,
                "enable_govinfo_cfr": False,
                "enable_agentic": False,
                "govinfo_available": False,
            },
        )
    }

    assert visible == {
        "research",
        "follow_up_research",
        "resolve_reference",
        "inspect_source",
        "get_runtime_status",
    }


@pytest.mark.asyncio
async def test_ecos_tools_route_through_recording_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ecos = RecordingEcosClient()
    monkeypatch.setattr(server, "ecos_client", ecos)
    monkeypatch.setattr(server, "enable_ecos", True)

    search_payload = _payload(
        await server.call_tool(
            "search_species_ecos",
            {"query": "California least tern"},
        )
    )
    profile_payload = _payload(
        await server.call_tool(
            "get_species_profile_ecos",
            {"species_id": "8104"},
        )
    )
    list_payload = _payload(
        await server.call_tool(
            "list_species_documents_ecos",
            {
                "species_id": "8104",
                "documentKinds": ["recovery_plan", "five_year_review"],
            },
        )
    )
    text_payload = _payload(
        await server.call_tool(
            "get_document_text_ecos",
            {
                "url": "https://ecosphere-documents-production-public.s3.amazonaws.com/sams/public_docs/species_nonpublish/30669.pdf"
            },
        )
    )

    assert ecos.calls == [
        (
            "search_species",
            {
                "query": "California least tern",
                "limit": 10,
                "match_mode": "auto",
            },
        ),
        ("get_species_profile", {"species_id": "8104"}),
        (
            "list_species_documents",
            {
                "species_id": "8104",
                "document_kinds": ["recovery_plan", "five_year_review"],
            },
        ),
        (
            "get_document_text",
            {
                "url": "https://ecosphere-documents-production-public.s3.amazonaws.com/sams/public_docs/species_nonpublish/30669.pdf"
            },
        ),
    ]
    assert search_payload["data"][0]["speciesId"] == "8104"
    assert profile_payload["species"]["speciesId"] == "8104"
    assert list_payload["data"][0]["documentKind"] == "five_year_review"
    assert text_payload["extractionStatus"] == "ok"
    _assert_additive_metadata(search_payload)
    _assert_additive_metadata(profile_payload)
    _assert_additive_metadata(list_payload)
    _assert_additive_metadata(text_payload)


@pytest.mark.asyncio
async def test_ecos_tools_raise_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "enable_ecos", False)

    with pytest.raises(ValueError, match="requires ECOS"):
        await server.call_tool(
            "search_species_ecos",
            {"query": "California least tern"},
        )


@pytest.mark.asyncio
async def test_search_papers_match_include_enrichment_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = RecordingSemanticClient()

    async def fake_match(**kwargs: dict) -> dict:
        fake_client.calls.append(("search_papers_match", kwargs))
        return {
            "paperId": "match-1",
            "title": "Best match",
            "year": 2024,
            "authors": [{"name": "Lead Author"}],
            "venue": "Journal of Tests",
            "matchFound": True,
            "matchStrategy": "exact_title",
        }

    class MatchingCrossrefClient(RecordingCrossrefClient):
        async def search_work(self, query: str) -> dict:
            self.calls.append(("search_work", {"query": query}))
            return {
                "doi": "10.1234/crossref-query",
                "title": "Best match",
                "authors": [{"name": "Lead Author"}],
                "venue": "Journal of Tests",
                "publisher": "Crossref Publisher",
                "publicationType": "journal-article",
                "publicationDate": "2024-05-01",
                "year": 2024,
                "url": "https://doi.org/10.1234/crossref-query",
                "citationCount": 7,
            }

    fake_client.search_papers_match = fake_match  # type: ignore[method-assign]
    crossref = MatchingCrossrefClient()
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
    assert enriched["enrichments"]["unpaywall"]["bestOaUrl"].endswith("10.1234/crossref-query")
    assert [call[0] for call in fake_client.calls] == [
        "search_papers_match",
        "search_papers_match",
        "get_paper_details",
    ]
    assert fake_client.calls[0][1]["query"] == "Best match"
    assert fake_client.calls[0][1]["fields"] is None
    assert fake_client.calls[1][1]["query"] == "Best match"
    assert fake_client.calls[1][1]["fields"] is None
    assert fake_client.calls[2] == (
        "get_paper_details",
        {
            "paper_id": "match-1",
            "fields": [
                "paperId",
                "title",
                "year",
                "authors",
                "venue",
                "publicationDate",
                "url",
                "externalIds",
            ],
        },
    )


@pytest.mark.asyncio
async def test_search_papers_match_include_enrichment_skips_untrusted_crossref_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = RecordingSemanticClient()

    async def fake_match(**kwargs: dict) -> dict:
        fake_client.calls.append(("search_papers_match", kwargs))
        return {
            "paperId": "match-1",
            "title": "Best match",
            "year": 2024,
            "authors": [{"name": "Lead Author"}],
            "venue": "Journal of Tests",
            "matchFound": True,
            "matchStrategy": "exact_title",
        }

    fake_client.search_papers_match = fake_match  # type: ignore[method-assign]
    crossref, unpaywall, _ = _install_recording_enrichment(monkeypatch)
    monkeypatch.setattr(server, "client", fake_client)

    enriched = _payload(
        await server.call_tool(
            "search_papers_match",
            {"query": "Best match", "includeEnrichment": True},
        )
    )

    assert "enrichments" not in enriched
    assert crossref.calls == [("search_work", {"query": "Best match"})]
    assert unpaywall.calls == []


@pytest.mark.asyncio
async def test_search_snippets_uses_search_papers_fallback_when_degraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FallbackSemanticClient(RecordingSemanticClient):
        async def search_snippets(self, **kwargs) -> dict:
            self.calls.append(("search_snippets", kwargs))
            return {
                "data": [],
                "degraded": True,
                "providerStatusCode": 400,
                "message": "Semantic Scholar snippet search could not serve this query.",
            }

        async def search_papers(self, **kwargs) -> dict:
            self.calls.append(("search_papers", kwargs))
            return {
                "total": 1,
                "offset": 0,
                "data": [
                    {
                        "paperId": "paper-1",
                        "title": "Anthropogenic noise effects on wildlife",
                        "year": 2015,
                        "url": "https://example.org/paper-1",
                        "abstract": (
                            "A synthesis of two decades of research documenting the effects of noise on wildlife."
                        ),
                    }
                ],
            }

    fake_client = FallbackSemanticClient()
    monkeypatch.setattr(server, "client", fake_client)

    payload = _payload(
        await server.call_tool(
            "search_snippets",
            {"query": '"anthropogenic noise wildlife review"', "limit": 3},
        )
    )

    assert payload["degraded"] is True
    assert payload["fallbackUsed"] == "search_papers"
    assert payload["providerStatusCode"] == 400
    assert payload["data"][0]["paper"]["paperId"] == "paper-1"
    assert payload["data"][0]["snippet"]["snippetKind"] == "fallback_paper_match"
    assert payload["data"][0]["snippet"]["section"] == "abstract"
    assert "two decades of research" in payload["data"][0]["snippet"]["text"]
    assert fake_client.calls == [
        (
            "search_snippets",
            {
                "query": '"anthropogenic noise wildlife review"',
                "limit": 3,
                "fields": None,
                "year": None,
                "publication_date_or_year": None,
                "fields_of_study": None,
                "min_citation_count": None,
                "venue": None,
            },
        ),
        (
            "search_papers",
            {
                "query": "anthropogenic noise wildlife review",
                "limit": 3,
                "fields": ["paperId", "title", "year", "url", "abstract"],
                "year": None,
                "publication_date_or_year": None,
                "fields_of_study": None,
                "min_citation_count": None,
                "venue": None,
            },
        ),
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
    crossref, unpaywall, _ = _install_recording_enrichment(monkeypatch)
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

    paper = _payload(await server.call_tool("get_paper_details_openalex", {"paper_id": "W42"}))
    author = _payload(await server.call_tool("get_author_info_openalex", {"author_id": "A42"}))
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
    assert decode_bulk_cursor(author_papers["pagination"]["nextCursor"]).provider == ("openalex")


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
    assert decode_bulk_cursor(entities["pagination"]["nextCursor"]).provider == ("openalex")
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
    assert decode_bulk_cursor(entity_papers["pagination"]["nextCursor"]).provider == ("openalex")


@pytest.mark.asyncio
async def test_scholarapi_tools_route_and_wrap_bulk_cursors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scholarapi = RecordingScholarApiClient()
    monkeypatch.setattr(server, "scholarapi_client", scholarapi)
    monkeypatch.setattr(server, "enable_scholarapi", True)

    search_payload = _payload(
        await server.call_tool(
            "search_papers_scholarapi",
            {"query": "graphene", "has_text": True},
        )
    )
    list_payload = _payload(
        await server.call_tool(
            "list_papers_scholarapi",
            {"query": "graphene", "has_pdf": True},
        )
    )
    text_payload = _payload(await server.call_tool("get_paper_text_scholarapi", {"paper_id": "sa-1"}))
    texts_payload = _payload(await server.call_tool("get_paper_texts_scholarapi", {"paper_ids": ["sa-1", "sa-2"]}))
    pdf_payload = _payload(await server.call_tool("get_paper_pdf_scholarapi", {"paper_id": "sa-1"}))

    search_cursor = decode_bulk_cursor(search_payload["pagination"]["nextCursor"])
    list_cursor = decode_bulk_cursor(list_payload["pagination"]["nextCursor"])

    assert scholarapi.calls == [
        (
            "search",
            {
                "query": "graphene",
                "limit": 10,
                "cursor": None,
                "indexed_after": None,
                "indexed_before": None,
                "published_after": None,
                "published_before": None,
                "has_text": True,
                "has_pdf": None,
            },
        ),
        (
            "list_papers",
            {
                "query": "graphene",
                "limit": 100,
                "cursor": None,
                "indexed_after": None,
                "indexed_before": None,
                "published_after": None,
                "published_before": None,
                "has_text": None,
                "has_pdf": True,
            },
        ),
        ("get_text", {"paper_id": "sa-1"}),
        ("get_texts", {"paper_ids": ["sa-1", "sa-2"]}),
        ("get_pdf", {"paper_id": "sa-1"}),
    ]
    assert search_cursor.provider == "scholarapi"
    assert search_cursor.tool == "search_papers_scholarapi"
    assert search_cursor.token == "sch-search-next"
    assert search_cursor.context_hash is not None
    assert list_cursor.provider == "scholarapi"
    assert list_cursor.tool == "list_papers_scholarapi"
    assert list_cursor.token == "2024-03-01T12:30:45.123Z"
    assert list_cursor.context_hash is not None
    assert "sorted by indexed_at" in list_payload["retrievalNote"]
    assert "search_papers_scholarapi" in list_payload["retrievalNote"]
    assert list_payload["agentHints"]["warnings"]
    assert search_payload["data"][0]["contentAccess"]["scholarapi"]["hasText"] is True
    assert list_payload["data"][0]["contentAccess"]["scholarapi"]["hasPdf"] is True
    assert text_payload["paperId"] == "ScholarAPI:sa-1"
    assert texts_payload["results"][1]["text"] is None
    assert pdf_payload["mimeType"] == "application/pdf"


@pytest.mark.asyncio
async def test_scholarapi_search_cursor_round_trips_and_rejects_cross_query_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scholarapi = RecordingScholarApiClient()
    monkeypatch.setattr(server, "scholarapi_client", scholarapi)
    monkeypatch.setattr(server, "enable_scholarapi", True)

    first_page = _payload(
        await server.call_tool(
            "search_papers_scholarapi",
            {"query": "graphene", "has_text": True},
        )
    )
    cursor = first_page["pagination"]["nextCursor"]

    await server.call_tool(
        "search_papers_scholarapi",
        {"query": "graphene", "has_text": True, "cursor": cursor},
    )

    assert scholarapi.calls[1] == (
        "search",
        {
            "query": "graphene",
            "limit": 10,
            "cursor": "sch-search-next",
            "indexed_after": None,
            "indexed_before": None,
            "published_after": None,
            "published_before": None,
            "has_text": True,
            "has_pdf": None,
        },
    )

    with pytest.raises(ValueError, match="INVALID_CURSOR") as exc_info:
        await server.call_tool(
            "search_papers_scholarapi",
            {"query": "boron nitride", "has_text": True, "cursor": cursor},
        )

    assert "different query context" in str(exc_info.value)
    assert len(scholarapi.calls) == 2


@pytest.mark.asyncio
async def test_scholarapi_list_cursor_round_trips_and_rejects_cross_filter_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scholarapi = RecordingScholarApiClient()
    monkeypatch.setattr(server, "scholarapi_client", scholarapi)
    monkeypatch.setattr(server, "enable_scholarapi", True)

    first_page = _payload(
        await server.call_tool(
            "list_papers_scholarapi",
            {"query": "graphene", "has_pdf": True},
        )
    )
    cursor = first_page["pagination"]["nextCursor"]

    await server.call_tool(
        "list_papers_scholarapi",
        {"query": "graphene", "has_pdf": True, "cursor": cursor},
    )

    assert scholarapi.calls[1] == (
        "list_papers",
        {
            "query": "graphene",
            "limit": 100,
            "cursor": "2024-03-01T12:30:45.123Z",
            "indexed_after": None,
            "indexed_before": None,
            "published_after": None,
            "published_before": None,
            "has_text": None,
            "has_pdf": True,
        },
    )

    with pytest.raises(ValueError, match="INVALID_CURSOR") as exc_info:
        await server.call_tool(
            "list_papers_scholarapi",
            {"query": "graphene", "has_pdf": False, "cursor": cursor},
        )

    assert "different query context" in str(exc_info.value)
    assert len(scholarapi.calls) == 2


@pytest.mark.asyncio
async def test_scholarapi_tools_require_explicit_enablement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "enable_scholarapi", False)

    with pytest.raises(ValueError, match="PAPER_CHASER_ENABLE_SCHOLARAPI"):
        await server.call_tool("search_papers_scholarapi", {"query": "graphene"})

    with pytest.raises(ValueError, match="PAPER_CHASER_ENABLE_SCHOLARAPI"):
        await server.call_tool("list_papers_scholarapi", {})

    with pytest.raises(ValueError, match="PAPER_CHASER_ENABLE_SCHOLARAPI"):
        await server.call_tool("get_paper_text_scholarapi", {"paper_id": "sa-1"})


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
    assert decode_cursor(cited_by["pagination"]["nextCursor"]).provider == ("serpapi_google_scholar")
    assert versions["data"][0]["paperId"] == "serp-version-1"
    assert decode_cursor(versions["pagination"]["nextCursor"]).provider == ("serpapi_google_scholar")
    assert author["authorId"] == "author-1"
    assert articles["authorId"] == "author-1"
    assert decode_cursor(articles["pagination"]["nextCursor"]).provider == ("serpapi_google_scholar")
    assert account["provider"] == "serpapi_google_scholar"
    provider_map = {item["provider"]: item for item in diagnostics["providers"] if isinstance(item, dict)}
    assert provider_map["openai"]["enabled"] is False
    assert provider_map["azure-openai"]["enabled"] is False
    assert provider_map["anthropic"]["enabled"] is False
    assert provider_map["nvidia"]["enabled"] is False
    assert provider_map["google"]["enabled"] is False
    assert provider_map["huggingface"]["enabled"] is False
    assert "huggingface" in diagnostics["providerOrder"]


@pytest.mark.asyncio
async def test_provider_diagnostics_can_surface_non_openai_smart_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeRuntime:
        def smart_provider_diagnostics(self) -> tuple[dict[str, bool], list[str]]:
            return (
                {
                    "openai": False,
                    "azure-openai": False,
                    "anthropic": False,
                    "nvidia": False,
                    "google": False,
                    "huggingface": True,
                },
                ["huggingface", "openai", "azure-openai", "anthropic", "nvidia", "google"],
            )

    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime())

    diagnostics = _payload(await server.call_tool("get_provider_diagnostics", {}))
    provider_map = {item["provider"]: item for item in diagnostics["providers"] if isinstance(item, dict)}

    assert provider_map["huggingface"]["enabled"] is True
    assert provider_map["huggingface"]["paywalled"] is True
    assert provider_map["anthropic"]["enabled"] is False
    assert provider_map["nvidia"]["enabled"] is False
    assert provider_map["openai"]["enabled"] is False
    assert diagnostics["providerOrder"].index("huggingface") < diagnostics["providerOrder"].index("openai")
    runtime = diagnostics["runtimeSummary"]
    assert runtime["smartLayerEnabled"] is True
    assert runtime["transportMode"] == server.settings.transport
    assert runtime["toolsHidden"] == server.settings.hide_disabled_tools
    assert runtime["embeddingsEnabled"] is (not server.settings.disable_embeddings)
    assert "stdio" in runtime["transportMode"]


@pytest.mark.asyncio
async def test_provider_diagnostics_runtime_summary_warns_on_narrow_provider_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "provider_order", ["semantic_scholar"])

    diagnostics = _payload(await server.call_tool("get_provider_diagnostics", {}))
    runtime = diagnostics["runtimeSummary"]

    assert runtime["providerOrderEffective"] == ["semantic_scholar"]
    assert runtime["activeProviderSet"]
    assert any("narrow" in warning.lower() for warning in runtime["warnings"])


@pytest.mark.asyncio
async def test_provider_diagnostics_runtime_summary_warns_on_hidden_tools_and_tls_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    monkeypatch.setattr(server, "hide_disabled_tools", True)
    monkeypatch.setattr(server, "ecos_client", server.ecos_client or types.SimpleNamespace())
    monkeypatch.setattr(server.ecos_client, "verify_tls", False, raising=False)

    diagnostics = _payload(await server.call_tool("get_provider_diagnostics", {}))
    warnings = diagnostics["runtimeSummary"]["warnings"]

    assert any("hidden" in warning.lower() for warning in warnings)
    assert any("guided profile is active" in warning.lower() for warning in warnings)
    assert any("tls verification is disabled" in warning.lower() for warning in warnings)


def test_guided_normalize_follow_up_arguments_warns_on_ambiguous_session_and_empty_question() -> None:
    isolated_registry = type(server.workspace_registry)()
    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-a",
        query="Attention Is All You Need",
        payload={"sources": [{"sourceId": "paper-a", "title": "Attention is All you Need"}]},
    )
    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-b",
        query="RAG",
        payload={"sources": [{"sourceId": "paper-b", "title": "RAG Overview"}]},
    )

    normalized, meta = dispatch_module._guided_normalize_follow_up_arguments(
        {"query": "   "},
        workspace_registry=isolated_registry,
    )

    assert not normalized["searchSessionId"]
    assert normalized["question"] == ""
    assert any("multiple active sessions exist" in warning.lower() for warning in meta["warnings"])
    assert any("question was empty" in warning.lower() for warning in meta["warnings"])


def test_guided_normalize_inspect_arguments_repairs_invalid_session_and_missing_source() -> None:
    isolated_registry = type(server.workspace_registry)()
    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-only-source",
        query="Attention Is All You Need",
        payload={"sources": [{"sourceId": "paper-a", "title": "Attention is All you Need"}]},
    )

    normalized, meta = dispatch_module._guided_normalize_inspect_arguments(
        {"searchSessionId": "missing-session", "sourceId": "   "},
        workspace_registry=isolated_registry,
    )

    assert normalized["searchSessionId"] == "ssn-only-source"
    assert normalized["sourceId"] == "paper-a"
    assert any("using active session" in warning.lower() for warning in meta["warnings"])
    assert any("inferred the only inspectable source" in warning.lower() for warning in meta["warnings"])


def test_guided_normalize_follow_up_arguments_repairs_invalid_session_to_unique_active_session() -> None:
    isolated_registry = type(server.workspace_registry)()
    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-only-follow-up",
        query="Attention Is All You Need",
        payload={"sources": [{"sourceId": "paper-a", "title": "Attention is All you Need"}]},
    )

    normalized, meta = dispatch_module._guided_normalize_follow_up_arguments(
        {"searchSessionId": "missing-session", "question": "Who wrote it?"},
        workspace_registry=isolated_registry,
    )

    assert normalized["searchSessionId"] == "ssn-only-follow-up"
    assert any("was unavailable; using active session" in warning.lower() for warning in meta["warnings"])


def test_guided_normalize_inspect_arguments_infers_source_bearing_session_from_source_id() -> None:
    isolated_registry = type(server.workspace_registry)()
    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-source-bearing",
        query="Attention Is All You Need",
        payload={"sources": [{"sourceId": "paper-a", "title": "Attention is All you Need"}]},
    )
    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-other",
        query="RAG",
        payload={"sources": [{"sourceId": "paper-b", "title": "RAG Overview"}]},
    )

    normalized, meta = dispatch_module._guided_normalize_inspect_arguments(
        {"sourceId": "paper-a"},
        workspace_registry=isolated_registry,
    )

    assert normalized["searchSessionId"] == "ssn-source-bearing"
    assert normalized["sourceId"] == "paper-a"
    assert any("inferred source-bearing session" in warning.lower() for warning in meta["warnings"])


def test_guided_source_metadata_answers_support_alias_and_multiple_fields() -> None:
    sources: list[dict[str, Any]] = [
        {
            "sourceId": "paper-a",
            "sourceAlias": "source-1",
            "title": "Attention is All you Need",
            "date": "2017",
            "citation": {
                "authors": ["Ashish Vaswani", "Noam Shazeer"],
                "journalOrPublisher": "Neural Information Processing Systems",
                "doi": "10.5555/3295222.3295349",
                "year": "2017",
            },
        },
        {
            "sourceId": "paper-b",
            "sourceAlias": "source-2",
            "title": "RAG Overview",
        },
    ]

    answers = dispatch_module._guided_source_metadata_answers(
        "For sourceId source-1, who wrote it, what venue, DOI, year, and title are listed?",
        sources,
    )

    joined = " ".join(answers)
    assert "Ashish Vaswani" in joined
    assert "Neural Information Processing Systems" in joined
    assert "10.5555/3295222.3295349" in joined
    assert "2017" in joined
    assert "Attention is All you Need" in joined


@pytest.mark.asyncio
async def test_get_runtime_status_surfaces_configured_and_active_smart_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeBundle:
        def selection_metadata(self) -> dict[str, object]:
            return {
                "configuredSmartProvider": "azure-openai",
                "activeSmartProvider": "deterministic",
                "plannerModel": "azure-openai:gpt-4o-mini",
                "plannerModelSource": "deterministic",
                "synthesisModel": "azure-openai:gpt-4o-mini",
                "synthesisModelSource": "deterministic",
            }

    class _FakeRuntime:
        _provider_bundle = _FakeBundle()

        @staticmethod
        def smart_provider_diagnostics() -> tuple[dict[str, bool], list[str]]:
            return (
                {
                    "openai": False,
                    "azure-openai": True,
                    "anthropic": False,
                    "nvidia": False,
                    "google": False,
                    "mistral": False,
                    "huggingface": False,
                },
                ["azure-openai", "openai", "anthropic", "nvidia", "google", "mistral", "huggingface"],
            )

    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime())

    payload = _payload(await server.call_tool("get_runtime_status", {}))

    assert payload["runtimeSummary"]["configuredSmartProvider"] == "azure-openai"
    assert payload["runtimeSummary"]["activeSmartProvider"] == "deterministic"
    assert payload["runtimeSummary"]["guidedPolicy"] == "quality_first"
    assert payload["runtimeSummary"]["guidedResearchLatencyProfile"] == server.settings.guided_research_latency_profile
    assert payload["runtimeSummary"]["guidedFollowUpLatencyProfile"] == server.settings.guided_follow_up_latency_profile
    assert payload["runtimeSummary"]["guidedAllowPaidProviders"] is server.settings.guided_allow_paid_providers
    assert payload["providers"]


@pytest.mark.asyncio
async def test_guided_research_escalates_once_when_initial_pass_is_too_weak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def search_papers_smart(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return {
                    "searchSessionId": "ssn-weak-pass",
                    "strategyMetadata": {
                        "intent": "review",
                        "configuredSmartProvider": "openai",
                        "activeSmartProvider": "deterministic",
                        "providerBudgetApplied": {"allowPaidProviders": True},
                    },
                    "structuredSources": [],
                    "candidateLeads": [],
                    "evidenceGaps": ["Initial pass did not recover trustworthy on-topic sources."],
                    "coverageSummary": {
                        "providersAttempted": ["semantic_scholar"],
                        "providersSucceeded": [],
                        "providersFailed": [],
                        "providersZeroResults": ["semantic_scholar"],
                        "likelyCompleteness": "unknown",
                        "searchMode": "literature_review",
                    },
                    "failureSummary": None,
                    "clarification": None,
                }
            return {
                "searchSessionId": "ssn-weak-pass",
                "strategyMetadata": {
                    "intent": "review",
                    "configuredSmartProvider": "openai",
                    "activeSmartProvider": "deterministic",
                    "providerBudgetApplied": {"allowPaidProviders": True},
                },
                "structuredSources": [
                    {
                        "sourceId": "paper-1",
                        "title": "Grounded review paper one",
                        "provider": "semantic_scholar",
                        "sourceType": "scholarly_article",
                        "verificationStatus": "verified_metadata",
                        "accessStatus": "abstract_only",
                        "topicalRelevance": "on_topic",
                        "confidence": "high",
                        "isPrimarySource": False,
                    },
                    {
                        "sourceId": "paper-2",
                        "title": "Grounded review paper two",
                        "provider": "openalex",
                        "sourceType": "scholarly_article",
                        "verificationStatus": "verified_metadata",
                        "accessStatus": "abstract_only",
                        "topicalRelevance": "on_topic",
                        "confidence": "high",
                        "isPrimarySource": False,
                    },
                ],
                "candidateLeads": [],
                "evidenceGaps": [],
                "coverageSummary": {
                    "providersAttempted": ["semantic_scholar", "openalex"],
                    "providersSucceeded": ["semantic_scholar", "openalex"],
                    "providersFailed": [],
                    "providersZeroResults": [],
                    "likelyCompleteness": "likely_complete",
                    "searchMode": "literature_review",
                },
                "failureSummary": None,
                "clarification": None,
            }

    runtime = _FakeRuntime()
    monkeypatch.setattr(server, "agentic_runtime", runtime)

    payload = _payload(await server.call_tool("research", {"query": "multimodal transformer alignment methods"}))

    assert [call["mode"] for call in runtime.calls] == ["auto", "review"]
    assert payload["executionProvenance"]["escalationAttempted"] is True
    assert payload["executionProvenance"]["passesRun"] == 2
    assert payload["executionProvenance"]["passModes"] == ["auto", "review"]
    assert payload["executionProvenance"]["latencyProfileApplied"] == server.settings.guided_research_latency_profile
    assert payload["status"] in {"succeeded", "partial"}


@pytest.mark.asyncio
async def test_follow_up_research_without_session_returns_session_candidates_and_policy_metadata() -> None:
    isolated_registry = type(server.workspace_registry)()
    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-a",
        query="Attention Is All You Need",
        payload={"sources": [{"sourceId": "paper-a", "title": "Attention is All you Need"}]},
    )
    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-b",
        query="RAG",
        payload={"sources": [{"sourceId": "paper-b", "title": "RAG Overview"}]},
    )

    original_registry = server.workspace_registry
    server.workspace_registry = isolated_registry
    try:
        payload = _payload(await server.call_tool("follow_up_research", {"question": "Who wrote it?"}))
    finally:
        server.workspace_registry = original_registry

    assert payload["answerStatus"] == "insufficient_evidence"
    assert payload["sessionResolution"]["resolutionMode"] == "ambiguous"
    assert len(payload["sessionResolution"]["candidates"]) == 2
    assert payload["executionProvenance"]["latencyProfileApplied"] == server.settings.guided_follow_up_latency_profile
    assert payload["abstentionDetails"]["category"] == "no_sources"


@pytest.mark.asyncio
async def test_follow_up_research_passes_server_owned_latency_profile_to_ask_result_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated_registry = type(server.workspace_registry)()
    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-follow-up",
        query="Attention Is All You Need",
        payload={"sources": [{"sourceId": "paper-a", "title": "Attention is All you Need"}]},
    )

    class _FakeRuntime:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def ask_result_set(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(kwargs)
            return {
                "answerStatus": "answered",
                "answer": "The saved result set supports a grounded answer.",
                "evidence": [],
                "unsupportedAsks": [],
                "followUpQuestions": [],
                "structuredSources": [
                    {
                        "sourceId": "paper-a",
                        "title": "Attention is All you Need",
                        "provider": "semantic_scholar",
                        "sourceType": "scholarly_article",
                        "verificationStatus": "verified_metadata",
                        "accessStatus": "abstract_only",
                        "topicalRelevance": "on_topic",
                        "confidence": "high",
                        "isPrimarySource": False,
                    }
                ],
                "candidateLeads": [],
                "evidenceGaps": [],
                "coverageSummary": {
                    "providersAttempted": ["semantic_scholar"],
                    "providersSucceeded": ["semantic_scholar"],
                    "providersFailed": [],
                    "providersZeroResults": [],
                    "likelyCompleteness": "likely_complete",
                    "searchMode": "grounded_follow_up",
                },
                "failureSummary": None,
            }

    runtime = _FakeRuntime()
    monkeypatch.setattr(server, "agentic_runtime", runtime)
    original_registry = server.workspace_registry
    server.workspace_registry = isolated_registry
    try:
        payload = _payload(
            await server.call_tool(
                "follow_up_research",
                {"searchSessionId": "ssn-follow-up", "question": "What does the saved session support?"},
            )
        )
    finally:
        server.workspace_registry = original_registry

    assert runtime.calls
    assert runtime.calls[0]["latency_profile"] == server.settings.guided_follow_up_latency_profile
    assert payload["executionProvenance"]["latencyProfileApplied"] == server.settings.guided_follow_up_latency_profile
    assert payload["answerStatus"] == "answered"


@pytest.mark.asyncio
async def test_inspect_source_returns_structured_disambiguation_when_session_is_ambiguous() -> None:
    isolated_registry = type(server.workspace_registry)()
    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-left",
        query="Attention Is All You Need",
        payload={"sources": [{"sourceId": "paper-shared", "title": "Attention is All you Need"}]},
    )
    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-right",
        query="RAG",
        payload={"sources": [{"sourceId": "paper-shared", "title": "RAG Overview"}]},
    )

    original_registry = server.workspace_registry
    server.workspace_registry = isolated_registry
    try:
        payload = _payload(await server.call_tool("inspect_source", {"sourceId": "paper-shared"}))
    finally:
        server.workspace_registry = original_registry

    assert payload["searchSessionId"] is None
    assert payload["source"] is None
    assert payload["sessionResolution"]["resolutionMode"] == "ambiguous"
    assert payload["sourceResolution"]["matchType"] == "missing_session_id"
    assert payload["resultState"]["status"] == "needs_disambiguation"


@pytest.mark.asyncio
async def test_inspect_source_returns_available_source_ids_when_source_is_unresolved() -> None:
    isolated_registry = type(server.workspace_registry)()
    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-source-list",
        query="Attention Is All You Need",
        payload={
            "sources": [
                {"sourceId": "paper-a", "title": "Attention is All you Need"},
                {"sourceId": "paper-b", "title": "The Annotated Transformer"},
            ]
        },
    )

    original_registry = server.workspace_registry
    server.workspace_registry = isolated_registry
    try:
        payload = _payload(
            await server.call_tool(
                "inspect_source",
                {"searchSessionId": "ssn-source-list", "sourceId": "missing-source"},
            )
        )
    finally:
        server.workspace_registry = original_registry

    assert payload["source"] is None
    assert payload["sourceResolution"]["matchType"] == "unresolved"
    assert set(payload["sourceResolution"]["availableSourceIds"]) == {"paper-a", "paper-b"}
    assert payload["resultState"]["status"] == "needs_disambiguation"


@pytest.mark.asyncio
async def test_resolve_reference_exact_identifier_reports_resolved_status(monkeypatch: pytest.MonkeyPatch) -> None:
    semantic = RecordingSemanticClient()
    monkeypatch.setattr(server, "client", semantic)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_openalex", False)

    payload = _payload(await server.call_tool("resolve_reference", {"reference": "10.1038/nrn3241"}))

    assert payload["status"] == "resolved"
    assert payload["resolutionType"] == "paper_identifier"
    assert payload["resolutionConfidence"] == "high"
    assert payload["bestMatch"]["paper"]["paperId"] == "DOI:10.1038/nrn3241"


@pytest.mark.asyncio
async def test_resolve_reference_arxiv_identifier_reports_resolved_status(monkeypatch: pytest.MonkeyPatch) -> None:
    semantic = RecordingSemanticClient()
    monkeypatch.setattr(server, "client", semantic)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_openalex", False)

    payload = _payload(await server.call_tool("resolve_reference", {"reference": "arXiv:1706.03762"}))

    assert payload["status"] == "resolved"
    assert payload["resolutionType"] == "paper_identifier"
    assert payload["resolutionConfidence"] == "high"
    assert payload["bestMatch"]["paper"]["paperId"] == "ARXIV:1706.03762"


@pytest.mark.asyncio
async def test_resolve_reference_semantic_scholar_url_reports_resolved_status(monkeypatch: pytest.MonkeyPatch) -> None:
    semantic = RecordingSemanticClient()
    monkeypatch.setattr(server, "client", semantic)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_openalex", False)

    payload = _payload(
        await server.call_tool(
            "resolve_reference",
            {
                "reference": (
                    "https://www.semanticscholar.org/paper/Attention-Is-All-You-Need/"
                    "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
                )
            },
        )
    )

    assert payload["status"] == "resolved"
    assert payload["resolutionType"] == "paper_identifier"
    assert payload["resolutionConfidence"] == "high"
    assert payload["bestMatch"]["paper"]["paperId"] == "204e3073870fae3d05bcbc2f6a8e263d9b72e776"


@pytest.mark.asyncio
async def test_resolve_reference_title_only_match_with_conflicting_author_year_venue_is_needs_disambiguation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: resolve_reference must not report 'resolved' when author/year/venue all conflict."""

    class ConflictingExactTitleClient(RecordingSemanticClient):
        async def search_papers_match(self, **kwargs: object) -> dict:
            self.calls.append(("search_papers_match", dict(kwargs)))
            return {
                "paperId": "planetary-2021",
                "title": "Planetary Boundaries",
                "year": 2021,
                "venue": "Some Journal 2021",
                "authors": [{"name": "Vincent Bellinkx"}],
                "matchFound": True,
                "matchStrategy": "exact_title",
                "matchConfidence": "high",
                "matchedFields": ["title"],
                "conflictingFields": ["author", "year", "venue"],
                "candidateCount": 1,
            }

        async def search_snippets(self, **kwargs: object) -> dict:
            self.calls.append(("search_snippets", dict(kwargs)))
            return {"data": []}

        async def search_papers(self, **kwargs: object) -> dict:
            self.calls.append(("search_papers", dict(kwargs)))
            return {"total": 0, "offset": 0, "data": []}

    semantic = ConflictingExactTitleClient()
    monkeypatch.setattr(server, "client", semantic)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_openalex", False)
    monkeypatch.setattr(server, "enable_core", False)
    monkeypatch.setattr(server, "enable_arxiv", False)
    monkeypatch.setattr(server, "enable_serpapi", False)

    payload = _payload(
        await server.call_tool(
            "resolve_reference",
            {"reference": "Rockstrom et al planetary boundaries 2009 Nature 461 472"},
        )
    )

    assert payload["status"] == "needs_disambiguation", (
        "title-only match with conflicting author/year/venue must not be 'resolved'"
    )
    assert payload["resolutionConfidence"] != "high", (
        "title-only match with 3 conflicting key fields must not be high confidence"
    )
    # next actions must guide toward research or manual disambiguation
    assert any("research" in action.lower() or "conflict" in action.lower() for action in payload["nextActions"])

    class _FakeRuntime:
        async def search_papers_smart(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {
                "searchSessionId": "ssn-guided-1",
                "strategyMetadata": {"intent": "regulatory"},
                "nextStepHint": "Inspect the primary source first.",
                "structuredSources": [
                    {
                        "sourceId": "condor-fr",
                        "title": "Critical Habitat Revision for California Condor",
                        "provider": "federal_register",
                        "sourceType": "primary_regulatory",
                        "verificationStatus": "verified_metadata",
                        "accessStatus": "full_text_verified",
                        "topicalRelevance": "on_topic",
                        "confidence": "high",
                        "isPrimarySource": True,
                    }
                ],
                "candidateLeads": [
                    {
                        "sourceId": "polar-bear-fr",
                        "title": "Designation of Critical Habitat for Polar Bear",
                        "provider": "federal_register",
                        "sourceType": "primary_regulatory",
                        "verificationStatus": "verified_metadata",
                        "accessStatus": "full_text_verified",
                        "topicalRelevance": "off_topic",
                        "confidence": "medium",
                        "isPrimarySource": True,
                        "note": "Filtered from verified timeline.",
                    }
                ],
                "evidenceGaps": [],
                "coverageSummary": {"searchMode": "regulatory_primary_source"},
                "failureSummary": None,
                "clarification": None,
                "regulatoryTimeline": {"events": []},
            }

        async def ask_result_set(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {
                "searchSessionId": "ssn-guided-1",
                "answerStatus": "abstained",
                "answer": None,
                "evidence": [],
                "unsupportedAsks": ["What evaluation tradeoffs show up here?"],
                "followUpQuestions": [],
                "structuredSources": [],
                "candidateLeads": [
                    {
                        "sourceId": "lead-1",
                        "title": "Materials Design with OpenFOAM",
                        "provider": "semantic_scholar",
                        "sourceType": "scholarly_article",
                        "verificationStatus": "verified_metadata",
                        "accessStatus": "access_unverified",
                        "topicalRelevance": "off_topic",
                        "confidence": "medium",
                        "isPrimarySource": False,
                    }
                ],
                "evidenceGaps": ["Grounded follow-up abstained."],
                "coverageSummary": {"searchMode": "grounded_follow_up"},
                "failureSummary": None,
            }

    isolated_registry = type(server.workspace_registry)()
    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-guided-1",
        query="regulatory history of california condor",
        payload={"searchSessionId": "ssn-guided-1", "sources": []},
    )

    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime())
    monkeypatch.setattr(server, "workspace_registry", isolated_registry)

    research = _payload(await server.call_tool("research", {"query": "regulatory history of california condor"}))
    follow_up = _payload(
        await server.call_tool(
            "follow_up_research",
            {
                "searchSessionId": "ssn-guided-1",
                "question": "What evaluation tradeoffs show up here?",
            },
        )
    )

    assert research["unverifiedLeads"][0]["sourceId"] == "polar-bear-fr"
    assert research["unverifiedLeads"][0]["topicalRelevance"] == "off_topic"
    assert research["verifiedFindings"]
    assert "failureSummary" in research
    assert "resultMeaning" in research
    assert "candidateLeads" not in research
    assert "findings" not in research
    assert "failure" not in research
    assert follow_up["unverifiedLeads"][0]["sourceId"] == "lead-1"
    assert follow_up["unverifiedLeads"][0]["topicalRelevance"] == "off_topic"
    assert "failureSummary" in follow_up


@pytest.mark.asyncio
async def test_guided_research_blends_regulatory_and_literature_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeRuntime:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def search_papers_smart(self, **kwargs: object) -> dict[str, object]:
            mode = str(kwargs.get("mode") or "auto")
            self.calls.append(mode)
            if mode == "regulatory":
                return {
                    "searchSessionId": "ssn-mixed-guided",
                    "strategyMetadata": {"intent": "regulatory"},
                    "structuredSources": [
                        {
                            "sourceId": "nleb-fr",
                            "title": "Endangered Species Status for the Northern Long-Eared Bat",
                            "provider": "federal_register",
                            "sourceType": "primary_regulatory",
                            "verificationStatus": "verified_metadata",
                            "accessStatus": "full_text_verified",
                            "topicalRelevance": "on_topic",
                            "confidence": "high",
                            "isPrimarySource": True,
                        }
                    ],
                    "candidateLeads": [],
                    "evidenceGaps": [],
                    "coverageSummary": {
                        "providersAttempted": ["ecos", "federal_register"],
                        "providersSucceeded": ["federal_register"],
                        "providersFailed": [],
                        "providersZeroResults": ["ecos"],
                        "likelyCompleteness": "partial",
                        "searchMode": "regulatory_primary_source",
                    },
                    "failureSummary": None,
                    "clarification": None,
                    "regulatoryTimeline": {"events": [{"title": "NLEB rule"}]},
                }
            if mode == "review":
                return {
                    "searchSessionId": "ssn-mixed-guided",
                    "strategyMetadata": {"intent": "review"},
                    "structuredSources": [
                        {
                            "sourceId": "nleb-paper",
                            "title": "Recent peer-reviewed review of northern long-eared bat conservation",
                            "provider": "semantic_scholar",
                            "sourceType": "scholarly_article",
                            "verificationStatus": "verified_metadata",
                            "accessStatus": "abstract_only",
                            "topicalRelevance": "on_topic",
                            "confidence": "medium",
                            "isPrimarySource": False,
                        }
                    ],
                    "candidateLeads": [],
                    "evidenceGaps": ["Literature coverage may still be incomplete."],
                    "coverageSummary": {
                        "providersAttempted": ["semantic_scholar", "openalex"],
                        "providersSucceeded": ["semantic_scholar"],
                        "providersFailed": [],
                        "providersZeroResults": ["openalex"],
                        "likelyCompleteness": "partial",
                        "searchMode": "smart_literature_review",
                    },
                    "failureSummary": None,
                    "clarification": None,
                }
            raise AssertionError(f"Unexpected mode: {mode}")

    runtime = _FakeRuntime()
    monkeypatch.setattr(server, "agentic_runtime", runtime)

    payload = _payload(
        await server.call_tool(
            "research",
            {
                "query": (
                    "Summarize recent peer-reviewed literature and regulatory history for the northern "
                    "long-eared bat under the ESA."
                )
            },
        )
    )

    assert runtime.calls == ["regulatory", "review"]
    assert payload["intent"] == "mixed"
    assert payload["coverage"]["searchMode"] == "guided_hybrid_research"
    assert len(payload["sources"]) == 2
    assert any(source["sourceId"] == "nleb-fr" for source in payload["sources"])
    assert any(source["sourceId"] == "nleb-paper" for source in payload["sources"])
    assert any("inspect_source" in action for action in payload["nextActions"])
    assert payload["regulatoryTimeline"]["events"][0]["title"] == "NLEB rule"


@pytest.mark.asyncio
async def test_guided_research_abstained_without_sources_omits_inspect_source(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeRuntime:
        async def search_papers_smart(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {
                "searchSessionId": "ssn-no-sources",
                "strategyMetadata": {"intent": "regulatory"},
                "structuredSources": [],
                "candidateLeads": [],
                "evidenceGaps": [
                    (
                        "No primary-source regulatory timeline could be reconstructed from the currently "
                        "enabled regulatory providers."
                    )
                ],
                "coverageSummary": {
                    "providersAttempted": ["ecos", "federal_register"],
                    "providersSucceeded": [],
                    "providersFailed": [],
                    "providersZeroResults": ["ecos", "federal_register"],
                    "likelyCompleteness": "unknown",
                    "searchMode": "regulatory_primary_source",
                },
                "failureSummary": None,
                "clarification": None,
                "regulatoryTimeline": {"events": []},
            }

    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime())

    payload = _payload(
        await server.call_tool(
            "research",
            {"query": "Northern long-eared bat ESA listing status and Federal Register history"},
        )
    )

    assert payload["status"] == "abstained"
    assert payload["sources"] == []
    assert not any("inspect_source" in action for action in payload["nextActions"])
    assert any("follow_up_research" in action for action in payload["nextActions"])


@pytest.mark.asyncio
async def test_inspect_source_surfaces_guided_v2_source_fields() -> None:
    record = server.workspace_registry.save_result_set(
        source_tool="research",
        search_session_id="ssn-guided-inspect",
        query="test query",
        payload={
            "structuredSources": [
                {
                    "sourceId": "src-1",
                    "title": "Habitat Connectivity for Listed Species",
                    "provider": "arxiv",
                    "sourceType": "repository_record",
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "abstract_only",
                    "topicalRelevance": "on_topic",
                    "confidence": "medium",
                    "isPrimarySource": False,
                    "canonicalUrl": "https://arxiv.org/abs/2401.12345",
                    "retrievedUrl": "https://arxiv.org/abs/2401.12345",
                    "fullTextObserved": False,
                    "abstractObserved": True,
                    "openAccessRoute": "repository_open_access",
                    "citationText": "arXiv:2401.12345",
                    "citation": {
                        "authors": ["Ada Lovelace"],
                        "year": "2024",
                        "title": "Habitat Connectivity for Listed Species",
                        "journalOrPublisher": "arXiv",
                        "doi": None,
                        "url": "https://arxiv.org/abs/2401.12345",
                        "sourceType": "repository_record",
                        "confidence": "medium",
                    },
                }
            ]
        },
    )
    assert record.search_session_id == "ssn-guided-inspect"

    payload = _payload(
        await server.call_tool(
            "inspect_source",
            {"searchSessionId": "ssn-guided-inspect", "sourceId": "src-1"},
        )
    )

    source = payload["source"]
    assert source["fullTextObserved"] is False
    assert source["abstractObserved"] is True
    assert source["openAccessRoute"] == "repository_open_access"
    assert source["citationText"] == "arXiv:2401.12345"
    assert source["citation"]["authors"] == ["Ada Lovelace"]
    assert all("get_paper_" not in item for item in payload["directReadRecommendations"])


@pytest.mark.asyncio
async def test_follow_up_research_answers_from_saved_session_metadata_when_regulatory_session_abstained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        async def ask_result_set(self, **kwargs: object) -> dict[str, object]:
            raise AssertionError(f"ask_result_set should not run for session introspection: {kwargs!r}")

    record = server.workspace_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-follow-up-regulatory",
        query="Regulatory history of the gray wolf delisting in the contiguous United States",
        payload={
            "searchSessionId": "ssn-follow-up-regulatory",
            "structuredSources": [],
            "candidateLeads": [
                {
                    "sourceId": "2026-05678",
                    "title": (
                        "Endangered and Threatened Wildlife and Plants; Designation of Critical "
                        "Habitat for 22 Species in the Commonwealth of the Northern Mariana Islands "
                        "and the Territory of Guam"
                    ),
                    "provider": "federal_register",
                    "sourceType": "primary_regulatory",
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "access_unverified",
                    "topicalRelevance": "off_topic",
                    "confidence": "medium",
                    "isPrimarySource": True,
                    "canonicalUrl": "https://www.federalregister.gov/documents/2026/03/24/2026-05678/example",
                    "retrievedUrl": "https://www.federalregister.gov/documents/2026/03/24/2026-05678/example",
                    "fullTextObserved": False,
                    "abstractObserved": False,
                    "openAccessRoute": "non_oa_or_unconfirmed",
                    "citationText": "2026-05678",
                    "note": "Filtered from verified regulatory timeline because it did not match the anchored subject.",
                }
            ],
            "evidenceGaps": [
                (
                    "No primary-source regulatory timeline could be reconstructed from the currently "
                    "enabled regulatory providers."
                ),
                "Filtered 5 Federal Register hit(s) that did not match the anchored regulatory subject.",
                "No ECOS species dossier match was found for the query.",
            ],
            "coverageSummary": {
                "providersAttempted": ["ecos", "federal_register"],
                "providersSucceeded": ["federal_register"],
                "providersFailed": [],
                "providersZeroResults": ["ecos"],
                "likelyCompleteness": "unknown",
                "searchMode": "regulatory_primary_source",
                "summaryLine": (
                    "2 provider(s) searched, 0 failed, 1 returned zero results, likely completeness: unknown."
                ),
            },
            "failureSummary": {
                "outcome": "no_failure",
                "whatStillWorked": (
                    "No provider failures were recorded, but the evidence was not strong enough to ground a result."
                ),
                "fallbackAttempted": False,
                "completenessImpact": (
                    "No primary-source regulatory timeline could be reconstructed from the currently "
                    "enabled regulatory providers."
                ),
                "recommendedNextAction": "research",
            },
        },
    )
    assert record.search_session_id == "ssn-follow-up-regulatory"
    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime())

    payload = _payload(
        await server.call_tool(
            "follow_up_research",
            {
                "searchSessionId": "ssn-follow-up-regulatory",
                "question": (
                    "Which providers were searched, and what specific evidence gap prevented a grounded answer?"
                ),
            },
        )
    )

    assert payload["answerStatus"] == "answered"
    assert "ecos, federal_register" in payload["answer"]
    assert "No primary-source regulatory timeline could be reconstructed" in payload["answer"]
    assert payload["coverage"]["searchMode"] == "regulatory_primary_source"
    assert payload["evidenceGaps"][0].startswith("No primary-source regulatory timeline")
    assert payload["unverifiedLeads"][0]["sourceId"] == "2026-05678"
    assert payload["resultState"]["status"] == "answered"
    assert payload["executionProvenance"]["executionMode"] == "session_introspection"


@pytest.mark.asyncio
async def test_follow_up_research_classifies_saved_sources_and_leads_for_relevance_triage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        async def ask_result_set(self, **kwargs: object) -> dict[str, object]:
            raise AssertionError(f"ask_result_set should not run for saved-session triage: {kwargs!r}")

    record = server.workspace_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-follow-up-triage",
        query="FDA AI/ML lifecycle guidance documents",
        payload={
            "searchSessionId": "ssn-follow-up-triage",
            "structuredSources": [
                {
                    "sourceId": "pcp-guidance",
                    "title": (
                        "Marketing Submission Recommendations for a Predetermined Change Control Plan for "
                        "Artificial Intelligence-Enabled Device Software Functions"
                    ),
                    "provider": "govinfo",
                    "sourceType": "primary_regulatory",
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "access_unverified",
                    "topicalRelevance": "on_topic",
                    "confidence": "high",
                    "isPrimarySource": True,
                },
                {
                    "sourceId": "framework-paper",
                    "title": (
                        "Proposed Regulatory Framework for Modifications to Artificial Intelligence/"
                        "Machine Learning-Based Software as a Medical Device"
                    ),
                    "provider": "federal_register",
                    "sourceType": "primary_regulatory",
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "access_unverified",
                    "topicalRelevance": "on_topic",
                    "confidence": "medium",
                    "isPrimarySource": True,
                },
            ],
            "candidateLeads": [
                {
                    "sourceId": "cds-guidance",
                    "title": "Clinical Decision Support Software Guidance for Industry and FDA Staff",
                    "provider": "govinfo",
                    "sourceType": "primary_regulatory",
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "access_unverified",
                    "topicalRelevance": "off_topic",
                    "confidence": "medium",
                    "isPrimarySource": True,
                    "note": (
                        "Filtered from grounded evidence because it did not match the requested AI/ML lifecycle scope."
                    ),
                }
            ],
            "evidenceGaps": ["Some returned agency records were filtered as off-topic to preserve precision."],
            "coverageSummary": {
                "providersAttempted": ["govinfo", "federal_register"],
                "providersSucceeded": ["govinfo", "federal_register"],
                "providersFailed": [],
                "providersZeroResults": [],
                "likelyCompleteness": "partial",
                "searchMode": "regulatory_primary_source",
            },
        },
    )
    assert record.search_session_id == "ssn-follow-up-triage"
    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime())

    payload = _payload(
        await server.call_tool(
            "follow_up_research",
            {
                "searchSessionId": "ssn-follow-up-triage",
                "question": (
                    "Which of these are actual FDA AI/ML lifecycle guidance documents, and which are off-target?"
                ),
            },
        )
    )

    assert payload["answerStatus"] == "answered"
    assert "Predetermined Change Control Plan" in payload["answer"]
    assert "Proposed Regulatory Framework" in payload["answer"]
    assert "Clinical Decision Support Software Guidance" in payload["answer"]
    assert "off-target" in payload["answer"].lower()
    assert payload["selectedLeadIds"] == ["cds-guidance"]
    assert payload["executionProvenance"]["executionMode"] == "session_introspection"


@pytest.mark.asyncio
async def test_guided_research_summary_leads_with_top_result(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeRuntime:
        async def search_papers_smart(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {
                "searchSessionId": "ssn-summary-guided",
                "strategyMetadata": {"intent": "regulatory"},
                "structuredSources": [
                    {
                        "sourceId": "pcp-guidance",
                        "title": (
                            "Marketing Submission Recommendations for a Predetermined Change Control Plan for "
                            "Artificial Intelligence-Enabled Device Software Functions"
                        ),
                        "provider": "govinfo",
                        "sourceType": "primary_regulatory",
                        "verificationStatus": "verified_metadata",
                        "accessStatus": "access_unverified",
                        "topicalRelevance": "on_topic",
                        "confidence": "high",
                        "isPrimarySource": True,
                    }
                ],
                "candidateLeads": [],
                "evidenceGaps": [],
                "coverageSummary": {"searchMode": "regulatory_primary_source"},
                "failureSummary": None,
                "clarification": None,
                "regulatoryTimeline": {"events": []},
            }

    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime())

    payload = _payload(
        await server.call_tool(
            "research",
            {"query": "recent FDA AI/ML lifecycle guidance documents"},
        )
    )

    assert payload["summary"].startswith("Top result:")
    assert "Predetermined Change Control Plan" in payload["summary"]


@pytest.mark.asyncio
async def test_follow_up_research_uses_question_shaped_answer_mode_over_saved_claim_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeRuntime:
        async def ask_result_set(self, **kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return {
                "searchSessionId": "ssn-follow-up-mode",
                "answerStatus": "answered",
                "answer": "Validation limitations were summarized from the saved evidence.",
                "evidence": [],
                "unsupportedAsks": [],
                "followUpQuestions": [],
                "structuredSources": [],
                "candidateLeads": [],
                "evidenceGaps": [],
                "coverageSummary": {"searchMode": "grounded_follow_up"},
                "failureSummary": None,
            }

    record = server.workspace_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-follow-up-mode",
        query="freshwater eDNA biodiversity monitoring",
        metadata={"strategyMetadata": {"followUpMode": "claim_check"}},
        payload={
            "searchSessionId": "ssn-follow-up-mode",
            "structuredSources": [
                {
                    "sourceId": "edna-review",
                    "title": "Freshwater eDNA validation for biodiversity monitoring",
                    "provider": "semantic_scholar",
                    "sourceType": "scholarly_article",
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "abstract_only",
                    "topicalRelevance": "on_topic",
                    "confidence": "medium",
                    "isPrimarySource": False,
                }
            ],
        },
    )
    assert record.search_session_id == "ssn-follow-up-mode"
    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime())

    payload = _payload(
        await server.call_tool(
            "follow_up_research",
            {
                "searchSessionId": "ssn-follow-up-mode",
                "question": "What limitations or validation issues appear most often in the returned evidence?",
            },
        )
    )

    assert payload["answerStatus"] == "answered"
    assert captured.get("answerMode", captured.get("answer_mode")) == "qa"


@pytest.mark.asyncio
async def test_follow_up_research_relevance_triage_answers_weak_matches_versus_off_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        async def ask_result_set(self, **kwargs: object) -> dict[str, object]:
            raise AssertionError(f"ask_result_set should not run for saved-session relevance triage: {kwargs!r}")

    record = server.workspace_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-follow-up-weak-vs-offtopic",
        query="remote sensing algal blooms freshwater lakes",
        payload={
            "searchSessionId": "ssn-follow-up-weak-vs-offtopic",
            "structuredSources": [
                {
                    "sourceId": "remote-review",
                    "title": "Remote sensing for mapping algal blooms in freshwater lakes: a review",
                    "provider": "semantic_scholar",
                    "sourceType": "scholarly_article",
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "abstract_only",
                    "topicalRelevance": "weak_match",
                    "confidence": "medium",
                    "isPrimarySource": False,
                }
            ],
            "candidateLeads": [
                {
                    "sourceId": "shellfish-safety",
                    "title": "Shellfish safety surveillance guidance",
                    "provider": "govinfo",
                    "sourceType": "primary_regulatory",
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "access_unverified",
                    "topicalRelevance": "off_topic",
                    "confidence": "medium",
                    "isPrimarySource": True,
                }
            ],
        },
    )
    assert record.search_session_id == "ssn-follow-up-weak-vs-offtopic"
    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime())

    payload = _payload(
        await server.call_tool(
            "follow_up_research",
            {
                "searchSessionId": "ssn-follow-up-weak-vs-offtopic",
                "question": (
                    "Which returned items were treated as weak matches versus off-topic, based on the saved evidence?"
                ),
            },
        )
    )

    assert payload["answerStatus"] == "answered"
    assert "Remote sensing for mapping algal blooms" in payload["answer"]
    assert "Shellfish safety surveillance guidance" in payload["answer"]
    assert "weaker" in payload["answer"].lower() or "weak" in payload["answer"].lower()


@pytest.mark.asyncio
async def test_follow_up_research_execution_provenance_uses_live_runtime_provider_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ProviderBundle:
        def selection_metadata(self) -> dict[str, Any]:
            return {
                "configuredSmartProvider": "openai",
                "activeSmartProvider": "deterministic",
                "plannerModel": "openai:deterministic-planner",
                "plannerModelSource": "deterministic",
                "synthesisModel": "openai:deterministic-synthesizer",
                "synthesisModelSource": "deterministic",
            }

    class _FakeRuntime:
        def __init__(self) -> None:
            self._provider_bundle = _ProviderBundle()

        async def ask_result_set(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {
                "searchSessionId": "ssn-follow-up-runtime-provider",
                "answerStatus": "answered",
                "answer": "Grounded answer.",
                "evidence": [],
                "unsupportedAsks": [],
                "followUpQuestions": [],
                "structuredSources": [],
                "candidateLeads": [],
                "evidenceGaps": [],
                "coverageSummary": {"searchMode": "grounded_follow_up"},
                "failureSummary": None,
            }

    record = server.workspace_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-follow-up-runtime-provider",
        query="environmental follow-up",
        metadata={"strategyMetadata": {"configuredSmartProvider": "openai", "activeSmartProvider": "openai"}},
        payload={"searchSessionId": "ssn-follow-up-runtime-provider", "structuredSources": []},
    )
    assert record.search_session_id == "ssn-follow-up-runtime-provider"
    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime())

    payload = _payload(
        await server.call_tool(
            "follow_up_research",
            {
                "searchSessionId": "ssn-follow-up-runtime-provider",
                "question": "What do the saved sources say?",
            },
        )
    )

    assert payload["executionProvenance"]["configuredSmartProvider"] == "openai"
    assert payload["executionProvenance"]["activeSmartProvider"] == "deterministic"


@pytest.mark.asyncio
async def test_follow_up_research_uses_saved_unverified_leads_for_source_overview_when_no_sources_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        async def ask_result_set(self, **kwargs: object) -> dict[str, object]:
            raise AssertionError(f"ask_result_set should not run for session source overview: {kwargs!r}")

    record = server.workspace_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-follow-up-leads",
        query="Regulatory history of the gray wolf delisting in the contiguous United States",
        payload={
            "searchSessionId": "ssn-follow-up-leads",
            "structuredSources": [],
            "candidateLeads": [
                {
                    "sourceId": "2026-05678",
                    "title": (
                        "Endangered and Threatened Wildlife and Plants; Designation of Critical "
                        "Habitat for 22 Species in the Commonwealth of the Northern Mariana Islands "
                        "and the Territory of Guam"
                    ),
                    "provider": "federal_register",
                    "sourceType": "primary_regulatory",
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "access_unverified",
                    "topicalRelevance": "off_topic",
                    "confidence": "medium",
                    "isPrimarySource": True,
                },
                {
                    "sourceId": "2025-19732",
                    "title": (
                        "Endangered and Threatened Wildlife and Plants; Review of Species That Are "
                        "Candidates for Listing as Endangered or Threatened; Annual Notification of "
                        "Findings on Resubmitted Petitions; Annual Description of Progress on "
                        "Listing Actions"
                    ),
                    "provider": "federal_register",
                    "sourceType": "primary_regulatory",
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "access_unverified",
                    "topicalRelevance": "off_topic",
                    "confidence": "medium",
                    "isPrimarySource": True,
                },
            ],
            "evidenceGaps": [
                (
                    "No primary-source regulatory timeline could be reconstructed from the currently "
                    "enabled regulatory providers."
                )
            ],
            "coverageSummary": {
                "providersAttempted": ["ecos", "federal_register"],
                "providersSucceeded": ["federal_register"],
                "providersZeroResults": ["ecos"],
                "likelyCompleteness": "unknown",
                "searchMode": "regulatory_primary_source",
            },
        },
    )
    assert record.search_session_id == "ssn-follow-up-leads"
    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime())

    payload = _payload(
        await server.call_tool(
            "follow_up_research",
            {
                "searchSessionId": "ssn-follow-up-leads",
                "question": "What records or source leads were found in this session?",
            },
        )
    )

    assert payload["answerStatus"] == "answered"
    assert payload["answer"].startswith("Saved source leads included:")
    assert "Designation of Critical Habitat for 22 Species" in payload["answer"]
    assert "Annual Notification of Findings on Resubmitted Petitions" in payload["answer"]


@pytest.mark.asyncio
async def test_follow_up_research_infers_single_active_session_when_search_session_id_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        async def ask_result_set(self, **kwargs: object) -> dict[str, object]:
            raise AssertionError(f"ask_result_set should not run for inferred session introspection: {kwargs!r}")

    isolated_registry = type(server.workspace_registry)()
    monkeypatch.setattr(server, "workspace_registry", isolated_registry)

    record = isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-inferred-follow-up",
        query="Tool-using agents for literature review",
        payload={
            "searchSessionId": "ssn-inferred-follow-up",
            "structuredSources": [
                {
                    "sourceId": "paper-1",
                    "title": "Tool-Using Agents for Literature Review",
                    "provider": "semantic_scholar",
                    "sourceType": "scholarly_article",
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "abstract_only",
                    "topicalRelevance": "on_topic",
                    "confidence": "high",
                    "canonicalUrl": "https://example.org/paper-1",
                }
            ],
            "coverageSummary": {"searchMode": "smart_literature_review"},
        },
    )
    assert record.search_session_id == "ssn-inferred-follow-up"
    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime())

    payload = _payload(
        await server.call_tool(
            "follow_up_research",
            {
                "question": "Which sources were found in this session?",
            },
        )
    )

    assert payload["searchSessionId"] == "ssn-inferred-follow-up"
    assert payload["answerStatus"] == "answered"
    assert payload["inputNormalization"]["warnings"]
    assert "inferred active session" in payload["inputNormalization"]["warnings"][0].lower()


@pytest.mark.asyncio
async def test_inspect_source_accepts_index_alias_without_search_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated_registry = type(server.workspace_registry)()
    monkeypatch.setattr(server, "workspace_registry", isolated_registry)

    record = isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-inspect-alias",
        query="California condor regulatory history",
        payload={
            "searchSessionId": "ssn-inspect-alias",
            "structuredSources": [
                {
                    "sourceId": "condor-fr",
                    "title": "Critical Habitat Revision for California Condor",
                    "provider": "federal_register",
                    "sourceType": "primary_regulatory",
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "full_text_verified",
                    "topicalRelevance": "on_topic",
                    "confidence": "high",
                    "canonicalUrl": "https://example.org/condor-fr",
                }
            ],
        },
    )
    assert record.search_session_id == "ssn-inspect-alias"

    payload = _payload(
        await server.call_tool(
            "inspect_source",
            {
                "sourceId": "source-1",
            },
        )
    )

    assert payload["searchSessionId"] == "ssn-inspect-alias"
    assert payload["source"]["sourceId"] == "condor-fr"
    assert payload["sourceResolution"]["matchType"] == "index_alias"


@pytest.mark.asyncio
async def test_follow_up_research_requires_explicit_session_when_multiple_are_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated_registry = type(server.workspace_registry)()
    monkeypatch.setattr(server, "workspace_registry", isolated_registry)

    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-follow-up-a",
        query="Attention Is All You Need",
        payload={"sources": [{"sourceId": "paper-a", "title": "Attention is All you Need"}]},
    )
    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-follow-up-b",
        query="Retrieval-Augmented Generation",
        payload={"sources": [{"sourceId": "paper-b", "title": "RAG Overview"}]},
    )

    payload = _payload(
        await server.call_tool(
            "follow_up_research",
            {
                "question": "What venue is listed for this paper?",
            },
        )
    )

    assert payload["answerStatus"] == "insufficient_evidence"
    assert payload["searchSessionId"] is None
    assert "unique saved search session" in payload["evidenceGaps"][0].lower()
    assert "multiple active sessions exist" in payload["inputNormalization"]["warnings"][0].lower()


@pytest.mark.asyncio
async def test_inspect_source_requires_explicit_session_when_multiple_are_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated_registry = type(server.workspace_registry)()
    monkeypatch.setattr(server, "workspace_registry", isolated_registry)

    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-inspect-a",
        query="Attention Is All You Need",
        payload={"sources": [{"sourceId": "paper-a", "title": "Attention is All you Need"}]},
    )
    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-inspect-b",
        query="Retrieval-Augmented Generation",
        payload={"sources": [{"sourceId": "paper-b", "title": "RAG Overview"}]},
    )

    payload = _payload(
        await server.call_tool(
            "inspect_source",
            {
                "sourceId": "source-1",
            },
        )
    )

    assert payload["searchSessionId"] is None
    assert payload["sessionResolution"]["resolutionMode"] == "ambiguous"
    assert payload["resultState"]["status"] == "needs_disambiguation"


@pytest.mark.asyncio
async def test_follow_up_research_answers_from_saved_session_metadata_when_partial_session_has_verified_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        async def ask_result_set(self, **kwargs: object) -> dict[str, object]:
            raise AssertionError(f"ask_result_set should not run for session introspection: {kwargs!r}")

    record = server.workspace_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-follow-up-partial",
        query="Recent overview of retrieval-augmented generation evaluation methods for scholarly QA",
        payload={
            "searchSessionId": "ssn-follow-up-partial",
            "verifiedFindings": ["Ragas: Automated Evaluation of Retrieval Augmented Generation"],
            "structuredSources": [
                {
                    "sourceId": "2309.15217",
                    "title": "Ragas: Automated Evaluation of Retrieval Augmented Generation",
                    "provider": "arxiv",
                    "sourceType": "repository_record",
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "full_text_verified",
                    "topicalRelevance": "on_topic",
                    "confidence": "medium",
                    "isPrimarySource": False,
                    "canonicalUrl": "https://arxiv.org/abs/2309.15217v2",
                    "retrievedUrl": "https://arxiv.org/pdf/2309.15217v2",
                    "fullTextObserved": True,
                    "abstractObserved": True,
                    "openAccessRoute": "repository_open_access",
                    "citationText": "2309.15217",
                    "note": "Ragas matched the overview request.",
                }
            ],
            "candidateLeads": [
                {
                    "sourceId": "2411.18583",
                    "title": (
                        "Automated Literature Review Using NLP Techniques and LLM-Based Retrieval-Augmented Generation"
                    ),
                    "provider": "arxiv",
                    "sourceType": "repository_record",
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "full_text_verified",
                    "topicalRelevance": "weak_match",
                    "confidence": "medium",
                    "isPrimarySource": False,
                    "canonicalUrl": "https://arxiv.org/abs/2411.18583v1",
                    "retrievedUrl": "https://arxiv.org/pdf/2411.18583v1",
                    "fullTextObserved": True,
                    "abstractObserved": True,
                    "openAccessRoute": "repository_open_access",
                    "citationText": "2411.18583",
                    "note": "Useful but weaker match.",
                }
            ],
            "evidenceGaps": [
                (
                    "Smart provider 'huggingface' fell back to deterministic mode; inspect "
                    "providerOutcomes before trusting planning or expansion quality."
                )
            ],
            "coverageSummary": {
                "providersAttempted": ["scholarapi", "semantic_scholar", "openalex", "arxiv", "core"],
                "providersSucceeded": ["arxiv", "openalex"],
                "providersFailed": ["core"],
                "providersZeroResults": ["semantic_scholar"],
                "likelyCompleteness": "partial",
                "searchMode": "smart_literature_review",
                "summaryLine": (
                    "5 provider(s) searched, 1 failed, 1 returned zero results, likely completeness: partial."
                ),
            },
            "failureSummary": {
                "outcome": "fallback_success",
                "whatFailed": "One or more smart-search providers or provider-side stages failed.",
                "whatStillWorked": "The smart workflow returned the strongest available partial result set.",
                "fallbackAttempted": True,
                "fallbackMode": "smart_provider_fallback",
                "primaryPathFailureReason": "core",
                "completenessImpact": "Coverage may be partial because these providers or stages failed: core.",
                "recommendedNextAction": "review_partial_results",
            },
        },
    )
    assert record.search_session_id == "ssn-follow-up-partial"
    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime())

    payload = _payload(
        await server.call_tool(
            "follow_up_research",
            {
                "searchSessionId": "ssn-follow-up-partial",
                "question": (
                    "Which provider failure made completeness partial, and what was the strongest verified finding?"
                ),
            },
        )
    )

    assert payload["answerStatus"] == "answered"
    assert "Failed providers: core." in payload["answer"]
    assert (
        "The strongest verified finding in the saved session was: Ragas: Automated Evaluation "
        "of Retrieval Augmented Generation."
    ) in payload["answer"]
    assert payload["failureSummary"]["primaryPathFailureReason"] == "core"
    assert payload["verifiedFindings"][0]["claim"] == "Ragas: Automated Evaluation of Retrieval Augmented Generation"


@pytest.mark.asyncio
async def test_follow_up_research_answers_authors_and_venue_from_saved_source_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        async def ask_result_set(self, **kwargs: object) -> dict[str, object]:
            raise AssertionError(f"ask_result_set should not run for source metadata introspection: {kwargs!r}")

    isolated_registry = type(server.workspace_registry)()
    monkeypatch.setattr(server, "workspace_registry", isolated_registry)
    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime())

    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-follow-up-source-metadata",
        query="Attention Is All You Need",
        payload={
            "searchSessionId": "ssn-follow-up-source-metadata",
            "sources": [
                {
                    "sourceId": "attention-paper",
                    "title": "Attention is All you Need",
                    "provider": "semantic_scholar",
                    "sourceType": "scholarly_article",
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "access_unverified",
                    "topicalRelevance": "on_topic",
                    "confidence": "high",
                    "citation": {
                        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
                        "year": "2017",
                        "title": "Attention is All you Need",
                        "journalOrPublisher": "Neural Information Processing Systems",
                        "doi": None,
                        "url": "https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776",
                        "sourceType": "scholarly_article",
                        "confidence": "medium",
                    },
                    "date": "2017-06-12",
                }
            ],
        },
    )

    payload = _payload(
        await server.call_tool(
            "follow_up_research",
            {
                "searchSessionId": "ssn-follow-up-source-metadata",
                "question": "Who are the authors and what venue is listed for this paper?",
            },
        )
    )

    assert payload["answerStatus"] == "answered"
    assert "Ashish Vaswani" in payload["answer"]
    assert "Neural Information Processing Systems" in payload["answer"]


@pytest.mark.asyncio
async def test_follow_up_research_does_not_treat_stopwords_as_source_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        async def ask_result_set(self, **kwargs: object) -> dict[str, object]:
            raise AssertionError(f"ask_result_set should not run for session source overview: {kwargs!r}")

    isolated_registry = type(server.workspace_registry)()
    monkeypatch.setattr(server, "workspace_registry", isolated_registry)
    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime())

    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-follow-up-source-overview",
        query="Northern long-eared bat ESA status",
        payload={
            "searchSessionId": "ssn-follow-up-source-overview",
            "structuredSources": [
                {
                    "sourceId": "50 CFR 17.11",
                    "title": "50 CFR 17.11 Endangered and threatened wildlife",
                    "provider": "govinfo",
                    "sourceType": "primary_regulatory",
                    "verificationStatus": "verified_primary_source",
                    "accessStatus": "full_text_verified",
                    "topicalRelevance": "on_topic",
                    "confidence": "high",
                    "isPrimarySource": True,
                },
                {
                    "sourceId": "2022-25998",
                    "title": "Endangered Species Status for the Northern Long-Eared Bat",
                    "provider": "federal_register",
                    "sourceType": "primary_regulatory",
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "access_unverified",
                    "topicalRelevance": "on_topic",
                    "confidence": "medium",
                    "isPrimarySource": True,
                },
            ],
        },
    )

    payload = _payload(
        await server.call_tool(
            "follow_up_research",
            {
                "searchSessionId": "ssn-follow-up-source-overview",
                "question": "Which source is the authoritative current-text source in this session?",
            },
        )
    )

    assert payload["answerStatus"] == "answered"
    assert "No saved source matched 'is'" not in payload["answer"]
    assert "Saved sources included:" in payload["answer"]
    assert payload["selectedEvidenceIds"] == ["50 CFR 17.11", "2022-25998"]
    assert payload["executionProvenance"]["answerSource"] == "saved_session_metadata"


@pytest.mark.asyncio
async def test_follow_up_research_uses_evidence_contract_records_when_sources_are_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        async def ask_result_set(self, **kwargs: object) -> dict[str, object]:
            raise AssertionError(f"ask_result_set should not run for evidence-only session introspection: {kwargs!r}")

    isolated_registry = type(server.workspace_registry)()
    monkeypatch.setattr(server, "workspace_registry", isolated_registry)
    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime())

    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-follow-up-evidence-only",
        query="50 CFR 17.11 northern long-eared bat current text",
        payload={
            "searchSessionId": "ssn-follow-up-evidence-only",
            "intent": "regulatory",
            "resultStatus": "succeeded",
            "answerability": "grounded",
            "routingSummary": {"intent": "regulatory", "anchorType": "cfr_citation"},
            "evidence": [
                {
                    "evidenceId": "50 CFR 17.11",
                    "title": "50 CFR 17.11 Endangered and threatened wildlife",
                    "provider": "govinfo",
                    "sourceType": "primary_regulatory",
                    "verificationStatus": "verified_primary_source",
                    "accessStatus": "full_text_verified",
                    "topicalRelevance": "on_topic",
                    "isPrimarySource": True,
                    "canonicalUrl": "https://www.govinfo.gov/example/cfr",
                    "whyIncluded": "Authoritative current text for this CFR section.",
                    "date": "2024-10-01",
                }
            ],
            "leads": [
                {
                    "evidenceId": "2022-25998",
                    "title": "Endangered Species Status for the Northern Long-Eared Bat",
                    "provider": "federal_register",
                    "sourceType": "primary_regulatory",
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "access_unverified",
                    "topicalRelevance": "weak_match",
                    "isPrimarySource": True,
                    "whyIncluded": "Related but not current codified text.",
                    "whyNotVerified": "Topical relevance was weak_match.",
                }
            ],
            "evidenceGaps": [],
            "coverageSummary": {
                "providersAttempted": ["govinfo", "federal_register"],
                "providersSucceeded": ["govinfo", "federal_register"],
                "providersFailed": [],
                "providersZeroResults": [],
                "searchMode": "regulatory_primary_source",
            },
        },
    )

    payload = _payload(
        await server.call_tool(
            "follow_up_research",
            {
                "searchSessionId": "ssn-follow-up-evidence-only",
                "question": "Which sources were found in this session?",
            },
        )
    )

    assert payload["answerStatus"] == "answered"
    assert payload["answer"].startswith("Saved sources included:")
    assert "50 CFR 17.11 Endangered and threatened wildlife" in payload["answer"]
    assert payload["selectedEvidenceIds"] == ["50 CFR 17.11"]
    assert payload["routingSummary"]["intent"] == "regulatory"
    assert payload["answerability"] == "grounded"


@pytest.mark.asyncio
async def test_inspect_source_reads_evidence_contract_records_without_legacy_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated_registry = type(server.workspace_registry)()
    monkeypatch.setattr(server, "workspace_registry", isolated_registry)

    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-inspect-evidence-only",
        query="50 CFR 17.11 northern long-eared bat current text",
        payload={
            "searchSessionId": "ssn-inspect-evidence-only",
            "intent": "regulatory",
            "routingSummary": {"intent": "regulatory", "anchorType": "cfr_citation"},
            "coverageSummary": {"searchMode": "regulatory_primary_source"},
            "evidence": [
                {
                    "evidenceId": "50 CFR 17.11",
                    "title": "50 CFR 17.11 Endangered and threatened wildlife",
                    "provider": "govinfo",
                    "sourceType": "primary_regulatory",
                    "verificationStatus": "verified_primary_source",
                    "accessStatus": "full_text_verified",
                    "topicalRelevance": "on_topic",
                    "isPrimarySource": True,
                    "canonicalUrl": "https://www.govinfo.gov/example/cfr",
                    "whyIncluded": "Authoritative current text for this CFR section.",
                }
            ],
            "leads": [],
            "evidenceGaps": [],
        },
    )

    payload = _payload(
        await server.call_tool(
            "inspect_source",
            {
                "searchSessionId": "ssn-inspect-evidence-only",
                "sourceId": "50 CFR 17.11",
            },
        )
    )

    assert payload["source"]["sourceId"] == "50 CFR 17.11"
    assert payload["evidenceId"] == "50 CFR 17.11"
    assert payload["selectedEvidenceIds"] == ["50 CFR 17.11"]
    assert payload["routingSummary"]["intent"] == "regulatory"
    assert payload["coverageSummary"]["searchMode"] == "regulatory_primary_source"
    assert payload["evidence"]
    assert payload["evidence"][0]["evidenceId"] == "50 CFR 17.11"


@pytest.mark.asyncio
async def test_follow_up_research_recovers_when_agentic_answer_is_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        async def ask_result_set(self, **kwargs: object) -> dict[str, object]:
            return {
                "answerStatus": "answered",
                "answer": " : ",
                "structuredSources": [
                    {
                        "sourceId": "attention-paper",
                        "title": "Attention is All you Need",
                        "provider": "semantic_scholar",
                        "sourceType": "scholarly_article",
                        "verificationStatus": "verified_metadata",
                        "accessStatus": "access_unverified",
                        "topicalRelevance": "on_topic",
                        "confidence": "high",
                        "citation": {
                            "authors": ["Ashish Vaswani", "Noam Shazeer"],
                            "year": "2017",
                            "title": "Attention is All you Need",
                            "journalOrPublisher": "Neural Information Processing Systems",
                        },
                    }
                ],
                "candidateLeads": [],
                "evidence": [],
                "unsupportedAsks": [],
                "followUpQuestions": [],
                "evidenceGaps": [],
                "coverageSummary": None,
                "failureSummary": None,
            }

    isolated_registry = type(server.workspace_registry)()
    monkeypatch.setattr(server, "workspace_registry", isolated_registry)
    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime())

    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-follow-up-blank-agentic-answer",
        query="Attention Is All You Need",
        payload={
            "searchSessionId": "ssn-follow-up-blank-agentic-answer",
            "sources": [
                {
                    "sourceId": "attention-paper",
                    "title": "Attention is All you Need",
                    "provider": "semantic_scholar",
                    "sourceType": "scholarly_article",
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "access_unverified",
                    "topicalRelevance": "on_topic",
                    "confidence": "high",
                    "citation": {
                        "authors": ["Ashish Vaswani", "Noam Shazeer"],
                        "year": "2017",
                        "title": "Attention is All you Need",
                        "journalOrPublisher": "Neural Information Processing Systems",
                    },
                }
            ],
        },
    )

    payload = _payload(
        await server.call_tool(
            "follow_up_research",
            {
                "searchSessionId": "ssn-follow-up-blank-agentic-answer",
                "question": "What venue is listed for this paper?",
            },
        )
    )

    assert payload["answerStatus"] == "answered"
    assert payload["answer"] != " : "
    assert "Neural Information Processing Systems" in payload["answer"]


@pytest.mark.asyncio
async def test_get_serpapi_account_status_sanitizes_upstream_response_in_public_tool_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serpapi_payload = {
        "account_id": "acct-123",
        "api_key": "SECRET_API_KEY",
        "account_email": "demo@serpapi.com",
        "plan_id": "bigdata",
        "plan_name": "Big Data Plan",
        "plan_monthly_price": 250.0,
        "searches_per_month": 30000,
        "plan_searches_left": 5958,
        "extra_credits": 5,
        "total_searches_left": 5963,
        "this_month_usage": 24042,
        "last_hour_searches": 42,
        "account_rate_limit_per_hour": 6000,
        "secret_token": "should-not-leak",
    }
    dummy = DummySerpApiAsyncClient(DummyResponse(status_code=200, payload=serpapi_payload))
    monkeypatch.setattr(serpapi_client_module.httpx, "AsyncClient", lambda timeout: dummy)
    monkeypatch.setattr(server, "enable_serpapi", True)
    monkeypatch.setattr(server, "serpapi_client", SerpApiScholarClient(api_key="test-key"))

    account = _payload(await server.call_tool("get_serpapi_account_status", {}))

    assert account["provider"] == "serpapi_google_scholar"
    assert account["planId"] == "bigdata"
    assert account["planSearchesLeft"] == 5958
    assert account["totalSearchesLeft"] == 5963
    assert account["accountRateLimitPerHour"] == 6000
    assert "api_key" not in account
    assert "account_email" not in account
    assert "account_id" not in account
    assert "secret_token" not in account


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
    provider_map = {item["provider"]: item for item in diagnostics["providers"] if isinstance(item, dict)}

    assert provider_map["crossref"]["enabled"] is True
    assert provider_map["unpaywall"]["enabled"] is True
    assert provider_map["crossref"]["lastEndpoint"] == "get_work"
    assert provider_map["unpaywall"]["lastEndpoint"] == "get_open_access"
    assert provider_map["crossref"]["recentOutcomes"]
    assert provider_map["unpaywall"]["recentOutcomes"]


@pytest.mark.asyncio
async def test_provider_diagnostics_surface_explicit_scholarapi_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scholarapi = RecordingScholarApiClient()
    monkeypatch.setattr(server, "scholarapi_client", scholarapi)
    monkeypatch.setattr(server, "enable_scholarapi", True)

    await server.call_tool("list_papers_scholarapi", {"query": "graphene", "has_pdf": True})
    await server.call_tool("get_paper_text_scholarapi", {"paper_id": "sa-1"})
    await server.call_tool("get_paper_texts_scholarapi", {"paper_ids": ["sa-1", "sa-2"]})
    await server.call_tool("get_paper_pdf_scholarapi", {"paper_id": "sa-1"})

    diagnostics = _payload(await server.call_tool("get_provider_diagnostics", {}))
    provider_map = {item["provider"]: item for item in diagnostics["providers"] if isinstance(item, dict)}

    scholarapi_diag = provider_map["scholarapi"]
    endpoints = [item["endpoint"] for item in scholarapi_diag["recentOutcomes"]]

    assert scholarapi_diag["enabled"] is True
    assert scholarapi_diag["lastEndpoint"] == "pdf"
    assert "list" in endpoints
    assert "text" in endpoints
    assert "texts" in endpoints
    assert "pdf" in endpoints


def test_package_import_and_module_entrypoints_keep_expected_targets() -> None:
    """Package import exposes server.main while python -m routes through cli.main."""
    assert paper_chaser_mcp.main is server.main
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
            {"citation": "Vaswani A, Shazeer N, Parmar N, et al. 2017. Attention Is All You Need. NeurIPS."},
            "search_papers_match",
            lambda p: "resolutionConfidence" in p and "candidateCount" in p,
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
    payload = _payload(await server.call_tool("search_papers_smart", {"query": "transformers"}))

    assert payload["error"] == "FEATURE_NOT_CONFIGURED"
    assert "fallbackTools" in payload
    assert payload["agentHints"]["nextToolCandidates"]


# ---------------------------------------------------------------------------
# Topical-relevance classification — regression for single-token-only matches
# ---------------------------------------------------------------------------


def test_topical_relevance_requires_facet_or_majority_coverage_for_on_topic() -> None:
    """_topical_relevance_from_signals must not classify a paper as on_topic when
    the only signal is a single-token title hit and similarity ≤ 0.5.

    Before the fix, `query_similarity >= 0.25 and title_has_anchor` was sufficient,
    so a query where one out of two distinctive terms matched the title (50 % coverage,
    no phrase match) would produce "on_topic" — admitting it as grounded evidence.
    The tightened rule requires either a multi-token facet phrase match plus the low
    threshold, or a strict majority (> 0.5) of distinctive terms without a phrase match.
    """
    fn = dispatch_module._topical_relevance_from_signals  # type: ignore[attr-defined]

    # 1. No facet match, similarity exactly at the old threshold boundary (0.5) → weak_match
    assert (
        fn(
            query_similarity=0.5,
            title_facet_coverage=0.0,
            title_anchor_coverage=0.5,
            query_facet_coverage=0.0,
            query_anchor_coverage=0.5,
        )
        == "weak_match"
    )

    # 2. No facet match, similarity strictly above the new threshold → on_topic
    assert (
        fn(
            query_similarity=0.6,
            title_facet_coverage=0.0,
            title_anchor_coverage=0.6,
            query_facet_coverage=0.0,
            query_anchor_coverage=0.6,
        )
        == "on_topic"
    )

    # 3. Facet match present, standard low threshold still reaches on_topic
    assert (
        fn(
            query_similarity=0.3,
            title_facet_coverage=0.5,
            title_anchor_coverage=0.5,
            query_facet_coverage=0.0,
            query_anchor_coverage=0.5,
        )
        == "on_topic"
    )

    # 4. Similarity below off_topic floor → off_topic regardless of anchor
    assert (
        fn(
            query_similarity=0.1,
            title_facet_coverage=0.0,
            title_anchor_coverage=0.5,
            query_facet_coverage=0.0,
            query_anchor_coverage=0.5,
        )
        == "off_topic"
    )

    # 5. No title or body signal at all → off_topic
    assert (
        fn(
            query_similarity=0.4,
            title_facet_coverage=0.0,
            title_anchor_coverage=0.0,
            query_facet_coverage=0.0,
            query_anchor_coverage=0.0,
        )
        == "off_topic"
    )


def test_paper_topical_relevance_single_token_match_without_phrase_is_weak_match() -> None:
    """A paper whose title contains only one of two distinctive query terms, without
    the multi-token query phrase appearing anywhere, must be classified as
    'weak_match', not 'on_topic'.

    This is the guided-research false-grounding regression: a query such as
    'temperature precipitation humidity' should not classify a paper titled
    'A Study of Temperature Effects' as on_topic because 'temperature' is the
    only match and no facet phrase is present.
    """
    fn = dispatch_module._paper_topical_relevance  # type: ignore[attr-defined]

    # Two distinctive terms; only one matches the title — no phrase match.
    relevance = fn(
        "temperature precipitation humidity",
        {
            "title": "A Study of Temperature Effects",
            "abstract": "We examine how temperature influences regional climate.",
        },
    )
    assert relevance == "weak_match", (
        f"Expected 'weak_match' for single-term title hit without phrase match, got {relevance!r}"
    )


def test_paper_topical_relevance_full_phrase_match_is_on_topic() -> None:
    """When the query's multi-token facet phrase appears in the paper title,
    even moderate similarity must still reach on_topic.
    """
    fn = dispatch_module._paper_topical_relevance  # type: ignore[attr-defined]

    relevance = fn(
        "temperature precipitation humidity",
        {
            "title": "Temperature Precipitation and Humidity Interactions",
            "abstract": "This paper studies temperature, precipitation, and humidity jointly.",
        },
    )
    assert relevance == "on_topic", f"Expected 'on_topic' when full phrase tokens appear in title, got {relevance!r}"


@pytest.mark.asyncio
async def test_guided_research_raw_broker_does_not_ground_single_term_match_as_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Via the raw-broker fallback path (no agentic_runtime), the guided research
    tool must not place a paper in the evidence array when only one distinctive
    query term matches the title and no multi-token phrase is present.

    The paper should appear in leads, and the evidence array must be empty.
    """

    class _MinimalSemanticClient:
        async def search_papers(self, **_: Any) -> dict[str, Any]:
            return {
                "total": 1,
                "offset": 0,
                "data": [
                    {
                        "paperId": "paper-temp-1",
                        "title": "A Study of Temperature Effects",
                        "abstract": "We examine how temperature influences regional climate.",
                        "publicationDate": "2022-01-01",
                        "citationCount": 10,
                    }
                ],
            }

    original_runtime = server.agentic_runtime
    original_client = server.client
    monkeypatch.setattr(server, "agentic_runtime", None)
    monkeypatch.setattr(server, "client", _MinimalSemanticClient())
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_core", False)
    monkeypatch.setattr(server, "enable_arxiv", False)

    try:
        payload = _payload(
            await server.call_tool(
                "research",
                {"query": "temperature precipitation humidity", "limit": 3},
            )
        )
    finally:
        server.agentic_runtime = original_runtime
        server.client = original_client

    evidence = payload.get("evidence", [])
    leads = payload.get("leads", [])

    assert evidence == [], (
        f"Expected empty evidence when only one distinctive term matches the title "
        f"(no phrase match), but got: {evidence}"
    )
    assert any(lead.get("evidenceId") == "paper-temp-1" for lead in leads), (
        "Expected the partial-match paper to appear in leads, but it was not found there."
    )
