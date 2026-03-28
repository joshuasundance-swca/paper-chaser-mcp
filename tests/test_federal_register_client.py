from __future__ import annotations

import httpx
import pytest

from paper_chaser_mcp.clients.federal_register import FederalRegisterClient


@pytest.mark.asyncio
async def test_federal_register_search_normalizes_filters_and_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/documents.json"
        assert request.url.params["conditions[term]"] == "least tern"
        assert request.url.params["conditions[agencies][]"] == "fish-and-wildlife-service"
        assert request.url.params["conditions[type][]"] == "NOTICE"
        return httpx.Response(
            200,
            json={
                "count": 1,
                "results": [
                    {
                        "document_number": "2024-23830",
                        "title": "Initiation of 5-Year Status Reviews for 59 Pacific Southwest Species",
                        "abstract": "Notice abstract.",
                        "type": "NOTICE",
                        "publication_date": "2024-10-16",
                        "volume": 89,
                        "start_page": 83510,
                        "end_page": 83514,
                        "body_html_url": "https://www.federalregister.gov/documents/full_text/html/2024-23830",
                        "pdf_url": "https://www.govinfo.gov/content/pkg/FR-2024-10-16/pdf/2024-23830.pdf",
                        "agencies": [{"name": "Fish and Wildlife Service", "slug": "fish-and-wildlife-service"}],
                        "cfr_references": [{"title": 50, "part": 17}],
                    }
                ],
            },
            request=request,
        )

    client = FederalRegisterClient()
    client._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)

    payload = await client.search_documents(
        query="least tern",
        agencies=["fish-and-wildlife-service"],
        document_types=["NOTICE"],
    )

    assert payload["total"] == 1
    assert payload["data"][0]["documentNumber"] == "2024-23830"
    assert payload["data"][0]["citation"] == "89 FR 83510"
    assert payload["data"][0]["govInfoLink"] == "https://www.govinfo.gov/link/fr/89/83510"
    assert payload["data"][0]["cfrReferences"] == ["50 CFR 17"]
    assert payload["data"][0]["agencies"][0]["slug"] == "fish-and-wildlife-service"

    await client.aclose()


@pytest.mark.asyncio
async def test_federal_register_get_document_normalizes_one_record() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/documents/2024-23830.json"
        return httpx.Response(
            200,
            json={
                "document_number": "2024-23830",
                "title": "Initiation of 5-Year Status Reviews for 59 Pacific Southwest Species",
                "type": "NOTICE",
                "publication_date": "2024-10-16",
                "volume": 89,
                "start_page": 83510,
                "end_page": 83514,
                "body_html_url": "https://www.federalregister.gov/documents/full_text/html/2024-23830",
            },
            request=request,
        )

    client = FederalRegisterClient()
    client._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)

    payload = await client.get_document("2024-23830")

    assert payload is not None
    assert payload["documentNumber"] == "2024-23830"
    assert payload["citation"] == "89 FR 83510"
    assert payload["bodyHtmlUrl"].endswith("2024-23830")

    await client.aclose()


@pytest.mark.asyncio
async def test_federal_register_search_uses_direct_lookup_for_document_number_with_query_filters() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/documents/2024-23830.json"
        return httpx.Response(
            200,
            json={
                "document_number": "2024-23830",
                "title": "Initiation of 5-Year Status Reviews for 59 Pacific Southwest Species",
                "abstract": "Notice abstract.",
                "type": "NOTICE",
                "publication_date": "2024-10-16",
                "volume": 89,
                "start_page": 83510,
                "agencies": [{"name": "Fish and Wildlife Service", "slug": "fish-and-wildlife-service"}],
            },
            request=request,
        )

    client = FederalRegisterClient()
    client._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)

    payload = await client.search_documents(
        query="status reviews pacific southwest",
        limit=5,
        agencies=["fish-and-wildlife-service"],
        document_types=["NOTICE"],
        document_number="2024-23830",
    )

    assert payload["total"] == 1
    assert payload["data"][0]["documentNumber"] == "2024-23830"

    await client.aclose()


@pytest.mark.asyncio
async def test_federal_register_search_retries_without_cfr_api_filter_and_filters_locally() -> None:
    calls: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params.multi_items()))
        assert request.url.path == "/api/v1/documents.json"
        if "conditions[cfr][citation]" in request.url.params:
            return httpx.Response(400, json={"error": "unsupported filter combination"}, request=request)
        return httpx.Response(
            200,
            json={
                "count": 2,
                "results": [
                    {
                        "document_number": "2011-7943",
                        "title": (
                            "Endangered and Threatened Wildlife and Plants; "
                            "Determination of Nine Distinct Population Segments of "
                            "Loggerhead Sea Turtles"
                        ),
                        "type": "RULE",
                        "publication_date": "2011-09-22",
                        "volume": 76,
                        "start_page": 58868,
                        "agencies": [
                            {"name": "Fish and Wildlife Service", "slug": "fish-and-wildlife-service"},
                            {"name": "National Marine Fisheries Service", "slug": "national-marine-fisheries-service"},
                        ],
                        "cfr_references": [{"title": 50, "part": 17}],
                    },
                    {
                        "document_number": "2011-0001",
                        "title": "Unrelated notice",
                        "type": "NOTICE",
                        "publication_date": "2011-09-22",
                        "volume": 76,
                        "start_page": 1000,
                        "agencies": [{"name": "Other Agency", "slug": "other-agency"}],
                    },
                ],
            },
            request=request,
        )

    client = FederalRegisterClient()
    client._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)

    payload = await client.search_documents(
        query="loggerhead sea turtle distinct population segment listing",
        limit=5,
        agencies=["fish-and-wildlife-service", "national-marine-fisheries-service"],
        document_types=["RULE"],
        publication_date_from="2011-01-01",
        publication_date_to="2013-12-31",
        cfr_title=50,
        cfr_part=17,
    )

    assert len(calls) == 2
    assert calls[0]["conditions[cfr][citation]"] == "50 CFR 17"
    assert "conditions[cfr][citation]" not in calls[1]
    assert payload["total"] == 1
    assert payload["data"][0]["documentNumber"] == "2011-7943"

    await client.aclose()
