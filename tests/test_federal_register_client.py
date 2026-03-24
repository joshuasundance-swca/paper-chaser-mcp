from __future__ import annotations

import httpx
import pytest

from scholar_search_mcp.clients.federal_register import FederalRegisterClient


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