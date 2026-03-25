from __future__ import annotations

import time
import types
from typing import Any, cast

import httpx
import pytest

from paper_chaser_mcp.clients.govinfo.client import GovInfoAuthError, GovInfoClient


@pytest.mark.asyncio
async def test_govinfo_get_cfr_text_uses_x_api_key_and_prefers_xml() -> None:
    client = GovInfoClient(api_key="gov-key")
    client._markdown_converter = cast(Any, types.SimpleNamespace(convert=lambda **kwargs: "# CFR\n\nSection text."))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "gov-key"
        if request.method == "POST" and request.url.path == "/search":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "packageId": "CFR-2025-title40-vol24",
                            "granuleId": "CFR-2025-title40-vol24-sec122-26",
                            "download": {
                                "xmlLink": "https://api.govinfo.gov/packages/CFR-2025-title40-vol24/granules/CFR-2025-title40-vol24-sec122-26/xml"
                            },
                        }
                    ]
                },
                request=request,
            )
        if request.method == "GET" and request.url.path.endswith("/xml"):
            return httpx.Response(
                200,
                content=b"<SECTION><P>Section text.</P></SECTION>",
                headers={"content-type": "application/xml"},
                request=request,
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    client._http_client = httpx.AsyncClient(transport=transport, follow_redirects=True)
    client._document_client = httpx.AsyncClient(transport=transport, follow_redirects=True)

    payload = await client.get_cfr_text(title_number=40, part_number=122, section_number="26", revision_year=2025)

    assert payload["citation"] == "40 CFR 122.26"
    assert payload["packageId"] == "CFR-2025-title40-vol24"
    assert payload["granuleId"] == "CFR-2025-title40-vol24-sec122-26"
    assert payload["resolvedVolume"] == 24
    assert payload["contentSource"] == "govinfo_xml"
    assert payload["markdown"] == "Section text."

    await client.aclose()


@pytest.mark.asyncio
async def test_govinfo_get_cfr_text_extracts_cfr_xml_without_markitdown() -> None:
    client = GovInfoClient(api_key="gov-key")
    client._markdown_converter = cast(
        Any,
        types.SimpleNamespace(
            convert=lambda **kwargs: (_ for _ in ()).throw(AssertionError("converter should not run"))
        ),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/search":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "packageId": "CFR-2024-title50-vol2",
                            "granuleId": "CFR-2024-title50-vol2-sec17-11",
                            "download": {
                                "xmlLink": "https://api.govinfo.gov/packages/CFR-2024-title50-vol2/granules/CFR-2024-title50-vol2-sec17-11/xml"
                            },
                        }
                    ]
                },
                request=request,
            )
        if request.method == "GET" and request.url.path.endswith("/xml"):
            return httpx.Response(
                200,
                content=(
                    b"<SECTION><SECTNO>Sec. 17.11</SECTNO><SUBJECT>Definitions.</SUBJECT>"
                    b"<P>Section text.</P><P>More detail.</P></SECTION>"
                ),
                headers={"content-type": "application/xml"},
                request=request,
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    client._http_client = httpx.AsyncClient(transport=transport, follow_redirects=True)
    client._document_client = httpx.AsyncClient(transport=transport, follow_redirects=True)

    payload = await client.get_cfr_text(title_number=50, part_number=17, section_number="11", revision_year=2024)

    assert payload["contentSource"] == "govinfo_xml"
    assert payload["warnings"] == []
    assert payload["markdown"] == "## Sec. 17.11\n\n### Definitions.\n\nSection text.\n\nMore detail."

    await client.aclose()


@pytest.mark.asyncio
async def test_govinfo_get_cfr_text_returns_warning_when_conversion_times_out() -> None:
    client = GovInfoClient(api_key="gov-key", document_timeout=0.01)

    def delayed_convert(**kwargs: object) -> str:
        time.sleep(0.05)
        return "# CFR\n\nDelayed text"

    client._markdown_converter = cast(
        Any,
        types.SimpleNamespace(
            convert=delayed_convert,
        ),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/search":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "packageId": "CFR-2025-title40-vol24",
                            "granuleId": "CFR-2025-title40-vol24-part122",
                            "download": {
                                "txtLink": "https://api.govinfo.gov/packages/CFR-2025-title40-vol24/granules/CFR-2025-title40-vol24-part122/txt"
                            },
                        }
                    ]
                },
                request=request,
            )
        if request.method == "GET" and request.url.path.endswith("/txt"):
            return httpx.Response(
                200,
                content=b"Plain text CFR body",
                headers={"content-type": "text/plain"},
                request=request,
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    client._http_client = httpx.AsyncClient(transport=transport, follow_redirects=True)
    client._document_client = httpx.AsyncClient(transport=transport, follow_redirects=True)

    payload = await client.get_cfr_text(title_number=40, part_number=122, revision_year=2025)

    assert payload["contentSource"] == "govinfo_html"
    assert payload["markdown"] is None
    assert payload["warnings"]
    assert "timed out" in payload["warnings"][0].lower()

    await client.aclose()


@pytest.mark.asyncio
async def test_govinfo_get_federal_register_document_prefers_hard_timeout_converter_when_available() -> None:
    class StubFederalRegisterClient:
        async def get_document(self, document_number: str) -> dict[str, object]:
            assert document_number == "2024-23830"
            return {
                "documentNumber": "2024-23830",
                "title": "Pacific Southwest Species Reviews",
                "citation": "89 FR 83510",
                "bodyHtmlUrl": "https://www.federalregister.gov/documents/full_text/html/2024-23830",
            }

    class RecordingConverter:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def convert(self, **kwargs: object) -> str:
            raise AssertionError("convert() should not be used when convert_with_timeout() is available")

        def convert_with_timeout(self, **kwargs: object) -> str:
            self.calls.append(dict(kwargs))
            return "Fallback HTML"

    client = GovInfoClient(api_key=None, document_timeout=0.01)
    converter = RecordingConverter()
    client._markdown_converter = cast(Any, converter)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/2024-23830")
        return httpx.Response(
            200,
            content=b"<html><body><h1>Notice</h1></body></html>",
            headers={"content-type": "text/html"},
            request=request,
        )

    client._document_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)

    payload = await client.get_federal_register_document(
        identifier="2024-23830",
        federal_register_client=StubFederalRegisterClient(),
    )

    assert payload["contentSource"] == "federal_register_html"
    assert payload["markdown"] == "Fallback HTML"
    assert converter.calls
    assert converter.calls[0]["timeout_seconds"] == 0.01

    await client.aclose()


@pytest.mark.asyncio
async def test_govinfo_get_federal_register_document_falls_back_to_federal_register_html_without_key() -> None:
    class StubFederalRegisterClient:
        async def get_document(self, document_number: str) -> dict[str, object]:
            assert document_number == "2024-23830"
            return {
                "documentNumber": "2024-23830",
                "title": "Pacific Southwest Species Reviews",
                "citation": "89 FR 83510",
                "bodyHtmlUrl": "https://www.federalregister.gov/documents/full_text/html/2024-23830",
            }

    client = GovInfoClient(api_key=None)
    client._markdown_converter = cast(Any, types.SimpleNamespace(convert=lambda **kwargs: "# Notice\n\nFallback HTML"))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/2024-23830")
        return httpx.Response(
            200,
            content=b"<html><body><h1>Notice</h1><p>Fallback HTML</p></body></html>",
            headers={"content-type": "text/html"},
            request=request,
        )

    client._document_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)

    payload = await client.get_federal_register_document(
        identifier="2024-23830",
        federal_register_client=StubFederalRegisterClient(),
    )

    assert payload["contentSource"] == "federal_register_html"
    assert payload["document"]["documentNumber"] == "2024-23830"
    assert payload["markdown"] == "# Notice\n\nFallback HTML"
    assert payload["warnings"]

    await client.aclose()


@pytest.mark.asyncio
async def test_govinfo_get_federal_register_document_falls_back_to_fr_html_when_govinfo_granule_fetch_fails() -> None:
    class StubFederalRegisterClient:
        async def get_document(self, document_number: str) -> dict[str, object] | None:
            assert document_number == "2011-7943"
            return {
                "documentNumber": "2011-7943",
                "title": "Loggerhead turtle listing rule",
                "citation": "77 FR 4632",
                "bodyHtmlUrl": "https://www.federalregister.gov/documents/full_text/html/2011-7943",
            }

        async def search_documents(self, *, query: str, limit: int = 10, **_: object) -> dict[str, object]:
            assert query == "77 FR 4632"
            assert limit == 5
            return {
                "total": 1,
                "data": [
                    {
                        "documentNumber": "2011-7943",
                        "title": "Loggerhead turtle listing rule",
                        "citation": "77 FR 4632",
                        "bodyHtmlUrl": "https://www.federalregister.gov/documents/full_text/html/2011-7943",
                    }
                ],
            }

    client = GovInfoClient(api_key="gov-key")
    client._markdown_converter = cast(Any, types.SimpleNamespace(convert=lambda **kwargs: "# Rule\n\nFallback HTML"))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/search":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "packageId": "FR-2012-01-31",
                            "granuleId": "2011-7943",
                            "title": "Loggerhead turtle listing rule",
                            "dateIssued": "2012-01-31",
                        }
                    ]
                },
                request=request,
            )
        if request.method == "GET" and request.url.path.endswith("/summary"):
            return httpx.Response(404, json={"message": "NOT_FOUND"}, request=request)
        if request.method == "GET" and request.url.path.endswith("/2011-7943"):
            return httpx.Response(
                200,
                content=b"<html><body><h1>Rule</h1><p>Fallback HTML</p></body></html>",
                headers={"content-type": "text/html"},
                request=request,
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    client._http_client = httpx.AsyncClient(transport=transport, follow_redirects=True)
    client._document_client = httpx.AsyncClient(transport=transport, follow_redirects=True)

    payload = await client.get_federal_register_document(
        identifier="Federal Register (2012). Loggerhead sea turtle DPS listing. 77 FR 4632.",
        federal_register_client=StubFederalRegisterClient(),
    )

    assert payload["contentSource"] == "federal_register_html"
    assert payload["document"]["documentNumber"] == "2011-7943"
    assert payload["markdown"] == "# Rule\n\nFallback HTML"
    assert any("FederalRegister.gov HTML" in warning for warning in payload["warnings"])

    await client.aclose()


@pytest.mark.asyncio
async def test_govinfo_get_federal_register_document_tries_alternate_fr_search_queries_for_citation() -> None:
    class StubFederalRegisterClient:
        def __init__(self) -> None:
            self.queries: list[str] = []

        async def get_document(self, document_number: str) -> dict[str, object] | None:
            return None

        async def search_documents(self, *, query: str, limit: int = 10, **_: object) -> dict[str, object]:
            self.queries.append(query)
            if query == "89 FR 48437":
                return {"total": 0, "data": []}
            if query == "89 48437":
                return {
                    "total": 1,
                    "data": [
                        {
                            "documentNumber": "2024-12370",
                            "title": "Southeastern beach mouse notice",
                            "citation": "89 FR 48437",
                            "bodyHtmlUrl": "https://www.federalregister.gov/documents/full_text/html/2024-12370",
                        }
                    ],
                }
            return {"total": 0, "data": []}

    client = GovInfoClient(api_key=None)
    client._markdown_converter = cast(Any, types.SimpleNamespace(convert=lambda **kwargs: "# Notice\n\nHTML fallback"))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/2024-12370")
        return httpx.Response(
            200,
            content=b"<html><body><h1>Notice</h1><p>HTML fallback</p></body></html>",
            headers={"content-type": "text/html"},
            request=request,
        )

    fr_client = StubFederalRegisterClient()
    client._document_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)

    payload = await client.get_federal_register_document(
        identifier="89 FR 48437",
        federal_register_client=fr_client,
    )

    assert fr_client.queries == ["89 FR 48437", "89 48437"]
    assert payload["document"]["documentNumber"] == "2024-12370"
    assert payload["contentSource"] == "federal_register_html"

    await client.aclose()


@pytest.mark.asyncio
async def test_govinfo_auth_errors_are_normalized() -> None:
    client = GovInfoClient(api_key="bad-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"errorCode": "API_KEY_INVALID", "message": "The provided API key is invalid."},
            request=request,
        )

    client._http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)

    with pytest.raises(GovInfoAuthError, match="API_KEY_INVALID"):
        await client.get_cfr_text(title_number=40, part_number=122, section_number="26", revision_year=2025)

    await client.aclose()
