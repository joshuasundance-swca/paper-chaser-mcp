import ssl
import sys
import time
import types
from typing import Any, cast

import httpx
import pytest

from scholar_search_mcp.clients.ecos import EcosClient
from scholar_search_mcp.ecos_markdown import EcosMarkdownConverter
from scholar_search_mcp.provider_runtime import ProviderDiagnosticsRegistry

CALIFORNIA_LEAST_TERN_PROFILE = {
    "id": 8104,
    "cn": "California least tern",
    "sn": "Sternula antillarum browni",
    "group": "Birds",
    "family": "Laridae",
    "kingdom": "Animalia",
    "tsn": "176974",
    "uri": "urn:ecos:species:8104",
    "life_history": "Ground-nesting seabird that breeds on beaches and salt flats.",
    "historical_range_state": "CA",
    "species_entities": [
        {
            "id": 96,
            "abbrev": "F",
            "agency": "FWS",
            "alt_status": None,
            "desc": "Wherever found",
            "dps": None,
            "status": "Endangered",
            "status_category": "Animal",
            "listing_date": "06-02-1970",
            "lead_fws_region": "Region 8",
            "recovery_priority_number": "12C",
            "more_info_url": "/ecp/species/8104",
            "current_range_country": "United States",
            "current_range_state": "CA",
            "current_range_county": "San Diego",
            "current_range_refuge": "San Diego Bay NWR",
            "range_envelope": "bbox",
            "range_shapefile": "shape.zip",
            "shapefile_last_updated": "2025-01-01",
            "biological_opinion": [
                {
                    "final_date": "06/12/2012",
                    "lead_offices_csv": "Sacramento Fish and Wildlife Office",
                    "activity_titles_csv": (
                        "Cargill Salt Ponds Operation and Maintenance"
                    ),
                    "activity_codes_csv": "81420-2010-F-0519",
                    "work_types_csv": "DREDGE / EXCAVATION",
                    "locations_csv": "Alameda (CA), San Mateo (CA)",
                    "lead_agencies_csv": "Army Corps of Engineers",
                    "category": "Biological Opinion Rendered (Final)",
                    "event_code": "08ESMF00-2012-E-02227",
                    "file_name": {
                        "value": (
                            "Biological Opinion Rendered (Final) 08ESMF00-2012-E-02227"
                        ),
                        "url": "/tails/pub/document/527831",
                    },
                }
            ],
        }
    ],
    "recoveryPlans": [
        {
            "entity_id": 96,
            "source_id": 400056,
            "doc_title": {
                "value": "Revised California Least Tern Recovery Plan",
                "url": "https://ecos.fws.gov/docs/recovery_plan/850927_w signature.pdf",
            },
            "doc_date": "09/27/1985",
            "doc_type_qualifier": "Final Revision 1",
            "recovery_plan_id": 400282,
            "doc_type": "Recovery Plan",
        }
    ],
    "fiveYearReviews": [
        {
            "entity_id": 96,
            "source_id": 25776,
            "doc_title": {
                "value": "California Least Tern 5YR 2025",
                "url": "https://ecosphere-documents-production-public.s3.amazonaws.com/sams/public_docs/species_nonpublish/30669.pdf",
            },
            "doc_date": "08/28/2025",
            "doc_type_qualifier": "Final",
            "recovery_plan_id": None,
            "doc_type": "Five Year Review",
        },
        {
            "entity_id": 96,
            "source_id": 2663,
            "doc_title": {
                "value": "California least tern 5-year review",
                "url": "https://ecosphere-documents-production-public.s3.amazonaws.com/sams/public_docs/species_nonpublish/3520.pdf",
            },
            "doc_date": "07/09/2020",
            "doc_type_qualifier": "Final",
            "recovery_plan_id": None,
            "doc_type": "Five Year Review",
        },
    ],
    "federal_register_document": [
        {
            "id": 63911,
            "pub_id": 11849,
            "publication_date": "10/16/2024",
            "publication_page": "89 FR 83510 83514",
            "publication_title": {
                "value": (
                    "Initiation of 5-Year Status Reviews for 59 Pacific "
                    "Southwest Species"
                ),
                "url": "https://www.govinfo.gov/link/fr/89/83510",
            },
            "associated_document": [],
        }
    ],
    "otherRecoveryDocs": [
        {
            "id": 2978,
            "pub_id": 861,
            "publication_date": "06/18/2018",
            "publication_page": "83 FR 28251 28254",
            "publication_title": {
                "value": (
                    "Initiation of 5-Year Status Reviews of 50 Species in "
                    "California, Nevada, and the Klamath Basin of Oregon"
                ),
                "url": "https://www.govinfo.gov/link/fr/83/28251?link-type=pdf",
            },
            "doc_types": ["Five Year Review Notice, Information Solicitation"],
        }
    ],
    "conservationPlans": {
        "hasConservationPlans": True,
        "hcpDocs": [
            {
                "plan_id": 610,
                "plan_title": {
                    "value": "MHCP, City of Carlsbad Habitat Management Plan",
                    "url": "/ecp/report/conservation-plan?plan_id=610",
                },
                "plan_type": "HCP",
            }
        ],
        "shaDocs": [],
    },
}


@pytest.mark.asyncio
async def test_ecos_search_species_auto_prefers_exact_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = EcosClient()

    async def fake_pullreports_export(filter_expression: str) -> dict:
        if " = " in filter_expression:
            return {
                "data": [
                    [
                        "California least tern",
                        {
                            "value": "Sternula antillarum browni",
                            "url": "https://ecos.fws.gov/ecp/species/8104",
                        },
                        "Endangered",
                        "Wherever found",
                        "06-02-1970",
                    ]
                ]
            }
        return {
            "data": [
                [
                    "California least tern colony",
                    {
                        "value": "Sternula antillarum browni colony",
                        "url": "https://ecos.fws.gov/ecp/species/9999",
                    },
                    "Endangered",
                    "Wherever found",
                    "06-02-1970",
                ]
            ]
        }

    async def fake_fetch_species_payload(species_id: str) -> dict:
        if species_id == "8104":
            return CALIFORNIA_LEAST_TERN_PROFILE
        return {
            "id": 9999,
            "group": "Birds",
            "species_entities": [{"agency": "FWS", "status_category": "Animal"}],
        }

    monkeypatch.setattr(client, "_pullreports_export", fake_pullreports_export)
    monkeypatch.setattr(client, "_fetch_species_payload", fake_fetch_species_payload)

    payload = await client.search_species(query="California least tern")

    assert payload["data"][0]["speciesId"] == "8104"
    assert payload["data"][0]["group"] == "Birds"
    assert payload["data"][0]["leadAgency"] == "FWS"
    assert payload["data"][1]["speciesId"] == "9999"


@pytest.mark.asyncio
async def test_ecos_species_profile_normalizes_grouped_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = EcosClient()

    async def fake_fetch_species_payload(species_id: str) -> dict:
        assert species_id == "8104"
        return CALIFORNIA_LEAST_TERN_PROFILE

    monkeypatch.setattr(client, "_fetch_species_payload", fake_fetch_species_payload)

    payload = await client.get_species_profile(
        species_id="https://ecos.fws.gov/ecp/species/8104"
    )

    assert payload["species"]["speciesId"] == "8104"
    assert payload["species"]["group"] == "Birds"
    assert payload["speciesEntities"][0]["entityId"] == 96
    assert payload["range"]["currentRangeStates"] == ["CA"]
    assert payload["documents"]["recoveryPlans"][0]["url"].startswith("https://")
    assert payload["documents"]["biologicalOpinions"][0]["url"].endswith(
        "/tails/pub/document/527831"
    )
    assert payload["conservationPlanLinks"][0]["url"].startswith(
        "https://ecos.fws.gov/ecp/report/conservation-plan"
    )


@pytest.mark.asyncio
async def test_ecos_list_species_documents_filters_and_sorts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = EcosClient()

    async def fake_fetch_species_payload(species_id: str) -> dict:
        assert species_id == "8104"
        return CALIFORNIA_LEAST_TERN_PROFILE

    monkeypatch.setattr(client, "_fetch_species_payload", fake_fetch_species_payload)

    payload = await client.list_species_documents(
        species_id="8104",
        document_kinds=["recovery_plan", "five_year_review", "biological_opinion"],
    )

    assert payload["total"] == 4
    assert [item["documentKind"] for item in payload["data"]] == [
        "five_year_review",
        "five_year_review",
        "biological_opinion",
        "recovery_plan",
    ]
    assert payload["data"][0]["title"] == "California Least Tern 5YR 2025"
    assert payload["data"][-1]["title"] == "Revised California Least Tern Recovery Plan"


def test_ecos_markdown_converter_uses_markitdown_stream_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeMarkItDown:
        def __init__(self, *, enable_plugins: bool) -> None:
            captured["enable_plugins"] = enable_plugins

        def convert_stream(self, stream, *, file_extension: str, url: str):
            captured["content"] = stream.read()
            captured["file_extension"] = file_extension
            captured["url"] = url
            return types.SimpleNamespace(text_content="# Review\n\nConverted.")

    monkeypatch.setitem(
        sys.modules,
        "markitdown",
        types.SimpleNamespace(MarkItDown=FakeMarkItDown),
    )

    converter = EcosMarkdownConverter()
    markdown = converter.convert(
        content=b"%PDF-1.4 sample",
        source_url="https://ecos.fws.gov/docs/example.pdf",
        content_type="application/pdf",
        filename="example.pdf",
    )

    assert markdown == "# Review\n\nConverted."
    assert captured["enable_plugins"] is False
    assert captured["file_extension"] == ".pdf"
    assert captured["url"] == "https://ecos.fws.gov/docs/example.pdf"
    assert captured["content"] == b"%PDF-1.4 sample"


@pytest.mark.asyncio
async def test_ecos_get_document_text_follows_redirect_and_records_diagnostics() -> (
    None
):
    registry = ProviderDiagnosticsRegistry()
    client = EcosClient(provider_registry=registry)
    client._markdown_converter = cast(
        Any,
        types.SimpleNamespace(
            convert=lambda **kwargs: (
                "# Five-Year Review\n\n## Recommendation\n\nRetain endangered status."
            )
        ),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tails/pub/document/527831":
            return httpx.Response(
                302,
                headers={"location": "https://ecos.fws.gov/docs/example.pdf"},
                request=request,
            )
        if request.url.path == "/docs/example.pdf":
            return httpx.Response(
                200,
                content=b"%PDF-1.4 test",
                headers={
                    "content-type": "application/pdf; charset=ISO-8859-1",
                    "content-disposition": (
                        'attachment; filename="california_least_tern_2025.pdf"'
                    ),
                },
                request=request,
            )
        raise AssertionError(f"Unexpected request: {request.url}")

    client._document_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
    )

    payload = await client.get_document_text(url="/tails/pub/document/527831")
    snapshot = registry.snapshot(enabled={"ecos": True}, provider_order=["ecos"])

    assert payload["extractionStatus"] == "ok"
    assert payload["document"]["url"] == "https://ecos.fws.gov/docs/example.pdf"
    assert payload["contentType"].startswith("application/pdf")
    assert snapshot["providers"][0]["lastEndpoint"] == "document.fetch"
    assert snapshot["providers"][0]["recentOutcomes"]

    await client.aclose()


@pytest.mark.asyncio
async def test_ecos_get_document_text_logs_fetch_and_conversion_steps(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = EcosClient()
    client._markdown_converter = cast(
        Any,
        types.SimpleNamespace(
            convert=lambda **kwargs: (
                "# Review\n\nRetain listed status with updated monitoring."
            )
        ),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"%PDF-1.4 test",
            headers={"content-type": "application/pdf"},
            request=request,
        )

    client._document_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
    )

    with caplog.at_level("INFO", logger="scholar-search-mcp"):
        payload = await client.get_document_text(
            url="https://ecos.fws.gov/docs/example.pdf"
        )

    assert payload["extractionStatus"] == "ok"
    assert "ECOS document fetch started" in caplog.text
    assert "ECOS document fetch completed" in caplog.text
    assert "ECOS document conversion started" in caplog.text
    assert "ECOS document conversion completed" in caplog.text

    await client.aclose()


@pytest.mark.asyncio
async def test_ecos_get_document_text_reports_structured_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = EcosClient()

    async def fake_download_unsupported(url: str) -> dict:
        return {
            "finalUrl": url,
            "contentType": "application/octet-stream",
            "filename": "blob.bin",
            "content": b"\x00\x01\x02",
            "status": "ok",
        }

    monkeypatch.setattr(client, "_download_document", fake_download_unsupported)
    unsupported = await client.get_document_text(url="https://example.com/blob.bin")
    assert unsupported["extractionStatus"] == "unsupported_type"

    async def fake_download_too_large(url: str) -> dict:
        return {
            "finalUrl": url,
            "contentType": "application/pdf",
            "filename": "large.pdf",
            "status": "too_large",
        }

    monkeypatch.setattr(client, "_download_document", fake_download_too_large)
    too_large = await client.get_document_text(url="https://example.com/large.pdf")
    assert too_large["extractionStatus"] == "too_large"

    async def fake_download_short(url: str) -> dict:
        return {
            "finalUrl": url,
            "contentType": "application/pdf",
            "filename": "scan.pdf",
            "content": b"%PDF-1.4 tiny",
            "status": "ok",
        }

    monkeypatch.setattr(client, "_download_document", fake_download_short)
    client._markdown_converter = cast(
        Any,
        types.SimpleNamespace(convert=lambda **kwargs: "short"),
    )
    near_empty = await client.get_document_text(url="https://example.com/scan.pdf")
    assert near_empty["extractionStatus"] == "empty_or_image_only"


@pytest.mark.asyncio
async def test_ecos_get_document_text_times_out_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = EcosClient(document_conversion_timeout=0.01)

    async def fake_download(url: str) -> dict:
        return {
            "finalUrl": url,
            "contentType": "application/pdf",
            "filename": "slow.pdf",
            "content": b"%PDF-1.4 slow",
            "status": "ok",
        }

    def slow_convert(**kwargs: Any) -> str:
        time.sleep(0.1)
        return "# Converted"

    monkeypatch.setattr(client, "_download_document", fake_download)
    client._markdown_converter = cast(
        Any,
        types.SimpleNamespace(convert=slow_convert),
    )

    payload = await client.get_document_text(url="https://example.com/slow.pdf")

    assert payload["extractionStatus"] == "conversion_timed_out"
    assert payload["markdown"] is None
    assert payload["warnings"]


@pytest.mark.asyncio
async def test_ecos_api_retries_with_system_trust_store_on_tls_verify_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed_clients: list[dict[str, Any]] = []

    class FakeAsyncClient:
        def __init__(
            self,
            *,
            timeout: float,
            follow_redirects: bool,
            verify: Any,
        ) -> None:
            constructed_clients.append(
                {
                    "timeout": timeout,
                    "follow_redirects": follow_redirects,
                    "verify": verify,
                }
            )
            self.verify = verify

        async def get(
            self, url: str, params: dict[str, Any] | None = None
        ) -> httpx.Response:
            request = httpx.Request("GET", url, params=params)
            if self.verify is True:
                raise httpx.ConnectError(
                    (
                        "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify "
                        "failed: unable to get local issuer certificate "
                        "(_ssl.c:1032)"
                    ),
                    request=request,
                )
            return httpx.Response(200, json={"data": []}, request=request)

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(
        "scholar_search_mcp.clients.ecos.client.httpx.AsyncClient",
        FakeAsyncClient,
    )

    client = EcosClient()
    payload = await client._get_json("/ecp/test", params={"format": "json"})

    assert payload == {"data": []}
    assert constructed_clients[0]["verify"] is True
    assert isinstance(constructed_clients[1]["verify"], ssl.SSLContext)


def test_ecos_client_uses_explicit_tls_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed_clients: list[dict[str, Any]] = []

    class FakeAsyncClient:
        def __init__(
            self,
            *,
            timeout: float,
            follow_redirects: bool,
            verify: Any,
        ) -> None:
            constructed_clients.append(
                {
                    "timeout": timeout,
                    "follow_redirects": follow_redirects,
                    "verify": verify,
                }
            )

    monkeypatch.setattr(
        "scholar_search_mcp.clients.ecos.client.httpx.AsyncClient",
        FakeAsyncClient,
    )

    insecure_client = EcosClient(verify_tls=False)
    insecure_client._get_api_client()

    bundle_client = EcosClient(ca_bundle="C:/certs/ecos-ca.pem")
    bundle_client._get_api_client()

    assert constructed_clients[0]["verify"] is False
    assert constructed_clients[1]["verify"] == "C:/certs/ecos-ca.pem"
