from __future__ import annotations

import pytest

import paper_chaser_mcp.clients.scholarapi.client as scholarapi_client_module
from paper_chaser_mcp.clients.scholarapi import ScholarApiClient, ScholarApiError, ScholarApiQuotaError
from tests.helpers import DummyResponse


@pytest.mark.asyncio
async def test_scholarapi_search_normalizes_results_and_uses_auth_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []

    class CapturingAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, url: str, **kwargs):
            captured.append({"url": url, **kwargs})
            return DummyResponse(
                status_code=200,
                payload={
                    "results": [
                        {
                            "id": "96f3e91",
                            "title": "A fluorescent-protein spin qubit",
                            "authors": ["Feder, Jacob S.", "Soloway, Benjamin S."],
                            "abstract": "Example abstract",
                            "doi": "10.1038/example",
                            "journal": "Nature",
                            "published_date": "2023-09-14",
                            "published_date_raw": "2023-09-14",
                            "indexed_at": "2024-03-01T12:30:45.123Z",
                            "has_text": True,
                            "has_pdf": True,
                            "url": "https://journal.example.org/article/96f3e91",
                        }
                    ],
                    "next_cursor": "AoE/H4ANVGVzdDo2NDBhZDIxNw==",
                },
                headers={
                    "X-Request-Id": "req-123",
                    "X-Request-Cost": "3",
                },
            )

    monkeypatch.setattr(scholarapi_client_module.httpx, "AsyncClient", lambda timeout: CapturingAsyncClient())

    result = await ScholarApiClient(api_key="sch-test-key").search(
        query="qubit",
        limit=25,
        has_text=True,
    )

    paper = result["data"][0]
    assert captured[0]["url"].endswith("/search")
    assert captured[0]["headers"]["X-API-Key"] == "sch-test-key"
    assert captured[0]["params"]["q"] == "qubit"
    assert captured[0]["params"]["limit"] == 25
    assert captured[0]["params"]["has_text"] is True
    assert paper["paperId"] == "ScholarAPI:96f3e91"
    assert paper["source"] == "scholarapi"
    assert paper["sourceId"] == "96f3e91"
    assert paper["canonicalId"] == "10.1038/example"
    assert paper["recommendedExpansionId"] == "10.1038/example"
    assert paper["expansionIdStatus"] == "portable"
    assert paper["hasText"] is True
    assert paper["hasPdf"] is True
    assert paper["indexedAt"] == "2024-03-01T12:30:45.123Z"
    assert paper["contentAccess"]["scholarapi"]["paperId"] == "ScholarAPI:96f3e91"
    assert paper["contentAccess"]["scholarapi"]["hasText"] is True
    assert paper["contentAccess"]["scholarapi"]["hasPdf"] is True
    assert result["requestId"] == "req-123"
    assert result["pagination"]["nextCursor"] == "AoE/H4ANVGVzdDo2NDBhZDIxNw=="
    assert result["requestCost"] == "3"


@pytest.mark.asyncio
async def test_scholarapi_list_uses_index_cursor_and_preserves_non_doi_portability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []

    class CapturingAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, url: str, **kwargs):
            captured.append({"url": url, **kwargs})
            return DummyResponse(
                status_code=200,
                payload={
                    "results": [
                        {
                            "id": "846a45f",
                            "title": "Paraneoplastic pemphigus case study",
                            "authors": ["E. R. Novak"],
                            "published_date": "2023-09-14",
                            "indexed_at": "2024-03-01T12:30:45.123Z",
                            "has_text": False,
                            "has_pdf": True,
                        }
                    ],
                    "next_indexed_after": "2024-03-01T12:30:45.123Z",
                },
                headers={
                    "X-Request-Id": "req-list-123",
                    "X-Request-Cost": "2",
                },
            )

    monkeypatch.setattr(scholarapi_client_module.httpx, "AsyncClient", lambda timeout: CapturingAsyncClient())

    result = await ScholarApiClient(api_key="sch-test-key").list_papers(
        query=None,
        cursor="2024-02-01T00:00:00Z",
        has_pdf=True,
    )

    paper = result["data"][0]
    assert captured[0]["url"].endswith("/list")
    assert captured[0]["params"]["indexed_after"] == "2024-02-01T00:00:00Z"
    assert captured[0]["params"]["has_pdf"] is True
    assert paper["paperId"] == "ScholarAPI:846a45f"
    assert paper["recommendedExpansionId"] is None
    assert paper["expansionIdStatus"] == "not_portable"
    assert paper["contentAccess"]["scholarapi"]["paperId"] == "ScholarAPI:846a45f"
    assert paper["contentAccess"]["scholarapi"]["hasPdf"] is True
    assert result["requestId"] == "req-list-123"
    assert result["requestCost"] == "2"
    assert result["pagination"]["nextCursor"] == "2024-03-01T12:30:45.123Z"
    assert "sorted by indexed_at" in result["retrievalNote"]
    assert "search_papers_scholarapi" in result["retrievalNote"]


@pytest.mark.asyncio
async def test_scholarapi_get_texts_preserves_order_and_nulls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    class CapturingAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, url: str, **kwargs):
            captured.append(url)
            return DummyResponse(
                status_code=200,
                payload={"results": ["text one", None]},
            )

    monkeypatch.setattr(scholarapi_client_module.httpx, "AsyncClient", lambda timeout: CapturingAsyncClient())

    result = await ScholarApiClient(api_key="sch-test-key").get_texts(["ScholarAPI:a1", "a2"])

    assert captured[0].endswith("/texts/a1,a2")
    assert result["results"] == [
        {"paperId": "ScholarAPI:a1", "source": "scholarapi", "text": "text one"},
        {"paperId": "ScholarAPI:a2", "source": "scholarapi", "text": None},
    ]


@pytest.mark.asyncio
async def test_scholarapi_get_pdf_base64_encodes_binary_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    class PdfResponse:
        status_code = 200
        headers = {"Content-Type": "application/pdf"}
        content = b"%PDF-1.4"

        def raise_for_status(self) -> None:
            return None

    class CapturingAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, url: str, **kwargs):
            captured.append(url)
            return PdfResponse()

    monkeypatch.setattr(scholarapi_client_module.httpx, "AsyncClient", lambda timeout: CapturingAsyncClient())

    result = await ScholarApiClient(api_key="sch-test-key").get_pdf("ScholarAPI:a1")

    assert captured[0].endswith("/pdf/a1")
    assert result["paperId"] == "ScholarAPI:a1"
    assert result["mimeType"] == "application/pdf"
    assert result["contentBase64"] == "JVBERi0xLjQ="
    assert result["byteLength"] == 8


@pytest.mark.asyncio
async def test_scholarapi_get_text_accepts_namespaced_paper_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    class TextResponse:
        status_code = 200
        headers: dict[str, str] = {}
        text = "full text"

        def raise_for_status(self) -> None:
            return None

    class CapturingAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, url: str, **kwargs):
            captured.append(url)
            return TextResponse()

    monkeypatch.setattr(scholarapi_client_module.httpx, "AsyncClient", lambda timeout: CapturingAsyncClient())

    result = await ScholarApiClient(api_key="sch-test-key").get_text("ScholarAPI:a1")

    assert captured[0].endswith("/text/a1")
    assert result["paperId"] == "ScholarAPI:a1"
    assert result["text"] == "full text"


@pytest.mark.asyncio
async def test_scholarapi_credit_exhaustion_raises_specific_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CapturingAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, url: str, **kwargs):
            return DummyResponse(
                status_code=402,
                payload={"code": "payment_required", "message": "Insufficient credits"},
                headers={"X-Request-Id": "req-quota-1", "X-Request-Cost": "9"},
            )

    monkeypatch.setattr(scholarapi_client_module.httpx, "AsyncClient", lambda timeout: CapturingAsyncClient())

    with pytest.raises(ScholarApiQuotaError, match="Insufficient credits"):
        await ScholarApiClient(api_key="sch-test-key").search("qubit")


@pytest.mark.asyncio
async def test_scholarapi_rate_limit_error_surfaces_retry_after_and_request_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CapturingAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, url: str, **kwargs):
            return DummyResponse(
                status_code=429,
                payload={"message": "Too many requests"},
                headers={"Retry-After": "12", "X-Request-Id": "req-rate-1"},
            )

    monkeypatch.setattr(scholarapi_client_module.httpx, "AsyncClient", lambda timeout: CapturingAsyncClient())

    with pytest.raises(ScholarApiQuotaError, match="Retry after 12"):
        await ScholarApiClient(api_key="sch-test-key").search("qubit")


@pytest.mark.asyncio
async def test_scholarapi_missing_text_surfaces_404_request_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CapturingAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, url: str, **kwargs):
            return DummyResponse(
                status_code=404,
                payload={"message": "Not found"},
                headers={"X-Request-Id": "req-missing-1"},
            )

    monkeypatch.setattr(scholarapi_client_module.httpx, "AsyncClient", lambda timeout: CapturingAsyncClient())

    with pytest.raises(ScholarApiError, match="req-missing-1"):
        await ScholarApiClient(api_key="sch-test-key").get_text("missing-id")
