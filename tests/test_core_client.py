import pytest

from paper_chaser_mcp import server
from tests.helpers import DummyResponse


def test_core_response_to_merged_preserves_total_and_limit() -> None:
    result = server._core_response_to_merged(
        {
            "total": 10,
            "entries": [
                {"paperId": "1", "title": "One", "url": "https://example.com/1"},
                {"paperId": "2", "title": "Two", "url": "https://example.com/2"},
            ],
        },
        limit=1,
    )

    assert result["total"] == 10
    assert result["offset"] == 0
    assert len(result["data"]) == 1
    assert result["data"][0]["paperId"] == "1"
    assert result["data"][0]["title"] == "One"
    assert result["data"][0]["url"] == "https://example.com/1"
    assert result["data"][0]["sourceType"] == "repository_record"
    assert result["data"][0]["verificationStatus"] == "verified_metadata"
    assert result["data"][0]["accessStatus"] == "access_unverified"
    assert result["data"][0]["canonicalUrl"] == "https://example.com/1"
    assert result["data"][0]["retrievedUrl"] == "https://example.com/1"


def test_core_result_to_paper_prefers_doi_url_and_normalizes_metadata() -> None:
    paper = server.CoreApiClient()._result_to_paper(
        {
            "id": 42,
            "doi": "10.1000/example-doi",
            "title": "Example paper",
            "abstract": "Example abstract",
            "publishedDate": "2023-05-01",
            "authors": [{"name": "Author One"}, "Author Two"],
            "journals": [{"title": "Journal A"}, {"title": "Journal B"}],
            "documentType": ["article"],
            "downloadUrl": "https://downloads.example/paper.pdf",
            "citationCount": 7,
        }
    )

    assert paper is not None
    assert paper["paperId"] == "42"
    assert paper["title"] == "Example paper"
    assert paper["abstract"] == "Example abstract"
    assert paper["year"] == 2023
    assert paper["authors"] == [{"name": "Author One"}, {"name": "Author Two"}]
    assert paper["citationCount"] == 7
    assert paper["venue"] == "Journal A, Journal B"
    assert paper["publicationTypes"] == ["article"]
    assert paper["publicationDate"] == "2023-05-01"
    assert paper["url"] == "https://doi.org/10.1000/example-doi"
    assert paper["pdfUrl"] == "https://downloads.example/paper.pdf"
    assert paper["source"] == "core"
    assert paper["sourceId"] == "42"
    assert paper["canonicalId"] == "10.1000/example-doi"
    assert paper["recommendedExpansionId"] == "10.1000/example-doi"
    assert paper["expansionIdStatus"] == "portable"
    assert paper["sourceType"] is None
    assert paper["verificationStatus"] is None
    assert paper["accessStatus"] is None
    assert paper["canonicalUrl"] is None
    assert paper["retrievedUrl"] is None


def test_core_result_to_paper_wraps_scalar_document_type_in_list() -> None:
    paper = server.CoreApiClient()._result_to_paper(
        {
            "id": 99,
            "title": "Scalar type test",
            "downloadUrl": "https://example.com/paper.pdf",
            "documentType": "research",
        }
    )

    assert paper is not None
    assert paper["publicationTypes"] == ["research"]


def test_core_result_to_paper_uses_nested_download_url_variants() -> None:
    paper = server.CoreApiClient()._result_to_paper(
        {
            "id": "core-1",
            "title": "Nested download url",
            "downloadUrl": {"urls": [{"link": "https://downloads.example/from-urls.pdf"}]},
            "authors": [{"name": "Author One"}, {"orcid": "missing-name"}],
        }
    )

    assert paper is not None
    assert paper["url"] == "https://downloads.example/from-urls.pdf"
    assert paper["pdfUrl"] is None
    assert paper["authors"] == [{"name": "Author One"}]
    assert paper["paperId"] == "core-1"


def test_core_result_to_paper_uses_source_fulltext_url_variants() -> None:
    paper = server.CoreApiClient()._result_to_paper(
        {
            "id": "core-2",
            "title": "Source fulltext url",
            "sourceFulltextUrls": {"urls": ["https://fulltext.example/paper"]},
            "downloadUrl": {"url": "https://downloads.example/paper.pdf"},
            "depositedDate": "2022-03-02",
        }
    )

    assert paper is not None
    assert paper["url"] == "https://downloads.example/paper.pdf"
    assert paper["pdfUrl"] == "https://downloads.example/paper.pdf"
    assert paper["publicationDate"] == "2022-03-02"
    assert paper["year"] == 2022


def test_core_result_to_paper_returns_none_without_required_fields() -> None:
    client = server.CoreApiClient()

    assert client._result_to_paper({"id": "core-3", "downloadUrl": "https://x"}) is None
    assert client._result_to_paper({"title": "Missing url"}) is None


def test_merge_search_results_deduplicates_arxiv_entries() -> None:
    merged = server._merge_search_results(
        {
            "offset": 3,
            "data": [
                {
                    "paperId": "semantic-1",
                    "title": "Known paper",
                    "externalIds": {"ArXiv": "1234.5678"},
                }
            ],
        },
        {
            "entries": [
                {"paperId": "1234.5678", "title": "Known paper from arXiv"},
                {"paperId": "9999.0001", "title": "Unique arXiv paper"},
            ]
        },
        limit=5,
    )

    assert merged["offset"] == 3
    assert merged["total"] == 2
    assert [paper["paperId"] for paper in merged["data"]] == [
        "semantic-1",
        "9999.0001",
    ]
    assert merged["data"][0]["source"] == "semantic_scholar"


def test_core_paper_has_provenance_fields_with_doi() -> None:
    """CORE papers with a DOI must prefer the DOI as canonicalId."""
    paper = server.CoreApiClient()._result_to_paper(
        {
            "id": 98765,
            "doi": "10.1234/core-test",
            "title": "CORE Provenance Test",
            "downloadUrl": "https://core.ac.uk/download/pdf/98765.pdf",
        }
    )

    assert paper is not None
    assert paper["source"] == "core"
    assert paper["sourceId"] == "98765"
    assert paper["canonicalId"] == "10.1234/core-test"


def test_core_paper_without_doi_is_marked_not_portable_for_expansion() -> None:
    """CORE-native fallback IDs must not be presented as expansion-safe."""
    paper = server.CoreApiClient()._result_to_paper(
        {
            "id": 11111,
            "title": "CORE No-DOI Paper",
            "downloadUrl": "https://core.ac.uk/download/pdf/11111.pdf",
        }
    )

    assert paper is not None
    assert paper["source"] == "core"
    assert paper["sourceId"] == "11111"
    assert paper["canonicalId"] == "11111"
    assert paper["expansionIdStatus"] == "not_portable"
    assert paper["recommendedExpansionId"] is None


@pytest.mark.asyncio
async def test_core_search_follows_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

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
                    "total_hits": 1,
                    "results": [
                        {
                            "id": "core-redirect",
                            "title": "Redirect-safe CORE result",
                            "downloadUrl": "https://example.com/paper.pdf",
                        }
                    ],
                },
            )

    monkeypatch.setattr(
        server.httpx,
        "AsyncClient",
        lambda timeout: CapturingAsyncClient(),
    )

    result = await server.CoreApiClient().search("transformers")

    assert result["entries"][0]["paperId"] == "core-redirect"
    assert captured[0]["follow_redirects"] is True


@pytest.mark.asyncio
async def test_core_search_retries_transient_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        DummyResponse(status_code=500, payload={"message": "temporary failure"}),
        DummyResponse(
            status_code=200,
            payload={
                "total_hits": 1,
                "results": [
                    {
                        "id": "core-retry",
                        "title": "Recovered CORE result",
                        "downloadUrl": "https://example.com/recovered.pdf",
                    }
                ],
            },
        ),
    ]
    sleep_calls: list[float] = []

    class RetryAsyncClient:
        def __init__(self, queued_responses: list[DummyResponse]) -> None:
            self._responses = queued_responses
            self.calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, url: str, **kwargs):
            response = self._responses[self.calls]
            self.calls += 1
            return response

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    retry_client = RetryAsyncClient(responses)
    monkeypatch.setattr(server.httpx, "AsyncClient", lambda timeout: retry_client)
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    result = await server.CoreApiClient().search("transformers")

    assert retry_client.calls == 2
    assert sleep_calls == [0.5]
    assert result["entries"][0]["paperId"] == "core-retry"
