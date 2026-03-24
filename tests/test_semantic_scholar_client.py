import pytest

from scholar_search_mcp import server
from scholar_search_mcp.transport import httpx
from tests.helpers import DummyAsyncClient, DummyResponse


@pytest.mark.asyncio
async def test_semantic_scholar_request_retries_after_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        DummyResponse(status_code=429, headers={"Retry-After": "0"}),
        DummyResponse(status_code=200, payload={"data": [{"paperId": "ok"}]}),
    ]
    dummy_client = DummyAsyncClient(responses)
    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda timeout: dummy_client)
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    client = server.SemanticScholarClient(api_key="test-key")
    result = await client._request("GET", "paper/search", params={"query": "test"})

    assert result == {"data": [{"paperId": "ok"}]}
    assert dummy_client.calls == 2
    assert 1.0 in sleep_calls  # the 429 back-off delay was honored
    # _pace() fires before every attempt, so there must be a second sleep
    # for the pacing guard on the retry.
    assert len(sleep_calls) == 2


@pytest.mark.asyncio
async def test_semantic_scholar_rate_limiter_paces_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared rate limiter must insert a sleep when requests arrive faster
    than the configured minimum interval."""
    responses = [
        DummyResponse(status_code=200, payload={"data": [{"paperId": "r1"}]}),
        DummyResponse(status_code=200, payload={"data": [{"paperId": "r2"}]}),
    ]
    dummy_client = DummyAsyncClient(responses)
    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    call_count = 0

    def fake_monotonic() -> float:
        # Simulate two calls within 0.1 s of each other to trigger pacing.
        # First call (inside _pace for request-1): now = 0.0 → no prior request,
        # sets _last_request_time = 0.0.
        # Second call (inside _pace for request-2): now = 0.0 → elapsed = 0.0
        # → sleep needed.  Subsequent calls return a large value so _pace exits.
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return 0.0
        return 100.0

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda timeout: dummy_client)
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        "scholar_search_mcp.clients.semantic_scholar.client.time.monotonic",
        fake_monotonic,
    )

    sc = server.SemanticScholarClient(api_key="test-key")
    # First request – sets _last_request_time to 0.0; no sleep needed.
    await sc._request("GET", "paper/search", params={"query": "a"})
    # Second request – monotonic() still returns 0.0 so elapsed < MIN_INTERVAL.
    await sc._request("GET", "paper/search", params={"query": "b"})

    # At least one pacing sleep must have been issued.
    assert any(s > 0 for s in sleep_calls)


@pytest.mark.asyncio
async def test_get_recommendations_uses_recommendations_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_recommendations must target the recommendations API, not the graph API."""
    captured: list[str] = []

    async def fake_sleep(_: float) -> None:
        pass

    class CapturingAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, *, method, url, **kwargs):
            captured.append(url)
            return DummyResponse(
                status_code=200,
                payload={"recommendedPapers": [{"paperId": "rec-1"}]},
            )

    monkeypatch.setattr(
        server.httpx,
        "AsyncClient",
        lambda timeout: CapturingAsyncClient(),
    )
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    await sc.get_recommendations("paper-xyz")

    assert len(captured) == 1
    assert "recommendations" in captured[0]
    assert "graph" not in captured[0]


@pytest.mark.asyncio
async def test_semantic_scholar_client_reuses_lazy_async_client_and_closes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_clients: list["ReusableAsyncClient"] = []

    async def fake_sleep(_: float) -> None:
        pass

    class ReusableAsyncClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.closed = False

        async def request(self, **kwargs):
            self.calls.append(kwargs)
            return DummyResponse(
                status_code=200,
                payload={"data": [{"paperId": f"ok-{len(self.calls)}"}]},
            )

        async def aclose(self) -> None:
            self.closed = True

    def _factory(timeout: float) -> ReusableAsyncClient:
        del timeout
        client = ReusableAsyncClient()
        created_clients.append(client)
        return client

    monkeypatch.setattr(server.httpx, "AsyncClient", _factory)
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    client = server.SemanticScholarClient(api_key="test-key")
    await client._request("GET", "paper/search", params={"query": "alpha"})
    await client._request("GET", "paper/search", params={"query": "beta"})

    assert len(created_clients) == 1
    assert len(created_clients[0].calls) == 2

    await client.aclose()

    assert created_clients[0].closed is True


@pytest.mark.asyncio
async def test_search_papers_bulk_truncates_provider_oversized_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bulk search must honor small limits even if the provider ignores them."""

    async def fake_sleep(_: float) -> None:
        pass

    oversized_batch = [{"paperId": f"bulk-{index}", "title": f"Paper {index}"} for index in range(10)]

    class BulkAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, **kwargs):
            return DummyResponse(
                status_code=200,
                payload={
                    "total": 5000,
                    "token": "tok-next",
                    "data": oversized_batch,
                },
            )

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda timeout: BulkAsyncClient())
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    result = await sc.search_papers_bulk("graph neural networks", limit=5)

    assert len(result["data"]) == 5
    assert [paper["paperId"] for paper in result["data"]] == [
        "bulk-0",
        "bulk-1",
        "bulk-2",
        "bulk-3",
        "bulk-4",
    ]
    assert result["pagination"]["hasMore"] is True
    assert result["pagination"]["nextCursor"] == "tok-next"
    # The raw provider token must NOT be exposed in the public response.
    assert "token" not in result
    # All returned papers must have expansion ID portability fields populated.
    for paper in result["data"]:
        assert paper["recommendedExpansionId"] == paper["paperId"]
        assert paper["expansionIdStatus"] == "portable"
        assert paper["source"] == "semantic_scholar"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_paper", "expected_expansion_id"),
    [
        # paperId only → recommendedExpansionId falls back to paperId
        (
            {"paperId": "aabbcc1234", "title": "Paper A"},
            "aabbcc1234",
        ),
        # DOI present → recommendedExpansionId prefers DOI
        (
            {
                "paperId": "aabbcc1234",
                "title": "Paper B",
                "externalIds": {"DOI": "10.1000/xyz"},
            },
            "10.1000/xyz",
        ),
        # No DOI but ArXiv present → falls back to paperId (paperId wins over arXiv)
        (
            {
                "paperId": "aabbcc1234",
                "title": "Paper C",
                "externalIds": {"ArXiv": "2111.99999"},
            },
            "aabbcc1234",
        ),
    ],
)
async def test_search_papers_bulk_enriches_expansion_id_fields(
    monkeypatch: pytest.MonkeyPatch,
    raw_paper: dict,
    expected_expansion_id: str,
) -> None:
    """search_papers_bulk must populate recommendedExpansionId and expansionIdStatus."""

    async def fake_sleep(_: float) -> None:
        pass

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, **kwargs):
            return DummyResponse(
                status_code=200,
                payload={"total": 1, "data": [raw_paper]},
            )

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda timeout: FakeClient())
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    result = await sc.search_papers_bulk("test query", limit=5)

    assert len(result["data"]) == 1
    paper = result["data"][0]
    assert paper["recommendedExpansionId"] == expected_expansion_id
    assert paper["expansionIdStatus"] == "portable"
    assert paper["source"] == "semantic_scholar"


@pytest.mark.asyncio
async def test_search_papers_bulk_400_with_custom_fields_retries_with_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When search_papers_bulk gets a 400 with a custom fields list, it must
    retry with DEFAULT_PAPER_FIELDS and add a fieldsDropped flag to the result.

    The Semantic Scholar /paper/search/bulk endpoint supports fewer fields than
    /paper/search, so agents that pass unsupported fields (e.g. 'tldr',
    'embedding') should still get results rather than an unhandled 400 error.
    """

    async def fake_sleep(_: float) -> None:
        pass

    captured_params: list[dict] = []
    responses = [
        # First attempt (with custom fields) returns 400.
        httpx.Response(
            status_code=400,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/bulk"),
        ),
        # Retry (with default fields) succeeds.
        httpx.Response(
            status_code=200,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/bulk"),
            json={
                "total": 1,
                "data": [{"paperId": "bulk-fallback", "title": "Fallback Paper"}],
            },
        ),
    ]

    class CapturingAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, *, params, **kwargs):
            captured_params.append(dict(params))
            return responses.pop(0)

    monkeypatch.setattr(
        server.httpx,
        "AsyncClient",
        lambda timeout: CapturingAsyncClient(),
    )
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    result = await sc.search_papers_bulk("attention mechanism", fields=["paperId", "title", "tldr"])

    # The result must contain the fallback paper, not raise.
    assert len(result["data"]) == 1
    assert result["data"][0]["paperId"] == "bulk-fallback"
    # The degradation flags must be set.
    assert result["fieldsDropped"] is True
    assert "default field set" in result["message"]
    # First attempt used the custom fields; retry used DEFAULT_PAPER_FIELDS.
    from scholar_search_mcp.constants import DEFAULT_PAPER_FIELDS

    assert set(captured_params[0]["fields"].split(",")) == {"paperId", "title", "tldr"}
    assert set(captured_params[1]["fields"].split(",")) == set(DEFAULT_PAPER_FIELDS)


@pytest.mark.asyncio
async def test_semantic_scholar_request_retries_with_pacing_on_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pacing guard must fire before every attempt, including 429 retries."""
    responses = [
        DummyResponse(status_code=429, headers={"Retry-After": "0"}),
        DummyResponse(status_code=200, payload={"data": [{"paperId": "ok"}]}),
    ]
    dummy_client = DummyAsyncClient(responses)
    pace_calls: list[str] = []
    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    original_pace = server.SemanticScholarClient._pace

    async def recording_pace(self: server.SemanticScholarClient) -> None:
        pace_calls.append("pace")
        await original_pace(self)

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda timeout: dummy_client)
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(server.SemanticScholarClient, "_pace", recording_pace)

    sc = server.SemanticScholarClient(api_key="test-key")
    result = await sc._request("GET", "paper/search", params={"query": "test"})

    assert result == {"data": [{"paperId": "ok"}]}
    # _pace must be called once for the initial attempt and once for the retry.
    assert len(pace_calls) == 2, f"Expected 2 _pace calls (initial + retry), got {len(pace_calls)}"
    # The 429 back-off sleep and the pacing sleep for the retry must both fire.
    assert 1.0 in sleep_calls, "429 back-off sleep was not honoured"
    assert len(sleep_calls) == 2, f"Expected 2 sleep calls (429 back-off + pacing), got {len(sleep_calls)}"


@pytest.mark.asyncio
async def test_get_paper_citations_response_includes_pagination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_paper_citations end-to-end: pagination fields must survive the full path."""

    async def fake_sleep(_: float) -> None:
        pass

    class PaginatingAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, **kwargs):
            return DummyResponse(
                status_code=200,
                payload={
                    "offset": 0,
                    "next": 100,
                    "data": [{"paperId": "citing-paper-1", "title": "Citing paper"}],
                },
            )

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda timeout: PaginatingAsyncClient())
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    result = await sc.get_paper_citations("paper-xyz", limit=100)

    assert result["offset"] == 0
    assert result["next"] == 100
    assert result["data"][0]["paperId"] == "citing-paper-1"
    # Uniform pagination envelope must be present
    assert result["pagination"]["hasMore"] is True
    assert result["pagination"]["nextCursor"] == "100"


@pytest.mark.asyncio
async def test_get_paper_citations_unwraps_nested_citing_paper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_sleep(_: float) -> None:
        pass

    class NestedCitationsAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, **kwargs):
            return DummyResponse(
                status_code=200,
                payload={
                    "offset": 0,
                    "data": [
                        {
                            "paperId": None,
                            "title": None,
                            "citingPaper": {
                                "paperId": "citing-paper-1",
                                "title": "Wrapped citing paper",
                            },
                        }
                    ],
                },
            )

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda timeout: NestedCitationsAsyncClient())
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    result = await sc.get_paper_citations("paper-xyz", limit=1)

    assert result["data"][0]["paperId"] == "citing-paper-1"
    assert result["data"][0]["title"] == "Wrapped citing paper"
    assert "citingPaper" not in result["data"][0]


@pytest.mark.asyncio
async def test_get_paper_references_unwraps_nested_cited_paper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_sleep(_: float) -> None:
        pass

    class NestedReferencesAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, **kwargs):
            return DummyResponse(
                status_code=200,
                payload={
                    "offset": 0,
                    "data": [
                        {
                            "paperId": None,
                            "title": None,
                            "citedPaper": {
                                "paperId": "referenced-paper-1",
                                "title": "Wrapped reference",
                            },
                        }
                    ],
                },
            )

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda timeout: NestedReferencesAsyncClient())
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    result = await sc.get_paper_references("paper-xyz", limit=1)

    assert result["data"][0]["paperId"] == "referenced-paper-1"
    assert result["data"][0]["title"] == "Wrapped reference"
    assert "citedPaper" not in result["data"][0]


@pytest.mark.asyncio
async def test_get_author_papers_response_includes_pagination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_author_papers end-to-end: pagination fields must survive the full path."""

    async def fake_sleep(_: float) -> None:
        pass

    class PaginatingAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, **kwargs):
            return DummyResponse(
                status_code=200,
                payload={
                    "offset": 50,
                    "next": 100,
                    "data": [{"paperId": "author-paper-1", "title": "Author paper"}],
                },
            )

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda timeout: PaginatingAsyncClient())
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    result = await sc.get_author_papers("author-abc", limit=50, offset=50)

    assert result["offset"] == 50
    assert result["next"] == 100
    assert result["data"][0]["paperId"] == "author-paper-1"
    # Uniform pagination envelope must be present
    assert result["pagination"]["hasMore"] is True
    assert result["pagination"]["nextCursor"] == "100"


@pytest.mark.asyncio
async def test_get_author_papers_normalizes_trailing_hyphen_in_date_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_author_papers must normalize 'YYYY-' to 'YYYY:' before calling the API.

    Regression test for the broken golden path that advertised
    publicationDateOrYear="2022-" (year-parameter style with a trailing hyphen)
    when the correct open-ended format for publicationDateOrYear is "2022:"
    (colon separator).  The client should silently normalize the trailing hyphen
    so that agents using the documented form do not receive a misleading 400.
    """

    async def fake_sleep(_: float) -> None:
        pass

    captured_params: list[dict] = []

    class CapturingAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, **kwargs):
            captured_params.append(kwargs.get("params", {}))
            return DummyResponse(
                status_code=200,
                payload={
                    "offset": 0,
                    "data": [{"paperId": "paper-x", "title": "Recent work"}],
                },
            )

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda timeout: CapturingAsyncClient())
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    result = await sc.get_author_papers("1751762", limit=5, publication_date_or_year="2022-")

    # The request should have succeeded and returned paper data.
    assert result["data"][0]["paperId"] == "paper-x"
    # The trailing hyphen must have been rewritten to a colon before the call.
    assert captured_params[0]["publicationDateOrYear"] == "2022:"


@pytest.mark.asyncio
async def test_get_author_papers_400_with_date_filter_mentions_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 400 on get_author_papers with publicationDateOrYear set should blame
    the filter, not the author ID, so agents are not sent on a dead-end
    "check your authorId" path when the ID is actually valid.
    """

    async def fake_sleep(_: float) -> None:
        pass

    request = httpx.Request(
        "GET",
        "https://api.semanticscholar.org/graph/v1/author/1751762/papers",
    )
    bad_response = httpx.Response(status_code=400, request=request)

    class RejectingFilterClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, **kwargs):
            return bad_response

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda timeout: RejectingFilterClient())
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    with pytest.raises(ValueError, match="publicationDateOrYear"):
        await sc.get_author_papers("1751762", limit=5, publication_date_or_year="bad-filter-value")


@pytest.mark.asyncio
async def test_get_author_papers_400_without_date_filter_mentions_author_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 400 on get_author_papers without a date filter should still tell the
    agent to verify the authorId, since that is the most likely cause.
    """

    async def fake_sleep(_: float) -> None:
        pass

    request = httpx.Request(
        "GET",
        "https://api.semanticscholar.org/graph/v1/author/bad-id/papers",
    )
    bad_response = httpx.Response(status_code=400, request=request)

    class RejectingAuthorClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, **kwargs):
            return bad_response

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda timeout: RejectingAuthorClient())
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    with pytest.raises(ValueError, match="search_authors or get_paper_authors"):
        await sc.get_author_papers("bad-id", limit=5)


@pytest.mark.asyncio
async def test_search_papers_response_includes_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """search_papers end-to-end: the next field must be preserved in the response."""

    async def fake_sleep(_: float) -> None:
        pass

    class NextPageAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, **kwargs):
            return DummyResponse(
                status_code=200,
                payload={
                    "total": 300,
                    "offset": 10,
                    "next": 20,
                    "data": [{"paperId": "s2-page2-1", "title": "Result"}],
                },
            )

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda timeout: NextPageAsyncClient())
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    result = await sc.search_papers("transformers", limit=10, offset=10)

    assert result["total"] == 300
    assert result["offset"] == 10
    assert result["next"] == 20
    # Uniform pagination envelope must be present
    assert result["pagination"]["hasMore"] is True
    assert result["pagination"]["nextCursor"] == "20"


def test_search_papers_match_unwraps_nested_match_payload() -> None:
    client = server.SemanticScholarClient()

    normalized = client._normalize_match_response(
        {
            "paperId": None,
            "title": None,
            "data": [
                {
                    "paperId": "match-123",
                    "title": "Wrapped best match",
                }
            ],
        }
    )

    assert normalized.paper_id == "match-123"
    assert normalized.title == "Wrapped best match"
    assert "data" not in normalized.model_dump(by_alias=True, exclude_none=True)


@pytest.mark.asyncio
async def test_search_papers_match_falls_back_to_fuzzy_search_on_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_sleep(_: float) -> None:
        pass

    requests: list[tuple[str, str]] = []
    # The primary /paper/search/match endpoint is tried for each candidate query
    # produced by _title_lookup_queries: original, punctuation-normalised,
    # lowercase, and lowercase-punct-normalised.  All four return 404 here so
    # the fuzzy-search fallback runs.
    responses = [
        httpx.Response(
            status_code=404,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/match"),
        ),
        httpx.Response(
            status_code=404,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/match"),
        ),
        httpx.Response(
            status_code=404,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/match"),
        ),
        httpx.Response(
            status_code=404,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/match"),
        ),
        httpx.Response(
            status_code=200,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search"),
            json={
                "total": 1,
                "offset": 0,
                "data": [
                    {
                        "paperId": "paper-1",
                        "title": (
                            "Oyster Cultch Recruit Patterns Provide New Insight "
                            "Into the Restoration and Management of a Critical "
                            "Resource"
                        ),
                    }
                ],
            },
        ),
    ]

    class SequencedAsyncClient:
        def __init__(self, queued_responses: list[httpx.Response]) -> None:
            self._responses = queued_responses

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, *, url: str, params, **kwargs):
            requests.append((url, params["query"]))
            return self._responses.pop(0)

    monkeypatch.setattr(
        server.httpx,
        "AsyncClient",
        lambda timeout: SequencedAsyncClient(responses),
    )
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    result = await sc.search_papers_match(
        "Oyster Cultch-Recruit Patterns Provide New Insight Into the Restoration and Management of a Critical Resource"
    )

    assert result["paperId"] == "paper-1"
    assert result["matchFound"] is True
    assert result["matchStrategy"] == "fuzzy_search"
    assert requests[0][0].endswith("/paper/search/match")
    assert requests[1][0].endswith("/paper/search/match")
    assert requests[2][0].endswith("/paper/search/match")
    assert requests[3][0].endswith("/paper/search/match")
    assert requests[4][0].endswith("/paper/search")


@pytest.mark.asyncio
async def test_search_papers_match_returns_structured_no_match_payload_after_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_sleep(_: float) -> None:
        pass

    # Four /paper/search/match attempts (original, punct-normalised, lowercase,
    # lowercase-punct-normalised) all return 404; then the fuzzy-search fallback
    # tries each of the four candidate queries via /paper/search (unquoted) then
    # each again in quoted-phrase form, none with a usable title match.
    responses = [
        httpx.Response(
            status_code=404,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/match"),
        ),
        httpx.Response(
            status_code=404,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/match"),
        ),
        httpx.Response(
            status_code=404,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/match"),
        ),
        httpx.Response(
            status_code=404,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/match"),
        ),
        # Unquoted fallback searches (4 variants) — no match
        httpx.Response(
            status_code=200,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search"),
            json={"total": 1, "offset": 0, "data": [{"title": "Unrelated result"}]},
        ),
        httpx.Response(
            status_code=200,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search"),
            json={"total": 0, "offset": 0, "data": []},
        ),
        httpx.Response(
            status_code=200,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search"),
            json={"total": 0, "offset": 0, "data": []},
        ),
        httpx.Response(
            status_code=200,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search"),
            json={"total": 0, "offset": 0, "data": []},
        ),
        # Quoted-phrase fallback searches (4 variants) — still no match
        httpx.Response(
            status_code=200,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search"),
            json={"total": 0, "offset": 0, "data": []},
        ),
        httpx.Response(
            status_code=200,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search"),
            json={"total": 0, "offset": 0, "data": []},
        ),
        httpx.Response(
            status_code=200,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search"),
            json={"total": 0, "offset": 0, "data": []},
        ),
        httpx.Response(
            status_code=200,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search"),
            json={"total": 0, "offset": 0, "data": []},
        ),
        # Citation-ranked bulk fallback (4 variants) — no title match either
        httpx.Response(
            status_code=200,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/bulk"),
            json={"total": 0, "data": []},
        ),
        httpx.Response(
            status_code=200,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/bulk"),
            json={"total": 0, "data": []},
        ),
        httpx.Response(
            status_code=200,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/bulk"),
            json={"total": 0, "data": []},
        ),
        httpx.Response(
            status_code=200,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/bulk"),
            json={"total": 0, "data": []},
        ),
    ]

    class SequencedAsyncClient:
        def __init__(self, queued_responses: list[httpx.Response]) -> None:
            self._responses = queued_responses

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, **kwargs):
            return self._responses.pop(0)

    monkeypatch.setattr(
        server.httpx,
        "AsyncClient",
        lambda timeout: SequencedAsyncClient(responses),
    )
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    result = await sc.search_papers_match("ezMCDA: An Interactive Dashboard")

    assert result["paperId"] is None
    assert result["matchFound"] is False
    assert result["matchStrategy"] == "none"
    assert "outside the indexed paper surface" in result["message"]
    # normalizedQueriesTried now includes all 8 queries tried:
    # 4 unquoted variants + 4 quoted-phrase variants.
    assert result["normalizedQueriesTried"] == [
        "ezMCDA: An Interactive Dashboard",
        "ezMCDA An Interactive Dashboard",
        "ezmcda: an interactive dashboard",
        "ezmcda an interactive dashboard",
        '"ezMCDA: An Interactive Dashboard"',
        '"ezMCDA An Interactive Dashboard"',
        '"ezmcda: an interactive dashboard"',
        '"ezmcda an interactive dashboard"',
    ]


@pytest.mark.asyncio
async def test_search_papers_match_primary_success_includes_match_found_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful 200 from the primary match endpoint must include matchFound=True
    and matchStrategy='exact_title' so agents can distinguish it from a no-match."""

    async def fake_sleep(_: float) -> None:
        pass

    class SuccessfulMatchAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, **kwargs):
            return httpx.Response(
                status_code=200,
                request=httpx.Request(
                    "GET",
                    "https://api.semanticscholar.org/graph/v1/paper/search/match",
                ),
                json={
                    "paperId": "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
                    "title": "Attention is All you Need",
                },
            )

    monkeypatch.setattr(
        server.httpx,
        "AsyncClient",
        lambda timeout: SuccessfulMatchAsyncClient(),
    )
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    result = await sc.search_papers_match("Attention Is All You Need")

    assert result["matchFound"] is True
    assert result["matchStrategy"] == "exact_title"
    assert result["paperId"] == "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    assert result["title"] == "Attention is All you Need"


@pytest.mark.asyncio
async def test_search_papers_match_200_null_paper_triggers_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 200 response with paperId=null (empty data) must trigger the fallback
    so agents never receive a confusing null-paper payload instead of a
    structured no-match with recovery hints."""

    async def fake_sleep(_: float) -> None:
        pass

    # "Attention Is All You Need" has no punctuation to strip, so _title_lookup_queries
    # produces two unique candidates: the original and its lowercase form.  The first
    # primary attempt returns a null-paperId 200; the second (lowercase) gets a 404;
    # then the fuzzy-search fallback finds the paper.
    responses = [
        httpx.Response(
            status_code=200,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/match"),
            json={"paperId": None, "title": None, "data": []},
        ),
        httpx.Response(
            status_code=404,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/match"),
        ),
        httpx.Response(
            status_code=200,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search"),
            json={
                "total": 1,
                "offset": 0,
                "data": [
                    {
                        "paperId": "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
                        "title": "Attention is All you Need",
                    }
                ],
            },
        ),
    ]

    class SequencedAsyncClient:
        def __init__(self, queued_responses: list[httpx.Response]) -> None:
            self._responses = queued_responses

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, **kwargs):
            return self._responses.pop(0)

    monkeypatch.setattr(
        server.httpx,
        "AsyncClient",
        lambda timeout: SequencedAsyncClient(responses),
    )
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    result = await sc.search_papers_match("Attention Is All You Need")

    assert result["matchFound"] is True
    assert result["matchStrategy"] == "fuzzy_search"
    assert result["paperId"] == "204e3073870fae3d05bcbc2f6a8e263d9b72e776"


@pytest.mark.asyncio
async def test_search_papers_match_fallback_finds_famous_paper_by_exact_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fuzzy-search fallback must find a famous paper like 'Attention Is All
    You Need' when the primary match endpoint returns 404 but search_papers
    returns the canonical paper in its result set."""

    async def fake_sleep(_: float) -> None:
        pass

    # _title_lookup_queries produces two unique candidates for this query:
    # the original and its lowercase form.  Both primary attempts return 404 so
    # the fuzzy-search fallback fires and finds the paper in the search results.
    responses = [
        httpx.Response(
            status_code=404,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/match"),
        ),
        httpx.Response(
            status_code=404,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/match"),
        ),
        httpx.Response(
            status_code=200,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search"),
            json={
                "total": 5,
                "offset": 0,
                "data": [
                    {"paperId": "other-1", "title": "Deep Residual Learning"},
                    {"paperId": "other-2", "title": "ImageNet Classification"},
                    {
                        "paperId": "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
                        "title": "Attention is All you Need",
                    },
                    {"paperId": "other-3", "title": "BERT Pre-training"},
                    {"paperId": "other-4", "title": "GPT Language Models"},
                ],
            },
        ),
    ]

    class SequencedAsyncClient:
        def __init__(self, queued_responses: list[httpx.Response]) -> None:
            self._responses = queued_responses

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, **kwargs):
            return self._responses.pop(0)

    monkeypatch.setattr(
        server.httpx,
        "AsyncClient",
        lambda timeout: SequencedAsyncClient(responses),
    )
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    result = await sc.search_papers_match("Attention Is All You Need")

    assert result["matchFound"] is True
    assert result["matchStrategy"] == "fuzzy_search"
    assert result["paperId"] == "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    assert result["title"] == "Attention is All you Need"


@pytest.mark.asyncio
async def test_search_papers_match_fallback_uses_quoted_phrase_when_unquoted_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quoted-phrase fallback must find the paper when unquoted keyword search
    returns no relevant candidates.

    Semantic Scholar's /paper/search endpoint treats unquoted words as separate
    keywords so a query like 'Attention Is All You Need' may rank generic
    attention-mechanism papers above the Vaswani et al. paper.  Wrapping the
    query in double quotes enables exact phrase matching and should surface the
    specific paper even when unquoted variants fail.

    Scenario:
      - Both primary /paper/search/match attempts return 404.
        ('Attention Is All You Need' has no punctuation so _title_lookup_queries
        produces exactly two unique variants: the original and its lowercase form.)
      - Unquoted search_papers for both title variants return no relevant
        candidates (_pick_title_match_candidate returns None).
      - Quoted search '"Attention Is All You Need"' returns the canonical paper.
    """

    async def fake_sleep(_: float) -> None:
        pass

    captured_queries: list[str] = []

    # Sequence:
    #   [0] primary match 404 for "Attention Is All You Need"
    #   [1] primary match 404 for "attention is all you need"
    #   [2] unquoted search "Attention Is All You Need" → irrelevant results
    #   [3] unquoted search "attention is all you need" → irrelevant results
    #   [4] quoted search '"Attention Is All You Need"' → correct paper
    responses = [
        httpx.Response(
            status_code=404,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/match"),
        ),
        httpx.Response(
            status_code=404,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/match"),
        ),
        # Unquoted search — returns attention papers but not the specific one
        httpx.Response(
            status_code=200,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search"),
            json={
                "total": 3,
                "offset": 0,
                "data": [
                    {"paperId": "other-1", "title": "Attention Mechanisms in NLP"},
                    {"paperId": "other-2", "title": "Self-Attention Networks"},
                    {"paperId": "other-3", "title": "Multi-Head Attention"},
                ],
            },
        ),
        # Lowercase unquoted search — still no match
        httpx.Response(
            status_code=200,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search"),
            json={
                "total": 3,
                "offset": 0,
                "data": [
                    {"paperId": "other-1", "title": "Attention Mechanisms in NLP"},
                    {"paperId": "other-2", "title": "Self-Attention Networks"},
                ],
            },
        ),
        # Quoted phrase search — returns the canonical paper
        httpx.Response(
            status_code=200,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search"),
            json={
                "total": 1,
                "offset": 0,
                "data": [
                    {
                        "paperId": "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
                        "title": "Attention is All you Need",
                    }
                ],
            },
        ),
    ]

    class SequencedAsyncClient:
        def __init__(self, queued_responses: list[httpx.Response]) -> None:
            self._responses = queued_responses

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, *, url: str, params, **kwargs):
            captured_queries.append(params.get("query", ""))
            return self._responses.pop(0)

    monkeypatch.setattr(
        server.httpx,
        "AsyncClient",
        lambda timeout: SequencedAsyncClient(responses),
    )
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    result = await sc.search_papers_match("Attention Is All You Need")

    assert result["matchFound"] is True
    assert result["matchStrategy"] == "fuzzy_search"
    assert result["paperId"] == "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    assert result["title"] == "Attention is All you Need"
    # The normalizedQuery must reflect the quoted phrase that was actually used.
    assert result.get("normalizedQuery") == '"Attention Is All You Need"'
    # Verify the quoted query was tried after the two unquoted search attempts.
    search_queries = [q for q in captured_queries if q]
    # First two are primary match attempts: original title-case + lowercase.
    # (No punctuation in this title so only 2 unique variants are produced.)
    assert search_queries[0] == "Attention Is All You Need"
    assert search_queries[1] == "attention is all you need"
    # Next two are unquoted search fallback attempts (same two variants)
    assert search_queries[2] == "Attention Is All You Need"
    assert search_queries[3] == "attention is all you need"
    # Fifth is the quoted phrase fallback
    assert search_queries[4] == '"Attention Is All You Need"'


@pytest.mark.asyncio
async def test_search_papers_match_fallback_citation_ranked_bulk_last_resort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Citation-sorted bulk search must find the paper when all other fallbacks fail.

    This is the sentinel regression test for 'Attention Is All You Need'.  In
    production the Semantic Scholar /paper/search endpoint sometimes buries this
    paper behind topic-specific attention papers even with a quoted phrase, and
    the /paper/search/match endpoint returns 404.  The fix adds a final fallback
    that issues a /paper/search/bulk request sorted by citationCount:desc; the
    canonical Vaswani et al. transformer paper has 100,000+ citations and will
    always appear at the top of citation-sorted results.

    Scenario:
      - Both primary /paper/search/match attempts return 404.
      - All four /paper/search attempts (2 unquoted + 2 quoted) return 400
        (simulating the Semantic Scholar API rejecting quoted phrase syntax),
        so the fuzzy_search strategy cannot find the paper.
      - The /paper/search/bulk call (citation-sorted) returns the canonical paper.
    """

    async def fake_sleep(_: float) -> None:
        pass

    captured_bulk_params: list[dict] = []

    # Sequence:
    #   [0]  primary match 404 for "Attention Is All You Need"
    #   [1]  primary match 404 for "attention is all you need"
    #   [2]  search_papers unquoted "Attention Is All You Need" → 400
    #   [3]  search_papers unquoted "attention is all you need" → 400
    #   [4]  search_papers quoted '"Attention Is All You Need"' → 400
    #   [5]  search_papers quoted '"attention is all you need"' → 400
    #   [6]  search_papers_bulk "Attention Is All You Need" citationCount:desc → hit
    match_404 = httpx.Response(
        status_code=404,
        request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/match"),
    )
    search_400 = httpx.Response(
        status_code=400,
        request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search"),
    )
    bulk_hit = httpx.Response(
        status_code=200,
        request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/bulk"),
        json={
            "total": 1,
            "data": [
                {
                    "paperId": "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
                    "title": "Attention is All you Need",
                }
            ],
        },
    )
    responses = [
        match_404,
        match_404,
        search_400,
        search_400,
        search_400,
        search_400,
        bulk_hit,
    ]

    class SequencedAsyncClient:
        def __init__(self, queued_responses: list[httpx.Response]) -> None:
            self._responses = queued_responses
            self.called_urls: list[str] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, *, url: str, params, **kwargs):
            self.called_urls.append(url)
            if "bulk" in url:
                captured_bulk_params.append(dict(params))
            return self._responses.pop(0)

    sequenced_client = SequencedAsyncClient(responses)
    monkeypatch.setattr(
        server.httpx,
        "AsyncClient",
        lambda timeout: sequenced_client,
    )
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    result = await sc.search_papers_match("Attention Is All You Need")

    assert result["matchFound"] is True
    assert result["matchStrategy"] == "citation_ranked"
    assert result["paperId"] == "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    assert result["title"] == "Attention is All you Need"
    # Verify the bulk request used citation-count sorting
    assert captured_bulk_params, "Expected at least one bulk search call"
    assert captured_bulk_params[0].get("sort") == "citationCount:desc"
    # All queued responses were consumed in the expected order:
    # 2 match, 4 search (all 400), 1 bulk
    assert not responses, "Not all queued responses were consumed"
    called_suffixes = [u.split("/graph/v1/")[-1] for u in sequenced_client.called_urls]
    assert called_suffixes[:2] == ["paper/search/match", "paper/search/match"]
    assert all(s == "paper/search" for s in called_suffixes[2:6])
    assert called_suffixes[6] == "paper/search/bulk"


@pytest.mark.asyncio
async def test_search_papers_match_title_case_variant_succeeds_via_lowercase_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: 'Attention Is All You Need' (title case) must resolve to the
    same paper as 'Attention is All you Need' even when the Semantic Scholar
    /paper/search/match endpoint is case-sensitive.

    The fix: search_papers_match now iterates over _title_lookup_queries
    (original → punctuation-normalised → lowercase) on the primary endpoint so
    a simple capitalization difference does not produce a false no-match.
    """

    async def fake_sleep(_: float) -> None:
        pass

    captured_queries: list[tuple[str, str]] = []

    # The primary endpoint returns 404 for the original title-case query but
    # succeeds for the lowercase variant, reflecting the real Semantic Scholar
    # API sensitivity observed during smoke testing.
    responses = [
        httpx.Response(
            status_code=404,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/match"),
        ),
        httpx.Response(
            status_code=200,
            request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search/match"),
            json={
                "paperId": "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
                "title": "Attention Is All You Need",
            },
        ),
    ]

    class SequencedAsyncClient:
        def __init__(self, queued_responses: list[httpx.Response]) -> None:
            self._responses = queued_responses

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, *, url: str, params, **kwargs):
            captured_queries.append((url, params["query"]))
            return self._responses.pop(0)

    monkeypatch.setattr(
        server.httpx,
        "AsyncClient",
        lambda timeout: SequencedAsyncClient(responses),
    )
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    result = await sc.search_papers_match("Attention Is All You Need")

    assert result["matchFound"] is True
    assert result["matchStrategy"] == "exact_title"
    assert result["paperId"] == "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    # The normalizedQuery field must be set because the successful query differed
    # from the original input.
    assert result.get("normalizedQuery") == "attention is all you need"
    # The first attempt used the original title-case query; the second used
    # the lowercase variant produced by _title_lookup_queries.
    assert captured_queries[0][0].endswith("/paper/search/match")
    assert captured_queries[0][1] == "Attention Is All You Need"
    assert captured_queries[1][0].endswith("/paper/search/match")
    assert captured_queries[1][1] == "attention is all you need"


def test_title_lookup_queries_includes_lowercase_variant() -> None:
    """_title_lookup_queries must emit a lowercase variant so that both the
    primary endpoint retry loop and the fuzzy-search fallback can handle
    API endpoints that are sensitive to title capitalisation."""
    client = server.SemanticScholarClient()

    # Plain title with no punctuation: original + lowercase (2 unique values)
    queries = client._title_lookup_queries("Attention Is All You Need")
    assert queries == [
        "Attention Is All You Need",
        "attention is all you need",
    ]

    # Title with punctuation: original + punct-normalised + lowercase +
    # lowercase-punct-normalised (up to 4 unique values).  The
    # lowercase-punct-normalised variant is important when the API rejects
    # colons or other punctuation in lowercase queries.
    queries = client._title_lookup_queries("ezMCDA: An Interactive Dashboard")
    assert queries == [
        "ezMCDA: An Interactive Dashboard",
        "ezMCDA An Interactive Dashboard",
        "ezmcda: an interactive dashboard",
        "ezmcda an interactive dashboard",
    ]


def test_no_match_message_suggests_get_paper_details() -> None:
    """The no-match message must suggest get_paper_details as a recovery path.

    This is a regression guard for the smoke-run finding that the no-match
    response left agents stranded with no mention of the identifier-based
    recovery tool.
    """
    import inspect

    source = inspect.getsource(
        server.SemanticScholarClient._search_papers_match_fallback,
    )
    assert "get_paper_details" in source


def test_normalize_author_search_query_falls_back_to_original_when_empty() -> None:
    client = server.SemanticScholarClient()

    normalized = client._normalize_author_search_query('  ""  ')

    assert normalized == '""'


@pytest.mark.asyncio
async def test_search_authors_normalizes_exact_name_punctuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """search_authors should sanitize exact-name punctuation before the request."""
    captured_queries: list[str] = []

    async def fake_sleep(_: float) -> None:
        pass

    class CapturingAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, *, params, **kwargs):
            captured_queries.append(params["query"])
            return DummyResponse(
                status_code=200,
                payload={"total": 1, "offset": 0, "data": [{"authorId": "9191855"}]},
            )

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda timeout: CapturingAsyncClient())
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    result = await sc.search_authors("Ryan L. Perroy")

    assert captured_queries == ["Ryan L Perroy"]
    assert result["data"][0]["authorId"] == "9191855"


@pytest.mark.asyncio
async def test_search_authors_400_error_mentions_common_name_disambiguation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_sleep(_: float) -> None:
        pass

    response = httpx.Response(
        status_code=400,
        request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/author/search"),
    )

    class RejectingAuthorSearchAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, **kwargs):
            return response

    monkeypatch.setattr(
        server.httpx,
        "AsyncClient",
        lambda timeout: RejectingAuthorSearchAsyncClient(),
    )
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    with pytest.raises(ValueError, match="affiliation, coauthor, venue, or topic"):
        await sc.search_authors('"Matthew Richardson" "SWCA"')


@pytest.mark.asyncio
async def test_get_author_info_surfaces_actionable_404_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_sleep(_: float) -> None:
        pass

    request = httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/author/9191855")
    response = httpx.Response(status_code=404, request=request)

    class MissingAuthorAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, **kwargs):
            return response

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda timeout: MissingAuthorAsyncClient())
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    with pytest.raises(ValueError, match="Use a Semantic Scholar authorId"):
        await sc.get_author_info("9191855")


@pytest.mark.asyncio
async def test_get_paper_authors_surfaces_portability_hint_for_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_sleep(_: float) -> None:
        pass

    request = httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/170189535/authors")
    response = httpx.Response(status_code=404, request=request)

    class MissingPaperAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, **kwargs):
            return response

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda timeout: MissingPaperAsyncClient())
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    with pytest.raises(ValueError, match="paper.recommendedExpansionId"):
        await sc.get_paper_authors("170189535")


@pytest.mark.asyncio
async def test_search_snippets_degrades_provider_400_to_empty_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_sleep(_: float) -> None:
        pass

    response = httpx.Response(
        status_code=400,
        request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/snippet/search"),
    )

    class RejectingSnippetAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, **kwargs):
            return response

    monkeypatch.setattr(
        server.httpx,
        "AsyncClient",
        lambda timeout: RejectingSnippetAsyncClient(),
    )
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    result = await sc.search_snippets('"exact phrase query"')

    assert result["data"] == []
    assert result["degraded"] is True
    assert result["providerStatusCode"] == 400
    assert "search_papers_match/search_papers" in result["message"]


@pytest.mark.asyncio
async def test_search_snippets_degrades_provider_429_without_backoff_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    response = httpx.Response(
        status_code=429,
        request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/snippet/search"),
    )

    class RateLimitedSnippetAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, **kwargs):
            return response

    monkeypatch.setattr(
        server.httpx,
        "AsyncClient",
        lambda timeout: RateLimitedSnippetAsyncClient(),
    )
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    result = await sc.search_snippets("attention")

    assert result["data"] == []
    assert result["degraded"] is True
    assert result["providerStatusCode"] == 429
    assert sleep_calls == []


# ---------------------------------------------------------------------------
# _normalize_paper_id
# ---------------------------------------------------------------------------


def test_normalize_paper_id_arxiv_mixed_case() -> None:
    """arXiv: prefix in mixed case should be normalized to ARXIV:."""
    sc = server.SemanticScholarClient()
    assert sc._normalize_paper_id("arXiv:1706.03762") == "ARXIV:1706.03762"
    assert sc._normalize_paper_id("arxiv:1706.03762") == "ARXIV:1706.03762"
    assert sc._normalize_paper_id("ARXIV:1706.03762") == "ARXIV:1706.03762"


def test_normalize_paper_id_bare_new_style_arxiv() -> None:
    """A bare new-style arXiv ID (YYMM.NNNNN) should get the ARXIV: prefix."""
    sc = server.SemanticScholarClient()
    assert sc._normalize_paper_id("1706.03762") == "ARXIV:1706.03762"
    assert sc._normalize_paper_id("1706.03762v1") == "ARXIV:1706.03762v1"
    assert sc._normalize_paper_id("2301.00001") == "ARXIV:2301.00001"


def test_normalize_paper_id_bare_old_style_arxiv() -> None:
    """A bare old-style arXiv ID (category/NNNNNNN) should get ARXIV: prefix."""
    sc = server.SemanticScholarClient()
    assert sc._normalize_paper_id("hep-ph/9705253") == "ARXIV:hep-ph/9705253"
    assert sc._normalize_paper_id("cs/0301023v2") == "ARXIV:cs/0301023v2"


def test_normalize_paper_id_arxiv_url() -> None:
    """arxiv.org abs and pdf URLs should be normalized to ARXIV:<id>."""
    sc = server.SemanticScholarClient()
    assert sc._normalize_paper_id("https://arxiv.org/abs/1706.03762") == "ARXIV:1706.03762"
    assert sc._normalize_paper_id("https://arxiv.org/pdf/1706.03762") == "ARXIV:1706.03762"


def test_normalize_paper_id_passthrough() -> None:
    """Non-arXiv identifiers must pass through unchanged."""
    sc = server.SemanticScholarClient()
    # Raw Semantic Scholar paperId hash
    assert (
        sc._normalize_paper_id("649def34f8be52c8b66281af98ae884c09aef38b") == "649def34f8be52c8b66281af98ae884c09aef38b"
    )
    # DOI prefix already in correct form
    assert sc._normalize_paper_id("DOI:10.48550/arXiv.1706.03762") == "DOI:10.48550/arXiv.1706.03762"
    # CorpusId
    assert sc._normalize_paper_id("CorpusId:215416146") == "CorpusId:215416146"


# ---------------------------------------------------------------------------
# get_paper_details error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_paper_details_surfaces_actionable_404_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 404 from get_paper_details should surface a ValueError with the
    portability hint so agents know to try recommendedExpansionId."""

    async def fake_sleep(_: float) -> None:
        pass

    request = httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/ARXIV:1706.03762")
    response = httpx.Response(status_code=404, request=request)

    class MissingPaperClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, **kwargs):
            return response

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda timeout: MissingPaperClient())
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    with pytest.raises(ValueError, match="recommendedExpansionId"):
        await sc.get_paper_details("arXiv:1706.03762")


@pytest.mark.asyncio
async def test_get_paper_details_surfaces_actionable_400_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 400 from get_paper_details should surface a ValueError with the
    portability hint so agents know the identifier format is invalid."""

    async def fake_sleep(_: float) -> None:
        pass

    request = httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/bad-id")
    response = httpx.Response(status_code=400, request=request)

    class RejectingPaperClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def request(self, **kwargs):
            return response

    monkeypatch.setattr(server.httpx, "AsyncClient", lambda timeout: RejectingPaperClient())
    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)

    sc = server.SemanticScholarClient()
    with pytest.raises(ValueError, match="get_paper_details"):
        await sc.get_paper_details("bad-id")
