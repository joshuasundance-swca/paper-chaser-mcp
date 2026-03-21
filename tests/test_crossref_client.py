import pytest

from scholar_search_mcp.clients.crossref import CrossrefClient
from tests.helpers import DummyResponse


@pytest.mark.asyncio
async def test_crossref_get_work_normalizes_response_and_sets_contact_headers(
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
                    "message": {
                        "DOI": "10.1234/example",
                        "title": ["Crossref title"],
                        "author": [{"given": "Ada", "family": "Lovelace"}],
                        "container-title": ["Journal of Testing"],
                        "publisher": "Crossref Publisher",
                        "type": "journal-article",
                        "URL": "https://doi.org/10.1234/example",
                        "is-referenced-by-count": 12,
                        "issued": {"date-parts": [[2024, 5, 1]]},
                    }
                },
            )

    monkeypatch.setattr(
        "scholar_search_mcp.clients.crossref.client.httpx.AsyncClient",
        lambda timeout: CapturingAsyncClient(),
    )

    result = await CrossrefClient(mailto="ops@example.com").get_work(
        "https://doi.org/10.1234/example"
    )

    assert result is not None
    assert result["doi"] == "10.1234/example"
    assert result["title"] == "Crossref title"
    assert result["authors"][0]["name"] == "Ada Lovelace"
    assert result["citationCount"] == 12
    assert captured[0]["params"]["mailto"] == "ops@example.com"
    assert captured[0]["params"]["select"]
    assert "mailto:ops@example.com" in captured[0]["headers"]["User-Agent"]
    assert captured[0]["follow_redirects"] is True


@pytest.mark.asyncio
async def test_crossref_get_work_retries_transient_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        DummyResponse(status_code=500, payload={"message": "temporary failure"}),
        DummyResponse(
            status_code=200,
            payload={
                "message": {
                    "DOI": "10.1234/recovered",
                    "title": ["Recovered Crossref title"],
                    "issued": {"date-parts": [[2023]]},
                }
            },
        ),
    ]
    sleep_calls: list[float] = []

    class QueueAsyncClient:
        def __init__(self) -> None:
            self.calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, url: str, **kwargs):
            response = responses[self.calls]
            self.calls += 1
            return response

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    queue = QueueAsyncClient()
    monkeypatch.setattr(
        "scholar_search_mcp.clients.crossref.client.httpx.AsyncClient",
        lambda timeout: queue,
    )
    monkeypatch.setattr(
        "scholar_search_mcp.clients.crossref.client.asyncio.sleep",
        fake_sleep,
    )

    result = await CrossrefClient(max_retries=1).get_work("10.1234/recovered")

    assert queue.calls == 2
    assert sleep_calls == [0.5]
    assert result is not None
    assert result["doi"] == "10.1234/recovered"


@pytest.mark.asyncio
async def test_crossref_get_work_returns_none_on_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NotFoundAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, url: str, **kwargs):
            return DummyResponse(status_code=404)

    monkeypatch.setattr(
        "scholar_search_mcp.clients.crossref.client.httpx.AsyncClient",
        lambda timeout: NotFoundAsyncClient(),
    )

    result = await CrossrefClient().get_work("10.1234/missing")

    assert result is None


@pytest.mark.asyncio
async def test_crossref_client_reuses_lazy_async_client_and_closes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_clients: list["ReusableAsyncClient"] = []

    class ReusableAsyncClient:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.closed = False

        async def get(self, url: str, **kwargs):
            del kwargs
            self.calls.append(url)
            suffix = self.calls[-1].rsplit("/", 1)[-1]
            return DummyResponse(
                status_code=200,
                payload={"message": {"DOI": suffix, "title": ["Reusable"]}},
            )

        async def aclose(self) -> None:
            self.closed = True

    def _factory(timeout: float) -> ReusableAsyncClient:
        del timeout
        client = ReusableAsyncClient()
        created_clients.append(client)
        return client

    monkeypatch.setattr(
        "scholar_search_mcp.clients.crossref.client.httpx.AsyncClient",
        _factory,
    )

    client = CrossrefClient()
    await client.get_work("10.1234/first")
    result = await client.get_work("10.1234/second")

    assert len(created_clients) == 1
    assert result is not None
    assert created_clients[0].calls == [
        "https://api.crossref.org/works/10.1234%2Ffirst",
        "https://api.crossref.org/works/10.1234%2Fsecond",
    ]

    await client.aclose()

    assert created_clients[0].closed is True
