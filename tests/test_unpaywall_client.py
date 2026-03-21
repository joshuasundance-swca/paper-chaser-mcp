import pytest

from scholar_search_mcp.clients.unpaywall import UnpaywallClient
from tests.helpers import DummyResponse


@pytest.mark.asyncio
async def test_unpaywall_get_open_access_requires_email() -> None:
    with pytest.raises(ValueError, match="UNPAYWALL_EMAIL"):
        await UnpaywallClient(email=None).get_open_access("10.1234/example")


@pytest.mark.asyncio
async def test_unpaywall_get_open_access_normalizes_payload(
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
                    "doi": "10.1234/example",
                    "is_oa": True,
                    "oa_status": "gold",
                    "journal_is_in_doaj": True,
                    "best_oa_location": {
                        "url": "https://oa.example/landing",
                        "url_for_pdf": "https://oa.example/file.pdf",
                        "license": "cc-by",
                    },
                    "oa_locations": [],
                },
            )

    monkeypatch.setattr(
        "scholar_search_mcp.clients.unpaywall.client.httpx.AsyncClient",
        lambda timeout: CapturingAsyncClient(),
    )

    result = await UnpaywallClient(email="oa@example.com").get_open_access(
        "https://doi.org/10.1234/example"
    )

    assert result is not None
    assert result["doi"] == "10.1234/example"
    assert result["isOa"] is True
    assert result["bestOaUrl"] == "https://oa.example/landing"
    assert result["pdfUrl"] == "https://oa.example/file.pdf"
    assert result["license"] == "cc-by"
    assert captured[0]["params"] == {"email": "oa@example.com"}
    assert captured[0]["follow_redirects"] is True


@pytest.mark.asyncio
async def test_unpaywall_get_open_access_retries_transient_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        DummyResponse(status_code=503, payload={"error": "temporary failure"}),
        DummyResponse(
            status_code=200,
            payload={
                "doi": "10.1234/recovered",
                "is_oa": False,
                "oa_status": "closed",
                "best_oa_location": None,
                "oa_locations": [],
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
        "scholar_search_mcp.clients.unpaywall.client.httpx.AsyncClient",
        lambda timeout: queue,
    )
    monkeypatch.setattr(
        "scholar_search_mcp.clients.unpaywall.client.asyncio.sleep",
        fake_sleep,
    )

    result = await UnpaywallClient(
        email="oa@example.com",
        max_retries=1,
    ).get_open_access("10.1234/recovered")

    assert queue.calls == 2
    assert sleep_calls == [0.5]
    assert result is not None
    assert result["doi"] == "10.1234/recovered"
    assert result["oaStatus"] == "closed"


@pytest.mark.asyncio
async def test_unpaywall_get_open_access_returns_none_on_404(
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
        "scholar_search_mcp.clients.unpaywall.client.httpx.AsyncClient",
        lambda timeout: NotFoundAsyncClient(),
    )

    result = await UnpaywallClient(email="oa@example.com").get_open_access(
        "10.1234/missing"
    )

    assert result is None


@pytest.mark.asyncio
async def test_unpaywall_client_reuses_lazy_async_client_and_closes_it(
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
                payload={
                    "doi": suffix.replace("%2F", "/"),
                    "is_oa": True,
                    "oa_status": "gold",
                    "best_oa_location": None,
                    "oa_locations": [],
                },
            )

        async def aclose(self) -> None:
            self.closed = True

    def _factory(timeout: float) -> ReusableAsyncClient:
        del timeout
        client = ReusableAsyncClient()
        created_clients.append(client)
        return client

    monkeypatch.setattr(
        "scholar_search_mcp.clients.unpaywall.client.httpx.AsyncClient",
        _factory,
    )

    client = UnpaywallClient(email="oa@example.com")
    await client.get_open_access("10.1234/first")
    result = await client.get_open_access("10.1234/second")

    assert len(created_clients) == 1
    assert result is not None
    assert created_clients[0].calls == [
        "https://api.unpaywall.org/v2/10.1234%2Ffirst",
        "https://api.unpaywall.org/v2/10.1234%2Fsecond",
    ]

    await client.aclose()

    assert created_clients[0].closed is True
