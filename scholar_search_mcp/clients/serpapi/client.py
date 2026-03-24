"""SerpApi Google Scholar HTTP client."""

import logging
from typing import Any, Optional

from ...transport import httpx, maybe_close_async_resource
from .errors import (
    SerpApiError,
    SerpApiKeyMissingError,
    SerpApiQuotaError,
    SerpApiUpstreamError,
)
from .normalize import (
    _parse_year_range,
    normalize_author_article_result,
    normalize_organic_result,
)

SERPAPI_BASE_URL = "https://serpapi.com/search"
SERPAPI_ACCOUNT_URL = "https://serpapi.com/account.json"

logger = logging.getLogger("scholar-search-mcp")


class SerpApiScholarClient:
    """SerpApi Google Scholar client (https://serpapi.com/google-scholar-api).

    All requests are cache-friendly by default: ``no_cache`` is never set to
    ``true`` so SerpApi serves cached results for identical queries within the
    1-hour cache window.  This avoids burning quota on repeated identical
    requests.

    This is a **paid** upstream API.  Enable it explicitly by setting
    ``SCHOLAR_SEARCH_ENABLE_SERPAPI=true`` and providing ``SERPAPI_API_KEY``.
    """

    def __init__(self, api_key: Optional[str] = None, timeout: float = 30.0) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self._http_client: Any | None = None

    def _get_http_client(self) -> Any:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self._http_client

    def _check_key(self) -> None:
        """Raise ``SerpApiKeyMissingError`` if no API key is configured."""
        if not self.api_key:
            raise SerpApiKeyMissingError(
                "SERPAPI_API_KEY is not configured. "
                "Set the SERPAPI_API_KEY environment variable to use "
                "SerpApi Google Scholar. SerpApi is a paid service — "
                "visit https://serpapi.com to obtain a key. "
                "To disable this provider set SCHOLAR_SEARCH_ENABLE_SERPAPI=false."
            )

    async def _get_json(
        self,
        *,
        url: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform an authenticated GET to the SerpApi endpoint.

        Injects ``api_key`` and translates HTTP/upstream errors into typed
        ``SerpApiError`` subclasses so callers can handle them uniformly.
        """
        self._check_key()
        # Build a clean copy so we don't mutate the caller's dict.
        request_params = dict(params)
        request_params["api_key"] = self.api_key

        try:
            client = self._get_http_client()
            response = await client.get(url, params=request_params)
        except Exception as exc:
            logger.warning("SerpApi request failed: %s", exc)
            raise SerpApiUpstreamError(
                f"SerpApi request failed: {exc}. This may be a transient network error — try again later."
            ) from exc

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "a while")
            raise SerpApiQuotaError(
                f"SerpApi quota or rate limit reached (HTTP 429). "
                f"Retry after {retry_after}. "
                "Consider reducing request frequency or upgrading your SerpApi plan."
            )
        if response.status_code >= 500:
            raise SerpApiUpstreamError(
                f"SerpApi returned HTTP {response.status_code}. This is a transient upstream error — try again later."
            )
        try:
            response.raise_for_status()
        except Exception as exc:
            raise SerpApiError(
                f"SerpApi returned HTTP {response.status_code}. Check your API key and request parameters."
            ) from exc

        data: dict[str, Any] = response.json()

        # Translate SerpApi application-level errors.
        error_msg = data.get("error")
        if error_msg:
            msg = str(error_msg).lower()
            if "api_key" in msg or "invalid" in msg or "unauthorized" in msg:
                raise SerpApiKeyMissingError(
                    f"SerpApi authentication error: {error_msg}. Check your SERPAPI_API_KEY value."
                )
            if "quota" in msg or "limit" in msg or "credits" in msg:
                raise SerpApiQuotaError(f"SerpApi quota error: {error_msg}. Consider upgrading your SerpApi plan.")
            raise SerpApiError(f"SerpApi returned an error: {error_msg}. Check your request parameters and try again.")

        return data

    async def aclose(self) -> None:
        """Close the shared HTTP client, if one has been created."""
        client, self._http_client = self._http_client, None
        await maybe_close_async_resource(client)

    async def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        return await self._get_json(url=SERPAPI_BASE_URL, params=params)

    @staticmethod
    def _normalize_query(value: str) -> str:
        return " ".join(value.strip().split())

    @staticmethod
    def _pagination(next_start: int | None) -> dict[str, Any]:
        return {
            "pagination": {
                "hasMore": next_start is not None,
                "nextCursor": str(next_start) if next_start is not None else None,
            }
        }

    @staticmethod
    def _next_start(current_start: int, page_size: int, returned: int) -> int | None:
        if returned < page_size:
            return None
        return current_start + returned

    @staticmethod
    def _normalize_search_results(
        organic_results: list[dict[str, Any]],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        papers: list[dict[str, Any]] = []
        for result in organic_results[:limit]:
            normalized = normalize_organic_result(result)
            if normalized is not None:
                papers.append(normalized)
        return papers

    async def search(
        self,
        query: str,
        limit: int = 10,
        year: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Search Google Scholar via SerpApi.

        Returns a list of normalized paper dicts (up to *limit*, max 20 per
        call since SerpApi Scholar caps organic results at 20 per page).

        *year* accepts the same format used by ``search_papers``:
        ``"2023"`` or ``"2020-2023"``.  It is translated to ``as_ylo`` /
        ``as_yhi`` SerpApi parameters.
        """
        num = min(limit, 20)  # SerpApi Scholar max organic results per page
        params: dict[str, Any] = {
            "engine": "google_scholar",
            # Normalize whitespace for consistent cache hits.
            "q": self._normalize_query(query),
            "num": num,
            "hl": "en",
        }
        if year:
            year_low, year_high = _parse_year_range(year)
            if year_low is not None:
                params["as_ylo"] = year_low
            if year_high is not None:
                params["as_yhi"] = year_high

        data = await self._get(params)
        organic_results: list[dict[str, Any]] = data.get("organic_results") or []
        return self._normalize_search_results(organic_results, limit=limit)

    async def search_cited_by(
        self,
        cites_id: str,
        *,
        query: str | None = None,
        limit: int = 10,
        year: Optional[str] = None,
        start: int = 0,
    ) -> dict[str, Any]:
        """Search within the citing-articles view for one Scholar cites_id."""

        num = min(limit, 20)
        params: dict[str, Any] = {
            "engine": "google_scholar",
            "cites": cites_id.strip(),
            "num": num,
            "start": max(start, 0),
            "hl": "en",
        }
        if query:
            params["q"] = self._normalize_query(query)
        if year:
            year_low, year_high = _parse_year_range(year)
            if year_low is not None:
                params["as_ylo"] = year_low
            if year_high is not None:
                params["as_yhi"] = year_high
        data = await self._get(params)
        organic_results: list[dict[str, Any]] = data.get("organic_results") or []
        papers = self._normalize_search_results(organic_results, limit=limit)
        next_start = self._next_start(start, num, len(organic_results))
        return {
            "provider": "serpapi_google_scholar",
            "total": len(papers),
            "offset": start,
            "citesId": cites_id.strip(),
            "data": papers,
            **self._pagination(next_start),
        }

    async def search_versions(
        self,
        cluster_id: str,
        *,
        limit: int = 10,
        start: int = 0,
    ) -> dict[str, Any]:
        """Return all-versions results for one Scholar cluster_id."""

        num = min(limit, 20)
        params: dict[str, Any] = {
            "engine": "google_scholar",
            "cluster": cluster_id.strip(),
            "num": num,
            "start": max(start, 0),
            "hl": "en",
        }
        data = await self._get(params)
        organic_results: list[dict[str, Any]] = data.get("organic_results") or []
        papers = self._normalize_search_results(organic_results, limit=limit)
        next_start = self._next_start(start, num, len(organic_results))
        return {
            "provider": "serpapi_google_scholar",
            "total": len(papers),
            "offset": start,
            "clusterId": cluster_id.strip(),
            "data": papers,
            **self._pagination(next_start),
        }

    async def get_author_profile(self, author_id: str) -> dict[str, Any]:
        """Return structured Google Scholar author profile metadata."""

        data = await self._get(
            {
                "engine": "google_scholar_author",
                "author_id": author_id.strip(),
                "hl": "en",
            }
        )
        author = data.get("author") or {}
        cited_by = data.get("cited_by") or {}
        table = cited_by.get("table") or []
        total_citations: int | None = None
        if isinstance(table, list):
            for row in table:
                if not isinstance(row, dict):
                    continue
                if str(row.get("citations") or "").strip().lower() == "all":
                    try:
                        citation_value = row.get("value")
                        total_citations = int(str(citation_value))
                    except (TypeError, ValueError):
                        total_citations = None
                    break
        affiliations = author.get("affiliations")
        interests = [
            interest.get("title")
            for interest in author.get("interests") or []
            if isinstance(interest, dict) and interest.get("title")
        ]
        coauthors = [
            {
                "authorId": item.get("author_id"),
                "name": item.get("name"),
                "affiliations": item.get("affiliations"),
                "link": item.get("link"),
            }
            for item in data.get("co_authors") or []
            if isinstance(item, dict)
        ]
        return {
            "provider": "serpapi_google_scholar",
            "authorId": author.get("author_id") or author_id.strip(),
            "name": author.get("name"),
            "affiliations": [affiliations] if isinstance(affiliations, str) and affiliations.strip() else [],
            "homepage": author.get("website"),
            "citationCount": total_citations,
            "interests": interests,
            "coAuthors": coauthors,
        }

    async def get_author_articles(
        self,
        author_id: str,
        *,
        limit: int = 10,
        start: int = 0,
        sort: str | None = None,
    ) -> dict[str, Any]:
        """Return normalized papers from one Scholar author profile."""

        num = min(limit, 20)
        params: dict[str, Any] = {
            "engine": "google_scholar_author",
            "author_id": author_id.strip(),
            "hl": "en",
            "num": num,
            "start": max(start, 0),
        }
        if sort:
            params["sort"] = sort
        data = await self._get(params)
        raw_articles: list[dict[str, Any]] = data.get("articles") or []
        papers: list[dict[str, Any]] = []
        for article in raw_articles[:limit]:
            normalized = normalize_author_article_result(article)
            if normalized is not None:
                papers.append(normalized)
        next_start = self._next_start(start, num, len(raw_articles))
        return {
            "provider": "serpapi_google_scholar",
            "authorId": author_id.strip(),
            "total": len(papers),
            "offset": start,
            "data": papers,
            **self._pagination(next_start),
        }

    async def get_account_status(self) -> dict[str, Any]:
        """Return SerpApi account and quota metadata without consuming credits."""

        data = await self._get_json(url=SERPAPI_ACCOUNT_URL, params={})
        return {"provider": "serpapi_google_scholar", **data}

    async def get_citation_formats(
        self,
        result_id: str,
    ) -> dict[str, Any]:
        """Return citation export formats for a Google Scholar paper.

        Uses the ``google_scholar_cite`` SerpApi engine.  *result_id* must be
        the Scholar ``result_id`` from a previous ``search_papers`` call — it
        is available as ``paper.scholarResultId`` on any
        ``serpapi_google_scholar`` result.

        **Do not use** ``paper.sourceId`` here: ``sourceId`` follows the
        priority ``result_id → cluster_id → cites_id`` and may be a
        ``cluster_id`` or ``cites_id`` rather than a ``result_id``.  Only a
        genuine ``result_id`` is accepted by ``google_scholar_cite``.

        Returns the raw SerpApi response dict containing ``citations`` (text
        formats) and ``links`` (export download links).  The caller is
        responsible for reshaping this into the normalized tool response.

        This is a **paid** SerpApi request (results cached for 1 hour).
        """
        if not result_id or not result_id.strip():
            raise ValueError(
                "result_id must not be empty. "
                "Provide the Scholar result_id from a search_papers result — "
                "it is available as sourceId on serpapi_google_scholar papers."
            )
        params: dict[str, Any] = {
            "engine": "google_scholar_cite",
            "q": result_id.strip(),
        }
        return await self._get(params)
