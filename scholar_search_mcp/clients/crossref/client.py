"""Crossref API client for explicit paper enrichment."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

from ...identifiers import normalize_doi
from ...models import Author, CrossrefEnrichment, CrossrefWorkSummary, dump_jsonable
from ...transport import asyncio, httpx, maybe_close_async_resource

logger = logging.getLogger("scholar-search-mcp")

CROSSREF_API_BASE = "https://api.crossref.org"
CROSSREF_SELECT = (
    "DOI,title,author,container-title,publisher,type,URL,"
    "is-referenced-by-count,published,issued,published-online,published-print"
)


class CrossrefClient:
    """Thin Crossref REST client for DOI-first metadata enrichment."""

    def __init__(
        self,
        *,
        mailto: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 1,
        base_delay: float = 0.5,
    ) -> None:
        self.mailto = (
            mailto.strip() if isinstance(mailto, str) and mailto.strip() else None
        )
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._http_client: Any | None = None

    def _headers(self) -> dict[str, str]:
        user_agent = "ScholarSearchMCP/1.0"
        if self.mailto:
            user_agent = f"{user_agent} (mailto:{self.mailto})"
        return {
            "Accept": "application/json",
            "User-Agent": user_agent,
        }

    def _get_http_client(self) -> Any:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self._http_client

    async def _request(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        url = f"{CROSSREF_API_BASE}{endpoint}"
        request_params = dict(params or {})
        if self.mailto:
            request_params.setdefault("mailto", self.mailto)

        client = self._get_http_client()
        for attempt in range(self.max_retries + 1):
            response = await client.get(
                url,
                params=request_params,
                headers=self._headers(),
                follow_redirects=True,
            )
            if response.status_code == 404:
                return None
            if (
                response.status_code in {429, 500, 502, 503, 504}
                and attempt < self.max_retries
            ):
                delay = self.base_delay * (2**attempt)
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = max(delay, float(retry_after))
                logger.warning(
                    "Crossref request to %s returned %s, retrying in %.1fs (%s/%s)",
                    url,
                    response.status_code,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                await asyncio.sleep(delay)
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                return None
            message = payload.get("message")
            return message if isinstance(message, dict) else None
        raise RuntimeError("Crossref request retry loop exited unexpectedly.")

    async def aclose(self) -> None:
        """Close the shared HTTP client, if one has been created."""
        client, self._http_client = self._http_client, None
        await maybe_close_async_resource(client)

    @staticmethod
    def _author_name(author: dict[str, Any]) -> str | None:
        given = str(author.get("given") or "").strip()
        family = str(author.get("family") or "").strip()
        sequence = [part for part in (given, family) if part]
        if sequence:
            return " ".join(sequence)
        name = str(author.get("name") or "").strip()
        return name or None

    @classmethod
    def _authors(cls, raw_authors: Any) -> list[Author]:
        authors: list[Author] = []
        if not isinstance(raw_authors, list):
            return authors
        for item in raw_authors:
            if not isinstance(item, dict):
                continue
            name = cls._author_name(item)
            if name:
                authors.append(Author(name=name))
        return authors

    @staticmethod
    def _first_text(value: Any) -> str | None:
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item.strip()
            return None
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    @classmethod
    def _publication_date(
        cls,
        message: dict[str, Any],
    ) -> tuple[str | None, int | None]:
        for field in ("published-print", "published-online", "published", "issued"):
            payload = message.get(field)
            if not isinstance(payload, dict):
                continue
            date_parts = payload.get("date-parts")
            if not isinstance(date_parts, list) or not date_parts:
                continue
            first = date_parts[0]
            if not isinstance(first, list) or not first:
                continue
            parts = [int(part) for part in first[:3] if isinstance(part, int)]
            if not parts:
                continue
            year = parts[0]
            if len(parts) == 1:
                return f"{year:04d}", year
            if len(parts) == 2:
                return f"{year:04d}-{parts[1]:02d}", year
            return f"{year:04d}-{parts[1]:02d}-{parts[2]:02d}", year
        return None, None

    @classmethod
    def _normalize_work(cls, message: dict[str, Any]) -> CrossrefWorkSummary | None:
        doi = normalize_doi(message.get("DOI"))
        title = cls._first_text(message.get("title"))
        if not doi and not title:
            return None
        publication_date, year = cls._publication_date(message)
        return CrossrefWorkSummary(
            doi=doi,
            title=title,
            authors=cls._authors(message.get("author")),
            venue=cls._first_text(message.get("container-title")),
            publisher=cls._first_text(message.get("publisher")),
            publicationType=cls._first_text(message.get("type")),
            publicationDate=publication_date,
            year=year,
            url=cls._first_text(message.get("URL"))
            or (f"https://doi.org/{doi}" if doi else None),
            citationCount=message.get("is-referenced-by-count"),
        )

    @staticmethod
    def to_enrichment(work: CrossrefWorkSummary | None) -> CrossrefEnrichment | None:
        if work is None:
            return None
        return CrossrefEnrichment(
            doi=work.doi,
            publisher=work.publisher,
            venue=work.venue,
            publicationType=work.publication_type,
            publicationDate=work.publication_date,
            year=work.year,
            url=work.url,
            citationCount=work.citation_count,
        )

    async def get_work(self, doi: str) -> dict[str, Any] | None:
        """Return one normalized Crossref work by DOI."""

        normalized_doi = normalize_doi(doi)
        if not normalized_doi:
            raise ValueError("Crossref lookups require a valid DOI or DOI URL.")
        payload = await self._request(f"/works/{quote(normalized_doi, safe='')}")
        if payload is None:
            return None
        work = self._normalize_work(payload)
        return dump_jsonable(work) if work is not None else None

    async def search_work(self, query: str) -> dict[str, Any] | None:
        """Return the top normalized Crossref work for a title/bibliographic query."""

        normalized_query = str(query or "").strip()
        if not normalized_query:
            raise ValueError("Crossref query fallback requires a non-empty query.")
        payload = await self._request(
            "/works",
            params={
                "query.bibliographic": normalized_query,
                "rows": 1,
                "select": CROSSREF_SELECT,
            },
        )
        if payload is None:
            return None
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            return None
        first = items[0]
        if not isinstance(first, dict):
            return None
        work = self._normalize_work(first)
        return dump_jsonable(work) if work is not None else None
