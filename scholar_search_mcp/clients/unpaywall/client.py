"""Unpaywall API client for explicit paper OA enrichment."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

from ...identifiers import normalize_doi
from ...models import UnpaywallEnrichment, dump_jsonable
from ...transport import asyncio, httpx, maybe_close_async_resource

logger = logging.getLogger("scholar-search-mcp")

UNPAYWALL_API_BASE = "https://api.unpaywall.org/v2"


class UnpaywallClient:
    """Thin Unpaywall client for DOI-based OA and PDF discovery."""

    def __init__(
        self,
        *,
        email: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 1,
        base_delay: float = 0.5,
    ) -> None:
        self.email = email.strip() if isinstance(email, str) and email.strip() else None
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._http_client: Any | None = None

    def _get_http_client(self) -> Any:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self._http_client

    async def _request(self, doi: str) -> dict[str, Any] | None:
        if not self.email:
            raise ValueError("UNPAYWALL_EMAIL must be configured to use Unpaywall enrichment.")
        normalized_doi = normalize_doi(doi)
        if not normalized_doi:
            raise ValueError("Unpaywall lookups require a valid DOI or DOI URL.")
        url = f"{UNPAYWALL_API_BASE}/{quote(normalized_doi, safe='')}"
        client = self._get_http_client()
        for attempt in range(self.max_retries + 1):
            response = await client.get(
                url,
                params={"email": self.email},
                headers={"Accept": "application/json"},
                follow_redirects=True,
            )
            if response.status_code == 404:
                return None
            if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                delay = self.base_delay * (2**attempt)
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = max(delay, float(retry_after))
                logger.warning(
                    ("Unpaywall request to %s returned %s, retrying in %.1fs (%s/%s)"),
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
            return payload if isinstance(payload, dict) else None
        raise RuntimeError("Unpaywall request retry loop exited unexpectedly.")

    @staticmethod
    def _best_oa_url(location: dict[str, Any] | None) -> str | None:
        if not isinstance(location, dict):
            return None
        url = location.get("url")
        return url.strip() if isinstance(url, str) and url.strip() else None

    @classmethod
    def _pdf_url(cls, payload: dict[str, Any]) -> str | None:
        locations = []
        best_oa_location = payload.get("best_oa_location")
        if isinstance(best_oa_location, dict):
            locations.append(best_oa_location)
        raw_locations = payload.get("oa_locations")
        if isinstance(raw_locations, list):
            locations.extend(item for item in raw_locations if isinstance(item, dict))
        for location in locations:
            pdf_url = location.get("url_for_pdf")
            if isinstance(pdf_url, str) and pdf_url.strip():
                return pdf_url.strip()
        return None

    def to_enrichment(self, payload: dict[str, Any]) -> UnpaywallEnrichment:
        best_oa_location = payload.get("best_oa_location")
        return UnpaywallEnrichment(
            doi=normalize_doi(payload.get("doi")),
            isOa=payload.get("is_oa"),
            oaStatus=payload.get("oa_status"),
            bestOaUrl=self._best_oa_url(best_oa_location),
            pdfUrl=self._pdf_url(payload),
            license=(best_oa_location.get("license") if isinstance(best_oa_location, dict) else payload.get("license")),
            journalIsInDoaj=payload.get("journal_is_in_doaj"),
        )

    async def get_open_access(self, doi: str) -> dict[str, Any] | None:
        """Return one normalized Unpaywall OA payload for a DOI."""

        payload = await self._request(doi)
        if payload is None:
            return None
        return dump_jsonable(self.to_enrichment(payload))

    async def aclose(self) -> None:
        """Close the shared HTTP client, if one has been created."""
        client, self._http_client = self._http_client, None
        await maybe_close_async_resource(client)
