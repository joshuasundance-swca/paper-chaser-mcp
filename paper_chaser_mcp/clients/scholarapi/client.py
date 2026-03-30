"""ScholarAPI HTTP client."""

from __future__ import annotations

import base64
from typing import Any, Literal

from ...models import Author, Paper, dump_jsonable
from ...transport import httpx, maybe_close_async_resource
from .errors import (
    ScholarApiError,
    ScholarApiKeyMissingError,
    ScholarApiQuotaError,
    ScholarApiUpstreamError,
)

SCHOLARAPI_BASE_URL = "https://scholarapi.net/api/v1"
SCHOLARAPI_PAPER_ID_PREFIX = "ScholarAPI:"


class ScholarApiClient:
    """ScholarAPI client for explicit discovery and full-text workflows."""

    def __init__(self, api_key: str | None = None, timeout: float = 30.0) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self._http_client: Any | None = None

    def _get_http_client(self) -> Any:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self._http_client

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ScholarApiKeyMissingError(
                "SCHOLARAPI_API_KEY is not configured. Set SCHOLARAPI_API_KEY and "
                "PAPER_CHASER_ENABLE_SCHOLARAPI=true to use ScholarAPI tools."
            )
        return {"X-API-Key": self.api_key}

    @staticmethod
    def _response_context(response: Any) -> str:
        request_id = str(response.headers.get("X-Request-Id") or "").strip()
        request_cost = str(response.headers.get("X-Request-Cost") or "").strip()
        details: list[str] = []
        if request_id:
            details.append(f"request id {request_id}")
        if request_cost:
            details.append(f"request cost {request_cost}")
        if not details:
            return ""
        return f" ({'; '.join(details)})"

    async def aclose(self) -> None:
        client, self._http_client = self._http_client, None
        await maybe_close_async_resource(client)

    @staticmethod
    def _normalize_doi(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        if not normalized:
            return None
        lowered = normalized.lower()
        if lowered.startswith("doi:"):
            normalized = normalized[4:].strip()
        elif lowered.startswith("https://doi.org/"):
            normalized = normalized[16:]
        elif lowered.startswith("http://doi.org/"):
            normalized = normalized[15:]
        return normalized.strip().lstrip("/") or None

    @staticmethod
    def _year_from_date(value: Any) -> int | None:
        if not isinstance(value, str) or len(value) < 4 or not value[:4].isdigit():
            return None
        return int(value[:4])

    @staticmethod
    def _authors(names: Any) -> list[Author]:
        if not isinstance(names, list):
            return []
        return [Author(name=name.strip()) for name in names if isinstance(name, str) and name.strip()]

    @staticmethod
    def _raw_paper_id(paper_id: Any) -> str:
        normalized = str(paper_id or "").strip()
        if normalized.lower().startswith(SCHOLARAPI_PAPER_ID_PREFIX.lower()):
            return normalized[len(SCHOLARAPI_PAPER_ID_PREFIX) :].strip()
        return normalized

    @classmethod
    def _normalized_paper_id(cls, paper_id: Any) -> str:
        raw_id = cls._raw_paper_id(paper_id)
        if not raw_id:
            return raw_id
        return f"{SCHOLARAPI_PAPER_ID_PREFIX}{raw_id}"

    @classmethod
    def _paper_from_result(cls, result: dict[str, Any]) -> dict[str, Any]:
        raw_id = str(result.get("id") or "").strip()
        source_id = raw_id or None
        paper_id = cls._normalized_paper_id(raw_id) or None
        doi = cls._normalize_doi(result.get("doi"))
        recommended_expansion_id = doi or None
        expansion_id_status: Literal["portable", "not_portable"] = (
            "portable" if recommended_expansion_id else "not_portable"
        )
        paper = Paper(
            paperId=paper_id,
            title=result.get("title"),
            abstract=result.get("abstract"),
            year=cls._year_from_date(result.get("published_date")),
            authors=cls._authors(result.get("authors")),
            venue=result.get("journal"),
            publicationDate=result.get("published_date"),
            url=result.get("url"),
            source="scholarapi",
            sourceId=source_id,
            canonicalId=doi or source_id,
            recommendedExpansionId=recommended_expansion_id,
            expansionIdStatus=expansion_id_status,
        )
        return dump_jsonable(
            paper.model_copy(
                update={
                    "hasText": bool(result.get("has_text")) if result.get("has_text") is not None else None,
                    "hasPdf": bool(result.get("has_pdf")) if result.get("has_pdf") is not None else None,
                    "indexedAt": result.get("indexed_at"),
                    "publishedDateRaw": result.get("published_date_raw"),
                    "journalPublisher": result.get("journal_publisher"),
                    "journalIssn": result.get("journal_issn"),
                    "journalIssue": result.get("journal_issue"),
                    "journalPages": result.get("journal_pages"),
                }
            )
        )

    async def _get(self, endpoint: str, *, params: dict[str, Any] | None = None) -> Any:
        try:
            response = await self._get_http_client().get(
                f"{SCHOLARAPI_BASE_URL}{endpoint}",
                headers=self._headers(),
                params=params or None,
            )
        except ScholarApiKeyMissingError:
            raise
        except Exception as exc:
            raise ScholarApiUpstreamError(f"ScholarAPI request failed: {exc}") from exc

        context = self._response_context(response)

        if response.status_code == 401 or response.status_code == 403:
            raise ScholarApiKeyMissingError(f"ScholarAPI authentication failed. Check SCHOLARAPI_API_KEY.{context}")
        if response.status_code == 402:
            payload = response.json() if hasattr(response, "json") else {}
            message = str(payload.get("message") or payload.get("detail") or "Insufficient credits")
            raise ScholarApiQuotaError(f"ScholarAPI credit exhaustion: {message}{context}")
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "later")
            raise ScholarApiQuotaError(f"ScholarAPI rate limit reached. Retry after {retry_after}.{context}")
        if response.status_code >= 500:
            raise ScholarApiUpstreamError(f"ScholarAPI returned HTTP {response.status_code}.{context}")
        if response.status_code == 404:
            raise ScholarApiError(f"ScholarAPI resource not found or content unavailable.{context}")
        if response.status_code >= 400:
            raise ScholarApiError(f"ScholarAPI returned HTTP {response.status_code}.{context}")
        return response

    @staticmethod
    def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in params.items() if value is not None}

    async def search(
        self,
        query: str,
        limit: int = 10,
        cursor: str | None = None,
        indexed_after: str | None = None,
        indexed_before: str | None = None,
        published_after: str | None = None,
        published_before: str | None = None,
        has_text: bool | None = None,
        has_pdf: bool | None = None,
    ) -> dict[str, Any]:
        response = await self._get(
            "/search",
            params=self._clean_params(
                {
                    "q": query,
                    "limit": limit,
                    "cursor": cursor,
                    "indexed_after": indexed_after,
                    "indexed_before": indexed_before,
                    "published_after": published_after,
                    "published_before": published_before,
                    "has_text": has_text,
                    "has_pdf": has_pdf,
                }
            ),
        )
        payload = response.json()
        return {
            "provider": "scholarapi",
            "total": len(payload.get("results") or []),
            "offset": 0,
            "data": [self._paper_from_result(item) for item in payload.get("results") or [] if isinstance(item, dict)],
            "pagination": {
                "hasMore": bool(payload.get("next_cursor")),
                "nextCursor": payload.get("next_cursor"),
            },
            "requestId": response.headers.get("X-Request-Id"),
            "requestCost": response.headers.get("X-Request-Cost"),
        }

    async def list_papers(
        self,
        *,
        query: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
        indexed_after: str | None = None,
        indexed_before: str | None = None,
        published_after: str | None = None,
        published_before: str | None = None,
        has_text: bool | None = None,
        has_pdf: bool | None = None,
    ) -> dict[str, Any]:
        response = await self._get(
            "/list",
            params=self._clean_params(
                {
                    "q": query,
                    "limit": limit,
                    "indexed_after": cursor or indexed_after,
                    "indexed_before": indexed_before,
                    "published_after": published_after,
                    "published_before": published_before,
                    "has_text": has_text,
                    "has_pdf": has_pdf,
                }
            ),
        )
        payload = response.json()
        return {
            "provider": "scholarapi",
            "total": len(payload.get("results") or []),
            "offset": 0,
            "data": [self._paper_from_result(item) for item in payload.get("results") or [] if isinstance(item, dict)],
            "pagination": {
                "hasMore": bool(payload.get("next_indexed_after")),
                "nextCursor": payload.get("next_indexed_after"),
            },
            "requestId": response.headers.get("X-Request-Id"),
            "requestCost": response.headers.get("X-Request-Cost"),
        }

    async def get_text(self, paper_id: str) -> dict[str, Any]:
        raw_paper_id = self._raw_paper_id(paper_id)
        response = await self._get(f"/text/{raw_paper_id}")
        return {
            "provider": "scholarapi",
            "paperId": self._normalized_paper_id(raw_paper_id),
            "source": "scholarapi",
            "text": response.text if hasattr(response, "text") else response.json(),
        }

    async def get_texts(self, paper_ids: list[str]) -> dict[str, Any]:
        raw_paper_ids = [self._raw_paper_id(paper_id) for paper_id in paper_ids]
        response = await self._get(f"/texts/{','.join(raw_paper_ids)}")
        payload = response.json()
        texts = payload.get("results") or []
        return {
            "provider": "scholarapi",
            "results": [
                {
                    "paperId": self._normalized_paper_id(paper_id),
                    "source": "scholarapi",
                    "text": texts[index] if index < len(texts) else None,
                }
                for index, paper_id in enumerate(raw_paper_ids)
            ],
        }

    async def get_pdf(self, paper_id: str) -> dict[str, Any]:
        raw_paper_id = self._raw_paper_id(paper_id)
        response = await self._get(f"/pdf/{raw_paper_id}")
        content = bytes(getattr(response, "content", b""))
        return {
            "provider": "scholarapi",
            "paperId": self._normalized_paper_id(raw_paper_id),
            "source": "scholarapi",
            "mimeType": response.headers.get("Content-Type", "application/pdf").split(";", 1)[0],
            "contentBase64": base64.b64encode(content).decode("ascii"),
            "byteLength": len(content),
        }
