"""GovInfo authoritative retrieval client for Federal Register and CFR."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from ...document_markdown import DocumentMarkdownConverter
from ...models import CfrTextResponse, FederalRegisterDocument, FederalRegisterDocumentTextResponse, dump_jsonable
from ...transport import httpx, maybe_close_async_resource

GOVINFO_API_BASE = "https://api.govinfo.gov"
GOVINFO_FR_LINK_RE = re.compile(r"/link/fr/(?P<volume>\d+)/(?P<page>\d+)", re.IGNORECASE)
FR_CITATION_RE = re.compile(r"^(?P<volume>\d+)\s*F\.?R\.?\s*(?P<page>\d+)$", re.IGNORECASE)


class GovInfoApiError(RuntimeError):
    """Base GovInfo API error."""


class GovInfoAuthError(GovInfoApiError):
    """Raised when GovInfo authentication fails."""


class GovInfoRateLimitError(GovInfoApiError):
    """Raised when GovInfo rate limits the caller."""


class GovInfoClient:
    """Thin GovInfo client for authoritative FR and CFR retrieval."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = GOVINFO_API_BASE,
        timeout: float = 30.0,
        document_timeout: float = 60.0,
        max_document_size_mb: int = 25,
        markdown_converter: DocumentMarkdownConverter | None = None,
    ) -> None:
        self.api_key = str(api_key or "").strip() or None
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.document_timeout = document_timeout
        self.max_document_bytes = max(int(max_document_size_mb), 1) * 1024 * 1024
        self._markdown_converter: Any = markdown_converter or DocumentMarkdownConverter()
        self._http_client: Any | None = None
        self._document_client: Any | None = None

    def _get_http_client(self) -> Any:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.timeout, follow_redirects=True)
        return self._http_client

    def _get_document_client(self) -> Any:
        if self._document_client is None:
            self._document_client = httpx.AsyncClient(timeout=self.document_timeout, follow_redirects=True)
        return self._document_client

    async def aclose(self) -> None:
        http_client, self._http_client = self._http_client, None
        document_client, self._document_client = self._document_client, None
        await maybe_close_async_resource(http_client)
        await maybe_close_async_resource(document_client)

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-Api-Key"] = self.api_key
        return headers

    @staticmethod
    def _extract_api_error(payload: dict[str, Any]) -> str | None:
        for key in ("errorCode", "error", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = await self._get_http_client().request(
            method,
            f"{self.base_url}{path}",
            params=params,
            json=json_payload,
            headers=self._headers(),
        )
        payload = response.json() if "json" in response.headers.get("content-type", "") else None
        if response.status_code == 429:
            detail = self._extract_api_error(payload) if isinstance(payload, dict) else "OVER_RATE_LIMIT"
            raise GovInfoRateLimitError(detail or "OVER_RATE_LIMIT")
        if response.status_code == 403:
            detail = self._extract_api_error(payload) if isinstance(payload, dict) else "API_KEY_INVALID"
            raise GovInfoAuthError(detail or "API_KEY_INVALID")
        if response.status_code == 404:
            raise GovInfoApiError("NOT_FOUND")
        response.raise_for_status()
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object from GovInfo.")
        return payload

    async def _request_content(self, url: str) -> tuple[bytes, str | None]:
        response = await self._get_document_client().get(url, headers=self._headers())
        if response.status_code == 429:
            raise GovInfoRateLimitError("OVER_RATE_LIMIT")
        if response.status_code == 403:
            raise GovInfoAuthError("API_KEY_INVALID")
        if response.status_code == 404:
            raise GovInfoApiError("NOT_FOUND")
        response.raise_for_status()
        return response.content, response.headers.get("content-type")

    @staticmethod
    def _fr_issue_package_id(publication_date: str | None) -> str | None:
        if not publication_date:
            return None
        try:
            issued = datetime.strptime(publication_date, "%Y-%m-%d").date()
        except ValueError:
            return None
        return f"FR-{issued.isoformat()}"

    @staticmethod
    def _parse_fr_identifier(identifier: str) -> dict[str, str] | None:
        normalized = str(identifier or "").strip()
        if not normalized:
            return None
        link_match = GOVINFO_FR_LINK_RE.search(normalized)
        if link_match:
            return {"citation": f"{link_match.group('volume')} FR {link_match.group('page')}"}
        citation_match = FR_CITATION_RE.match(normalized)
        if citation_match:
            return {"citation": f"{citation_match.group('volume')} FR {citation_match.group('page')}"}
        if re.fullmatch(r"\d{4}-\d{4,6}", normalized):
            return {"documentNumber": normalized}
        return None

    async def _govinfo_search(self, query: str) -> dict[str, Any]:
        return await self._request_json(
            "POST",
            "/search",
            json_payload={
                "query": query,
                "pageSize": 10,
                "offsetMark": "*",
                "sorts": [{"field": "score", "sortOrder": "DESC"}],
            },
        )

    async def _resolve_fr_from_citation(self, citation: str) -> tuple[str | None, str | None, dict[str, Any] | None]:
        search = await self._govinfo_search(f'collection:(FR) and frcitation:"{citation}"')
        results = search.get("results") or []
        if not isinstance(results, list) or not results:
            return None, None, None
        first = results[0]
        if not isinstance(first, dict):
            return None, None, None
        return (
            str(first.get("packageId") or "").strip() or None,
            str(first.get("granuleId") or "").strip() or None,
            first,
        )

    @staticmethod
    def _cfr_citation(title_number: int, part_number: int, section_number: str | None) -> str:
        if section_number:
            return f"{title_number} CFR {part_number}.{section_number}"
        return f"{title_number} CFR Part {part_number}"

    async def get_federal_register_document(
        self,
        *,
        identifier: str,
        federal_register_client: Any | None = None,
    ) -> dict[str, Any]:
        parsed = self._parse_fr_identifier(identifier)
        if parsed is None:
            raise ValueError("identifier must be an FR document number, FR citation, or GovInfo FR link.")

        document_payload: dict[str, Any] | None = None
        if parsed.get("documentNumber") and federal_register_client is not None:
            document_payload = await federal_register_client.get_document(parsed["documentNumber"])

        package_id = None
        granule_id = None
        if self.api_key:
            if parsed.get("citation"):
                package_id, granule_id, result = await self._resolve_fr_from_citation(parsed["citation"])
                if document_payload is None and isinstance(result, dict):
                    document_payload = {
                        "documentNumber": str(result.get("granuleId") or "").strip() or None,
                        "title": str(result.get("title") or "").strip() or None,
                        "publicationDate": str(result.get("dateIssued") or "").strip() or None,
                        "citation": parsed["citation"],
                        "govInfoPackageId": package_id,
                        "govInfoGranuleId": granule_id,
                    }
            elif document_payload is not None:
                package_id = self._fr_issue_package_id(document_payload.get("publicationDate"))
                granule_id = document_payload.get("documentNumber")

        normalized_document = FederalRegisterDocument.model_validate(document_payload or {})
        if package_id and granule_id:
            summary = await self._request_json(
                "GET",
                f"/packages/{package_id}/granules/{granule_id}/summary",
            )
            download: dict[str, Any] = summary["download"] if isinstance(summary.get("download"), dict) else {}
            for key, source in (("xmlLink", "govinfo_xml"), ("txtLink", "govinfo_html"), ("pdfLink", "govinfo_pdf")):
                url = download.get(key)
                if not isinstance(url, str) or not url:
                    continue
                content, content_type = await self._request_content(url)
                markdown = self._markdown_converter.convert(
                    content=content,
                    source_url=url,
                    content_type=content_type,
                    filename=f"{granule_id}.{key.removesuffix('Link').lower()}",
                )
                response = FederalRegisterDocumentTextResponse(
                    document=FederalRegisterDocument.model_validate(
                        {
                            **normalized_document.model_dump(by_alias=True),
                            "govInfoPackageId": package_id,
                            "govInfoGranuleId": granule_id,
                        }
                    ),
                    markdown=markdown,
                    contentSource=source,
                    contentType=content_type,
                    authoritativeUrl=url,
                )
                return dump_jsonable(response)

        if federal_register_client is None or document_payload is None:
            raise ValueError(
                "GovInfo could not resolve this Federal Register document. "
                "Provide GOVINFO_API_KEY or a Federal Register document number "
                "with FederalRegister.gov metadata available."
            )
        body_html_url = document_payload.get("bodyHtmlUrl")
        if not isinstance(body_html_url, str) or not body_html_url:
            raise ValueError("FederalRegister.gov did not provide a fallback bodyHtmlUrl for this document.")
        content, content_type = await self._request_content(body_html_url)
        markdown = self._markdown_converter.convert(
            content=content,
            source_url=body_html_url,
            content_type=content_type,
            filename=f"{normalized_document.document_number or 'federal-register'}.html",
        )
        response = FederalRegisterDocumentTextResponse(
            document=normalized_document,
            markdown=markdown,
            contentSource="federal_register_html",
            contentType=content_type,
            authoritativeUrl=body_html_url,
            warnings=["Returned FederalRegister.gov HTML because GovInfo authoritative resolution was unavailable."],
        )
        return dump_jsonable(response)

    async def get_cfr_text(
        self,
        *,
        title_number: int,
        part_number: int,
        section_number: str | None = None,
        revision_year: int | None = None,
        effective_date: str | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError("get_cfr_text requires GOVINFO_API_KEY for authoritative GovInfo retrieval.")

        resolved_revision_year = revision_year
        if resolved_revision_year is None and effective_date:
            match = re.match(r"(?P<year>\d{4})", effective_date)
            if match:
                resolved_revision_year = int(match.group("year"))
        citation = self._cfr_citation(title_number, part_number, section_number)
        query = f'collection:(CFR) and citation:"{citation}"'
        if resolved_revision_year is not None:
            query += f" and dateIssued:{resolved_revision_year}-01-01"
        search = await self._govinfo_search(query)
        results = search.get("results") or []
        if not isinstance(results, list) or not results:
            raise ValueError(f"No GovInfo CFR result matched {citation}.")
        first = results[0]
        if not isinstance(first, dict):
            raise ValueError(f"No GovInfo CFR result matched {citation}.")
        package_id = str(first.get("packageId") or "").strip() or None
        granule_id = str(first.get("granuleId") or "").strip() or None
        download: dict[str, Any] = first["download"] if isinstance(first.get("download"), dict) else {}
        resolved_volume = None
        if package_id:
            match = re.search(r"-vol(?P<volume>\d+)", package_id)
            if match:
                resolved_volume = int(match.group("volume"))
        for key, source in (("xmlLink", "govinfo_xml"), ("txtLink", "govinfo_html"), ("pdfLink", "govinfo_pdf")):
            url = download.get(key)
            if not isinstance(url, str) or not url:
                continue
            content, content_type = await self._request_content(url)
            markdown = self._markdown_converter.convert(
                content=content,
                source_url=url,
                content_type=content_type,
                filename=f"{granule_id or package_id or 'cfr'}.{key.removesuffix('Link').lower()}",
            )
            response = CfrTextResponse(
                titleNumber=title_number,
                partNumber=part_number,
                sectionNumber=section_number,
                revisionYear=resolved_revision_year,
                effectiveDate=effective_date,
                citation=citation,
                packageId=package_id,
                granuleId=granule_id,
                resolvedVolume=resolved_volume,
                contentSource=source,
                contentType=content_type,
                sourceUrl=url,
                markdown=markdown,
            )
            return dump_jsonable(response)
        raise ValueError(
            f"GovInfo returned a CFR record for {citation}, but no retrievable "
            "content links were available."
        )