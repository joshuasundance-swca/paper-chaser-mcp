"""GovInfo authoritative retrieval client for Federal Register and CFR."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from typing import Any

from defusedxml import ElementTree as ET

from ...document_markdown import DocumentMarkdownConverter
from ...models import CfrTextResponse, FederalRegisterDocument, FederalRegisterDocumentTextResponse, dump_jsonable
from ...transport import httpx, maybe_close_async_resource

GOVINFO_API_BASE = "https://api.govinfo.gov"
GOVINFO_FR_LINK_RE = re.compile(r"/link/fr/(?P<volume>\d+)/(?P<page>\d+)", re.IGNORECASE)
FR_CITATION_RE = re.compile(
    r"(?P<volume>\d+)\s*(?:F\.?\s*R\.?|FED(?:ERAL)?\.?\s+REG(?:ISTER)?\.?)\s*(?P<page>\d+)",
    re.IGNORECASE,
)
logger = logging.getLogger("scholar-search-mcp")


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
        self.document_conversion_timeout = max(float(document_timeout), 0.001)
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
        citation_match = FR_CITATION_RE.search(normalized)
        if citation_match:
            return {"citation": f"{citation_match.group('volume')} FR {citation_match.group('page')}"}
        if re.fullmatch(r"\d{4}-\d{4,6}", normalized):
            return {"documentNumber": normalized}
        return None

    async def _find_federal_register_document(
        self,
        *,
        parsed: dict[str, str],
        federal_register_client: Any | None,
    ) -> dict[str, Any] | None:
        if federal_register_client is None:
            return None
        document_number = parsed.get("documentNumber")
        if document_number:
            return await federal_register_client.get_document(document_number)
        citation = parsed.get("citation")
        if not citation:
            return None
        search_queries = [citation]
        citation_match = FR_CITATION_RE.search(citation)
        if citation_match:
            volume = citation_match.group("volume")
            page = citation_match.group("page")
            search_queries.extend(
                [
                    f"{volume} {page}",
                    page,
                ]
            )

        seen_document_numbers: set[str] = set()
        fallback_candidates: list[dict[str, Any]] = []
        for query_text in search_queries:
            search_payload = None
            try:
                search_payload = await federal_register_client.search_documents(query=query_text, limit=5)
            except Exception as exc:
                logger.debug("Federal Register search fallback failed for %s: %s", query_text, exc)
            if not isinstance(search_payload, dict):
                continue
            for candidate in search_payload.get("data") or []:
                if not isinstance(candidate, dict):
                    continue
                normalized_document_number = str(candidate.get("documentNumber") or "").strip()
                if normalized_document_number and normalized_document_number in seen_document_numbers:
                    continue
                if normalized_document_number:
                    seen_document_numbers.add(normalized_document_number)
                if str(candidate.get("citation") or "").strip().lower() == citation.lower():
                    return candidate
                fallback_candidates.append(candidate)
        for candidate in fallback_candidates:
            if isinstance(candidate, dict):
                return candidate
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

    @staticmethod
    def _looks_like_xml(
        *,
        content_type: str | None,
        source_url: str,
        filename: str | None,
    ) -> bool:
        normalized_type = (content_type or "").split(";", 1)[0].strip().lower()
        if normalized_type in {"application/xml", "text/xml"}:
            return True
        lowered_url = source_url.lower()
        if lowered_url.endswith("/xml") or lowered_url.endswith(".xml"):
            return True
        lowered_filename = str(filename or "").strip().lower()
        return lowered_filename.endswith(".xml")

    @staticmethod
    def _xml_local_name(tag: Any) -> str:
        if not isinstance(tag, str):
            return ""
        return tag.rsplit("}", 1)[-1].upper()

    @staticmethod
    def _normalize_xml_text(value: str) -> str:
        return " ".join(value.split())

    @classmethod
    def _extract_cfr_xml_markdown(
        cls,
        *,
        content: bytes,
        content_type: str | None,
        source_url: str,
        filename: str | None,
    ) -> str | None:
        if not cls._looks_like_xml(content_type=content_type, source_url=source_url, filename=filename):
            return None
        try:
            root = ET.fromstring(content)
        except ET.ParseError:
            return None

        blocks: list[str] = []
        seen: set[str] = set()
        heading_tags = {"TITLE", "SECTNO", "SUBJECT", "HEAD", "HD"}
        paragraph_tags = {"P", "FP", "HD1", "HD2", "HD3"}

        for element in root.iter():
            tag = cls._xml_local_name(element.tag)
            if tag not in heading_tags and tag not in paragraph_tags:
                continue
            text = cls._normalize_xml_text(" ".join(element.itertext()))
            if not text or text in seen:
                continue
            seen.add(text)
            if tag in heading_tags:
                prefix = "##" if tag in {"TITLE", "SECTNO"} else "###"
                blocks.append(f"{prefix} {text}")
            else:
                blocks.append(text)

        if not blocks:
            flattened = cls._normalize_xml_text(" ".join(root.itertext()))
            if not flattened:
                return None
            return flattened
        return "\n\n".join(blocks).strip()

    async def _convert_download_to_markdown(
        self,
        *,
        content: bytes,
        source_url: str,
        content_type: str | None,
        filename: str | None,
        prefer_cfr_xml: bool = False,
    ) -> tuple[str | None, list[str]]:
        if len(content) > self.max_document_bytes:
            logger.warning(
                "GovInfo document exceeded size limit before conversion: url=%s bytes=%d limit=%d",
                source_url,
                len(content),
                self.max_document_bytes,
            )
            return None, ["GovInfo document exceeded the configured size limit before Markdown extraction."]

        if prefer_cfr_xml:
            extracted_xml = self._extract_cfr_xml_markdown(
                content=content,
                content_type=content_type,
                source_url=source_url,
                filename=filename,
            )
            if extracted_xml:
                return extracted_xml, []

        try:
            convert_method = getattr(self._markdown_converter, "convert_with_timeout", None)
            if callable(convert_method):
                markdown = await asyncio.to_thread(
                    convert_method,
                    content=content,
                    source_url=source_url,
                    content_type=content_type,
                    timeout_seconds=self.document_conversion_timeout,
                    filename=filename,
                )
            else:
                markdown = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._markdown_converter.convert,
                        content=content,
                        source_url=source_url,
                        content_type=content_type,
                        filename=filename,
                    ),
                    timeout=self.document_conversion_timeout,
                )
        except asyncio.TimeoutError:
            logger.warning(
                "GovInfo document conversion timed out after %.3f s: %s",
                self.document_conversion_timeout,
                source_url,
            )
            return None, ["GovInfo document conversion timed out before Markdown extraction finished."]
        return markdown, []

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
        document_payload = await self._find_federal_register_document(
            parsed=parsed,
            federal_register_client=federal_register_client,
        )

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
            try:
                summary = await self._request_json(
                    "GET",
                    f"/packages/{package_id}/granules/{granule_id}/summary",
                )
                download: dict[str, Any] = summary["download"] if isinstance(summary.get("download"), dict) else {}
                for key, source in (
                    ("xmlLink", "govinfo_xml"),
                    ("txtLink", "govinfo_html"),
                    ("pdfLink", "govinfo_pdf"),
                ):
                    url = download.get(key)
                    if not isinstance(url, str) or not url:
                        continue
                    content, content_type = await self._request_content(url)
                    markdown, warnings = await self._convert_download_to_markdown(
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
                        warnings=warnings,
                    )
                    return dump_jsonable(response)
            except (GovInfoApiError, httpx.HTTPError, ValueError) as error:
                logger.warning(
                    "GovInfo FR authoritative retrieval failed for %s (%s/%s): %s",
                    identifier,
                    package_id,
                    granule_id,
                    error,
                )
                if federal_register_client is not None and (
                    document_payload is None or not isinstance(document_payload.get("bodyHtmlUrl"), str)
                ):
                    document_payload = await self._find_federal_register_document(
                        parsed=parsed,
                        federal_register_client=federal_register_client,
                    )
                    normalized_document = FederalRegisterDocument.model_validate(document_payload or {})

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
        markdown, warnings = await self._convert_download_to_markdown(
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
            warnings=[
                "Returned FederalRegister.gov HTML because GovInfo authoritative resolution was unavailable.",
                *warnings,
            ],
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
            markdown, warnings = await self._convert_download_to_markdown(
                content=content,
                source_url=url,
                content_type=content_type,
                filename=f"{granule_id or package_id or 'cfr'}.{key.removesuffix('Link').lower()}",
                prefer_cfr_xml=True,
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
                warnings=warnings,
            )
            return dump_jsonable(response)
        raise ValueError(
            f"GovInfo returned a CFR record for {citation}, but no retrievable content links were available."
        )
