"""FederalRegister.gov discovery client."""

from __future__ import annotations

import re
from typing import Any

from ...models import FederalRegisterAgency, FederalRegisterDocument, FederalRegisterSearchResponse, dump_jsonable
from ...transport import httpx, maybe_close_async_resource

FEDERAL_REGISTER_API_BASE = "https://www.federalregister.gov/api/v1"
_QUERY_TOKEN_RE = re.compile(r"[a-z0-9]+")


class FederalRegisterClient:
    """Thin FederalRegister.gov client for discovery and document metadata."""

    def __init__(self, *, base_url: str = FEDERAL_REGISTER_API_BASE, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._http_client: Any | None = None

    def _get_http_client(self) -> Any:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.timeout, follow_redirects=True)
        return self._http_client

    async def aclose(self) -> None:
        client, self._http_client = self._http_client, None
        await maybe_close_async_resource(client)

    async def _get_json(self, path: str, *, params: dict[str, Any]) -> dict[str, Any]:
        response = await self._get_http_client().get(f"{self.base_url}{path}", params=params)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object from FederalRegister.gov.")
        return payload

    @staticmethod
    def _normalize_document(payload: dict[str, Any]) -> FederalRegisterDocument:
        volume = payload.get("volume")
        start_page = payload.get("start_page")
        end_page = payload.get("end_page")
        citation = None
        govinfo_link = None
        if isinstance(volume, int) and isinstance(start_page, int):
            citation = f"{volume} FR {start_page}"
            govinfo_link = f"https://www.govinfo.gov/link/fr/{volume}/{start_page}"
        cfr_references: list[str] = []
        for reference in payload.get("cfr_references") or []:
            if not isinstance(reference, dict):
                continue
            title = reference.get("title")
            parts = reference.get("part") or reference.get("parts")
            if title is None or parts is None:
                continue
            if isinstance(parts, list):
                for part in parts:
                    cfr_references.append(f"{title} CFR {part}")
            else:
                cfr_references.append(f"{title} CFR {parts}")

        agencies = [
            FederalRegisterAgency(
                name=str(item.get("name") or "").strip() or None,
                slug=str(item.get("slug") or "").strip() or None,
            )
            for item in (payload.get("agencies") or [])
            if isinstance(item, dict)
        ]

        return FederalRegisterDocument(
            documentNumber=str(payload.get("document_number") or "").strip() or None,
            title=str(payload.get("title") or "").strip() or None,
            abstract=str(payload.get("abstract") or "").strip() or None,
            documentType=str(payload.get("type") or payload.get("document_type") or "").strip() or None,
            publicationDate=str(payload.get("publication_date") or "").strip() or None,
            startPage=start_page if isinstance(start_page, int) else None,
            endPage=end_page if isinstance(end_page, int) else None,
            citation=citation,
            agencies=agencies,
            bodyHtmlUrl=str(payload.get("body_html_url") or "").strip() or None,
            htmlUrl=str(payload.get("html_url") or payload.get("full_text_xml_url") or "").strip() or None,
            pdfUrl=str(payload.get("pdf_url") or "").strip() or None,
            rawTextUrl=str(payload.get("raw_text_url") or "").strip() or None,
            publicInspectionPdfUrl=str(payload.get("public_inspection_pdf_url") or "").strip() or None,
            cfrReferences=sorted(dict.fromkeys(value for value in cfr_references if value)),
            govInfoLink=govinfo_link,
        )

    @staticmethod
    def _normalized_cfr_filter(
        *,
        cfr_citation: str | None,
        cfr_title: int | None,
        cfr_part: int | None,
    ) -> str | None:
        if isinstance(cfr_citation, str) and cfr_citation.strip():
            return cfr_citation.strip()
        if cfr_title is not None and cfr_part is not None:
            return f"{cfr_title} CFR {cfr_part}"
        return None

    @staticmethod
    def _matches_date_bounds(
        publication_date: str | None,
        *,
        publication_date_from: str | None,
        publication_date_to: str | None,
    ) -> bool:
        if not publication_date:
            return publication_date_from is None and publication_date_to is None
        if publication_date_from and publication_date < publication_date_from:
            return False
        if publication_date_to and publication_date > publication_date_to:
            return False
        return True

    @classmethod
    def _document_matches_filters(
        cls,
        document: dict[str, Any],
        *,
        query: str,
        apply_query_filter: bool,
        agencies: list[str] | None,
        document_types: list[str] | None,
        publication_date_from: str | None,
        publication_date_to: str | None,
        cfr_citation: str | None,
        cfr_title: int | None,
        cfr_part: int | None,
        document_number: str | None,
    ) -> bool:
        normalized_query = str(query or "").strip().lower()
        if apply_query_filter and normalized_query:
            haystack_parts = [
                str(document.get("title") or ""),
                str(document.get("abstract") or ""),
                str(document.get("citation") or ""),
                str(document.get("documentNumber") or ""),
                " ".join(str(value) for value in document.get("cfrReferences") or []),
            ]
            haystack = " ".join(part.lower() for part in haystack_parts if part).strip()
            query_tokens = _QUERY_TOKEN_RE.findall(normalized_query)
            if haystack and normalized_query not in haystack:
                if not query_tokens or not all(token in haystack for token in query_tokens):
                    return False
            elif not haystack:
                return False

        if document_number:
            normalized_document_number = str(document.get("documentNumber") or "").strip()
            if normalized_document_number != str(document_number).strip():
                return False

        if agencies:
            requested_agencies = {agency.strip() for agency in agencies if isinstance(agency, str) and agency.strip()}
            document_agencies = {
                str(item.get("slug") or "").strip() for item in document.get("agencies") or [] if isinstance(item, dict)
            }
            if requested_agencies and requested_agencies.isdisjoint(document_agencies):
                return False

        if document_types:
            document_type = str(document.get("documentType") or "").strip().upper()
            requested_types = {doc_type.strip().upper() for doc_type in document_types if isinstance(doc_type, str)}
            if requested_types and document_type not in requested_types:
                return False

        if not cls._matches_date_bounds(
            str(document.get("publicationDate") or "").strip() or None,
            publication_date_from=publication_date_from,
            publication_date_to=publication_date_to,
        ):
            return False

        normalized_cfr_filter = cls._normalized_cfr_filter(
            cfr_citation=cfr_citation,
            cfr_title=cfr_title,
            cfr_part=cfr_part,
        )
        if normalized_cfr_filter:
            document_references = {str(value).strip() for value in document.get("cfrReferences") or [] if value}
            if normalized_cfr_filter not in document_references:
                return False

        return True

    @classmethod
    def _filter_documents_locally(
        cls,
        documents: list[dict[str, Any]],
        *,
        query: str,
        apply_query_filter: bool,
        agencies: list[str] | None,
        document_types: list[str] | None,
        publication_date_from: str | None,
        publication_date_to: str | None,
        cfr_citation: str | None,
        cfr_title: int | None,
        cfr_part: int | None,
        document_number: str | None,
    ) -> list[FederalRegisterDocument]:
        filtered: list[FederalRegisterDocument] = []
        for document in documents:
            if not cls._document_matches_filters(
                document,
                query=query,
                apply_query_filter=apply_query_filter,
                agencies=agencies,
                document_types=document_types,
                publication_date_from=publication_date_from,
                publication_date_to=publication_date_to,
                cfr_citation=cfr_citation,
                cfr_title=cfr_title,
                cfr_part=cfr_part,
                document_number=document_number,
            ):
                continue
            filtered.append(FederalRegisterDocument.model_validate(document))
        return filtered

    async def search_documents(
        self,
        *,
        query: str,
        limit: int = 10,
        agencies: list[str] | None = None,
        document_types: list[str] | None = None,
        publication_date_from: str | None = None,
        publication_date_to: str | None = None,
        cfr_citation: str | None = None,
        cfr_title: int | None = None,
        cfr_part: int | None = None,
        document_number: str | None = None,
    ) -> dict[str, Any]:
        normalized_cfr_filter = self._normalized_cfr_filter(
            cfr_citation=cfr_citation,
            cfr_title=cfr_title,
            cfr_part=cfr_part,
        )
        if document_number:
            document = await self.get_document(document_number)
            documents = [document] if isinstance(document, dict) else []
            filtered = self._filter_documents_locally(
                documents,
                query=query,
                apply_query_filter=True,
                agencies=agencies,
                document_types=document_types,
                publication_date_from=publication_date_from,
                publication_date_to=publication_date_to,
                cfr_citation=normalized_cfr_filter,
                cfr_title=cfr_title,
                cfr_part=cfr_part,
                document_number=document_number,
            )
            return dump_jsonable(
                FederalRegisterSearchResponse(
                    total=len(filtered),
                    data=filtered,
                )
            )

        params: dict[str, Any] = {
            "conditions[term]": query,
            "order": "newest",
            "per_page": limit,
        }
        if publication_date_from:
            params["conditions[publication_date][gte]"] = publication_date_from
        if publication_date_to:
            params["conditions[publication_date][lte]"] = publication_date_to
        if normalized_cfr_filter:
            params["conditions[cfr][citation]"] = normalized_cfr_filter
        elif cfr_title is not None:
            params["conditions[cfr][title]"] = cfr_title
        if normalized_cfr_filter is None and cfr_part is not None:
            params["conditions[cfr][part]"] = cfr_part
        if agencies:
            params["conditions[agencies][]"] = agencies
        if document_types:
            params["conditions[type][]"] = document_types

        try:
            payload = await self._get_json("/documents.json", params=params)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 400 or normalized_cfr_filter is None:
                raise
            retry_params = dict(params)
            retry_params.pop("conditions[cfr][citation]", None)
            payload = await self._get_json("/documents.json", params=retry_params)

        raw_results = [item for item in (payload.get("results") or []) if isinstance(item, dict)]
        normalized_results = [dump_jsonable(self._normalize_document(item)) for item in raw_results]
        results = self._filter_documents_locally(
            normalized_results,
            query=query,
            apply_query_filter=False,
            agencies=agencies,
            document_types=document_types,
            publication_date_from=publication_date_from,
            publication_date_to=publication_date_to,
            cfr_citation=normalized_cfr_filter,
            cfr_title=cfr_title,
            cfr_part=cfr_part,
            document_number=document_number,
        )
        response = FederalRegisterSearchResponse(
            total=len(results),
            data=results,
        )
        return dump_jsonable(response)

    async def get_document(self, document_number: str) -> dict[str, Any] | None:
        normalized = str(document_number or "").strip()
        if not normalized:
            raise ValueError("document_number must be non-empty.")
        payload = await self._get_json(f"/documents/{normalized}.json", params={})
        if payload.get("errors"):
            return None
        return dump_jsonable(self._normalize_document(payload))
