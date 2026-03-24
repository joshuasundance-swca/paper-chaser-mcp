"""FederalRegister.gov discovery client."""

from __future__ import annotations

from typing import Any

from ...models import FederalRegisterAgency, FederalRegisterDocument, FederalRegisterSearchResponse, dump_jsonable
from ...transport import httpx, maybe_close_async_resource

FEDERAL_REGISTER_API_BASE = "https://www.federalregister.gov/api/v1"


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
        params: dict[str, Any] = {
            "conditions[term]": query,
            "order": "newest",
            "per_page": limit,
        }
        if publication_date_from:
            params["conditions[publication_date][gte]"] = publication_date_from
        if publication_date_to:
            params["conditions[publication_date][lte]"] = publication_date_to
        if document_number:
            params["conditions[document_number]"] = document_number
        if cfr_citation:
            params["conditions[cfr][citation]"] = cfr_citation
        if cfr_title is not None:
            params["conditions[cfr][title]"] = cfr_title
        if cfr_part is not None:
            params["conditions[cfr][part]"] = cfr_part
        if agencies:
            params["conditions[agencies][]"] = agencies
        if document_types:
            params["conditions[type][]"] = document_types

        payload = await self._get_json("/documents.json", params=params)
        results = [self._normalize_document(item) for item in (payload.get("results") or []) if isinstance(item, dict)]
        response = FederalRegisterSearchResponse(
            total=int(payload.get("count") or len(results) or 0),
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