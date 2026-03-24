"""ECOS species and document client."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime
from typing import Any, Awaitable, Callable, TypeVar, cast

from ...ecos_markdown import (
    EcosDocumentConversionError,
    EcosMarkdownConverter,
    EcosUnsupportedDocumentTypeError,
    guess_document_title,
)
from ...models import (
    EcosDocument,
    EcosDocumentGroups,
    EcosDocumentTextResponse,
    EcosRangeSummary,
    EcosSpecies,
    EcosSpeciesDocumentListResponse,
    EcosSpeciesEntity,
    EcosSpeciesHit,
    EcosSpeciesProfile,
    dump_jsonable,
)
from ...models.ecos import EcosDocumentKind
from ...provider_runtime import ProviderDiagnosticsRegistry, execute_provider_call
from ...transport import (
    build_httpx_verify_config,
    httpx,
    is_tls_verification_error,
    maybe_close_async_resource,
)

SPECIES_REPORT_COLUMNS = "/species@cn,sn,status,desc,listing_date"
SPECIES_REPORT_SORT = "/species@cn asc;/species@sn asc"
SPECIES_PROFILE_RE = re.compile(r"/ecp/species/(?P<species_id>\d+)")
GOVINFO_FR_LINK_RE = re.compile(r"/link/fr/(?P<volume>\d+)/(?P<page>\d+)", re.IGNORECASE)
DATE_FORMATS = ("%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d")
_PayloadT = TypeVar("_PayloadT")
logger = logging.getLogger("scholar-search-mcp")


def _filename_from_headers(headers: Any) -> str | None:
    content_disposition = None
    if isinstance(headers, dict):
        content_disposition = headers.get("content-disposition")
    else:
        content_disposition = getattr(headers, "get", lambda *_args, **_kwargs: None)("content-disposition")
    if not isinstance(content_disposition, str):
        return None
    match = re.search(r'filename="?([^";]+)"?', content_disposition)
    if match:
        return match.group(1).strip()
    return None


class EcosClient:
    """Thin ECOS client for species search, dossiers, and documents."""

    def __init__(
        self,
        *,
        base_url: str = "https://ecos.fws.gov",
        timeout: float = 30.0,
        document_timeout: float = 60.0,
        document_conversion_timeout: float = 60.0,
        max_document_size_mb: int = 25,
        verify_tls: bool = True,
        ca_bundle: str | None = None,
        provider_registry: ProviderDiagnosticsRegistry | None = None,
        markdown_converter: EcosMarkdownConverter | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.document_timeout = document_timeout
        self.document_conversion_timeout = max(float(document_conversion_timeout), 0.001)
        self.max_document_bytes = max(int(max_document_size_mb), 1) * 1024 * 1024
        self.verify_tls = verify_tls
        self.ca_bundle = str(ca_bundle or "").strip() or None
        self._provider_registry = provider_registry
        self._markdown_converter = markdown_converter or EcosMarkdownConverter()
        self._api_client: Any | None = None
        self._document_client: Any | None = None
        self._api_uses_system_store = False
        self._document_uses_system_store = False

    def _build_client(self, *, timeout: float, prefer_system_store: bool = False) -> Any:
        return httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            verify=build_httpx_verify_config(
                verify_tls=self.verify_tls,
                ca_bundle=self.ca_bundle,
                prefer_system_store=prefer_system_store,
            ),
        )

    def _get_api_client(self) -> Any:
        if self._api_client is None:
            self._api_client = self._build_client(timeout=self.timeout)
        return self._api_client

    def _get_document_client(self) -> Any:
        if self._document_client is None:
            self._document_client = self._build_client(timeout=self.document_timeout)
        return self._document_client

    async def _swap_client_to_system_store(self, *, document_client: bool) -> Any:
        attribute = "_document_client" if document_client else "_api_client"
        timeout = self.document_timeout if document_client else self.timeout
        previous_client = getattr(self, attribute)
        replacement = self._build_client(timeout=timeout, prefer_system_store=True)
        setattr(self, attribute, replacement)
        if document_client:
            self._document_uses_system_store = True
        else:
            self._api_uses_system_store = True
        await maybe_close_async_resource(previous_client)
        return replacement

    def _should_retry_with_system_store(
        self,
        *,
        document_client: bool,
        error: Exception,
    ) -> bool:
        if not self.verify_tls or self.ca_bundle is not None:
            return False
        if document_client and self._document_uses_system_store:
            return False
        if not document_client and self._api_uses_system_store:
            return False
        return is_tls_verification_error(error)

    async def _with_tls_fallback(
        self,
        *,
        document_client: bool,
        operation: Callable[[Any], Awaitable[_PayloadT]],
    ) -> _PayloadT:
        client = self._get_document_client() if document_client else self._get_api_client()
        try:
            return await operation(client)
        except httpx.ConnectError as exc:
            if not self._should_retry_with_system_store(
                document_client=document_client,
                error=exc,
            ):
                raise
            client = await self._swap_client_to_system_store(
                document_client=document_client,
            )
            return await operation(client)

    async def aclose(self) -> None:
        """Close shared HTTP clients."""

        api_client, self._api_client = self._api_client, None
        document_client, self._document_client = self._document_client, None
        await maybe_close_async_resource(api_client)
        await maybe_close_async_resource(document_client)

    def resolve_url(self, value: str) -> str:
        """Resolve an ECOS-relative URL to an absolute URL."""

        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("ECOS URLs must be non-empty.")
        if normalized.startswith(("http://", "https://")):
            return normalized
        if normalized.startswith("/"):
            return f"{self.base_url}{normalized}"
        return f"{self.base_url}/{normalized.lstrip('/')}"

    def resolve_species_id(self, species_id: str | int) -> str:
        """Normalize a species identifier or ECOS species URL to the numeric id."""

        normalized = str(species_id).strip()
        if not normalized:
            raise ValueError("species_id must be a numeric ECOS species id or URL.")
        if normalized.isdigit():
            return normalized
        match = SPECIES_PROFILE_RE.search(normalized)
        if match:
            return match.group("species_id")
        raise ValueError("species_id must be a numeric ECOS species id or species URL.")

    @staticmethod
    def _escape_filter_literal(value: str) -> str:
        return value.replace("'", "''")

    @staticmethod
    def _parse_multi_value(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            values = [str(item).strip() for item in value if str(item).strip()]
            return sorted(dict.fromkeys(values))
        normalized = str(value).strip()
        if not normalized:
            return []
        values = [segment.strip() for segment in normalized.split(",") if segment.strip()]
        return sorted(dict.fromkeys(values))

    @staticmethod
    def _date_ordinal(value: str | None) -> int:
        if value is None:
            return 0
        normalized = value.strip()
        if not normalized:
            return 0
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(normalized, fmt).date().toordinal()
            except ValueError:
                continue
        return 0

    async def _get_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        async def _request(client: Any) -> dict[str, Any] | None:
            response = await client.get(self.resolve_url(path), params=params)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return payload
            raise ValueError("Expected a JSON object from ECOS.")

        return await self._with_tls_fallback(document_client=False, operation=_request)

    async def _pullreports_export(self, filter_expression: str) -> dict[str, Any]:
        lookup = await execute_provider_call(
            provider="ecos",
            endpoint="pullreports.export",
            operation=lambda: self._get_json(
                "/ecp/pullreports/catalog/species/report/species/export",
                params={
                    "format": "json",
                    "columns": SPECIES_REPORT_COLUMNS,
                    "sort": SPECIES_REPORT_SORT,
                    "filter": filter_expression,
                },
            ),
            registry=self._provider_registry,
            is_empty=lambda payload: not payload or not payload.get("data"),
        )
        if lookup.payload is None:
            if lookup.outcome.status_bucket in {"success", "empty"}:
                return {"meta": {"totalCount": 0}, "data": []}
            raise ValueError(lookup.outcome.error or lookup.outcome.fallback_reason or "ECOS species search failed.")
        return cast(dict[str, Any], lookup.payload)

    async def list_label_values(
        self,
        *,
        column_path: str = "/species@cn",
    ) -> list[str]:
        """Return one Pull Reports label value list for the species report."""

        lookup = await execute_provider_call(
            provider="ecos",
            endpoint="pullreports.label_value_list",
            operation=lambda: self._get_json(
                "/ecp/pullreports/catalog/species/report/species/labelValueList",
                params={"columns": column_path},
            ),
            registry=self._provider_registry,
            is_empty=lambda payload: not payload,
        )
        if lookup.payload is None:
            if lookup.outcome.status_bucket in {"success", "empty"}:
                return []
            raise ValueError(
                lookup.outcome.error or lookup.outcome.fallback_reason or "ECOS label value lookup failed."
            )
        payload = cast(dict[str, Any], lookup.payload)
        values = payload.get("data")
        if not isinstance(values, list):
            return []
        return [str(item).strip() for item in values if str(item).strip()]

    def _row_to_hit(self, row: list[Any]) -> EcosSpeciesHit | None:
        if len(row) < 2:
            return None
        common_name = str(row[0] or "").strip() or None
        scientific_cell = row[1]
        scientific_name: str | None = None
        profile_url: str | None = None
        if isinstance(scientific_cell, dict):
            scientific_name = str(scientific_cell.get("value") or "").strip() or None
            raw_url = scientific_cell.get("url")
            if isinstance(raw_url, str) and raw_url.strip():
                profile_url = self.resolve_url(raw_url)
        elif scientific_cell is not None:
            scientific_name = str(scientific_cell).strip() or None
        if profile_url is None:
            return None
        species_id = self.resolve_species_id(profile_url)
        return EcosSpeciesHit(
            speciesId=species_id,
            commonName=common_name,
            scientificName=scientific_name,
            listingStatus=str(row[2] or "").strip() or None if len(row) > 2 else None,
            profileUrl=profile_url,
        )

    @staticmethod
    def _rank_hit(hit: EcosSpeciesHit, query: str) -> int:
        normalized_query = query.casefold()
        common_name = (hit.common_name or "").casefold()
        scientific_name = (hit.scientific_name or "").casefold()
        if common_name == normalized_query:
            return 0
        if scientific_name == normalized_query:
            return 1
        if common_name.startswith(normalized_query):
            return 2
        if scientific_name.startswith(normalized_query):
            return 3
        return 4

    async def _fetch_species_payload(self, species_id: str) -> dict[str, Any]:
        lookup = await execute_provider_call(
            provider="ecos",
            endpoint="species.profile",
            operation=lambda: self._get_json(
                f"/ecp/species/{species_id}",
                params={"format": "json"},
            ),
            registry=self._provider_registry,
            is_empty=lambda payload: not payload or not payload.get("id"),
        )
        if lookup.payload is None:
            if lookup.outcome.status_bucket in {"success", "empty"}:
                raise ValueError(f"ECOS species {species_id} was not found.")
            raise ValueError(
                lookup.outcome.error
                or lookup.outcome.fallback_reason
                or f"ECOS species profile lookup failed for {species_id}."
            )
        return cast(dict[str, Any], lookup.payload)

    def _normalize_species(self, payload: dict[str, Any]) -> EcosSpecies:
        species_id = self.resolve_species_id(str(payload.get("id") or ""))
        return EcosSpecies(
            speciesId=species_id,
            commonName=str(payload.get("cn") or "").strip() or None,
            scientificName=str(payload.get("sn") or "").strip() or None,
            family=str(payload.get("family") or "").strip() or None,
            group=str(payload.get("group") or "").strip() or None,
            kingdom=str(payload.get("kingdom") or "").strip() or None,
            tsn=str(payload.get("tsn") or "").strip() or None,
            uri=str(payload.get("uri") or "").strip() or None,
            profileUrl=self.resolve_url(f"/ecp/species/{species_id}"),
        )

    def _normalize_species_entities(
        self,
        payload: dict[str, Any],
    ) -> list[EcosSpeciesEntity]:
        entities: list[EcosSpeciesEntity] = []
        for item in payload.get("species_entities") or []:
            if not isinstance(item, dict):
                continue
            entities.append(
                EcosSpeciesEntity(
                    entityId=item.get("id"),
                    abbreviation=str(item.get("abbrev") or "").strip() or None,
                    agency=str(item.get("agency") or "").strip() or None,
                    altStatus=str(item.get("alt_status") or "").strip() or None,
                    description=str(item.get("desc") or "").strip() or None,
                    dps=str(item.get("dps") or "").strip() or None,
                    status=str(item.get("status") or "").strip() or None,
                    statusCategory=str(item.get("status_category") or "").strip() or None,
                    listingDate=str(item.get("listing_date") or "").strip() or None,
                    leadFwsRegion=str(item.get("lead_fws_region") or "").strip() or None,
                    recoveryPriorityNumber=str(item.get("recovery_priority_number") or "").strip() or None,
                    moreInfoUrl=(
                        self.resolve_url(item["more_info_url"])
                        if isinstance(item.get("more_info_url"), str) and str(item["more_info_url"]).strip()
                        else None
                    ),
                    currentRangeCountries=self._parse_multi_value(item.get("current_range_country")),
                    currentRangeStates=self._parse_multi_value(item.get("current_range_state")),
                    currentRangeCounties=self._parse_multi_value(item.get("current_range_county")),
                    currentRangeRefuges=self._parse_multi_value(item.get("current_range_refuge")),
                    rangeEnvelope=str(item.get("range_envelope") or "").strip() or None,
                    rangeShapefile=str(item.get("range_shapefile") or "").strip() or None,
                    shapefileLastUpdated=str(item.get("shapefile_last_updated") or "").strip() or None,
                )
            )
        return entities

    def _normalize_range_summary(
        self,
        payload: dict[str, Any],
        entities: list[EcosSpeciesEntity],
    ) -> EcosRangeSummary:
        def _collect(attribute: str) -> list[str]:
            values: list[str] = []
            for entity in entities:
                items = getattr(entity, attribute)
                if isinstance(items, list):
                    values.extend(items)
            return sorted(dict.fromkeys(value for value in values if value))

        return EcosRangeSummary(
            historicalRangeStates=self._parse_multi_value(payload.get("historical_range_state")),
            currentRangeCountries=_collect("current_range_countries"),
            currentRangeStates=_collect("current_range_states"),
            currentRangeCounties=_collect("current_range_counties"),
            currentRangeRefuges=_collect("current_range_refuges"),
        )

    def _normalize_document_title_and_url(
        self,
        value: Any,
    ) -> tuple[str | None, str | None]:
        if isinstance(value, dict):
            title = str(value.get("value") or "").strip() or None
            url = value.get("url")
            normalized_url = self.resolve_url(url) if isinstance(url, str) and url.strip() else None
            return title, normalized_url
        if value is None:
            return None, None
        return str(value).strip() or None, None

    def _normalize_recovery_like_documents(
        self,
        documents: list[Any],
        *,
        document_kind: EcosDocumentKind,
    ) -> list[EcosDocument]:
        normalized: list[EcosDocument] = []
        for item in documents:
            if not isinstance(item, dict):
                continue
            title, url = self._normalize_document_title_and_url(item.get("doc_title"))
            normalized.append(
                EcosDocument(
                    documentKind=document_kind,
                    title=title,
                    url=url,
                    documentDate=str(item.get("doc_date") or "").strip() or None,
                    entityId=item.get("entity_id"),
                    sourceId=item.get("source_id"),
                    recoveryPlanId=item.get("recovery_plan_id"),
                    documentType=str(item.get("doc_type") or "").strip() or None,
                    qualifier=str(item.get("doc_type_qualifier") or "").strip() or None,
                )
            )
        return normalized

    def _normalize_federal_register_documents(
        self,
        documents: list[Any],
        *,
        document_kind: EcosDocumentKind,
    ) -> list[EcosDocument]:
        normalized: list[EcosDocument] = []
        for item in documents:
            if not isinstance(item, dict):
                continue
            title, url = self._normalize_document_title_and_url(item.get("publication_title"))
            fr_citation = None
            if isinstance(url, str):
                match = GOVINFO_FR_LINK_RE.search(url)
                if match:
                    fr_citation = f"{match.group('volume')} FR {match.group('page')}"
            normalized.append(
                EcosDocument(
                    documentKind=document_kind,
                    title=title,
                    url=url,
                    documentDate=str(item.get("publication_date") or "").strip() or None,
                    sourceId=item.get("id"),
                    publicationId=item.get("pub_id"),
                    publicationPage=str(item.get("publication_page") or "").strip() or None,
                    frCitation=fr_citation,
                    documentTypes=[str(value).strip() for value in (item.get("doc_types") or []) if str(value).strip()],
                )
            )
        return normalized

    def _normalize_biological_opinions(
        self,
        payload: dict[str, Any],
    ) -> list[EcosDocument]:
        normalized: list[EcosDocument] = []
        for entity in payload.get("species_entities") or []:
            if not isinstance(entity, dict):
                continue
            for item in entity.get("biological_opinion") or []:
                if not isinstance(item, dict):
                    continue
                title, url = self._normalize_document_title_and_url(item.get("file_name"))
                normalized.append(
                    EcosDocument(
                        documentKind="biological_opinion",
                        title=title,
                        url=url,
                        documentDate=str(item.get("final_date") or "").strip() or None,
                        entityId=entity.get("id"),
                        documentType=str(item.get("category") or "").strip() or None,
                        category=str(item.get("category") or "").strip() or None,
                        eventCode=str(item.get("event_code") or "").strip() or None,
                        leadAgencies=str(item.get("lead_agencies_csv") or "").strip() or None,
                        leadOffices=str(item.get("lead_offices_csv") or "").strip() or None,
                        activities=str(item.get("activity_titles_csv") or "").strip() or None,
                        locations=str(item.get("locations_csv") or "").strip() or None,
                        workTypes=str(item.get("work_types_csv") or "").strip() or None,
                    )
                )
        return normalized

    def _normalize_conservation_plan_links(
        self,
        payload: dict[str, Any],
    ) -> list[EcosDocument]:
        plans = payload.get("conservationPlans")
        if not isinstance(plans, dict):
            return []
        normalized: list[EcosDocument] = []
        for key, values in plans.items():
            if key == "hasConservationPlans" or not isinstance(values, list):
                continue
            for item in values:
                if not isinstance(item, dict):
                    continue
                title, url = self._normalize_document_title_and_url(item.get("plan_title"))
                normalized.append(
                    EcosDocument(
                        documentKind="conservation_plan_link",
                        title=title,
                        url=url,
                        planId=item.get("plan_id"),
                        planType=str(item.get("plan_type") or "").strip() or None,
                    )
                )
        return sorted(
            normalized,
            key=lambda document: (
                (document.title or "").lower(),
                document.plan_id or 0,
            ),
        )

    def _normalize_document_groups(self, payload: dict[str, Any]) -> EcosDocumentGroups:
        return EcosDocumentGroups(
            recoveryPlans=self._normalize_recovery_like_documents(
                list(payload.get("recoveryPlans") or []),
                document_kind="recovery_plan",
            ),
            fiveYearReviews=self._normalize_recovery_like_documents(
                list(payload.get("fiveYearReviews") or []),
                document_kind="five_year_review",
            ),
            biologicalOpinions=self._normalize_biological_opinions(payload),
            federalRegisterDocuments=self._normalize_federal_register_documents(
                list(payload.get("federal_register_document") or []),
                document_kind="federal_register",
            ),
            otherRecoveryDocs=self._normalize_federal_register_documents(
                list(payload.get("otherRecoveryDocs") or []),
                document_kind="other_recovery_doc",
            ),
        )

    async def search_species(
        self,
        *,
        query: str,
        limit: int = 10,
        match_mode: str = "auto",
    ) -> dict[str, Any]:
        normalized_query = str(query or "").strip()
        if not normalized_query:
            raise ValueError("search_species_ecos requires a non-empty query.")

        escaped = self._escape_filter_literal(normalized_query)
        exact_filter = f"/species@cn = '{escaped}' or /species@sn = '{escaped}'"
        prefix_filter = f"/species@cn like '{escaped}%' or /species@sn like '{escaped}%'"

        raw_hits: dict[str, tuple[int, EcosSpeciesHit]] = {}

        if match_mode in {"auto", "exact"}:
            exact_payload = await self._pullreports_export(exact_filter)
            for row in exact_payload.get("data") or []:
                if not isinstance(row, list):
                    continue
                hit = self._row_to_hit(row)
                if hit is None:
                    continue
                rank = self._rank_hit(hit, normalized_query)
                existing = raw_hits.get(hit.species_id)
                if existing is None or rank < existing[0]:
                    raw_hits[hit.species_id] = (rank, hit)

        if match_mode in {"auto", "prefix"}:
            prefix_payload = await self._pullreports_export(prefix_filter)
            for row in prefix_payload.get("data") or []:
                if not isinstance(row, list):
                    continue
                hit = self._row_to_hit(row)
                if hit is None:
                    continue
                rank = self._rank_hit(hit, normalized_query)
                existing = raw_hits.get(hit.species_id)
                if existing is None or rank < existing[0]:
                    raw_hits[hit.species_id] = (rank, hit)

        ordered_hits = sorted(
            (entry[1] for entry in raw_hits.values()),
            key=lambda hit: (
                self._rank_hit(hit, normalized_query),
                (hit.common_name or "").lower(),
                (hit.scientific_name or "").lower(),
                hit.species_id,
            ),
        )[:limit]

        for hit in ordered_hits:
            try:
                payload = await self._fetch_species_payload(hit.species_id)
            except ValueError:
                continue
            entities = payload.get("species_entities") or []
            first_entity = entities[0] if entities and isinstance(entities[0], dict) else {}
            hit.group = str(payload.get("group") or "").strip() or None
            hit.status_category = str(first_entity.get("status_category") or "").strip() or None
            hit.lead_agency = str(first_entity.get("agency") or "").strip() or None

        return {
            "query": normalized_query,
            "matchMode": match_mode,
            "total": len(ordered_hits),
            "data": dump_jsonable(ordered_hits),
        }

    async def get_species_profile(self, *, species_id: str | int) -> dict[str, Any]:
        normalized_species_id = self.resolve_species_id(species_id)
        payload = await self._fetch_species_payload(normalized_species_id)
        species = self._normalize_species(payload)
        entities = self._normalize_species_entities(payload)
        profile = EcosSpeciesProfile(
            species=species,
            speciesEntities=entities,
            lifeHistory=str(payload.get("life_history") or "").strip() or None,
            range=self._normalize_range_summary(payload, entities),
            documents=self._normalize_document_groups(payload),
            conservationPlanLinks=self._normalize_conservation_plan_links(payload),
        )
        return dump_jsonable(profile)

    @staticmethod
    def _iter_all_documents(profile: dict[str, Any]) -> list[dict[str, Any]]:
        documents = profile.get("documents") or {}
        flattened: list[dict[str, Any]] = []
        if isinstance(documents, dict):
            for key in (
                "recoveryPlans",
                "fiveYearReviews",
                "biologicalOpinions",
                "federalRegisterDocuments",
                "otherRecoveryDocs",
            ):
                values = documents.get(key) or []
                if isinstance(values, list):
                    flattened.extend(item for item in values if isinstance(item, dict))
        conservation_links = profile.get("conservationPlanLinks") or []
        if isinstance(conservation_links, list):
            flattened.extend(item for item in conservation_links if isinstance(item, dict))
        return flattened

    async def list_species_documents(
        self,
        *,
        species_id: str | int,
        document_kinds: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized_species_id = self.resolve_species_id(species_id)
        profile = await self.get_species_profile(species_id=normalized_species_id)
        documents = self._iter_all_documents(profile)
        normalized_kinds = list(dict.fromkeys(document_kinds or []))
        if normalized_kinds:
            allowed = set(normalized_kinds)
            documents = [document for document in documents if document.get("documentKind") in allowed]
        ordered = sorted(
            documents,
            key=lambda document: (
                -self._date_ordinal(cast(str | None, document.get("documentDate"))),
                str(document.get("title") or "").lower(),
            ),
        )
        response = EcosSpeciesDocumentListResponse(
            speciesId=normalized_species_id,
            total=len(ordered),
            documentKindsApplied=cast(list[Any], normalized_kinds),
            data=[EcosDocument.model_validate(document) for document in ordered],
        )
        return dump_jsonable(response)

    async def _download_document(self, url: str) -> dict[str, Any] | None:
        async def _request(client: Any) -> dict[str, Any] | None:
            async with client.stream("GET", self.resolve_url(url)) as response:
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                content_type = response.headers.get("content-type")
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > self.max_document_bytes:
                    return {
                        "finalUrl": str(response.url),
                        "contentType": content_type,
                        "filename": _filename_from_headers(response.headers),
                        "status": "too_large",
                    }
                total_bytes = 0
                chunks: list[bytes] = []
                async for chunk in response.aiter_bytes():
                    total_bytes += len(chunk)
                    if total_bytes > self.max_document_bytes:
                        return {
                            "finalUrl": str(response.url),
                            "contentType": content_type,
                            "filename": _filename_from_headers(response.headers),
                            "status": "too_large",
                        }
                    chunks.append(chunk)
                return {
                    "finalUrl": str(response.url),
                    "contentType": content_type,
                    "filename": _filename_from_headers(response.headers),
                    "content": b"".join(chunks),
                    "status": "ok",
                }

        return await self._with_tls_fallback(document_client=True, operation=_request)

    async def get_document_text(self, *, url: str) -> dict[str, Any]:
        absolute_url = self.resolve_url(url)
        fetch_started = time.perf_counter()
        logger.info("ECOS document fetch started: %s", absolute_url)
        lookup = await execute_provider_call(
            provider="ecos",
            endpoint="document.fetch",
            operation=lambda: self._download_document(absolute_url),
            registry=self._provider_registry,
            is_empty=lambda payload: payload is None,
        )
        fetch_elapsed_ms = int((time.perf_counter() - fetch_started) * 1000)
        if lookup.payload is None:
            logger.warning(
                "ECOS document fetch failed after %d ms: %s",
                fetch_elapsed_ms,
                absolute_url,
            )
            response = EcosDocumentTextResponse(
                document=EcosDocument(
                    title=guess_document_title(absolute_url),
                    url=absolute_url,
                ),
                contentType=None,
                extractionStatus="fetch_failed",
                warnings=[lookup.outcome.error or lookup.outcome.fallback_reason or "ECOS document fetch failed."],
            )
            return dump_jsonable(response)

        payload = cast(dict[str, Any], lookup.payload)
        final_url = str(payload.get("finalUrl") or absolute_url)
        filename = cast(str | None, payload.get("filename"))
        content_type = cast(str | None, payload.get("contentType"))
        document = EcosDocument(
            title=guess_document_title(final_url, filename),
            url=final_url,
        )

        logger.info(
            ("ECOS document fetch completed after %d ms: url=%s final_url=%s status=%s content_type=%s bytes=%s"),
            fetch_elapsed_ms,
            absolute_url,
            final_url,
            payload.get("status"),
            content_type,
            len(payload.get("content") or b""),
        )

        if payload.get("status") == "too_large":
            logger.warning(
                ("ECOS document conversion skipped because the document exceeded the size limit: %s"),
                final_url,
            )
            response = EcosDocumentTextResponse(
                document=document,
                contentType=content_type,
                extractionStatus="too_large",
                warnings=[("ECOS document exceeded the configured size limit before conversion.")],
            )
            return dump_jsonable(response)

        content = payload.get("content")
        if not isinstance(content, (bytes, bytearray)):
            logger.warning(
                "ECOS document fetch returned no body after %d ms: %s",
                fetch_elapsed_ms,
                final_url,
            )
            response = EcosDocumentTextResponse(
                document=document,
                contentType=content_type,
                extractionStatus="fetch_failed",
                warnings=["ECOS document body was empty after fetch."],
            )
            return dump_jsonable(response)

        conversion_started = time.perf_counter()
        logger.info(
            "ECOS document conversion started: url=%s content_type=%s bytes=%d",
            final_url,
            content_type,
            len(content),
        )
        try:
            markdown = await asyncio.wait_for(
                asyncio.to_thread(
                    self._markdown_converter.convert,
                    content=bytes(content),
                    source_url=final_url,
                    content_type=content_type,
                    filename=filename,
                ),
                timeout=self.document_conversion_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "ECOS document conversion timed out after %d ms: %s",
                int((time.perf_counter() - conversion_started) * 1000),
                final_url,
            )
            response = EcosDocumentTextResponse(
                document=document,
                contentType=content_type,
                extractionStatus="conversion_timed_out",
                warnings=[
                    ("ECOS document conversion exceeded the configured timeout before Markdown extraction finished.")
                ],
            )
            return dump_jsonable(response)
        except EcosUnsupportedDocumentTypeError as exc:
            logger.warning(
                "ECOS document conversion rejected unsupported type after %d ms: %s",
                int((time.perf_counter() - conversion_started) * 1000),
                final_url,
            )
            response = EcosDocumentTextResponse(
                document=document,
                contentType=content_type,
                extractionStatus="unsupported_type",
                warnings=[str(exc)],
            )
            return dump_jsonable(response)
        except EcosDocumentConversionError as exc:
            logger.warning(
                "ECOS document conversion failed after %d ms: %s",
                int((time.perf_counter() - conversion_started) * 1000),
                final_url,
            )
            response = EcosDocumentTextResponse(
                document=document,
                contentType=content_type,
                extractionStatus="fetch_failed",
                warnings=[str(exc)],
            )
            return dump_jsonable(response)

        conversion_elapsed_ms = int((time.perf_counter() - conversion_started) * 1000)
        logger.info(
            "ECOS document conversion completed after %d ms: url=%s markdown_chars=%d",
            conversion_elapsed_ms,
            final_url,
            len(markdown),
        )

        if len(markdown.strip()) < 50:
            logger.warning(
                ("ECOS document conversion produced nearly empty markdown after %d ms: %s"),
                conversion_elapsed_ms,
                final_url,
            )
            response = EcosDocumentTextResponse(
                document=document,
                markdown=markdown,
                contentType=content_type,
                extractionStatus="empty_or_image_only",
                warnings=[("Converted Markdown was nearly empty. The source may be image-only or poorly extractable.")],
            )
            return dump_jsonable(response)

        response = EcosDocumentTextResponse(
            document=document,
            markdown=markdown,
            contentType=content_type,
            extractionStatus="ok",
        )
        return dump_jsonable(response)
