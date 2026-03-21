"""Crossref + Unpaywall enrichment helpers shared across tool surfaces."""

from __future__ import annotations

import logging
from typing import Any

from .clients.crossref import CrossrefClient
from .clients.unpaywall import UnpaywallClient
from .identifiers import resolve_doi_inputs
from .models import (
    CrossrefEnrichment,
    CrossrefEnrichmentResult,
    CrossrefWorkSummary,
    DoiResolution,
    Paper,
    PaperEnrichmentResponse,
    PaperEnrichments,
    UnpaywallEnrichment,
    UnpaywallEnrichmentResult,
    dump_jsonable,
)
from .provider_runtime import ProviderDiagnosticsRegistry, execute_provider_call

logger = logging.getLogger("scholar-search-mcp")


class PaperEnrichmentService:
    """Reusable Crossref + Unpaywall enrichment service."""

    def __init__(
        self,
        *,
        crossref_client: CrossrefClient | None,
        unpaywall_client: UnpaywallClient | None,
        enable_crossref: bool,
        enable_unpaywall: bool,
        provider_registry: ProviderDiagnosticsRegistry | None = None,
    ) -> None:
        self._crossref_client = crossref_client
        self._unpaywall_client = unpaywall_client
        self._enable_crossref = enable_crossref
        self._enable_unpaywall = enable_unpaywall
        self._provider_registry = provider_registry

    async def get_crossref_metadata(
        self,
        *,
        paper_id: str | None = None,
        doi: str | None = None,
        paper: dict[str, Any] | Paper | None = None,
        query: str | None = None,
        request_id: str | None = None,
        request_outcomes: list[dict[str, Any]] | None = None,
        suppress_errors: bool = False,
    ) -> CrossrefEnrichmentResult:
        if not self._enable_crossref or self._crossref_client is None:
            if suppress_errors:
                return CrossrefEnrichmentResult(found=False)
            raise ValueError(
                "get_paper_metadata_crossref requires Crossref, which is disabled. "
                "Set SCHOLAR_SEARCH_ENABLE_CROSSREF=true to use this tool."
            )

        resolved_doi, _ = resolve_doi_inputs(doi=doi, paper_id=paper_id, paper=paper)
        try:
            if resolved_doi:
                lookup = await execute_provider_call(
                    provider="crossref",
                    endpoint="get_work",
                    operation=lambda: self._crossref_client.get_work(resolved_doi),
                    registry=self._provider_registry,
                    request_outcomes=request_outcomes,
                    request_id=request_id,
                    is_empty=lambda payload: payload is None,
                )
            else:
                normalized_query = str(query or "").strip()
                if not normalized_query:
                    raise ValueError(
                        "Crossref enrichment needs a DOI-bearing identifier, an "
                        "explicit doi, or a non-empty query fallback."
                    )
                lookup = await execute_provider_call(
                    provider="crossref",
                    endpoint="search_work",
                    operation=lambda: self._crossref_client.search_work(
                        normalized_query
                    ),
                    registry=self._provider_registry,
                    request_outcomes=request_outcomes,
                    request_id=request_id,
                    is_empty=lambda payload: payload is None,
                )
        except Exception:
            if suppress_errors:
                logger.debug("Crossref enrichment failed", exc_info=True)
                return CrossrefEnrichmentResult(found=False, resolvedDoi=resolved_doi)
            raise

        if lookup.payload is None:
            if (
                lookup.outcome.status_bucket not in {"empty", "success"}
                and not suppress_errors
            ):
                raise ValueError(
                    lookup.outcome.error
                    or lookup.outcome.fallback_reason
                    or "Crossref enrichment failed."
                )
            return CrossrefEnrichmentResult(found=False, resolvedDoi=resolved_doi)

        work = CrossrefWorkSummary.model_validate(lookup.payload)
        resolved_doi = work.doi or resolved_doi
        enrichment = CrossrefClient.to_enrichment(work)
        return CrossrefEnrichmentResult(
            found=True,
            resolvedDoi=resolved_doi,
            work=work,
            enrichment=enrichment,
        )

    async def get_unpaywall_open_access(
        self,
        *,
        paper_id: str | None = None,
        doi: str | None = None,
        paper: dict[str, Any] | Paper | None = None,
        request_id: str | None = None,
        request_outcomes: list[dict[str, Any]] | None = None,
        suppress_errors: bool = False,
    ) -> UnpaywallEnrichmentResult:
        if not self._enable_unpaywall or self._unpaywall_client is None:
            if suppress_errors:
                return UnpaywallEnrichmentResult(found=False)
            raise ValueError(
                "get_paper_open_access_unpaywall requires Unpaywall, "
                "which is disabled. "
                "Set SCHOLAR_SEARCH_ENABLE_UNPAYWALL=true to use this tool."
            )

        resolved_doi, _ = resolve_doi_inputs(doi=doi, paper_id=paper_id, paper=paper)
        if not resolved_doi:
            if suppress_errors:
                return UnpaywallEnrichmentResult(found=False)
            raise ValueError(
                "Unpaywall enrichment needs a DOI-bearing identifier or explicit doi."
            )

        try:
            lookup = await execute_provider_call(
                provider="unpaywall",
                endpoint="get_open_access",
                operation=lambda: self._unpaywall_client.get_open_access(resolved_doi),
                registry=self._provider_registry,
                request_outcomes=request_outcomes,
                request_id=request_id,
                is_empty=lambda payload: payload is None,
            )
        except Exception:
            if suppress_errors:
                logger.debug("Unpaywall enrichment failed", exc_info=True)
                return UnpaywallEnrichmentResult(found=False, doi=resolved_doi)
            raise

        if lookup.payload is None:
            if (
                lookup.outcome.status_bucket not in {"empty", "success"}
                and not suppress_errors
            ):
                raise ValueError(
                    lookup.outcome.error
                    or lookup.outcome.fallback_reason
                    or "Unpaywall enrichment failed."
                )
            return UnpaywallEnrichmentResult(found=False, doi=resolved_doi)

        enrichment = UnpaywallEnrichment.model_validate(lookup.payload)
        return UnpaywallEnrichmentResult(
            found=True,
            doi=enrichment.doi or resolved_doi,
            isOa=enrichment.is_oa,
            oaStatus=enrichment.oa_status,
            bestOaUrl=enrichment.best_oa_url,
            pdfUrl=enrichment.pdf_url,
            license=enrichment.license,
            journalIsInDoaj=enrichment.journal_is_in_doaj,
            enrichment=enrichment,
        )

    async def enrich_paper(
        self,
        *,
        paper_id: str | None = None,
        doi: str | None = None,
        paper: dict[str, Any] | Paper | None = None,
        query: str | None = None,
        request_id: str | None = None,
        request_outcomes: list[dict[str, Any]] | None = None,
    ) -> PaperEnrichmentResponse:
        resolved_doi, resolution_source = resolve_doi_inputs(
            doi=doi,
            paper_id=paper_id,
            paper=paper,
        )
        crossref = await self.get_crossref_metadata(
            paper_id=paper_id,
            doi=resolved_doi,
            paper=paper,
            query=query,
            request_id=request_id,
            request_outcomes=request_outcomes,
            suppress_errors=True,
        )
        resolved_after_crossref = crossref.resolved_doi or resolved_doi
        unpaywall = await self.get_unpaywall_open_access(
            paper_id=paper_id,
            doi=resolved_after_crossref,
            paper=paper,
            request_id=request_id,
            request_outcomes=request_outcomes,
            suppress_errors=True,
        )
        enrichments = self._merge_enrichments(crossref=crossref, unpaywall=unpaywall)
        return PaperEnrichmentResponse(
            doiResolution=DoiResolution(
                resolvedDoi=resolved_after_crossref,
                resolutionSource=(
                    resolution_source
                    or (
                        "crossref"
                        if crossref.found and crossref.resolved_doi
                        else None
                    )
                ),
            ),
            crossref=crossref,
            unpaywall=unpaywall,
            enrichments=enrichments,
        )

    async def enrich_paper_payload(
        self,
        paper: dict[str, Any] | Paper,
        *,
        query: str | None = None,
        request_id: str | None = None,
        request_outcomes: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        paper_model = Paper.model_validate(paper)
        response = await self.enrich_paper(
            paper_id=paper_model.canonical_id
            or paper_model.recommended_expansion_id
            or paper_model.paper_id,
            paper=paper_model,
            query=query or paper_model.title,
            request_id=request_id,
            request_outcomes=request_outcomes,
        )
        if response.enrichments is None:
            return dump_jsonable(paper_model)
        merged = self._merge_paper_enrichments(
            existing=paper_model.enrichments,
            incoming=response.enrichments,
        )
        return dump_jsonable(paper_model.model_copy(update={"enrichments": merged}))

    @staticmethod
    def _merge_enrichments(
        *,
        crossref: CrossrefEnrichmentResult | None,
        unpaywall: UnpaywallEnrichmentResult | None,
    ) -> PaperEnrichments | None:
        payload: dict[str, Any] = {}
        if crossref is not None and isinstance(crossref.enrichment, CrossrefEnrichment):
            payload["crossref"] = crossref.enrichment
        if unpaywall is not None and isinstance(
            unpaywall.enrichment, UnpaywallEnrichment
        ):
            payload["unpaywall"] = unpaywall.enrichment
        return PaperEnrichments.model_validate(payload) if payload else None

    @staticmethod
    def _merge_paper_enrichments(
        *,
        existing: PaperEnrichments | None,
        incoming: PaperEnrichments | None,
    ) -> PaperEnrichments | None:
        if incoming is None:
            return existing
        if existing is None:
            return incoming
        merged = existing.model_dump(by_alias=False, exclude_none=True)
        merged.update(incoming.model_dump(by_alias=False, exclude_none=True))
        return PaperEnrichments.model_validate(merged)
