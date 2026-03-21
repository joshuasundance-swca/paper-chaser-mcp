"""Crossref + Unpaywall enrichment helpers shared across tool surfaces."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .clients.crossref import CrossrefClient
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
        crossref_client: Any | None,
        unpaywall_client: Any | None,
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
        crossref_client = self._crossref_client
        if not self._enable_crossref or crossref_client is None:
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
                    operation=lambda: crossref_client.get_work(resolved_doi),
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
                    operation=lambda: crossref_client.search_work(normalized_query),
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
        unpaywall_client = self._unpaywall_client
        if not self._enable_unpaywall or unpaywall_client is None:
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
                operation=lambda: unpaywall_client.get_open_access(resolved_doi),
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
        paper_model = Paper.model_validate(paper) if paper is not None else None
        existing_enrichments = (
            paper_model.enrichments if paper_model is not None else None
        )
        existing_crossref = (
            existing_enrichments.crossref if existing_enrichments is not None else None
        )
        existing_unpaywall = (
            existing_enrichments.unpaywall if existing_enrichments is not None else None
        )
        resolved_doi, resolution_source = resolve_doi_inputs(
            doi=doi,
            paper_id=paper_id,
            paper=paper,
        )
        if resolved_doi is None:
            for cached_doi in [
                existing_crossref.doi if existing_crossref is not None else None,
                existing_unpaywall.doi if existing_unpaywall is not None else None,
            ]:
                if cached_doi:
                    resolved_doi = cached_doi
                    resolution_source = resolution_source or "existing_enrichment"
                    break

        crossref = (
            CrossrefEnrichmentResult(
                found=True,
                resolvedDoi=existing_crossref.doi or resolved_doi,
                enrichment=existing_crossref,
            )
            if existing_crossref is not None
            else None
        )
        unpaywall = (
            UnpaywallEnrichmentResult(
                found=True,
                doi=existing_unpaywall.doi or resolved_doi,
                isOa=existing_unpaywall.is_oa,
                oaStatus=existing_unpaywall.oa_status,
                bestOaUrl=existing_unpaywall.best_oa_url,
                pdfUrl=existing_unpaywall.pdf_url,
                license=existing_unpaywall.license,
                journalIsInDoaj=existing_unpaywall.journal_is_in_doaj,
                enrichment=existing_unpaywall,
            )
            if existing_unpaywall is not None
            else None
        )

        if resolved_doi:
            crossref_task = (
                asyncio.create_task(
                    self.get_crossref_metadata(
                        paper_id=paper_id,
                        doi=resolved_doi,
                        paper=paper_model or paper,
                        query=query,
                        request_id=request_id,
                        request_outcomes=request_outcomes,
                        suppress_errors=True,
                    )
                )
                if crossref is None
                else None
            )
            unpaywall_task = (
                asyncio.create_task(
                    self.get_unpaywall_open_access(
                        paper_id=paper_id,
                        doi=resolved_doi,
                        paper=paper_model or paper,
                        request_id=request_id,
                        request_outcomes=request_outcomes,
                        suppress_errors=True,
                    )
                )
                if unpaywall is None
                else None
            )
            tasks = [
                task for task in [crossref_task, unpaywall_task] if task is not None
            ]
            if tasks:
                await asyncio.gather(*tasks)
            if crossref_task is not None:
                crossref = crossref_task.result()
            if unpaywall_task is not None:
                unpaywall = unpaywall_task.result()
        else:
            if crossref is None:
                crossref = await self.get_crossref_metadata(
                    paper_id=paper_id,
                    doi=resolved_doi,
                    paper=paper_model or paper,
                    query=query,
                    request_id=request_id,
                    request_outcomes=request_outcomes,
                    suppress_errors=True,
                )
            resolved_doi = (
                crossref.resolved_doi
                if crossref is not None and crossref.resolved_doi
                else resolved_doi
            )
            if unpaywall is None:
                unpaywall = await self.get_unpaywall_open_access(
                    paper_id=paper_id,
                    doi=resolved_doi,
                    paper=paper_model or paper,
                    request_id=request_id,
                    request_outcomes=request_outcomes,
                    suppress_errors=True,
                )

        if crossref is None:
            crossref = CrossrefEnrichmentResult(found=False, resolvedDoi=resolved_doi)
        resolved_after_crossref = crossref.resolved_doi or resolved_doi
        if unpaywall is None:
            unpaywall = UnpaywallEnrichmentResult(
                found=False,
                doi=resolved_after_crossref,
            )
        enrichments = self._merge_enrichments(crossref=crossref, unpaywall=unpaywall)
        return PaperEnrichmentResponse(
            doiResolution=DoiResolution(
                resolvedDoi=resolved_after_crossref,
                resolutionSource=(
                    resolution_source
                    or (
                        "crossref" if crossref.found and crossref.resolved_doi else None
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
