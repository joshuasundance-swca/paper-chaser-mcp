"""Crossref + Unpaywall enrichment helpers shared across tool surfaces."""

from __future__ import annotations

import asyncio
import logging
import re
from difflib import SequenceMatcher
from typing import Any

from .clients.crossref import CrossrefClient
from .identifiers import resolve_doi_from_paper_payload, resolve_doi_inputs
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

logger = logging.getLogger("paper-chaser-mcp")
_ENRICHMENT_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_ENRICHMENT_STOPWORDS = {
    "and",
    "for",
    "from",
    "that",
    "the",
    "this",
    "using",
    "with",
}
_ENRICHMENT_DETAIL_FIELDS = [
    "paperId",
    "title",
    "year",
    "authors",
    "venue",
    "publicationDate",
    "url",
    "externalIds",
]


def _normalized_enrichment_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _enrichment_token_overlap(left: str, right: str) -> float:
    left_tokens = {token for token in _ENRICHMENT_TOKEN_RE.findall(left) if token not in _ENRICHMENT_STOPWORDS}
    right_tokens = {token for token in _ENRICHMENT_TOKEN_RE.findall(right) if token not in _ENRICHMENT_STOPWORDS}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _author_surnames(authors: list[Any]) -> set[str]:
    surnames: set[str] = set()
    for author in authors:
        name = str(
            getattr(author, "name", None) or (author.get("name") if isinstance(author, dict) else "") or ""
        ).strip()
        tokens = _ENRICHMENT_TOKEN_RE.findall(name.lower())
        if tokens:
            surnames.add(tokens[-1])
    return surnames


def _trusted_query_fallback_match(
    *,
    paper: Paper,
    work: CrossrefWorkSummary,
) -> bool:
    paper_title = _normalized_enrichment_text(paper.title)
    work_title = _normalized_enrichment_text(work.title)
    if not paper_title or not work_title:
        return False

    title_similarity = max(
        SequenceMatcher(None, paper_title, work_title).ratio(),
        _enrichment_token_overlap(paper_title, work_title),
    )
    paper_authors = _author_surnames(list(paper.authors))
    work_authors = _author_surnames(list(work.authors))
    author_overlap = len(paper_authors & work_authors)

    venue_match = False
    if paper.venue and work.venue:
        paper_venue = _normalized_enrichment_text(paper.venue)
        work_venue = _normalized_enrichment_text(work.venue)
        venue_match = bool(paper_venue and work_venue and (paper_venue in work_venue or work_venue in paper_venue))

    year_match = paper.year is not None and work.year is not None and paper.year == work.year
    matched_signals = sum(
        [
            title_similarity >= 0.92,
            author_overlap > 0,
            year_match,
            venue_match,
        ]
    )
    if matched_signals >= 2 and title_similarity >= 0.92:
        return True
    if author_overlap > 0 and year_match and title_similarity >= 0.85:
        return True
    return False


def attach_enrichments_to_paper_payload(
    paper: dict[str, Any] | Paper,
    *,
    enriched_paper: dict[str, Any] | Paper,
) -> dict[str, Any]:
    """Attach additive enrichments without widening the original paper payload."""

    base = dump_jsonable(paper)
    enriched = dump_jsonable(enriched_paper)
    if not isinstance(base, dict):
        return enriched if isinstance(enriched, dict) else {}
    if isinstance(enriched, dict) and enriched.get("enrichments") is not None:
        base["enrichments"] = enriched["enrichments"]
    return base


async def hydrate_paper_for_enrichment(
    paper: dict[str, Any] | Paper,
    *,
    detail_client: Any | None,
) -> dict[str, Any]:
    """Fetch DOI-bearing paper details when the current payload is too thin."""

    paper_payload = dump_jsonable(Paper.model_validate(paper))
    resolved_doi, _ = resolve_doi_from_paper_payload(paper_payload)
    if resolved_doi or detail_client is None:
        return paper_payload

    detail_id = next(
        (
            candidate
            for candidate in (
                paper_payload.get("recommendedExpansionId"),
                paper_payload.get("canonicalId"),
                paper_payload.get("paperId"),
            )
            if isinstance(candidate, str) and candidate.strip()
        ),
        None,
    )
    if detail_id is None:
        return paper_payload

    try:
        detailed = await detail_client.get_paper_details(
            paper_id=detail_id,
            fields=_ENRICHMENT_DETAIL_FIELDS,
        )
    except Exception:
        logger.debug(
            "Paper-detail hydration for enrichment failed for %r.",
            detail_id,
            exc_info=True,
        )
        return paper_payload

    detailed_payload = dump_jsonable(detailed)
    if not isinstance(detailed_payload, dict):
        return paper_payload

    merged = dict(paper_payload)
    identifier_keys = {
        "canonicalId",
        "externalIds",
        "paperId",
        "recommendedExpansionId",
        "sourceId",
    }
    for key, value in detailed_payload.items():
        if value in (None, "", [], {}):
            continue
        if key in identifier_keys:
            merged[key] = value
            continue
        if merged.get(key) in (None, "", [], {}):
            merged[key] = value
    return merged


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
                "Set PAPER_CHASER_ENABLE_CROSSREF=true to use this tool."
            )

        resolved_doi, _ = resolve_doi_inputs(doi=doi, paper_id=paper_id, paper=paper)
        used_query_fallback = resolved_doi is None
        paper_model = Paper.model_validate(paper) if paper is not None else None
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
            if lookup.outcome.status_bucket not in {"empty", "success"} and not suppress_errors:
                raise ValueError(
                    lookup.outcome.error or lookup.outcome.fallback_reason or "Crossref enrichment failed."
                )
            return CrossrefEnrichmentResult(found=False, resolvedDoi=resolved_doi)

        work = CrossrefWorkSummary.model_validate(lookup.payload)
        if (
            used_query_fallback
            and paper_model is not None
            and not _trusted_query_fallback_match(paper=paper_model, work=work)
        ):
            logger.info(
                "Rejecting Crossref query fallback for %r because the returned work "
                "did not match the resolved paper strongly enough.",
                paper_model.title or paper_model.paper_id,
            )
            return CrossrefEnrichmentResult(found=False, resolvedDoi=resolved_doi)
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
                "Set PAPER_CHASER_ENABLE_UNPAYWALL=true to use this tool."
            )

        resolved_doi, _ = resolve_doi_inputs(doi=doi, paper_id=paper_id, paper=paper)
        if not resolved_doi:
            if suppress_errors:
                return UnpaywallEnrichmentResult(found=False)
            raise ValueError("Unpaywall enrichment needs a DOI-bearing identifier or explicit doi.")

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
            if lookup.outcome.status_bucket not in {"empty", "success"} and not suppress_errors:
                raise ValueError(
                    lookup.outcome.error or lookup.outcome.fallback_reason or "Unpaywall enrichment failed."
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
        existing_enrichments = paper_model.enrichments if paper_model is not None else None
        existing_crossref = existing_enrichments.crossref if existing_enrichments is not None else None
        existing_unpaywall = existing_enrichments.unpaywall if existing_enrichments is not None else None
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
            tasks = [task for task in [crossref_task, unpaywall_task] if task is not None]
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
            resolved_doi = crossref.resolved_doi if crossref is not None and crossref.resolved_doi else resolved_doi
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
                    resolution_source or ("crossref" if crossref.found and crossref.resolved_doi else None)
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
            paper_id=paper_model.canonical_id or paper_model.recommended_expansion_id or paper_model.paper_id,
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
        if unpaywall is not None and isinstance(unpaywall.enrichment, UnpaywallEnrichment):
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
