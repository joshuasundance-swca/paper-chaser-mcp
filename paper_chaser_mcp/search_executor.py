"""Shared provider contracts and execution helpers for paper search."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal
from urllib.parse import urlparse

from .models import (
    ArxivSearchResponse,
    BrokerAttempt,
    CoreSearchResponse,
    Paper,
    SearchResponse,
    SemanticSearchResponse,
)
from .models.tools import SearchProvider
from .provider_runtime import (
    ProviderBudgetState,
    ProviderCallResult,
    ProviderDiagnosticsRegistry,
    ProviderOutcomeEnvelope,
    ProviderStatusBucket,
    execute_provider_call,
    provider_attempt_reason,
    provider_status_to_attempt_status,
)
from .transport import asyncio

ProviderExecutorName = Literal[
    "semantic_scholar",
    "openalex",
    "core",
    "arxiv",
    "serpapi_google_scholar",
    "scholarapi",
]


@dataclass(frozen=True)
class ProviderCapabilities:
    supports_semantic_scholar_only_filters: bool = False
    hedge_eligible: bool = False
    supported_filter_aliases: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ProviderSearchRequest:
    query: str
    limit: int
    year: str | None = None
    fields: list[str] | None = None
    venue: list[str] | None = None
    publication_date_or_year: str | None = None
    fields_of_study: str | None = None
    publication_types: str | None = None
    open_access_pdf: bool | None = None
    min_citation_count: int | None = None


@dataclass(frozen=True)
class SearchClientBundle:
    core_client: Any = None
    semantic_client: Any = None
    openalex_client: Any = None
    arxiv_client: Any = None
    serpapi_client: Any = None
    scholarapi_client: Any = None


@dataclass(frozen=True)
class SearchProviderSpec:
    provider: ProviderExecutorName
    endpoint: str
    client_attr: str
    capabilities: ProviderCapabilities
    build_operation: Callable[[SearchClientBundle, ProviderSearchRequest], Callable[[], Awaitable[Any]]]
    parse_response: Callable[[Any, int], SearchResponse]
    propagate_exceptions: tuple[type[Exception], ...] | Callable[[], tuple[type[Exception], ...]] = ()


@dataclass(frozen=True)
class ProviderSearchResult:
    provider: ProviderExecutorName
    attempt: BrokerAttempt
    outcome: ProviderOutcomeEnvelope
    response: SearchResponse | None = None

    @property
    def has_results(self) -> bool:
        return self.response is not None and bool(self.response.data)


@dataclass(frozen=True)
class SearchExecutionTrace:
    effective_order: list[SearchProvider]
    ss_only_filters: list[str]
    attempts: list[BrokerAttempt]
    processed_providers: int
    provider_used: ProviderExecutorName | Literal["none"]
    result: SearchResponse | None


def enrich_semantic_scholar_paper(paper: Paper) -> Paper:
    """Return a Semantic Scholar paper enriched with provenance fields."""

    external_ids: dict[str, Any] = (paper.model_extra or {}).get("externalIds") or {}
    doi: str | None = external_ids.get("DOI") or None
    arxiv_id: str | None = external_ids.get("ArXiv") or None
    paper_id: str | None = paper.paper_id
    source_id = paper_id
    canonical_id: str | None = doi or paper_id or arxiv_id or source_id
    return paper.model_copy(
        update={
            "source": paper.source or "semantic_scholar",
            "source_id": source_id,
            "canonical_id": canonical_id,
            "recommended_expansion_id": canonical_id,
            "expansion_id_status": "portable",
        }
    )


_MIRROR_HOSTS = frozenset({"researchgate.net", "www.researchgate.net", "academia.edu", "www.academia.edu"})


def _paper_doi(paper: Paper) -> str | None:
    enrichments = paper.enrichments
    if enrichments is not None:
        for candidate in (
            enrichments.crossref.doi if enrichments.crossref is not None else None,
            enrichments.unpaywall.doi if enrichments.unpaywall is not None else None,
            enrichments.openalex.doi if enrichments.openalex is not None else None,
        ):
            if candidate:
                return candidate
    external_ids: dict[str, Any] = (paper.model_extra or {}).get("externalIds") or {}
    candidate = external_ids.get("DOI")
    return str(candidate) if candidate else None


def _paper_retrieved_url(paper: Paper) -> str | None:
    enrichments = paper.enrichments
    if paper.pdf_url:
        return paper.pdf_url
    if enrichments is not None and enrichments.unpaywall is not None:
        if enrichments.unpaywall.pdf_url:
            return enrichments.unpaywall.pdf_url
        if enrichments.unpaywall.best_oa_url:
            return enrichments.unpaywall.best_oa_url
    return paper.url


def _paper_canonical_url(paper: Paper, doi: str | None) -> str | None:
    if doi:
        return f"https://doi.org/{doi}"
    enrichments = paper.enrichments
    if enrichments is not None:
        for candidate in (
            enrichments.crossref.url if enrichments.crossref is not None else None,
            enrichments.openalex.url if enrichments.openalex is not None else None,
        ):
            if candidate:
                return candidate
    return paper.url


def _is_mirror_url(url: str | None) -> bool:
    if not url:
        return False
    hostname = urlparse(url).hostname or ""
    return hostname.lower() in _MIRROR_HOSTS


def _paper_source_type(paper: Paper) -> str:
    source = (paper.source or "").lower()
    if source in {"semantic_scholar", "scholarapi", "serpapi_google_scholar"}:
        return "scholarly_article"
    if source in {"arxiv", "core", "openalex"}:
        return "repository_record"
    if _is_mirror_url(paper.url):
        return "mirror"
    return "unknown"


def _paper_access_status(paper: Paper, retrieved_url: str | None) -> str:
    scholarapi = paper.content_access.scholarapi if paper.content_access is not None else None
    enrichments = paper.enrichments
    if scholarapi is not None and (scholarapi.has_text or scholarapi.has_pdf):
        return "full_text_verified"
    if paper.pdf_url:
        return "full_text_verified"
    if enrichments is not None and enrichments.unpaywall is not None:
        if enrichments.unpaywall.is_oa and (enrichments.unpaywall.pdf_url or enrichments.unpaywall.best_oa_url):
            return "oa_verified"
        if enrichments.unpaywall.is_oa:
            return "oa_uncertain"
    if _is_mirror_url(retrieved_url):
        return "mirror_only"
    if paper.abstract:
        return "abstract_only"
    return "access_unverified"


def _paper_open_access_route(paper: Paper, canonical_url: str | None, retrieved_url: str | None) -> str:
    enrichments = paper.enrichments
    unpaywall = enrichments.unpaywall if enrichments is not None else None
    if _is_mirror_url(retrieved_url):
        return "mirror_only"
    if unpaywall is not None and unpaywall.is_oa:
        if canonical_url and canonical_url.startswith("https://doi.org/"):
            return "canonical_open_access"
        return "repository_open_access"
    if paper.source in {"arxiv", "core", "openalex"} and (paper.pdf_url or paper.url):
        return "repository_open_access"
    if paper.access_status in {"full_text_verified", "oa_verified", "oa_uncertain", "abstract_only"}:
        return "non_oa_or_unconfirmed"
    return "unknown"


def annotate_paper_trust_metadata(paper: Paper) -> Paper:
    """Attach first-pass trust metadata to normalized scholarly paper results."""

    doi = _paper_doi(paper)
    canonical_url = _paper_canonical_url(paper, doi)
    retrieved_url = _paper_retrieved_url(paper)
    source_type = paper.source_type or _paper_source_type(paper)
    access_status = paper.access_status or _paper_access_status(paper, retrieved_url)
    full_text_observed = bool(
        paper.full_text_observed
        or paper.pdf_url
        or (
            paper.content_access is not None
            and paper.content_access.scholarapi is not None
            and (paper.content_access.scholarapi.has_text or paper.content_access.scholarapi.has_pdf)
        )
    )
    if paper.verification_status is not None:
        verification_status = paper.verification_status
    elif source_type == "mirror":
        verification_status = "search_hit_only"
    elif paper.source in {"semantic_scholar", "arxiv", "core", "openalex", "scholarapi"}:
        verification_status = "verified_metadata"
    else:
        verification_status = "search_hit_only"
    confidence = paper.confidence
    if confidence is None:
        confidence = (
            "high"
            if verification_status == "verified_primary_source"
            else ("medium" if verification_status == "verified_metadata" else "low")
        )
    return paper.model_copy(
        update={
            "source_type": source_type,
            "verification_status": verification_status,
            "access_status": access_status,
            "canonical_url": canonical_url,
            "retrieved_url": retrieved_url,
            "confidence": confidence,
            "is_primary_source": paper.is_primary_source if paper.is_primary_source is not None else False,
            "full_text_observed": full_text_observed,
            "abstract_observed": paper.abstract_observed
            if paper.abstract_observed is not None
            else bool(paper.abstract),
            "open_access_route": paper.open_access_route
            or _paper_open_access_route(paper, canonical_url, retrieved_url),
        }
    )


def core_result(core_response: dict[str, Any], limit: int) -> SearchResponse:
    core_search = CoreSearchResponse.model_validate(core_response)
    return SearchResponse(
        total=core_search.total or len(core_search.entries),
        offset=0,
        data=[
            annotate_paper_trust_metadata(paper.model_copy(update={"source": paper.source or "core"}))
            for paper in core_search.entries[:limit]
        ],
    )


def semantic_result(s2_response: dict[str, Any], limit: int) -> SearchResponse:
    semantic_search = SemanticSearchResponse.model_validate(s2_response)
    return SearchResponse(
        total=semantic_search.total or len(semantic_search.data),
        offset=semantic_search.offset,
        data=[
            annotate_paper_trust_metadata(enrich_semantic_scholar_paper(paper))
            for paper in semantic_search.data[:limit]
        ],
    )


def serpapi_result(serpapi_papers: list[dict[str, Any]], limit: int) -> SearchResponse:
    validated: list[Paper] = [Paper.model_validate(p) for p in serpapi_papers]
    return SearchResponse(
        total=len(validated),
        offset=0,
        data=[annotate_paper_trust_metadata(paper) for paper in validated[:limit]],
    )


def arxiv_result(arxiv_response: dict[str, Any], limit: int) -> SearchResponse:
    arxiv_search = ArxivSearchResponse.model_validate(arxiv_response)
    return SearchResponse(
        total=arxiv_search.total_results,
        offset=0,
        data=[annotate_paper_trust_metadata(paper) for paper in arxiv_search.entries[:limit]],
    )


def default_disabled_reason(provider: ProviderExecutorName) -> str:
    disabled_reasons = {
        "core": "Disabled by PAPER_CHASER_ENABLE_CORE=false.",
        "semantic_scholar": "Disabled by PAPER_CHASER_ENABLE_SEMANTIC_SCHOLAR=false.",
        "serpapi_google_scholar": "Disabled by PAPER_CHASER_ENABLE_SERPAPI=false.",
        "scholarapi": "Disabled by PAPER_CHASER_ENABLE_SCHOLARAPI=false.",
        "arxiv": "Disabled by PAPER_CHASER_ENABLE_ARXIV=false.",
        "openalex": "Disabled by PAPER_CHASER_ENABLE_OPENALEX=false.",
    }
    return disabled_reasons[provider]


def missing_client_reason(provider: ProviderExecutorName) -> str:
    reasons = {
        "core": "Enabled but no CORE client is configured.",
        "semantic_scholar": "Enabled but no Semantic Scholar client is configured.",
        "openalex": "Enabled but no OpenAlex client is configured.",
        "arxiv": "Enabled but no arXiv client is configured.",
        "serpapi_google_scholar": "Enabled but no SerpApi client is configured.",
        "scholarapi": "Enabled but no ScholarAPI client is configured.",
    }
    return reasons[provider]


def earlier_result_attempt(provider: ProviderExecutorName) -> BrokerAttempt:
    return BrokerAttempt(
        provider=provider,
        status="skipped",
        reason="Skipped because an earlier provider already returned results.",
    )


def provider_attempt_from_outcome(
    provider: ProviderExecutorName,
    *,
    status_bucket: ProviderStatusBucket,
    reason: str | None,
) -> BrokerAttempt:
    return BrokerAttempt(
        provider=provider,
        status=provider_status_to_attempt_status(status_bucket),
        reason=reason,
    )


def _core_operation(clients: SearchClientBundle, request: ProviderSearchRequest) -> Callable[[], Awaitable[Any]]:
    return lambda: clients.core_client.search(
        query=request.query,
        limit=request.limit,
        start=0,
        year=request.year,
    )


def _semantic_operation(clients: SearchClientBundle, request: ProviderSearchRequest) -> Callable[[], Awaitable[Any]]:
    return lambda: clients.semantic_client.search_papers(
        query=request.query,
        limit=request.limit,
        fields=request.fields,
        year=request.year,
        venue=request.venue,
        publication_date_or_year=request.publication_date_or_year,
        fields_of_study=request.fields_of_study,
        publication_types=request.publication_types,
        open_access_pdf=request.open_access_pdf,
        min_citation_count=request.min_citation_count,
    )


def _openalex_operation(clients: SearchClientBundle, request: ProviderSearchRequest) -> Callable[[], Awaitable[Any]]:
    return lambda: clients.openalex_client.search(
        query=request.query,
        limit=request.limit,
        year=request.year,
    )


def _arxiv_operation(clients: SearchClientBundle, request: ProviderSearchRequest) -> Callable[[], Awaitable[Any]]:
    return lambda: clients.arxiv_client.search(
        query=request.query,
        limit=request.limit,
        year=request.year,
    )


def _serpapi_operation(clients: SearchClientBundle, request: ProviderSearchRequest) -> Callable[[], Awaitable[Any]]:
    return lambda: clients.serpapi_client.search(
        query=request.query,
        limit=request.limit,
        year=request.year,
    )


def _scholarapi_operation(clients: SearchClientBundle, request: ProviderSearchRequest) -> Callable[[], Awaitable[Any]]:
    published_after = None
    published_before = None
    if request.year:
        normalized = str(request.year).strip()
        if re.fullmatch(r"\d{4}", normalized):
            published_after = f"{normalized}-01-01"
            published_before = f"{normalized}-12-31"
        elif re.fullmatch(r"\d{4}[-:]\d{4}", normalized):
            start_year, end_year = re.split(r"[-:]", normalized, maxsplit=1)
            published_after = f"{start_year}-01-01"
            published_before = f"{end_year}-12-31"
        elif re.fullmatch(r"\d{4}-", normalized):
            published_after = f"{normalized[:4]}-01-01"
        elif re.fullmatch(r"-\d{4}", normalized):
            published_before = f"{normalized[1:]}-12-31"
    return lambda: clients.scholarapi_client.search(
        query=request.query,
        limit=request.limit,
        published_after=published_after,
        published_before=published_before,
        has_pdf=True if request.open_access_pdf else None,
    )


def _serpapi_propagate_exceptions() -> tuple[type[Exception], ...]:
    from .clients.serpapi.errors import SerpApiKeyMissingError

    return (SerpApiKeyMissingError,)


def _scholarapi_propagate_exceptions() -> tuple[type[Exception], ...]:
    from .clients.scholarapi.errors import ScholarApiKeyMissingError

    return (ScholarApiKeyMissingError,)


SEARCH_PROVIDER_SPECS: dict[ProviderExecutorName, SearchProviderSpec] = {
    "core": SearchProviderSpec(
        provider="core",
        endpoint="search",
        client_attr="core_client",
        capabilities=ProviderCapabilities(
            supports_semantic_scholar_only_filters=False,
            hedge_eligible=True,
        ),
        build_operation=_core_operation,
        parse_response=core_result,
    ),
    "semantic_scholar": SearchProviderSpec(
        provider="semantic_scholar",
        endpoint="search_papers",
        client_attr="semantic_client",
        capabilities=ProviderCapabilities(
            supports_semantic_scholar_only_filters=True,
            hedge_eligible=True,
        ),
        build_operation=_semantic_operation,
        parse_response=semantic_result,
    ),
    "openalex": SearchProviderSpec(
        provider="openalex",
        endpoint="search",
        client_attr="openalex_client",
        capabilities=ProviderCapabilities(
            supports_semantic_scholar_only_filters=False,
            hedge_eligible=False,
        ),
        build_operation=_openalex_operation,
        parse_response=lambda payload, limit: SearchResponse(
            total=len(list((payload or {}).get("data") or [])),
            offset=0,
            data=[
                annotate_paper_trust_metadata(Paper.model_validate(paper))
                for paper in list((payload or {}).get("data") or [])[:limit]
            ],
        ),
    ),
    "arxiv": SearchProviderSpec(
        provider="arxiv",
        endpoint="search",
        client_attr="arxiv_client",
        capabilities=ProviderCapabilities(
            supports_semantic_scholar_only_filters=False,
            hedge_eligible=True,
        ),
        build_operation=_arxiv_operation,
        parse_response=arxiv_result,
    ),
    "serpapi_google_scholar": SearchProviderSpec(
        provider="serpapi_google_scholar",
        endpoint="search",
        client_attr="serpapi_client",
        capabilities=ProviderCapabilities(
            supports_semantic_scholar_only_filters=False,
            hedge_eligible=False,
        ),
        build_operation=_serpapi_operation,
        parse_response=serpapi_result,
        propagate_exceptions=_serpapi_propagate_exceptions,
    ),
    "scholarapi": SearchProviderSpec(
        provider="scholarapi",
        endpoint="search",
        client_attr="scholarapi_client",
        capabilities=ProviderCapabilities(
            supports_semantic_scholar_only_filters=False,
            hedge_eligible=False,
            supported_filter_aliases=frozenset({"openAccessPdf"}),
        ),
        build_operation=_scholarapi_operation,
        parse_response=lambda payload, limit: SearchResponse(
            total=int((payload or {}).get("total") or len(list((payload or {}).get("data") or []))),
            offset=int((payload or {}).get("offset") or 0),
            data=[
                annotate_paper_trust_metadata(Paper.model_validate(paper))
                for paper in list((payload or {}).get("data") or [])[:limit]
            ],
        ),
        propagate_exceptions=_scholarapi_propagate_exceptions,
    ),
}


class SearchExecutor:
    """Execute raw brokered and smart multi-provider search via shared specs."""

    def __init__(
        self,
        *,
        specs: Mapping[ProviderExecutorName, SearchProviderSpec] | None = None,
    ) -> None:
        self._specs = dict(specs or SEARCH_PROVIDER_SPECS)

    def effective_provider_order(
        self,
        *,
        preferred_provider: SearchProvider | None,
        provider_order: Sequence[SearchProvider] | None,
        default_order: Sequence[SearchProvider],
    ) -> list[SearchProvider]:
        order = list(provider_order or default_order)
        if preferred_provider is None:
            return order
        return [preferred_provider] + [provider for provider in order if provider != preferred_provider]

    def skip_for_semantic_scholar_only_filters(
        self,
        provider: SearchProvider,
        ss_only_filters: list[str],
    ) -> BrokerAttempt | None:
        spec = self._specs[provider]
        if spec.capabilities.supports_semantic_scholar_only_filters:
            return None
        if not ss_only_filters:
            return None
        unsupported_filters = [
            filter_name
            for filter_name in ss_only_filters
            if filter_name not in spec.capabilities.supported_filter_aliases
        ]
        if not unsupported_filters:
            return None
        reason = (
            "Skipped because Semantic Scholar-only filters were requested: " + ", ".join(ss_only_filters)
            if not spec.capabilities.supported_filter_aliases
            else "Skipped because this provider cannot honor requested advanced filters: "
            + ", ".join(unsupported_filters)
        )
        return BrokerAttempt(
            provider=provider,
            status="skipped",
            reason=reason,
        )

    def provider_enabled(
        self,
        provider: ProviderExecutorName,
        *,
        enabled: Mapping[ProviderExecutorName, bool],
        clients: SearchClientBundle,
    ) -> bool:
        if not enabled.get(provider, False):
            return False
        return getattr(clients, self._specs[provider].client_attr) is not None

    def disabled_attempt(
        self,
        provider: ProviderExecutorName,
        *,
        enabled: Mapping[ProviderExecutorName, bool],
        clients: SearchClientBundle,
    ) -> BrokerAttempt:
        if enabled.get(provider, False) and getattr(clients, self._specs[provider].client_attr) is None:
            return BrokerAttempt(
                provider=provider,
                status="skipped",
                reason=missing_client_reason(provider),
            )
        return BrokerAttempt(
            provider=provider,
            status="skipped",
            reason=default_disabled_reason(provider),
        )

    def hedge_target_index(
        self,
        *,
        effective_order: Sequence[SearchProvider],
        current_index: int,
        ss_only_filters: list[str],
        enabled: Mapping[ProviderExecutorName, bool],
        clients: SearchClientBundle,
    ) -> int | None:
        next_index = current_index + 1
        if next_index >= len(effective_order):
            return None
        provider = effective_order[next_index]
        spec = self._specs[provider]
        if not spec.capabilities.hedge_eligible:
            return None
        if self.skip_for_semantic_scholar_only_filters(provider, ss_only_filters):
            return None
        if not self.provider_enabled(provider, enabled=enabled, clients=clients):
            return None
        return next_index

    async def execute_provider(
        self,
        *,
        provider: ProviderExecutorName,
        request: ProviderSearchRequest,
        clients: SearchClientBundle,
        provider_registry: ProviderDiagnosticsRegistry | None = None,
        budget: ProviderBudgetState | None = None,
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> ProviderSearchResult:
        spec = self._specs[provider]
        propagate_exceptions = (
            spec.propagate_exceptions() if callable(spec.propagate_exceptions) else spec.propagate_exceptions
        )
        call_result: ProviderCallResult = await execute_provider_call(
            provider=provider,
            endpoint=spec.endpoint,
            operation=spec.build_operation(clients, request),
            registry=provider_registry,
            budget=budget,
            request_outcomes=request_outcomes,
            request_id=request_id,
            is_empty=lambda payload: not spec.parse_response(payload, request.limit).data,
            propagate_exceptions=propagate_exceptions,
        )
        response = spec.parse_response(call_result.payload, request.limit) if call_result.payload is not None else None
        return ProviderSearchResult(
            provider=provider,
            attempt=provider_attempt_from_outcome(
                provider,
                status_bucket=call_result.outcome.status_bucket,
                reason=provider_attempt_reason(call_result.outcome),
            ),
            outcome=call_result.outcome,
            response=response,
        )

    async def execute_parallel_requests(
        self,
        *,
        provider_requests: Sequence[tuple[ProviderExecutorName, ProviderSearchRequest]],
        clients: SearchClientBundle,
        provider_registry: ProviderDiagnosticsRegistry | None = None,
        budget: ProviderBudgetState | None = None,
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> list[ProviderSearchResult]:
        return await asyncio.gather(
            *[
                self.execute_provider(
                    provider=provider,
                    request=request,
                    clients=clients,
                    provider_registry=provider_registry,
                    budget=budget,
                    request_outcomes=request_outcomes,
                    request_id=request_id,
                )
                for provider, request in provider_requests
            ]
        )

    async def search_with_fallback(
        self,
        *,
        query: str,
        limit: int,
        year: str | None,
        fields: list[str] | None,
        venue: list[str] | None,
        preferred_provider: SearchProvider | None,
        provider_order: Sequence[SearchProvider] | None,
        default_provider_order: Sequence[SearchProvider],
        ss_only_filters: list[str],
        enabled: Mapping[ProviderExecutorName, bool],
        clients: SearchClientBundle,
        provider_registry: ProviderDiagnosticsRegistry | None = None,
        allow_default_hedging: bool = False,
        hedge_delay_seconds: float = 0.35,
        publication_date_or_year: str | None = None,
        fields_of_study: str | None = None,
        publication_types: str | None = None,
        open_access_pdf: bool | None = None,
        min_citation_count: int | None = None,
    ) -> SearchExecutionTrace:
        request = ProviderSearchRequest(
            query=query,
            limit=limit,
            year=year,
            fields=fields,
            venue=venue,
            publication_date_or_year=publication_date_or_year,
            fields_of_study=fields_of_study,
            publication_types=publication_types,
            open_access_pdf=open_access_pdf,
            min_citation_count=min_citation_count,
        )
        effective_order = self.effective_provider_order(
            preferred_provider=preferred_provider,
            provider_order=provider_order,
            default_order=default_provider_order,
        )
        result: SearchResponse | None = None
        provider_used: ProviderExecutorName | Literal["none"] = "none"
        attempts: list[BrokerAttempt] = []
        processed_providers = 0
        index = 0
        while index < len(effective_order):
            provider = effective_order[index]
            processed_providers = index + 1
            skip_for_filters = self.skip_for_semantic_scholar_only_filters(provider, ss_only_filters)
            if skip_for_filters is not None:
                attempts.append(skip_for_filters)
                index += 1
                continue
            if not self.provider_enabled(provider, enabled=enabled, clients=clients):
                attempts.append(self.disabled_attempt(provider, enabled=enabled, clients=clients))
                index += 1
                continue
            hedge_index = (
                self.hedge_target_index(
                    effective_order=effective_order,
                    current_index=index,
                    ss_only_filters=ss_only_filters,
                    enabled=enabled,
                    clients=clients,
                )
                if allow_default_hedging
                else None
            )
            hedge_started = asyncio.Event()
            current_task = asyncio.create_task(
                self.execute_provider(
                    provider=provider,
                    request=request,
                    clients=clients,
                    provider_registry=provider_registry,
                )
            )
            hedge_task: asyncio.Task[ProviderSearchResult] | None = None
            if hedge_index is not None:
                hedge_provider = effective_order[hedge_index]

                async def _run_hedged_provider() -> ProviderSearchResult:
                    await asyncio.sleep(hedge_delay_seconds)
                    if current_task.done():
                        raise asyncio.CancelledError
                    hedge_started.set()
                    return await self.execute_provider(
                        provider=hedge_provider,
                        request=request,
                        clients=clients,
                        provider_registry=provider_registry,
                    )

                hedge_task = asyncio.create_task(_run_hedged_provider())
            current_result = await current_task
            attempts.append(current_result.attempt)
            if current_result.has_results:
                result = current_result.response
                provider_used = provider
                if hedge_task is not None:
                    if not hedge_task.done():
                        hedge_task.cancel()
                    await asyncio.gather(hedge_task, return_exceptions=True)
                break
            result = None
            if hedge_index is None:
                index += 1
                continue
            if hedge_task is not None and hedge_started.is_set():
                hedge_result = await hedge_task
            else:
                if hedge_task is not None:
                    if not hedge_task.done():
                        hedge_task.cancel()
                    await asyncio.gather(hedge_task, return_exceptions=True)
                hedge_result = await self.execute_provider(
                    provider=effective_order[hedge_index],
                    request=request,
                    clients=clients,
                    provider_registry=provider_registry,
                )
            attempts.append(hedge_result.attempt)
            processed_providers = hedge_index + 1
            if hedge_result.has_results:
                result = hedge_result.response
                provider_used = hedge_result.provider
                break
            index = hedge_index + 1
        return SearchExecutionTrace(
            effective_order=effective_order,
            ss_only_filters=ss_only_filters,
            attempts=attempts,
            processed_providers=processed_providers,
            provider_used=provider_used,
            result=result,
        )


__all__ = [
    "ProviderCapabilities",
    "ProviderExecutorName",
    "ProviderSearchRequest",
    "ProviderSearchResult",
    "SEARCH_PROVIDER_SPECS",
    "SearchClientBundle",
    "SearchExecutionTrace",
    "SearchExecutor",
    "SearchProviderSpec",
    "arxiv_result",
    "core_result",
    "earlier_result_attempt",
    "enrich_semantic_scholar_paper",
    "provider_attempt_from_outcome",
    "semantic_result",
    "serpapi_result",
]
