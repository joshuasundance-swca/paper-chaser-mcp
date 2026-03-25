"""Shared provider contracts and execution helpers for paper search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

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
]


@dataclass(frozen=True)
class ProviderCapabilities:
    supports_semantic_scholar_only_filters: bool = False
    hedge_eligible: bool = False


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


def core_result(core_response: dict[str, Any], limit: int) -> SearchResponse:
    core_search = CoreSearchResponse.model_validate(core_response)
    return SearchResponse(
        total=core_search.total or len(core_search.entries),
        offset=0,
        data=core_search.entries[:limit],
    )


def semantic_result(s2_response: dict[str, Any], limit: int) -> SearchResponse:
    semantic_search = SemanticSearchResponse.model_validate(s2_response)
    return SearchResponse(
        total=semantic_search.total or len(semantic_search.data),
        offset=semantic_search.offset,
        data=[enrich_semantic_scholar_paper(paper) for paper in semantic_search.data[:limit]],
    )


def serpapi_result(serpapi_papers: list[dict[str, Any]], limit: int) -> SearchResponse:
    validated: list[Paper] = [Paper.model_validate(p) for p in serpapi_papers]
    return SearchResponse(
        total=len(validated),
        offset=0,
        data=validated[:limit],
    )


def arxiv_result(arxiv_response: dict[str, Any], limit: int) -> SearchResponse:
    arxiv_search = ArxivSearchResponse.model_validate(arxiv_response)
    return SearchResponse(
        total=arxiv_search.total_results,
        offset=0,
        data=arxiv_search.entries[:limit],
    )


def default_disabled_reason(provider: ProviderExecutorName) -> str:
    disabled_reasons = {
        "core": "Disabled by PAPER_CHASER_ENABLE_CORE=false.",
        "semantic_scholar": "Disabled by PAPER_CHASER_ENABLE_SEMANTIC_SCHOLAR=false.",
        "serpapi_google_scholar": "Disabled by PAPER_CHASER_ENABLE_SERPAPI=false.",
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


def _serpapi_propagate_exceptions() -> tuple[type[Exception], ...]:
    from .clients.serpapi.errors import SerpApiKeyMissingError

    return (SerpApiKeyMissingError,)


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
            data=[Paper.model_validate(paper) for paper in list((payload or {}).get("data") or [])[:limit]],
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
        return BrokerAttempt(
            provider=provider,
            status="skipped",
            reason=("Skipped because Semantic Scholar-only filters were requested: " + ", ".join(ss_only_filters)),
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
