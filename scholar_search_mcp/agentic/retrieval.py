"""Parallel multi-provider retrieval helpers for smart search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..provider_runtime import ProviderBudgetState, ProviderDiagnosticsRegistry
from ..search_executor import (
    ProviderExecutorName,
    ProviderSearchRequest,
    SearchClientBundle,
    SearchExecutor,
)
from .config import LatencyProfile

SMART_RETRIEVAL_FIELDS = [
    "paperId",
    "title",
    "abstract",
    "year",
    "authors",
    "citationCount",
    "referenceCount",
    "venue",
    "publicationDate",
    "url",
    "externalIds",
]

_SEARCH_EXECUTOR = SearchExecutor()


@dataclass
class RetrievedCandidate:
    """One candidate paper with provenance from provider and query variant."""

    paper: dict[str, Any]
    provider: str
    variant: str
    variant_source: str
    provider_rank: int


@dataclass
class RetrievalBatch:
    """All provider outputs for one query variant."""

    variant: str
    variant_source: str
    candidates: list[RetrievedCandidate]
    providers_used: list[str]
    provider_timings_ms: dict[str, int]
    provider_errors: dict[str, str]
    provider_outcomes: list[dict[str, Any]]


def provider_limits(
    *,
    intent: str,
    widened: bool = False,
    is_expansion: bool = False,
    latency_profile: LatencyProfile = "balanced",
) -> dict[str, int]:
    """Return conservative first-pass fetch sizes for each provider."""

    base_limits = {
        "semantic_scholar": 10,
        "openalex": 10,
        "core": 6,
        "arxiv": 6,
        "serpapi_google_scholar": 4,
    }
    if latency_profile == "fast":
        base_limits = {
            "semantic_scholar": 6,
            "openalex": 6,
            "core": 4,
            "arxiv": 4,
            "serpapi_google_scholar": 0,
        }
    elif latency_profile == "deep":
        base_limits = {
            "semantic_scholar": 12,
            "openalex": 12,
            "core": 8,
            "arxiv": 8,
            "serpapi_google_scholar": 6,
        }
    if intent == "review" or widened:
        if latency_profile == "fast":
            increment, cap = 2, 12
        elif latency_profile == "balanced":
            increment, cap = 4, 16
        else:
            increment, cap = 6, 20
        base_limits = {
            provider: min(limit + increment, cap)
            for provider, limit in base_limits.items()
        }
    if is_expansion:
        expansion_caps = {
            "fast": {
                "semantic_scholar": 4,
                "openalex": 4,
                "core": 3,
                "arxiv": 3,
                "serpapi_google_scholar": 0,
            },
            "balanced": {
                "semantic_scholar": 6,
                "openalex": 6,
                "core": 4,
                "arxiv": 4,
                "serpapi_google_scholar": 2,
            },
            "deep": {
                "semantic_scholar": 8,
                "openalex": 8,
                "core": 6,
                "arxiv": 6,
                "serpapi_google_scholar": 4,
            },
        }[latency_profile]
        return {
            provider: min(limit, expansion_caps[provider])
            for provider, limit in base_limits.items()
        }
    return base_limits


async def retrieve_variant(
    *,
    variant: str,
    variant_source: str,
    intent: str,
    year: str | None,
    venue: str | None,
    enable_core: bool,
    enable_semantic_scholar: bool,
    enable_openalex: bool,
    enable_arxiv: bool,
    enable_serpapi: bool,
    core_client: Any,
    semantic_client: Any,
    openalex_client: Any,
    arxiv_client: Any,
    serpapi_client: Any,
    widened: bool = False,
    is_expansion: bool = False,
    allow_serpapi: bool = True,
    latency_profile: LatencyProfile = "balanced",
    provider_registry: ProviderDiagnosticsRegistry | None = None,
    provider_budget: ProviderBudgetState | None = None,
    request_outcomes: list[dict[str, Any]] | None = None,
    request_id: str | None = None,
) -> RetrievalBatch:
    """Run one query variant across the configured provider family in parallel."""

    limits = provider_limits(
        intent=intent,
        widened=widened,
        is_expansion=is_expansion,
        latency_profile=latency_profile,
    )
    provider_calls: list[tuple[ProviderExecutorName, ProviderSearchRequest]] = []
    if enable_semantic_scholar:
        provider_calls.append(
            (
                "semantic_scholar",
                ProviderSearchRequest(
                    query=variant,
                    limit=limits["semantic_scholar"],
                    fields=SMART_RETRIEVAL_FIELDS,
                    year=year,
                    venue=[venue] if venue else None,
                ),
            )
        )
    if enable_openalex:
        provider_calls.append(
            (
                "openalex",
                ProviderSearchRequest(
                    query=variant,
                    limit=limits["openalex"],
                    year=year,
                ),
            )
        )
    if enable_core:
        provider_calls.append(
            (
                "core",
                ProviderSearchRequest(
                    query=variant,
                    limit=limits["core"],
                    year=year,
                ),
            )
        )
    if enable_arxiv:
        provider_calls.append(
            (
                "arxiv",
                ProviderSearchRequest(
                    query=variant,
                    limit=limits["arxiv"],
                    year=year,
                ),
            )
        )
    if (
        enable_serpapi
        and serpapi_client is not None
        and allow_serpapi
        and limits["serpapi_google_scholar"] > 0
    ):
        provider_calls.append(
            (
                "serpapi_google_scholar",
                ProviderSearchRequest(
                    query=variant,
                    limit=limits["serpapi_google_scholar"],
                    year=year,
                ),
            )
        )

    raw_results = await _SEARCH_EXECUTOR.execute_parallel_requests(
        provider_requests=provider_calls,
        clients=SearchClientBundle(
            core_client=core_client,
            semantic_client=semantic_client,
            openalex_client=openalex_client,
            arxiv_client=arxiv_client,
            serpapi_client=serpapi_client,
        ),
        provider_registry=provider_registry,
        budget=provider_budget,
        request_outcomes=request_outcomes,
        request_id=request_id,
    )

    candidates: list[RetrievedCandidate] = []
    providers_used: list[str] = []
    provider_timings_ms: dict[str, int] = {}
    provider_errors: dict[str, str] = {}
    provider_outcomes: list[dict[str, Any]] = []

    for index, call_result in enumerate(raw_results):
        provider = provider_calls[index][0]
        outcome = call_result.outcome
        provider_outcomes.append(outcome.to_dict())
        provider_timings_ms[provider] = outcome.latency_ms
        if outcome.error:
            provider_errors[provider] = outcome.error
        if call_result.response is None:
            continue
        provider_papers = call_result.response.model_dump(by_alias=True)["data"]
        if provider_papers:
            providers_used.append(provider)
        for rank, paper in enumerate(provider_papers, start=1):
            candidates.append(
                RetrievedCandidate(
                    paper=paper,
                    provider=provider,
                    variant=variant,
                    variant_source=variant_source,
                    provider_rank=rank,
                )
            )

    return RetrievalBatch(
        variant=variant,
        variant_source=variant_source,
        candidates=candidates,
        providers_used=providers_used,
        provider_timings_ms=provider_timings_ms,
        provider_errors=provider_errors,
        provider_outcomes=provider_outcomes,
    )
