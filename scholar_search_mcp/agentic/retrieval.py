"""Parallel multi-provider retrieval helpers for smart search."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from ..search import _arxiv_result, _core_result, _semantic_result, _serpapi_result


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


def provider_limits(*, intent: str, widened: bool = False) -> dict[str, int]:
    """Return conservative first-pass fetch sizes for each provider."""
    limits = {
        "semantic_scholar": 10,
        "openalex": 10,
        "core": 6,
        "arxiv": 6,
        "serpapi_google_scholar": 4,
    }
    if intent == "review" or widened:
        return {provider: min(limit + 4, 16) for provider, limit in limits.items()}
    return limits


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
    allow_serpapi: bool = True,
) -> RetrievalBatch:
    """Run one query variant across the configured provider family in parallel."""
    limits = provider_limits(intent=intent, widened=widened)
    provider_calls: list[tuple[str, Any]] = []
    if enable_semantic_scholar:
        provider_calls.append(
            (
                "semantic_scholar",
                semantic_client.search_papers(
                    query=variant,
                    limit=limits["semantic_scholar"],
                    year=year,
                    venue=[venue] if venue else None,
                ),
            )
        )
    if enable_openalex:
        provider_calls.append(
            (
                "openalex",
                openalex_client.search(
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
                core_client.search(
                    query=variant,
                    limit=limits["core"],
                    start=0,
                    year=year,
                ),
            )
        )
    if enable_arxiv:
        provider_calls.append(
            (
                "arxiv",
                arxiv_client.search(
                    query=variant,
                    limit=limits["arxiv"],
                    start=0,
                    year=year,
                ),
            )
        )
    if enable_serpapi and serpapi_client is not None and allow_serpapi:
        provider_calls.append(
            (
                "serpapi_google_scholar",
                serpapi_client.search(
                    query=variant,
                    limit=limits["serpapi_google_scholar"],
                    year=year,
                ),
            )
        )

    async def _timed_call(provider: str, coro: Any) -> tuple[str, int, Any]:
        started = time.perf_counter()
        result = await coro
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return provider, elapsed_ms, result

    raw_results = await asyncio.gather(
        *[_timed_call(provider, coro) for provider, coro in provider_calls],
        return_exceptions=True,
    )

    candidates: list[RetrievedCandidate] = []
    providers_used: list[str] = []
    provider_timings_ms: dict[str, int] = {}
    provider_errors: dict[str, str] = {}

    for index, raw_result in enumerate(raw_results):
        provider = provider_calls[index][0]
        if isinstance(raw_result, BaseException):
            provider_errors[provider] = str(raw_result)
            continue
        _, elapsed_ms, payload = raw_result
        provider_timings_ms[provider] = elapsed_ms
        provider_papers = _provider_papers(provider, payload, limits[provider])
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
    )


def _provider_papers(provider: str, payload: Any, limit: int) -> list[dict[str, Any]]:
    if provider == "semantic_scholar":
        return _semantic_result(payload, limit).model_dump(by_alias=True)["data"]
    if provider == "openalex":
        return list((payload or {}).get("data") or [])[:limit]
    if provider == "core":
        return _core_result(payload, limit).model_dump(by_alias=True)["data"]
    if provider == "arxiv":
        return _arxiv_result(payload, limit).model_dump(by_alias=True)["data"]
    if provider == "serpapi_google_scholar":
        return _serpapi_result(payload, limit).model_dump(by_alias=True)["data"]
    return []
