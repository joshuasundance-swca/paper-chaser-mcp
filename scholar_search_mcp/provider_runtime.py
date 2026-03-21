"""Shared provider execution policy, telemetry, and diagnostics."""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Literal

from .clients.serpapi import (
    SerpApiKeyMissingError,
    SerpApiQuotaError,
    SerpApiUpstreamError,
)

logger = logging.getLogger("scholar-search-mcp")

ProviderName = Literal[
    "semantic_scholar",
    "openalex",
    "core",
    "arxiv",
    "serpapi_google_scholar",
    "openai",
]
ProviderStatusBucket = Literal[
    "success",
    "empty",
    "rate_limited",
    "quota_exhausted",
    "auth_error",
    "provider_error",
    "suppressed",
    "skipped",
]

_RETRYABLE_STATUSES: frozenset[ProviderStatusBucket] = frozenset(
    {"rate_limited", "provider_error"}
)
_FAILURE_STATUSES: frozenset[ProviderStatusBucket] = frozenset(
    {"rate_limited", "quota_exhausted", "auth_error", "provider_error"}
)
_SUCCESSISH_STATUSES: frozenset[ProviderStatusBucket] = frozenset(
    {"success", "empty", "skipped"}
)


@dataclass(frozen=True)
class ProviderPolicy:
    """Shared execution policy for one provider family."""

    concurrency_limit: int = 2
    max_attempts: int = 1
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 4.0
    failure_threshold: int = 2
    suppression_seconds: float = 45.0
    paywalled: bool = False


DEFAULT_PROVIDER_POLICIES: dict[str, ProviderPolicy] = {
    "semantic_scholar": ProviderPolicy(
        concurrency_limit=1,
        max_attempts=1,
        failure_threshold=2,
        suppression_seconds=30.0,
    ),
    "openalex": ProviderPolicy(
        concurrency_limit=2,
        max_attempts=2,
        base_delay_seconds=0.5,
        suppression_seconds=30.0,
    ),
    "core": ProviderPolicy(
        concurrency_limit=1,
        max_attempts=1,
        failure_threshold=1,
        suppression_seconds=300.0,
    ),
    "arxiv": ProviderPolicy(
        concurrency_limit=2,
        max_attempts=1,
        failure_threshold=2,
        suppression_seconds=30.0,
    ),
    "serpapi_google_scholar": ProviderPolicy(
        concurrency_limit=1,
        max_attempts=1,
        failure_threshold=1,
        suppression_seconds=300.0,
        paywalled=True,
    ),
    "openai": ProviderPolicy(
        concurrency_limit=2,
        max_attempts=2,
        base_delay_seconds=0.5,
        failure_threshold=2,
        suppression_seconds=60.0,
        paywalled=True,
    ),
}


@dataclass
class ProviderOutcomeEnvelope:
    """Normalized execution envelope for one provider call."""

    provider: str
    endpoint: str
    status_bucket: ProviderStatusBucket
    latency_ms: int = 0
    retries: int = 0
    fallback_reason: str | None = None
    error: str | None = None
    cache_info: dict[str, Any] = field(default_factory=dict)
    quota_metadata: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "endpoint": self.endpoint,
            "statusBucket": self.status_bucket,
            "latencyMs": self.latency_ms,
            "retries": self.retries,
            "fallbackReason": self.fallback_reason,
            "error": self.error,
            "cacheInfo": self.cache_info,
            "quotaMetadata": self.quota_metadata,
            "requestId": self.request_id,
            "createdAt": self.created_at,
        }


@dataclass
class ProviderCallResult:
    """Wrapped provider result plus normalized telemetry."""

    payload: Any
    outcome: ProviderOutcomeEnvelope


@dataclass
class ProviderBudgetState:
    """Per-request budget guardrails for multi-provider retrieval."""

    max_total_calls: int | None = None
    max_semantic_scholar_calls: int | None = None
    max_openalex_calls: int | None = None
    max_core_calls: int | None = None
    max_arxiv_calls: int | None = None
    max_serpapi_calls: int | None = None
    allow_paid_providers: bool = True
    _counts: Counter[str] = field(default_factory=Counter, init=False, repr=False)
    _total_calls: int = field(default=0, init=False, repr=False)
    _lock: asyncio.Lock | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "ProviderBudgetState | None":
        if not raw:
            return None
        values = dict(raw)
        if (
            not any(
                values.get(key) is not None
                for key in (
                    "max_total_calls",
                    "max_semantic_scholar_calls",
                    "max_openalex_calls",
                    "max_core_calls",
                    "max_arxiv_calls",
                    "max_serpapi_calls",
                )
            )
            and values.get("allow_paid_providers", True) is True
        ):
            return None
        return cls(
            max_total_calls=values.get("max_total_calls"),
            max_semantic_scholar_calls=values.get("max_semantic_scholar_calls"),
            max_openalex_calls=values.get("max_openalex_calls"),
            max_core_calls=values.get("max_core_calls"),
            max_arxiv_calls=values.get("max_arxiv_calls"),
            max_serpapi_calls=values.get("max_serpapi_calls"),
            allow_paid_providers=bool(values.get("allow_paid_providers", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "maxTotalCalls": self.max_total_calls,
            "maxSemanticScholarCalls": self.max_semantic_scholar_calls,
            "maxOpenAlexCalls": self.max_openalex_calls,
            "maxCoreCalls": self.max_core_calls,
            "maxArxivCalls": self.max_arxiv_calls,
            "maxSerpApiCalls": self.max_serpapi_calls,
            "allowPaidProviders": self.allow_paid_providers,
            "appliedCounts": {
                "totalCalls": self._total_calls,
                **{provider: count for provider, count in sorted(self._counts.items())},
            },
        }

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def reserve(self, provider: str, *, paywalled: bool) -> str | None:
        lock = self._get_lock()
        async with lock:
            if paywalled and not self.allow_paid_providers:
                return "Skipped because providerBudget disallows paid providers."
            if (
                self.max_total_calls is not None
                and self._total_calls >= self.max_total_calls
            ):
                return "Skipped because providerBudget exhausted maxTotalCalls."
            per_provider_limit = {
                "semantic_scholar": self.max_semantic_scholar_calls,
                "openalex": self.max_openalex_calls,
                "core": self.max_core_calls,
                "arxiv": self.max_arxiv_calls,
                "serpapi_google_scholar": self.max_serpapi_calls,
            }.get(provider)
            if (
                per_provider_limit is not None
                and self._counts[provider] >= per_provider_limit
            ):
                return (
                    "Skipped because providerBudget exhausted "
                    f"the per-provider limit for {provider}."
                )
            self._total_calls += 1
            self._counts[provider] += 1
            return None


class ProviderDiagnosticsRegistry:
    """Cross-request provider health, suppression, and recent-outcome registry."""

    def __init__(self, *, recent_outcomes_per_provider: int = 20) -> None:
        self._recent_outcomes_per_provider = recent_outcomes_per_provider
        self._recent: dict[str, deque[ProviderOutcomeEnvelope]] = {}
        self._status_counts: dict[str, Counter[str]] = {}
        self._consecutive_failures: Counter[str] = Counter()
        self._suppressed_until: dict[str, float] = {}
        self._last_error: dict[str, str | None] = {}
        self._last_latency_ms: dict[str, int] = {}
        self._last_endpoint: dict[str, str | None] = {}
        self._last_outcome: dict[str, str | None] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._started_at = datetime.now(timezone.utc).isoformat()

    def semaphore(self, provider: str, limit: int) -> asyncio.Semaphore:
        semaphore = self._semaphores.get(provider)
        if semaphore is None:
            semaphore = asyncio.Semaphore(max(limit, 1))
            self._semaphores[provider] = semaphore
        return semaphore

    def is_suppressed(self, provider: str) -> bool:
        return self.suppressed_until(provider) is not None

    def suppressed_until(self, provider: str) -> str | None:
        value = self._suppressed_until.get(provider)
        if value is None:
            return None
        if value <= time.time():
            self._suppressed_until.pop(provider, None)
            return None
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()

    def record(
        self,
        outcome: ProviderOutcomeEnvelope,
        *,
        policy: ProviderPolicy,
    ) -> None:
        recent = self._recent.setdefault(
            outcome.provider,
            deque(maxlen=self._recent_outcomes_per_provider),
        )
        recent.append(outcome)
        self._status_counts.setdefault(outcome.provider, Counter())[
            outcome.status_bucket
        ] += 1
        self._last_latency_ms[outcome.provider] = outcome.latency_ms
        self._last_endpoint[outcome.provider] = outcome.endpoint
        self._last_outcome[outcome.provider] = outcome.status_bucket
        if outcome.status_bucket in _SUCCESSISH_STATUSES:
            self._consecutive_failures[outcome.provider] = 0
            if outcome.status_bucket == "success":
                self._last_error[outcome.provider] = None
        else:
            self._consecutive_failures[outcome.provider] += 1
            self._last_error[outcome.provider] = outcome.error
            if outcome.status_bucket == "quota_exhausted":
                self._suppressed_until[outcome.provider] = time.time() + max(
                    policy.suppression_seconds,
                    300.0,
                )
            elif (
                self._consecutive_failures[outcome.provider] >= policy.failure_threshold
            ):
                self._suppressed_until[outcome.provider] = (
                    time.time() + policy.suppression_seconds
                )

    def snapshot(
        self,
        *,
        enabled: dict[str, bool] | None = None,
        provider_order: list[str] | None = None,
    ) -> dict[str, Any]:
        providers = set(DEFAULT_PROVIDER_POLICIES)
        providers.update(self._recent)
        if enabled:
            providers.update(enabled)
        ordered_providers = list(provider_order or [])
        ordered_providers.extend(
            sorted(
                provider for provider in providers if provider not in ordered_providers
            )
        )
        return {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "startedAt": self._started_at,
            "providerOrder": ordered_providers,
            "providers": [
                {
                    "provider": provider,
                    "enabled": enabled.get(provider) if enabled else None,
                    "suppressed": self.is_suppressed(provider),
                    "suppressedUntil": self.suppressed_until(provider),
                    "consecutiveFailures": self._consecutive_failures.get(provider, 0),
                    "statusCounts": dict(self._status_counts.get(provider, Counter())),
                    "lastOutcome": self._last_outcome.get(provider),
                    "lastEndpoint": self._last_endpoint.get(provider),
                    "lastLatencyMs": self._last_latency_ms.get(provider),
                    "lastError": self._last_error.get(provider),
                    "recentOutcomes": [
                        envelope.to_dict()
                        for envelope in self._recent.get(provider, deque())
                    ],
                }
                for provider in ordered_providers
            ],
        }


def policy_for_provider(provider: str) -> ProviderPolicy:
    return DEFAULT_PROVIDER_POLICIES.get(provider, ProviderPolicy())


def _default_fallback_reason(status_bucket: ProviderStatusBucket) -> str | None:
    reasons = {
        "rate_limited": "Provider rate limited the request.",
        "quota_exhausted": "Provider quota was exhausted.",
        "auth_error": "Provider authentication failed.",
        "provider_error": "Provider returned an upstream error.",
        "suppressed": "Provider is temporarily suppressed after recent failures.",
        "skipped": "Provider call was skipped.",
    }
    return reasons.get(status_bucket)


def _classify_exception(exc: Exception) -> ProviderStatusBucket:
    if isinstance(exc, SerpApiKeyMissingError):
        return "auth_error"
    if isinstance(exc, SerpApiQuotaError):
        return "quota_exhausted"
    if isinstance(exc, SerpApiUpstreamError):
        return "provider_error"

    text = str(exc).lower()
    type_name = type(exc).__name__.lower()
    if "429" in text or "rate limit" in text or "too many requests" in text:
        return "rate_limited"
    if "quota" in text or "credits" in text:
        return "quota_exhausted"
    if any(
        token in text
        for token in ("api key", "unauthorized", "forbidden", "authentication")
    ):
        return "auth_error"
    if "timeout" in text or "timed out" in text:
        return "provider_error"
    if any(token in text for token in ("http 5", "500", "502", "503", "504")):
        return "provider_error"
    if (
        "connect" in text
        or "network" in text
        or "transport" in text
        or "remoteprotocolerror" in type_name
    ):
        return "provider_error"
    return "provider_error"


def _retry_delay_seconds(attempt: int, *, policy: ProviderPolicy) -> float:
    base = min(policy.base_delay_seconds * (2**attempt), policy.max_delay_seconds)
    return max(0.0, base + random.uniform(0.0, min(0.25, base / 4 or 0.05)))


def _empty_payload(payload: Any) -> bool:
    if payload is None:
        return True
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return len(payload.get("data") or []) == 0
        if isinstance(payload.get("recommendedPapers"), list):
            return len(payload.get("recommendedPapers") or []) == 0
        if isinstance(payload.get("matches"), list):
            return len(payload.get("matches") or []) == 0
        if isinstance(payload.get("results"), list):
            return len(payload.get("results") or []) == 0
    if isinstance(payload, list):
        return len(payload) == 0
    return False


def _quota_metadata_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    metadata: dict[str, Any] = {}
    search_metadata = payload.get("search_metadata")
    if isinstance(search_metadata, dict):
        if "id" in search_metadata:
            metadata["searchId"] = search_metadata.get("id")
        if "status" in search_metadata:
            metadata["searchStatus"] = search_metadata.get("status")
    account = payload.get("account")
    if isinstance(account, dict):
        for key in (
            "plan",
            "searches_left",
            "total_searches_left",
            "this_hour_searches_left",
        ):
            if key in account:
                metadata[key] = account.get(key)
    return metadata


def _shorten_runtime_log_text(text: str | None, *, limit: int = 120) -> str | None:
    if text is None:
        return None
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: max(limit - 3, 1)].rstrip()}..."


def _log_request_scoped_provider_event(
    *,
    request_id: str | None,
    provider: str,
    endpoint: str,
    status: str,
    latency_ms: int | None = None,
    retries: int | None = None,
    reason: str | None = None,
) -> None:
    if request_id is None:
        return
    parts = [f"status={status}"]
    if latency_ms is not None:
        parts.append(f"latency_ms={latency_ms}")
    if retries is not None:
        parts.append(f"retries={retries}")
    short_reason = _shorten_runtime_log_text(reason)
    if short_reason:
        parts.append(f"reason={short_reason!r}")
    logger.info(
        "provider-call[%s] provider=%s endpoint=%s %s",
        request_id,
        provider,
        endpoint,
        " ".join(parts),
    )


async def execute_provider_call(
    *,
    provider: str,
    endpoint: str,
    operation: Callable[[], Awaitable[Any]],
    registry: ProviderDiagnosticsRegistry | None = None,
    policy: ProviderPolicy | None = None,
    budget: ProviderBudgetState | None = None,
    request_outcomes: list[dict[str, Any]] | None = None,
    request_id: str | None = None,
    is_empty: Callable[[Any], bool] | None = None,
    metadata_extractor: Callable[[Any], dict[str, Any]] | None = None,
    propagate_exceptions: tuple[type[Exception], ...] = (),
) -> ProviderCallResult:
    """Run one async provider call through the shared policy wrapper."""

    resolved_policy = policy or policy_for_provider(provider)
    empty_checker = is_empty or _empty_payload
    metadata_reader = metadata_extractor or _quota_metadata_from_payload

    if registry is not None and registry.is_suppressed(provider):
        outcome = ProviderOutcomeEnvelope(
            provider=provider,
            endpoint=endpoint,
            status_bucket="suppressed",
            fallback_reason=_default_fallback_reason("suppressed"),
            request_id=request_id,
        )
        registry.record(outcome, policy=resolved_policy)
        if request_outcomes is not None:
            request_outcomes.append(outcome.to_dict())
        _log_request_scoped_provider_event(
            request_id=request_id,
            provider=provider,
            endpoint=endpoint,
            status="suppressed",
            reason=outcome.fallback_reason,
        )
        return ProviderCallResult(payload=None, outcome=outcome)

    if budget is not None:
        budget_reason = await budget.reserve(
            provider,
            paywalled=resolved_policy.paywalled,
        )
        if budget_reason is not None:
            outcome = ProviderOutcomeEnvelope(
                provider=provider,
                endpoint=endpoint,
                status_bucket="skipped",
                fallback_reason=budget_reason,
                request_id=request_id,
            )
            if registry is not None:
                registry.record(outcome, policy=resolved_policy)
            if request_outcomes is not None:
                request_outcomes.append(outcome.to_dict())
            _log_request_scoped_provider_event(
                request_id=request_id,
                provider=provider,
                endpoint=endpoint,
                status="skipped",
                reason=outcome.fallback_reason,
            )
            return ProviderCallResult(payload=None, outcome=outcome)

    semaphore = (
        registry.semaphore(provider, resolved_policy.concurrency_limit)
        if registry
        else None
    )
    if semaphore is not None:
        await semaphore.acquire()
    _log_request_scoped_provider_event(
        request_id=request_id,
        provider=provider,
        endpoint=endpoint,
        status="started",
    )
    try:
        for attempt in range(max(resolved_policy.max_attempts, 1)):
            started = time.perf_counter()
            try:
                payload = await operation()
            except Exception as exc:
                status_bucket = _classify_exception(exc)
                retries = attempt
                latency_ms = int((time.perf_counter() - started) * 1000)
                if status_bucket in _RETRYABLE_STATUSES and attempt + 1 < max(
                    resolved_policy.max_attempts, 1
                ):
                    retry_reason = str(exc)
                    _log_request_scoped_provider_event(
                        request_id=request_id,
                        provider=provider,
                        endpoint=endpoint,
                        status="retrying",
                        latency_ms=latency_ms,
                        retries=attempt + 1,
                        reason=retry_reason,
                    )
                    await asyncio.sleep(
                        _retry_delay_seconds(attempt, policy=resolved_policy)
                    )
                    continue
                outcome = ProviderOutcomeEnvelope(
                    provider=provider,
                    endpoint=endpoint,
                    status_bucket=status_bucket,
                    latency_ms=latency_ms,
                    retries=retries,
                    fallback_reason=_default_fallback_reason(status_bucket),
                    error=str(exc),
                    request_id=request_id,
                )
                if registry is not None:
                    registry.record(outcome, policy=resolved_policy)
                if request_outcomes is not None:
                    request_outcomes.append(outcome.to_dict())
                _log_request_scoped_provider_event(
                    request_id=request_id,
                    provider=provider,
                    endpoint=endpoint,
                    status=outcome.status_bucket,
                    latency_ms=latency_ms,
                    retries=retries,
                    reason=outcome.error or outcome.fallback_reason,
                )
                if isinstance(exc, propagate_exceptions):
                    raise
                return ProviderCallResult(payload=None, outcome=outcome)

            status_bucket = "empty" if empty_checker(payload) else "success"
            outcome = ProviderOutcomeEnvelope(
                provider=provider,
                endpoint=endpoint,
                status_bucket=status_bucket,
                latency_ms=int((time.perf_counter() - started) * 1000),
                retries=attempt,
                cache_info={},
                quota_metadata=metadata_reader(payload),
                request_id=request_id,
            )
            if registry is not None:
                registry.record(outcome, policy=resolved_policy)
            if request_outcomes is not None:
                request_outcomes.append(outcome.to_dict())
            _log_request_scoped_provider_event(
                request_id=request_id,
                provider=provider,
                endpoint=endpoint,
                status=outcome.status_bucket,
                latency_ms=outcome.latency_ms,
                retries=attempt,
            )
            return ProviderCallResult(payload=payload, outcome=outcome)
    finally:
        if semaphore is not None:
            semaphore.release()
    raise RuntimeError("Provider execution loop exited unexpectedly.")


def execute_provider_call_sync(
    *,
    provider: str,
    endpoint: str,
    operation: Callable[[], Any],
    registry: ProviderDiagnosticsRegistry | None = None,
    policy: ProviderPolicy | None = None,
    request_id: str | None = None,
    is_empty: Callable[[Any], bool] | None = None,
    metadata_extractor: Callable[[Any], dict[str, Any]] | None = None,
) -> ProviderCallResult:
    """Sync variant used by model-provider adapters that run inline."""

    resolved_policy = policy or policy_for_provider(provider)
    empty_checker = is_empty or _empty_payload
    metadata_reader = metadata_extractor or _quota_metadata_from_payload

    if registry is not None and registry.is_suppressed(provider):
        outcome = ProviderOutcomeEnvelope(
            provider=provider,
            endpoint=endpoint,
            status_bucket="suppressed",
            fallback_reason=_default_fallback_reason("suppressed"),
            request_id=request_id,
        )
        registry.record(outcome, policy=resolved_policy)
        return ProviderCallResult(payload=None, outcome=outcome)

    for attempt in range(max(resolved_policy.max_attempts, 1)):
        started = time.perf_counter()
        try:
            payload = operation()
        except Exception as exc:
            status_bucket = _classify_exception(exc)
            latency_ms = int((time.perf_counter() - started) * 1000)
            if status_bucket in _RETRYABLE_STATUSES and attempt + 1 < max(
                resolved_policy.max_attempts, 1
            ):
                time.sleep(_retry_delay_seconds(attempt, policy=resolved_policy))
                continue
            outcome = ProviderOutcomeEnvelope(
                provider=provider,
                endpoint=endpoint,
                status_bucket=status_bucket,
                latency_ms=latency_ms,
                retries=attempt,
                fallback_reason=_default_fallback_reason(status_bucket),
                error=str(exc),
                request_id=request_id,
            )
            if registry is not None:
                registry.record(outcome, policy=resolved_policy)
            return ProviderCallResult(payload=None, outcome=outcome)

        status_bucket = "empty" if empty_checker(payload) else "success"
        outcome = ProviderOutcomeEnvelope(
            provider=provider,
            endpoint=endpoint,
            status_bucket=status_bucket,
            latency_ms=int((time.perf_counter() - started) * 1000),
            retries=attempt,
            quota_metadata=metadata_reader(payload),
            request_id=request_id,
        )
        if registry is not None:
            registry.record(outcome, policy=resolved_policy)
        return ProviderCallResult(payload=payload, outcome=outcome)

    raise RuntimeError("Provider execution loop exited unexpectedly.")


def provider_attempt_reason(outcome: ProviderOutcomeEnvelope) -> str | None:
    return outcome.error or outcome.fallback_reason


def provider_status_to_attempt_status(
    status_bucket: ProviderStatusBucket,
) -> Literal["returned_results", "returned_no_results", "failed", "skipped"]:
    if status_bucket == "success":
        return "returned_results"
    if status_bucket == "empty":
        return "returned_no_results"
    if status_bucket in {"suppressed", "skipped"}:
        return "skipped"
    return "failed"


def parse_retry_after_seconds(text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None
