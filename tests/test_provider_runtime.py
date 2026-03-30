import asyncio
import logging
import time
from typing import Any

import pytest

from paper_chaser_mcp.clients.scholarapi import ScholarApiQuotaError
from paper_chaser_mcp.clients.serpapi import (
    SerpApiKeyMissingError,
    SerpApiQuotaError,
    SerpApiUpstreamError,
)
from paper_chaser_mcp.provider_runtime import (
    ProviderBudgetState,
    ProviderDiagnosticsRegistry,
    ProviderOutcomeEnvelope,
    ProviderPolicy,
    _classify_exception,
    _default_fallback_reason,
    _empty_payload,
    _format_exception,
    _log_request_scoped_provider_event,
    _provider_semaphore_key,
    _quota_metadata_from_payload,
    _retry_delay_seconds,
    _shorten_runtime_log_text,
    execute_provider_call,
    execute_provider_call_sync,
    parse_retry_after_seconds,
    policy_for_provider,
    provider_attempt_reason,
    provider_is_paywalled,
    provider_status_to_attempt_status,
)


@pytest.mark.asyncio
async def test_provider_budget_state_and_registry_snapshot() -> None:
    assert ProviderBudgetState.from_mapping(None) is None
    assert ProviderBudgetState.from_mapping({"allow_paid_providers": True}) is None

    paid_budget = ProviderBudgetState.from_mapping(
        {
            "allow_paid_providers": False,
            "max_total_calls": 3,
        }
    )
    assert paid_budget is not None
    assert (
        await paid_budget.reserve("openai", paywalled=True)
        == "Skipped because providerBudget disallows paid providers."
    )

    capped_budget = ProviderBudgetState.from_mapping(
        {
            "max_total_calls": 1,
            "max_openalex_calls": 1,
            "allow_paid_providers": True,
        }
    )
    assert capped_budget is not None
    assert await capped_budget.reserve("openalex", paywalled=False) is None
    assert (
        await capped_budget.reserve("openalex", paywalled=False)
        == "Skipped because providerBudget exhausted maxTotalCalls."
    )

    per_provider_budget = ProviderBudgetState.from_mapping(
        {
            "max_openalex_calls": 1,
        }
    )
    assert per_provider_budget is not None
    assert await per_provider_budget.reserve("openalex", paywalled=False) is None
    assert (
        await per_provider_budget.reserve("openalex", paywalled=False)
        == "Skipped because providerBudget exhausted the per-provider limit for "
        "openalex."
    )
    assert per_provider_budget.to_dict()["appliedCounts"] == {
        "totalCalls": 1,
        "openalex": 1,
    }

    registry = ProviderDiagnosticsRegistry(recent_outcomes_per_provider=2)
    registry.record(
        ProviderOutcomeEnvelope(
            provider="openalex",
            endpoint="works.search",
            status_bucket="success",
            latency_ms=12,
            quota_metadata={"requestId": "oa-req-1"},
            provider_request_id="oa-req-1",
        ),
        policy=ProviderPolicy(),
    )
    registry.record(
        ProviderOutcomeEnvelope(
            provider="openalex",
            endpoint="works.search",
            status_bucket="provider_error",
            error="RuntimeError: boom",
        ),
        policy=ProviderPolicy(failure_threshold=1, suppression_seconds=60.0),
    )
    registry.record(
        ProviderOutcomeEnvelope(
            provider="serpapi_google_scholar",
            endpoint="search",
            status_bucket="quota_exhausted",
            error="SerpApiQuotaError: no credits",
        ),
        policy=ProviderPolicy(suppression_seconds=10.0),
    )

    snapshot = registry.snapshot(
        enabled={"openalex": True},
        provider_order=["openalex"],
    )

    openalex_snapshot = snapshot["providers"][0]
    serpapi_snapshot = next(item for item in snapshot["providers"] if item["provider"] == "serpapi_google_scholar")

    assert snapshot["providerOrder"][0] == "openalex"
    assert openalex_snapshot["enabled"] is True
    assert openalex_snapshot["paywalled"] is False
    assert openalex_snapshot["consecutiveFailures"] == 1
    assert openalex_snapshot["lastQuotaMetadata"] == {"requestId": "oa-req-1"}
    assert openalex_snapshot["lastProviderRequestId"] == "oa-req-1"
    assert openalex_snapshot["statusCounts"] == {"success": 1, "provider_error": 1}
    assert openalex_snapshot["suppressed"] is True
    assert serpapi_snapshot["paywalled"] is True
    assert serpapi_snapshot["suppressed"] is True
    assert serpapi_snapshot["suppressedUntil"] is not None


def test_provider_runtime_helper_functions(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert policy_for_provider("openalex").concurrency_limit == 2
    assert policy_for_provider("azure-openai").paywalled is True
    assert policy_for_provider("anthropic").paywalled is True
    assert policy_for_provider("google").paywalled is True
    assert policy_for_provider("unknown").concurrency_limit == 2
    assert provider_is_paywalled("scholarapi") is True
    assert provider_is_paywalled("azure-openai") is True
    assert provider_is_paywalled("anthropic") is True
    assert provider_is_paywalled("google") is True
    assert provider_is_paywalled("openalex") is False
    assert _default_fallback_reason("auth_error") == "Provider authentication failed."
    assert _default_fallback_reason("success") is None
    assert _provider_semaphore_key("openai", "responses.parse:planner") == ("openai:responses.parse:planner")
    assert _provider_semaphore_key("openalex", "works.search") == "openalex"
    assert _classify_exception(SerpApiKeyMissingError("missing")) == "auth_error"
    assert _classify_exception(SerpApiQuotaError("quota")) == "quota_exhausted"
    assert _classify_exception(SerpApiUpstreamError("upstream")) == "provider_error"
    assert _classify_exception(ScholarApiQuotaError("quota")) == "quota_exhausted"
    assert _classify_exception(RuntimeError("429 too many requests")) == "rate_limited"
    assert _classify_exception(RuntimeError("quota exceeded")) == "quota_exhausted"
    assert _classify_exception(RuntimeError("forbidden")) == "auth_error"
    assert _classify_exception(RuntimeError("timeout while calling upstream")) == ("provider_error")
    assert _classify_exception(RuntimeError("HTTP 503")) == "provider_error"
    assert _classify_exception(RuntimeError("network transport failure")) == ("provider_error")

    monkeypatch.setattr("paper_chaser_mcp.provider_runtime.random.uniform", lambda a, b: 0.1)
    assert _retry_delay_seconds(2, policy=ProviderPolicy(base_delay_seconds=0.5)) == 2.1

    assert _empty_payload(None) is True
    assert _empty_payload({"data": []}) is True
    assert _empty_payload({"recommendedPapers": []}) is True
    assert _empty_payload({"matches": []}) is True
    assert _empty_payload({"results": []}) is True
    assert _empty_payload([]) is True
    assert _empty_payload({"data": [{"id": 1}]}) is False

    metadata = _quota_metadata_from_payload(
        {
            "requestId": "sch-req-1",
            "requestCost": "3",
            "pagination": {"hasMore": True, "nextCursor": "next-sch"},
            "search_metadata": {"id": "search-1", "status": "Success"},
            "account": {
                "plan": "paid",
                "searches_left": 9,
                "total_searches_left": 99,
                "this_hour_searches_left": 3,
            },
        }
    )
    assert metadata == {
        "requestId": "sch-req-1",
        "requestCost": "3",
        "hasMore": True,
        "nextCursor": "next-sch",
        "searchId": "search-1",
        "searchStatus": "Success",
        "plan": "paid",
        "searches_left": 9,
        "total_searches_left": 99,
        "this_hour_searches_left": 3,
    }
    assert _quota_metadata_from_payload("not-a-dict") == {}
    assert _shorten_runtime_log_text("  alpha   beta  ", limit=20) == "alpha beta"
    assert _shorten_runtime_log_text("alpha beta gamma", limit=10) == "alpha b..."
    assert _format_exception(RuntimeError("boom")) == "RuntimeError: boom"
    assert _format_exception(RuntimeError()) == "RuntimeError"

    caplog.set_level(logging.INFO, logger="paper-chaser-mcp")
    _log_request_scoped_provider_event(
        request_id=None,
        provider="openalex",
        endpoint="works.search",
        status="started",
    )
    _log_request_scoped_provider_event(
        request_id="req-1",
        provider="openalex",
        endpoint="works.search",
        status="failed",
        latency_ms=15,
        retries=1,
        reason="This reason is deliberately long enough to be shortened in logs.",
    )
    assert any("provider-call[req-1]" in record.getMessage() for record in caplog.records)

    outcome = ProviderOutcomeEnvelope(
        provider="openalex",
        endpoint="works.search",
        status_bucket="provider_error",
        error="RuntimeError: boom",
        fallback_reason="Provider returned an upstream error.",
    )
    assert provider_attempt_reason(outcome) == "RuntimeError: boom"
    assert provider_status_to_attempt_status("success") == "returned_results"
    assert provider_status_to_attempt_status("empty") == "returned_no_results"
    assert provider_status_to_attempt_status("skipped") == "skipped"
    assert provider_status_to_attempt_status("provider_error") == "failed"
    assert parse_retry_after_seconds("retry after 2.5 seconds") == 2.5
    assert parse_retry_after_seconds("no digits here") is None

    class _FakeMatch:
        def group(self, index: int) -> str:
            assert index == 1
            return "not-a-number"

    monkeypatch.setattr(
        "paper_chaser_mcp.provider_runtime.re.search",
        lambda pattern, text: _FakeMatch(),
    )
    assert parse_retry_after_seconds("retry after sometime") is None


def test_provider_diagnostics_registry_clears_expired_suppression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ProviderDiagnosticsRegistry()
    registry._suppressed_until["openalex"] = 10.0

    monkeypatch.setattr("paper_chaser_mcp.provider_runtime.time.time", lambda: 11.0)

    assert registry.suppressed_until("openalex") is None
    assert "openalex" not in registry._suppressed_until


@pytest.mark.asyncio
async def test_execute_provider_call_async_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("paper_chaser_mcp.provider_runtime.asyncio.sleep", _no_sleep)
    monkeypatch.setattr("paper_chaser_mcp.provider_runtime.random.uniform", lambda a, b: 0.0)

    registry = ProviderDiagnosticsRegistry()
    registry._suppressed_until["core"] = time.time() + 60.0
    suppressed = await execute_provider_call(
        provider="core",
        endpoint="search",
        operation=lambda: asyncio.sleep(0),
        registry=registry,
    )
    assert suppressed.outcome.status_bucket == "suppressed"

    skipped = await execute_provider_call(
        provider="openai",
        endpoint="responses.parse:planner",
        operation=lambda: asyncio.sleep(0),
        budget=ProviderBudgetState(allow_paid_providers=False),
    )
    assert skipped.outcome.status_bucket == "skipped"

    attempts = {"count": 0}

    async def _retry_then_succeed() -> dict[str, Any]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("429 too many requests")
        return {"data": [{"paperId": "paper-1"}]}

    outcomes: list[dict[str, Any]] = []
    retried = await execute_provider_call(
        provider="openalex",
        endpoint="works.search",
        operation=_retry_then_succeed,
        registry=ProviderDiagnosticsRegistry(),
        policy=ProviderPolicy(max_attempts=2, base_delay_seconds=0.0, max_delay_seconds=0.0),
        request_outcomes=outcomes,
        request_id="req-2",
    )
    assert retried.payload == {"data": [{"paperId": "paper-1"}]}
    assert retried.outcome.status_bucket == "success"
    assert retried.outcome.retries == 1
    assert attempts["count"] == 2
    assert outcomes[-1]["statusBucket"] == "success"

    async def _return_empty() -> dict[str, Any]:
        return {"data": []}

    empty = await execute_provider_call(
        provider="openalex",
        endpoint="works.search",
        operation=_return_empty,
        registry=ProviderDiagnosticsRegistry(),
    )
    assert empty.outcome.status_bucket == "empty"

    async def _raise_key_missing() -> Any:
        raise SerpApiKeyMissingError("missing key")

    with pytest.raises(SerpApiKeyMissingError):
        await execute_provider_call(
            provider="serpapi_google_scholar",
            endpoint="search",
            operation=_raise_key_missing,
            propagate_exceptions=(SerpApiKeyMissingError,),
        )


@pytest.mark.asyncio
async def test_execute_provider_call_records_budget_skips_and_waits_for_slots(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="paper-chaser-mcp")

    registry = ProviderDiagnosticsRegistry()
    outcomes: list[dict[str, Any]] = []
    skipped = await execute_provider_call(
        provider="openai",
        endpoint="responses.parse:planner",
        operation=lambda: asyncio.sleep(0),
        registry=registry,
        budget=ProviderBudgetState(allow_paid_providers=False),
        request_outcomes=outcomes,
        request_id="budget-1",
    )

    assert skipped.outcome.status_bucket == "skipped"
    assert skipped.outcome.paywalled is True
    assert outcomes[-1]["statusBucket"] == "skipped"
    assert any(
        item["lastOutcome"] == "skipped" for item in registry.snapshot()["providers"] if item["provider"] == "openai"
    )

    semaphore = registry.semaphore("openai:responses.parse:planner", 1)
    await semaphore.acquire()

    async def _operation() -> dict[str, Any]:
        return {"data": [{"paperId": "paper-1"}]}

    task = asyncio.create_task(
        execute_provider_call(
            provider="openai",
            endpoint="responses.parse:planner",
            operation=_operation,
            registry=registry,
            policy=ProviderPolicy(concurrency_limit=1, max_attempts=1),
            request_id="wait-1",
        )
    )
    await asyncio.sleep(0)
    semaphore.release()
    waited = await task

    assert waited.outcome.status_bucket == "success"
    assert waited.outcome.paywalled is True
    assert any("waiting_for_slot" in record.getMessage() for record in caplog.records)


def test_execute_provider_call_sync_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("paper_chaser_mcp.provider_runtime.time.sleep", lambda seconds: None)
    monkeypatch.setattr("paper_chaser_mcp.provider_runtime.random.uniform", lambda a, b: 0.0)

    registry = ProviderDiagnosticsRegistry()
    registry._suppressed_until["crossref"] = time.time() + 60.0
    suppressed = execute_provider_call_sync(
        provider="crossref",
        endpoint="works",
        operation=lambda: {"data": [{"doi": "10.1234/test"}]},
        registry=registry,
    )
    assert suppressed.outcome.status_bucket == "suppressed"

    attempts = {"count": 0}

    def _retry_then_succeed() -> dict[str, Any]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("500 upstream failure")
        return {"data": [{"doi": "10.1234/test"}]}

    success = execute_provider_call_sync(
        provider="crossref",
        endpoint="works",
        operation=_retry_then_succeed,
        registry=ProviderDiagnosticsRegistry(),
        policy=ProviderPolicy(max_attempts=2, base_delay_seconds=0.0, max_delay_seconds=0.0),
    )
    assert success.outcome.status_bucket == "success"
    assert success.outcome.retries == 1
    assert attempts["count"] == 2

    empty = execute_provider_call_sync(
        provider="crossref",
        endpoint="works",
        operation=lambda: {"data": []},
        registry=ProviderDiagnosticsRegistry(),
    )
    assert empty.outcome.status_bucket == "empty"

    failed = execute_provider_call_sync(
        provider="serpapi_google_scholar",
        endpoint="search",
        operation=lambda: (_ for _ in ()).throw(SerpApiQuotaError("out of credits")),
        registry=ProviderDiagnosticsRegistry(),
    )
    assert failed.payload is None
    assert failed.outcome.status_bucket == "quota_exhausted"
