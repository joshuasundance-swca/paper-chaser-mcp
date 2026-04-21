"""Runtime / provider-diagnostics helpers for MCP dispatch.

Extracted from :mod:`paper_chaser_mcp.dispatch._core` as part of the Phase 5
modularization. These helpers compose the provider-diagnostics snapshot that
``dispatch_tool`` returns for ``get_runtime_status`` and other tools that need
a consistent picture of configured providers, runtime health, and
smart-provider selection.

The module owns the following contract surfaces:

* ``_runtime_provider_order`` — canonical ordering of providers across the
  broker order, smart slate, and regulatory tail.
* ``_smart_runtime_provider_state`` — reads the smart-provider bundle safely
  and provides a deterministic default when the runtime is absent.
* ``_metadata_value_is_depleted`` / ``_provider_row_quota_limited`` — quota
  heuristics used to annotate provider rows.
* ``_annotate_runtime_provider_row`` — writes the
  ``runtimeAvailability``/``runtimeHealth``/``runtimeStateReason`` triple onto
  a provider row in place.
* ``_build_provider_diagnostics_snapshot`` — the biggest helper; builds the
  full runtime snapshot + ``RuntimeSummary`` model.

Behavior is preserved verbatim; only the module boundary moves. The
``GUIDED_POLICY_NAME`` constant continues to live in ``_core`` because other
guided submodules already import it from there via the
``from .._core import GUIDED_POLICY_NAME`` late-binding pattern. We read it
through a function-local import inside
``_build_provider_diagnostics_snapshot`` to keep ``_core``/``runtime`` free of
top-level circular imports.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any

from ..models import RuntimeSummary
from ..models.common import RuntimeHealthStatus
from ..models.tools import SearchProvider


def _runtime_provider_order(
    *,
    provider_order: list[SearchProvider] | None,
    smart_provider_order: list[str],
) -> list[str]:
    return [
        *(provider_order or []),
        "openalex",
        "scholarapi",
        "crossref",
        "unpaywall",
        *smart_provider_order,
        "ecos",
        "federal_register",
        "govinfo",
    ]


def _smart_runtime_provider_state(
    agentic_runtime: Any,
) -> tuple[
    dict[str, bool],
    list[str],
    str | None,
    str | None,
    bool,
]:
    smart_provider_enabled = {
        "openai": False,
        "azure-openai": False,
        "anthropic": False,
        "nvidia": False,
        "google": False,
        "mistral": False,
        "huggingface": False,
        "openrouter": False,
    }
    smart_provider_order = [
        "openai",
        "azure-openai",
        "anthropic",
        "nvidia",
        "google",
        "mistral",
        "huggingface",
        "openrouter",
    ]
    configured_smart_provider: str | None = None
    active_smart_provider: str | None = None
    provider_selection_settled = True
    if agentic_runtime is not None and hasattr(agentic_runtime, "smart_provider_diagnostics"):
        smart_provider_enabled, smart_provider_order = agentic_runtime.smart_provider_diagnostics()
    provider_bundle = getattr(agentic_runtime, "_provider_bundle", None)
    if provider_bundle is not None and hasattr(provider_bundle, "selection_metadata"):
        selection = provider_bundle.selection_metadata()
        configured_value = selection.get("configuredSmartProvider")
        active_value = selection.get("activeSmartProvider")
        configured_smart_provider = str(configured_value) if configured_value else None
        active_smart_provider = str(active_value) if active_value else None
        settled_reader = getattr(provider_bundle, "provider_selection_settled", None)
        if callable(settled_reader):
            try:
                provider_selection_settled = bool(settled_reader())
            except Exception:
                provider_selection_settled = True
        elif hasattr(provider_bundle, "_provider_selection_settled"):
            provider_selection_settled = bool(getattr(provider_bundle, "_provider_selection_settled"))
    return (
        smart_provider_enabled,
        smart_provider_order,
        configured_smart_provider,
        active_smart_provider,
        provider_selection_settled,
    )


_RUNTIME_FAILURE_PROVIDER_STATUSES: frozenset[str] = frozenset(
    {"rate_limited", "quota_exhausted", "auth_error", "provider_error"}
)
_QUOTA_METADATA_CAPACITY_KEYS: tuple[str, ...] = (
    "searches_left",
    "total_searches_left",
    "this_hour_searches_left",
    "requestsRemaining",
    "remainingRequests",
    "creditsRemaining",
    "remainingCredits",
)


def _metadata_value_is_depleted(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return value <= 0
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return False
        try:
            return float(stripped) <= 0
        except ValueError:
            return False
    return False


def _provider_row_quota_limited(provider_row: dict[str, Any]) -> bool:
    if provider_row.get("lastOutcome") == "quota_exhausted":
        return True
    quota_metadata = provider_row.get("lastQuotaMetadata")
    if not isinstance(quota_metadata, dict):
        return False
    return any(_metadata_value_is_depleted(quota_metadata.get(key)) for key in _QUOTA_METADATA_CAPACITY_KEYS)


def _annotate_runtime_provider_row(provider_row: dict[str, Any]) -> tuple[str, str]:
    enabled = provider_row.get("enabled") is True
    suppressed = provider_row.get("suppressed") is True
    last_outcome = provider_row.get("lastOutcome")
    consecutive_failures = provider_row.get("consecutiveFailures")
    quota_limited = _provider_row_quota_limited(provider_row)

    if not enabled:
        availability = "disabled"
        health = "ok"
        reason = "Provider is not currently enabled in the effective runtime."
    elif suppressed:
        availability = "suppressed"
        health = "quota_limited" if quota_limited else "ok"
        reason = "Provider is temporarily suppressed after recent failures."
    else:
        availability = "active"
        if quota_limited:
            health = "quota_limited"
            reason = "Provider reported quota exhaustion or zero remaining capacity."
        elif last_outcome in _RUNTIME_FAILURE_PROVIDER_STATUSES or (
            isinstance(consecutive_failures, int) and consecutive_failures > 0
        ):
            health = "degraded"
            if isinstance(last_outcome, str) and last_outcome:
                reason = f"Recent runtime outcome was {last_outcome}."
            else:
                reason = "Provider has recent runtime failures."
        else:
            health = "ok"
            reason = "Provider is currently eligible for calls."
    provider_row["runtimeAvailability"] = availability
    provider_row["runtimeHealth"] = health
    provider_row["runtimeStateReason"] = reason
    return availability, health


def _build_provider_diagnostics_snapshot(
    *,
    include_recent_outcomes: bool,
    provider_order: list[SearchProvider] | None,
    provider_registry: Any,
    agentic_runtime: Any,
    transport_mode: str,
    tool_profile: str,
    hide_disabled_tools: bool,
    session_ttl_seconds: int | None,
    embeddings_enabled: bool | None,
    guided_research_latency_profile: str,
    guided_follow_up_latency_profile: str,
    guided_allow_paid_providers: bool,
    guided_escalation_enabled: bool,
    guided_escalation_max_passes: int,
    guided_escalation_allow_paid_providers: bool,
    enable_core: bool,
    enable_semantic_scholar: bool,
    enable_openalex: bool,
    enable_arxiv: bool,
    enable_serpapi: bool,
    enable_scholarapi: bool,
    enable_crossref: bool,
    enable_unpaywall: bool,
    enable_ecos: bool,
    enable_federal_register: bool,
    enable_govinfo_cfr: bool,
    ecos_client: Any,
    serpapi_client: Any,
    scholarapi_client: Any,
) -> dict[str, Any]:
    # Deferred to avoid a circular import: ``_core`` imports this module at
    # the bottom of its initialization.
    from ._core import GUIDED_POLICY_NAME

    try:
        package_version_value = package_version("paper-chaser-mcp")
    except PackageNotFoundError:
        package_version_value = None

    (
        smart_provider_enabled,
        smart_provider_order,
        configured_smart_provider,
        active_smart_provider,
        provider_selection_settled,
    ) = _smart_runtime_provider_state(agentic_runtime)
    enabled_state = {
        "semantic_scholar": enable_semantic_scholar,
        "openalex": enable_openalex,
        "core": enable_core,
        "arxiv": enable_arxiv,
        "serpapi_google_scholar": enable_serpapi,
        "scholarapi": enable_scholarapi,
        "crossref": enable_crossref,
        "unpaywall": enable_unpaywall,
        **smart_provider_enabled,
        "ecos": enable_ecos,
        "federal_register": enable_federal_register,
        "govinfo": enable_govinfo_cfr,
    }
    provider_list_order = _runtime_provider_order(
        provider_order=provider_order,
        smart_provider_order=smart_provider_order,
    )
    provider_order_effective = [str(provider) for provider in (provider_order or [])]
    runtime_warnings: list[str] = []
    structured_warnings: list[dict[str, Any]] = []

    def _warn(
        message: str,
        *,
        code: str,
        severity: str = "warning",
        subject: str | None = None,
    ) -> None:
        runtime_warnings.append(message)
        structured_warnings.append(
            {
                "code": code,
                "severity": severity,
                "message": message,
                "subject": subject,
            }
        )

    if not enable_serpapi and serpapi_client is not None:
        _warn(
            "SerpApi client state is present but PAPER_CHASER_ENABLE_SERPAPI is "
            "false, so paid recall recovery is disabled.",
            code="provider_disabled",
            severity="info",
            subject="serpapi_google_scholar",
        )
    if not enable_scholarapi and scholarapi_client is not None:
        _warn(
            "ScholarAPI client state is present but PAPER_CHASER_ENABLE_"
            "SCHOLARAPI is false, so ScholarAPI discovery and full-text paths "
            "are inactive.",
            code="provider_disabled",
            severity="info",
            subject="scholarapi",
        )
    if hide_disabled_tools:
        _warn(
            "Disabled tools are hidden from list_tools output, which can make capability gaps harder to diagnose.",
            code="tools_hidden",
            severity="info",
        )
    if transport_mode == "stdio":
        _warn(
            "The current runtime is stdio, so HTTP deployment settings do not affect this invocation path.",
            code="stdio_transport",
            severity="info",
        )
    if getattr(ecos_client, "verify_tls", True) is False:
        _warn(
            "ECOS TLS verification is disabled. This should only be a temporary troubleshooting state.",
            code="ecos_tls_disabled",
            severity="warning",
            subject="ecos",
        )
    if tool_profile == "guided" and hide_disabled_tools:
        _warn(
            (
                "Guided profile is active while expert tools are hidden, "
                "so escalation paths are intentionally unavailable."
            ),
            code="guided_hides_expert",
            severity="info",
        )
    if configured_smart_provider == "huggingface":
        _warn(
            "Hugging Face is configured as a chat-only smart provider in this repo; embeddings stay disabled.",
            code="chat_only_smart_provider",
            severity="info",
            subject="huggingface",
        )
    if configured_smart_provider == "openrouter":
        _warn(
            "OpenRouter is configured as a chat-only smart provider in this repo; embeddings stay disabled.",
            code="chat_only_smart_provider",
            severity="info",
            subject="openrouter",
        )
    snapshot: dict[str, Any] | None = None
    if provider_registry is None:
        configured_provider_set = sorted([provider for provider, enabled in enabled_state.items() if enabled])
        active_provider_set = list(configured_provider_set)
        disabled_provider_set = sorted([provider for provider, enabled in enabled_state.items() if not enabled])
        suppressed_provider_set: list[str] = []
        degraded_provider_set: list[str] = []
        quota_limited_provider_set: list[str] = []
    else:
        snapshot = provider_registry.snapshot(
            enabled=enabled_state,
            provider_order=provider_list_order,
        )
        configured_provider_set = []
        active_provider_set = []
        disabled_provider_set = []
        suppressed_provider_set = []
        degraded_provider_set = []
        quota_limited_provider_set = []
        for provider_payload in snapshot.get("providers", []):
            if not isinstance(provider_payload, dict):
                continue
            provider_name = provider_payload.get("provider")
            if not isinstance(provider_name, str) or not provider_name:
                continue
            availability, health = _annotate_runtime_provider_row(provider_payload)
            if provider_payload.get("enabled") is True:
                configured_provider_set.append(provider_name)
            if availability == "active":
                active_provider_set.append(provider_name)
            elif availability == "disabled":
                disabled_provider_set.append(provider_name)
            elif availability == "suppressed":
                suppressed_provider_set.append(provider_name)
            if health == "degraded" and availability == "active":
                degraded_provider_set.append(provider_name)
            if health == "quota_limited":
                quota_limited_provider_set.append(provider_name)
            if not include_recent_outcomes:
                provider_payload["recentOutcomes"] = []
        configured_provider_set.sort()
        active_provider_set.sort()
        disabled_provider_set.sort()
        suppressed_provider_set.sort()
        degraded_provider_set.sort()
        quota_limited_provider_set.sort()
        if suppressed_provider_set:
            _warn(
                "Providers currently suppressed at runtime: "
                + ", ".join(suppressed_provider_set)
                + ". Check provider rows for suppressedUntil and lastOutcome.",
                code="provider_suppressed",
                severity="warning",
            )
        if quota_limited_provider_set:
            _warn(
                "Providers currently quota-limited: "
                + ", ".join(quota_limited_provider_set)
                + ". Check provider rows for lastQuotaMetadata and suppressedUntil.",
                code="provider_quota_limited",
                severity="warning",
            )
        if degraded_provider_set:
            _warn(
                "Providers with recent non-suppressed runtime degradation: "
                + ", ".join(degraded_provider_set)
                + ". Check provider rows for lastOutcome and consecutiveFailures.",
                code="provider_degraded",
                severity="warning",
            )
    enabled_raw_providers = [provider for provider in (provider_order or []) if provider in active_provider_set]
    if len(enabled_raw_providers) <= 1:
        _warn(
            "The effective broker order is very narrow, so no-result responses "
            "may reflect limited provider coverage rather than absence of evidence.",
            code="narrow_provider_order",
            severity="warning",
        )
    provisional_smart_provider = bool(
        configured_smart_provider
        and configured_smart_provider != "deterministic"
        and smart_provider_enabled.get(configured_smart_provider, False)
        and active_smart_provider == "deterministic"
        and not provider_selection_settled
    )
    if provisional_smart_provider:
        active_smart_provider = configured_smart_provider
        _warn(
            f"Configured smart provider '{configured_smart_provider}' has not completed a smart call yet, "
            "so activeSmartProvider is provisional until initialization settles.",
            code="smart_provider_unsettled",
            severity="info",
            subject=configured_smart_provider,
        )
    fallback_active = bool(
        configured_smart_provider
        and configured_smart_provider != "deterministic"
        and active_smart_provider == "deterministic"
    )
    fallback_reason: str | None = None
    if fallback_active:
        fallback_reason = (
            f"Configured smart provider '{configured_smart_provider}' is unavailable, "
            "so deterministic fallback is active."
        )
        _warn(
            fallback_reason,
            code="smart_provider_fallback",
            severity="critical",
            subject=configured_smart_provider,
        )

    _smart_provider_names = set(smart_provider_enabled.keys())
    _non_smart_disabled = [p for p in disabled_provider_set if p not in _smart_provider_names]
    if agentic_runtime is None and configured_smart_provider and configured_smart_provider != "deterministic":
        health_status: RuntimeHealthStatus = "critical"
    elif fallback_active:
        health_status = "fallback_active"
    elif provisional_smart_provider:
        health_status = "degraded"
    elif _non_smart_disabled or suppressed_provider_set or degraded_provider_set or quota_limited_provider_set:
        health_status = "degraded"
    else:
        health_status = "nominal"

    runtime_summary = RuntimeSummary(
        effectiveProfile=tool_profile,
        transportMode=transport_mode,
        smartLayerEnabled=agentic_runtime is not None,
        configuredProviderSet=configured_provider_set,
        activeProviderSet=active_provider_set,
        disabledProviderSet=disabled_provider_set,
        suppressedProviderSet=suppressed_provider_set,
        degradedProviderSet=degraded_provider_set,
        quotaLimitedProviderSet=quota_limited_provider_set,
        configuredSmartProvider=configured_smart_provider,
        activeSmartProvider=active_smart_provider,
        providerOrderEffective=provider_order_effective,
        toolsHidden=hide_disabled_tools,
        sessionTtlSeconds=session_ttl_seconds,
        embeddingsEnabled=embeddings_enabled,
        guidedPolicy=GUIDED_POLICY_NAME,
        guidedResearchLatencyProfile=guided_research_latency_profile,
        guidedFollowUpLatencyProfile=guided_follow_up_latency_profile,
        guidedAllowPaidProviders=guided_allow_paid_providers,
        guidedEscalationEnabled=guided_escalation_enabled,
        guidedEscalationMaxPasses=guided_escalation_max_passes,
        guidedEscalationAllowPaidProviders=guided_escalation_allow_paid_providers,
        version=package_version_value,
        warnings=runtime_warnings,
        healthStatus=health_status,
        fallbackActive=fallback_active,
        fallbackReason=fallback_reason,
        structuredWarnings=structured_warnings,
    )
    if snapshot is None:
        return {
            "generatedAt": None,
            "providerOrder": provider_list_order,
            "providers": [],
            "runtimeSummary": runtime_summary.model_dump(by_alias=True, exclude_none=True),
        }
    snapshot["runtimeSummary"] = runtime_summary.model_dump(by_alias=True, exclude_none=True)
    return snapshot
