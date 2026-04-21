"""Dispatch helpers for MCP tool routing."""

import logging
import re
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any, Callable, cast

from ..agentic.planner import (
    detect_regulatory_intent,
)
from ..citation_repair import (  # noqa: F401 — ``resolve_citation`` re-exported for tests that monkeypatch this name
    looks_like_citation_query,
    looks_like_paper_identifier,
    parse_citation,
    resolve_citation,
)
from ..clients.scholarapi import (
    ScholarApiError,
    ScholarApiKeyMissingError,
    ScholarApiQuotaError,
    ScholarApiUpstreamError,
)
from ..compat import augment_tool_result, build_clarification
from ..enrichment import (
    PaperEnrichmentService,
    attach_enrichments_to_paper_payload,
    hydrate_paper_for_enrichment,
)
from ..guided_semantic import (
    classify_answerability,
)
from ..models import TOOL_INPUT_MODELS, RuntimeSummary, dump_jsonable
from ..models.common import (
    RuntimeHealthStatus,
)
from ..models.tools import (
    FollowUpResearchArgs,
    GetRuntimeStatusArgs,
    InspectSourceArgs,
    ResearchArgs,
    ResolveReferenceArgs,
    SearchProvider,
)
from ..provider_runtime import ProviderOutcomeEnvelope, ProviderStatusBucket, policy_for_provider
from ..utils.cursor import (
    PROVIDER,
    SUPPORTED_VERSIONS,
    compute_context_hash,
    cursor_from_token,
    decode_bulk_cursor,
)
from .context import DispatchContext, build_dispatch_context
from .normalization import (  # noqa: F401 — re-exported for dispatch package namespace
    _guided_normalize_citation_surface,
    _guided_normalize_source_locator,
    _guided_normalize_whitespace,
    _guided_normalize_year_hint,
    _guided_strip_research_prefix,
)
from .paging import (  # noqa: F401 — ``_encode_next_cursor`` re-exported for backward-compat
    _cursor_to_offset,
    _encode_next_cursor,
)
from .relevance import (  # noqa: F401 — re-exported for dispatch package namespace
    _facet_match,
    _paper_topical_relevance,
    _tokenize_relevance_text,
    _topical_relevance_from_signals,
    compute_topical_relevance,
)
from .snippet_fallback import (  # noqa: F401 — re-exported for dispatch package namespace
    _maybe_fallback_snippet_search,
    _snippet_fallback_query,
    _snippet_fallback_results,
)

ToolArgBuilder = Callable[[dict[str, Any]], dict[str, Any]]
CURSOR_REUSE_HINT = (
    "Pass pagination.nextCursor back exactly as returned. Do not derive, edit, "
    "or fabricate cursors, and do not reuse them across a different tool or "
    "different query context."
)
SCHOLARAPI_LIST_RETRIEVAL_NOTE = (
    "ORDERING: list_papers_scholarapi follows ScholarAPI /list semantics and is sorted by indexed_at, "
    "not by topical relevance. Use it for monitoring or date-window scans, not ranked discovery. "
    "For topical search use search_papers_scholarapi, and pass pagination.nextCursor back exactly as "
    "returned to continue the same stream."
)

logger = logging.getLogger(__name__)

_GUIDED_REFERENCE_UNCERTAINTY_MARKERS: tuple[str, ...] = (
    "maybe",
    "perhaps",
    "possibly",
    "probably",
    "roughly",
    "approximately",
    "approx",
    "around",
    "something",
)
_GUIDED_REFERENCE_GENERIC_CANDIDATE_WORDS = {
    "article",
    "document",
    "paper",
    "policy",
    "report",
    "something",
    "study",
}


def _provider_error_text(exc: Exception) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _scholarapi_status_bucket(exc: Exception) -> ProviderStatusBucket:
    if isinstance(exc, ScholarApiKeyMissingError):
        return "auth_error"
    if isinstance(exc, ScholarApiQuotaError):
        return "quota_exhausted"
    if isinstance(exc, ScholarApiUpstreamError):
        return "provider_error"
    if isinstance(exc, ScholarApiError):
        return "provider_error"
    return "provider_error"


def _scholarapi_fallback_reason(status_bucket: ProviderStatusBucket) -> str | None:
    reasons = {
        "auth_error": "Provider authentication failed.",
        "quota_exhausted": "Provider quota was exhausted.",
        "provider_error": "Provider returned an upstream error.",
    }
    return reasons.get(status_bucket)


def _scholarapi_quota_metadata(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    metadata: dict[str, Any] = {}
    request_id = payload.get("requestId")
    if isinstance(request_id, str) and request_id.strip():
        metadata["requestId"] = request_id.strip()
    request_cost = payload.get("requestCost")
    if request_cost is not None:
        metadata["requestCost"] = request_cost
    pagination = payload.get("pagination")
    if isinstance(pagination, dict):
        if "hasMore" in pagination:
            metadata["hasMore"] = bool(pagination.get("hasMore"))
        next_cursor = pagination.get("nextCursor")
        if next_cursor is not None:
            metadata["nextCursor"] = next_cursor
    return metadata


def _scholarapi_payload_is_empty(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return payload is None
    if isinstance(payload.get("data"), list):
        return len(payload.get("data") or []) == 0
    if isinstance(payload.get("results"), list):
        return len(payload.get("results") or []) == 0
    return False


async def _call_explicit_scholarapi_tool(
    *,
    operation: Callable[[], Any],
    endpoint: str,
    provider_registry: Any,
    request_id: str,
) -> Any:
    started = time.perf_counter()
    try:
        payload = await operation()
    except Exception as exc:
        if provider_registry is not None:
            status_bucket = _scholarapi_status_bucket(exc)
            outcome = ProviderOutcomeEnvelope(
                provider="scholarapi",
                endpoint=endpoint,
                status_bucket=status_bucket,
                latency_ms=int((time.perf_counter() - started) * 1000),
                retries=0,
                fallback_reason=_scholarapi_fallback_reason(status_bucket),
                error=_provider_error_text(exc),
                paywalled=True,
                request_id=request_id,
            )
            provider_registry.record(outcome, policy=policy_for_provider("scholarapi"))
        raise

    if provider_registry is not None:
        quota_metadata = _scholarapi_quota_metadata(payload)
        outcome = ProviderOutcomeEnvelope(
            provider="scholarapi",
            endpoint=endpoint,
            status_bucket="empty" if _scholarapi_payload_is_empty(payload) else "success",
            latency_ms=int((time.perf_counter() - started) * 1000),
            retries=0,
            paywalled=True,
            quota_metadata=quota_metadata,
            provider_request_id=quota_metadata.get("requestId")
            if isinstance(quota_metadata.get("requestId"), str)
            else None,
            request_id=request_id,
        )
        provider_registry.record(outcome, policy=policy_for_provider("scholarapi"))
    return payload


def _cursor_to_bulk_token(
    cursor: str | None,
    *,
    tool: str,
    context_hash: str | None = None,
    expected_provider: str = PROVIDER,
) -> str | None:
    """Decode a structured bulk cursor to its provider token."""
    if cursor is None:
        return None
    try:
        state = decode_bulk_cursor(cursor)
    except ValueError:
        raise ValueError(
            f"Invalid pagination cursor {cursor!r}: cannot be decoded. "
            "code=INVALID_CURSOR. "
            f"{CURSOR_REUSE_HINT} Restart the request without a cursor."
        )
    if state.provider != expected_provider:
        raise ValueError(
            f"Invalid pagination cursor: cursor provider {state.provider!r} does not "
            f"match expected provider {expected_provider!r}. "
            "code=INVALID_CURSOR. "
            f"{CURSOR_REUSE_HINT} Restart the request without a cursor."
        )
    if state.version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"Invalid pagination cursor: cursor version {state.version} is not "
            f"supported (supported: {sorted(SUPPORTED_VERSIONS)}). "
            "code=INVALID_CURSOR. "
            f"{CURSOR_REUSE_HINT} Restart the request without a cursor."
        )
    if state.tool != tool:
        raise ValueError(
            f"Invalid pagination cursor: cursor was issued by tool {state.tool!r} "
            f"but is being used with tool {tool!r}. "
            "code=INVALID_CURSOR. "
            f"{CURSOR_REUSE_HINT} Restart the request without a cursor."
        )
    if context_hash is not None and state.context_hash is not None and state.context_hash != context_hash:
        raise ValueError(
            "Invalid pagination cursor: cursor was issued for a different query "
            "context and cannot be reused here. "
            "code=INVALID_CURSOR. "
            f"{CURSOR_REUSE_HINT} Restart the request without a cursor."
        )
    return state.token


def _encode_next_bulk_cursor(
    result: dict[str, Any],
    tool: str,
    context_hash: str | None = None,
    provider: str = PROVIDER,
) -> dict[str, Any]:
    """Wrap a raw bulk provider token in a structured server-issued cursor."""
    pagination = result.get("pagination")
    if not isinstance(pagination, dict):
        return result
    raw_cursor = pagination.get("nextCursor")
    if not isinstance(raw_cursor, str) or not raw_cursor:
        return result
    pagination["nextCursor"] = cursor_from_token(
        tool,
        raw_cursor,
        context_hash=context_hash,
        provider=provider,
    )
    return result


NON_SEARCH_TOOL_HANDLERS: dict[str, tuple[str, ToolArgBuilder]] = {
    "search_papers_match": (
        "search_papers_match",
        lambda a: {
            "query": a["query"],
            "fields": a.get("fields"),
        },
    ),
    "paper_autocomplete": (
        "paper_autocomplete",
        lambda a: {"query": a["query"]},
    ),
    "get_paper_details": (
        "get_paper_details",
        lambda a: {
            "paper_id": a["paper_id"],
            "fields": a.get("fields"),
        },
    ),
    "get_paper_citations": (
        "get_paper_citations",
        lambda a: {
            "paper_id": a["paper_id"],
            "limit": a.get("limit", 100),
            "fields": a.get("fields"),
            "offset": _cursor_to_offset(
                a.get("cursor"),
                "get_paper_citations",
                context_hash=compute_context_hash("get_paper_citations", a),
            ),
        },
    ),
    "get_paper_references": (
        "get_paper_references",
        lambda a: {
            "paper_id": a["paper_id"],
            "limit": a.get("limit", 100),
            "fields": a.get("fields"),
            "offset": _cursor_to_offset(
                a.get("cursor"),
                "get_paper_references",
                context_hash=compute_context_hash("get_paper_references", a),
            ),
        },
    ),
    "get_paper_authors": (
        "get_paper_authors",
        lambda a: {
            "paper_id": a["paper_id"],
            "limit": a.get("limit", 100),
            "fields": a.get("fields"),
            "offset": _cursor_to_offset(
                a.get("cursor"),
                "get_paper_authors",
                context_hash=compute_context_hash("get_paper_authors", a),
            ),
        },
    ),
    "get_author_info": (
        "get_author_info",
        lambda a: {
            "author_id": a["author_id"],
            "fields": a.get("fields"),
        },
    ),
    "get_author_papers": (
        "get_author_papers",
        lambda a: {
            "author_id": a["author_id"],
            "limit": a.get("limit", 100),
            "fields": a.get("fields"),
            "offset": _cursor_to_offset(
                a.get("cursor"),
                "get_author_papers",
                context_hash=compute_context_hash("get_author_papers", a),
            ),
            "publication_date_or_year": a.get("publication_date_or_year"),
        },
    ),
    "search_authors": (
        "search_authors",
        lambda a: {
            "query": a["query"],
            "limit": a.get("limit", 10),
            "fields": a.get("fields"),
            "offset": _cursor_to_offset(
                a.get("cursor"),
                "search_authors",
                context_hash=compute_context_hash("search_authors", a),
            ),
        },
    ),
    "batch_get_authors": (
        "batch_get_authors",
        lambda a: {
            "author_ids": a["author_ids"],
            "fields": a.get("fields"),
        },
    ),
    "search_snippets": (
        "search_snippets",
        lambda a: {
            "query": a["query"],
            "limit": a.get("limit", 10),
            "fields": a.get("fields"),
            "year": a.get("year"),
            "publication_date_or_year": a.get("publication_date_or_year"),
            "fields_of_study": a.get("fields_of_study"),
            "min_citation_count": a.get("min_citation_count"),
            "venue": a.get("venue"),
        },
    ),
    "get_paper_recommendations": (
        "get_recommendations",
        lambda a: {
            "paper_id": a["paper_id"],
            "limit": a.get("limit", 10),
            "fields": a.get("fields"),
        },
    ),
    "get_paper_recommendations_post": (
        "get_recommendations_post",
        lambda a: {
            "positive_paper_ids": a["positive_paper_ids"],
            "negative_paper_ids": a.get("negative_paper_ids"),
            "limit": a.get("limit", 10),
            "fields": a.get("fields"),
        },
    ),
    "batch_get_papers": (
        "batch_get_papers",
        lambda a: {
            "paper_ids": a["paper_ids"],
            "fields": a.get("fields"),
        },
    ),
}

PROVIDER_SEARCH_TOOLS: dict[str, SearchProvider] = {
    "search_papers_core": "core",
    "search_papers_semantic_scholar": "semantic_scholar",
    "search_papers_serpapi": "serpapi_google_scholar",
    "search_papers_scholarapi": "scholarapi",
    "search_papers_arxiv": "arxiv",
}

SMART_TOOLS = {
    "search_papers_smart",
    "ask_result_set",
    "map_research_landscape",
    "expand_research_graph",
}

GUIDED_TOOLS = {
    "research",
    "follow_up_research",
    "resolve_reference",
    "inspect_source",
    "get_runtime_status",
}

GUIDED_POLICY_NAME = "quality_first"


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


_REGULATORY_SOURCE_TYPES: frozenset[str] = frozenset(
    {
        "agency_report",
        "congressional_report",
        "executive_order",
        "federal_register_rule",
        "government_document",
        "government_report",
        "legislation",
        "policy_document",
        "primary_regulatory",
        "primary_source",
        "regulatory_document",
        "regulatory_guidance",
        "statute",
        "treaty",
    }
)
_ACCESS_STATUS_IMPLIES_BODY: frozenset[str] = frozenset(
    {"body_text_embedded", "qa_readable_text", "full_text_verified", "full_text_retrieved"}
)
_ACCESS_STATUS_IMPLIES_QA: frozenset[str] = frozenset({"qa_readable_text", "full_text_verified", "full_text_retrieved"})
_GUIDED_LITERATURE_TERMS = {
    "article",
    "articles",
    "citation",
    "citations",
    "doi",
    "evidence",
    "journal",
    "journals",
    "literature",
    "meta-analysis",
    "paper",
    "papers",
    "peer-reviewed",
    "review",
    "reviews",
    "scholarly",
    "science",
    "scientific",
    "study",
    "studies",
    "systematic review",
}


def _candidate_is_inspectable(candidate: dict[str, Any]) -> bool:
    """Mirror graphs._has_inspectable_sources: on-topic AND has URL/abstract."""
    if candidate.get("topicalRelevance") == "off_topic":
        return False
    return bool(
        candidate.get("canonicalUrl")
        or candidate.get("retrievedUrl")
        or candidate.get("fullTextUrlFound")
        or candidate.get("abstractObserved")
    )


def _normalize_author_key(name: str) -> tuple[str, str]:
    """Return (surname_lower, first_initial_lower) for dedup grouping."""
    parts = name.strip().split()
    if not parts:
        return ("", "")
    surname = parts[-1].lower().rstrip(".")
    given = parts[0].lower().rstrip(".") if len(parts) > 1 else ""
    initial = given[0] if given else ""
    return (surname, initial)


def _deduplicate_authors(authors: list[str]) -> list[str]:
    """Deduplicate authors that differ only by given-name completeness.

    Groups by (surname, first_initial) and keeps the longest given-name form.
    """
    groups: dict[tuple[str, str], list[str]] = {}
    for name in authors:
        key = _normalize_author_key(name)
        groups.setdefault(key, []).append(name)
    result: list[str] = []
    seen_keys: set[tuple[str, str]] = set()
    for name in authors:
        key = _normalize_author_key(name)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        # Keep the longest form (most complete given name)
        group = groups[key]
        best = max(group, key=len)
        result.append(best)
    return result


_AUTHORITATIVE_PROVIDERS: frozenset[str] = frozenset({"govinfo", "federal_register", "ecos", "gov_info"})


def _is_authoritative_source(source: dict[str, Any]) -> bool:
    provider = str(source.get("provider") or "").strip().lower()
    source_type = str(source.get("sourceType") or "").strip()
    if provider in _AUTHORITATIVE_PROVIDERS:
        return True
    if source_type in _REGULATORY_SOURCE_TYPES:
        return True
    if bool(source.get("isPrimarySource")):
        return True
    return False


def _compose_why_classified_weak_match(
    source: dict[str, Any],
    *,
    strategy_metadata: dict[str, Any] | None = None,
) -> str | None:
    """Compose a concise (<=200 char) human-readable rationale for why a source
    was classified as a weak match or off-topic.

    Reads, in priority order, from: ``classificationRationale`` (prior work),
    ``whyClassifiedAsWeakMatch``/``whyWeak`` (ws-regulatory-grounding),
    ``note``/``whyNotVerified``, and ``strategy_metadata.subjectChainGaps``.
    Returns ``None`` when no signal is present or the source is not weak/off-topic.
    """
    topical_relevance = str(source.get("topicalRelevance") or "").strip()
    if topical_relevance not in {"weak_match", "off_topic"}:
        return None
    fragments: list[str] = []
    seen: set[str] = set()

    def _add(value: Any) -> None:
        text = str(value or "").strip()
        if not text:
            return
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        fragments.append(text)

    _add(source.get("classificationRationale"))
    _add(source.get("whyClassifiedAsWeakMatch"))
    _add(source.get("whyWeak"))
    _add(source.get("note"))
    _add(source.get("whyNotVerified"))
    if isinstance(strategy_metadata, dict):
        gaps = strategy_metadata.get("subjectChainGaps")
        if isinstance(gaps, list):
            for gap in gaps[:2]:
                _add(gap)
    if not fragments:
        return None
    head = fragments[0].rstrip(".")
    if len(fragments) >= 2:
        tail = fragments[1].rstrip(".")
        combined = f"{head}; {tail}."
    else:
        combined = f"{head}."
    if len(combined) > 200:
        combined = combined[:197].rstrip() + "..."
    return combined


def _evidence_quality_detail(sources: list[dict[str, Any]]) -> str:
    """Classify the evidence pool into a qualitative profile.

    Returns one of ``strong_on_topic``, ``mixed``, ``weak_authoritative_only``,
    ``off_topic``, or ``insufficient``.
    """
    if not sources:
        return "insufficient"
    on_topic_primary = 0
    on_topic = 0
    weak = 0
    off_topic = 0
    authoritative_weak = 0
    for source in sources:
        relevance = str(source.get("topicalRelevance") or "").strip()
        if relevance == "on_topic":
            on_topic += 1
            if source.get("verificationStatus") == "verified_primary_source":
                on_topic_primary += 1
        elif relevance == "weak_match":
            weak += 1
            if _is_authoritative_source(source):
                authoritative_weak += 1
        elif relevance == "off_topic":
            off_topic += 1
    total = len(sources)
    if on_topic > 0 and (weak > 0 or off_topic > 0):
        return "mixed"
    if on_topic > 0:
        return "strong_on_topic"
    if on_topic == 0 and authoritative_weak > 0 and off_topic < total:
        return "weak_authoritative_only"
    if on_topic == 0 and off_topic >= max(weak, 1):
        return "off_topic"
    return "insufficient"


def _synthesis_path(
    *,
    status: str,
    sources: list[dict[str, Any]],
    evidence_gaps: list[str],
    synthesis_mode: str | None,
) -> str:
    """Return which synthesis path actually ran.

    Values: ``direct``, ``conservative``, ``abstained``, ``metadata_only``.
    """
    if status in {"abstained", "insufficient_evidence"} and not sources:
        return "abstained"
    normalized_mode = str(synthesis_mode or "").strip().lower()
    if normalized_mode == "source_audit" and sources:
        return "metadata_only"
    has_primary = any(item.get("verificationStatus") == "verified_primary_source" for item in sources)
    has_on_topic = any(item.get("topicalRelevance") == "on_topic" for item in sources)
    if status in {"answered", "succeeded"} and has_primary and has_on_topic and not evidence_gaps:
        return "direct"
    if status in {"answered", "succeeded", "partial"} and sources:
        return "conservative"
    if not sources:
        return "abstained"
    return "conservative"


def _trust_revision_narrative(
    *,
    sources: list[dict[str, Any]],
    evidence_gaps: list[str],
    degradation_reason: str | None,
) -> str | None:
    """Return a prose reason why the trust summary was downgraded, if any."""
    if not sources and not evidence_gaps and degradation_reason is None:
        return None
    has_on_topic = any(item.get("topicalRelevance") == "on_topic" for item in sources)
    authoritative_weak = [
        item
        for item in sources
        if _is_authoritative_source(item)
        and not has_on_topic
        and str(item.get("topicalRelevance") or "") in {"weak_match", "off_topic"}
    ]
    if degradation_reason == "deterministic_synthesis_fallback":
        return "Model-backed synthesis was unavailable; deterministic fallback summarized available evidence."
    if sources and not has_on_topic and authoritative_weak:
        return (
            "Authoritative primary sources were retrieved, but none specifically address "
            "the saved query. Treat as scope-limited support only."
        )
    if sources and not has_on_topic:
        return "Available evidence is weak or off-topic for the saved query."
    if evidence_gaps and sources and has_on_topic:
        return f"On-topic evidence is partial: {evidence_gaps[0]}"
    return None


def _authoritative_but_weak_source_ids(sources: list[dict[str, Any]]) -> list[str]:
    """Return source IDs in the authoritativeButWeak (missed-escalation) bucket.

    This bucket flags authoritative/primary-source records (e.g., Federal
    Register rules, regulatory documents, official agency pages) whose
    ``topicalRelevance`` is ``weak_match`` or ``off_topic``. It is a
    missed-escalation signal: the provider returned an authoritative record,
    but the subject-chain/topical grounding is weak, so the agent should not
    treat the source as grounded evidence. Clients should surface these IDs
    to a human or escalate to a disambiguation/primary-source workflow rather
    than fold them silently into evidence.
    """
    ids: list[str] = []
    for source in sources:
        relevance = str(source.get("topicalRelevance") or "").strip()
        if relevance not in {"weak_match", "off_topic"}:
            continue
        if not _is_authoritative_source(source):
            continue
        source_id = str(source.get("sourceId") or "").strip()
        if source_id:
            ids.append(source_id)
    return ids


def _find_record_source_with_resolution(
    *,
    workspace_registry: Any,
    search_session_id: str | None,
    source_id: str,
) -> tuple[dict[str, Any] | None, str]:
    if workspace_registry is None:
        return None, "workspace_unavailable"
    normalized_search_session_id = _guided_normalize_whitespace(search_session_id)
    if not normalized_search_session_id:
        return None, "missing_session_id"
    try:
        record = workspace_registry.get(normalized_search_session_id)
    except Exception:
        return None, "session_not_found"

    normalized_source_id = _guided_normalize_whitespace(source_id)
    if not normalized_source_id:
        return None, "empty_source_id"

    direct = _find_record_source(
        workspace_registry=workspace_registry,
        search_session_id=normalized_search_session_id,
        source_id=normalized_source_id,
    )
    if direct is not None:
        return direct, "exact_id"

    sources = _guided_record_source_candidates(record)
    if not sources:
        return None, "no_session_sources"

    index_match = re.fullmatch(r"(?:source|src|lead|paper)?\s*[-_#]?\s*(\d{1,3})", normalized_source_id, re.IGNORECASE)
    if index_match:
        index = int(index_match.group(1))
        if 1 <= index <= len(sources):
            return sources[index - 1], "index_alias"

    lowered_source_id = normalized_source_id.lower()
    normalized_locator = _guided_normalize_source_locator(normalized_source_id)
    for source in sources:
        for candidate in (
            source.get("sourceId"),
            source.get("sourceAlias"),
            source.get("citationText"),
            source.get("canonicalUrl"),
            source.get("retrievedUrl"),
            source.get("title"),
        ):
            normalized_candidate = _guided_normalize_whitespace(candidate).lower()
            if normalized_candidate and normalized_candidate == lowered_source_id:
                return source, "casefold_exact"
            if normalized_locator and _guided_normalize_source_locator(candidate) == normalized_locator:
                return source, "normalized_locator"

    partial_matches = [
        source
        for source in sources
        if lowered_source_id in _guided_normalize_whitespace(source.get("title")).lower()
        or lowered_source_id in _guided_normalize_whitespace(source.get("canonicalUrl")).lower()
        or lowered_source_id in _guided_normalize_whitespace(source.get("citationText")).lower()
        or (normalized_locator and normalized_locator in _guided_normalize_source_locator(source.get("canonicalUrl")))
        or (normalized_locator and normalized_locator in _guided_normalize_source_locator(source.get("retrievedUrl")))
    ]
    if len(partial_matches) == 1:
        return partial_matches[0], "unique_partial_match"

    return None, "unresolved"


def _find_record_source(
    *,
    workspace_registry: Any,
    search_session_id: str | None,
    source_id: str,
) -> dict[str, Any] | None:
    if workspace_registry is None:
        return None
    normalized_search_session_id = _guided_normalize_whitespace(search_session_id)
    if not normalized_search_session_id:
        return None
    record = workspace_registry.get(normalized_search_session_id)
    normalized_source_id = _guided_normalize_whitespace(source_id)
    if not normalized_source_id:
        return None
    for source in _guided_record_source_candidates(record):
        candidate_ids = {
            _guided_normalize_whitespace(source.get("sourceId")),
            _guided_normalize_whitespace(source.get("citationText")),
            _guided_normalize_whitespace(source.get("canonicalUrl")),
            _guided_normalize_whitespace(source.get("retrievedUrl")),
        }
        if normalized_source_id in candidate_ids:
            return source
    return None


def _direct_read_recommendations(source: dict[str, Any], *, tool_profile: str) -> list[str]:
    return [entry["recommendation"] for entry in _direct_read_recommendation_entries(source, tool_profile=tool_profile)]


def _direct_read_recommendation_details(source: dict[str, Any], *, tool_profile: str) -> list[dict[str, Any]]:
    """Parallel to :func:`_direct_read_recommendations` returning dicts with
    trustLevel/whyRecommended/cautions fields. Order matches.
    """
    return [
        {k: v for k, v in entry.items() if k != "recommendation"} | {"recommendation": entry["recommendation"]}
        for entry in _direct_read_recommendation_entries(source, tool_profile=tool_profile)
    ]


def _direct_read_recommendation_entries(source: dict[str, Any], *, tool_profile: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    topical_relevance = str(source.get("topicalRelevance") or "").strip()
    verification_status = str(source.get("verificationStatus") or "").strip()
    weak_match_reason = str(source.get("whyClassifiedAsWeakMatch") or source.get("note") or "").strip()
    is_authoritative = _is_authoritative_source(source)
    cautions: list[str] = []
    if topical_relevance == "on_topic" and verification_status == "verified_primary_source":
        trust_level = "high"
        why = "Authoritative and directly responsive to the saved query; safe to cite with direct-read verification."
        text = (
            "This source is authoritative and directly responsive to the query; direct reading is the safest next step."
        )
    elif topical_relevance == "on_topic" and verification_status == "verified_metadata":
        trust_level = "medium"
        why = "On-topic and metadata-verified; direct reading confirms specific claims."
        text = (
            "This source is on-topic and metadata-verified, but direct reading is still useful "
            "before relying on specific claims."
        )
        cautions.append("metadata-verified only; full text not yet observed")
    elif topical_relevance == "weak_match":
        scope_reason = (
            weak_match_reason or "It is related to the topic but not directly responsive enough to stand alone."
        )
        if is_authoritative:
            trust_level = "low_authoritative_but_weak"
            why = (
                "Authoritative primary source, but not directly on-topic. Useful for context or "
                "regulatory framing only."
            )
            cautions.append("authoritative but not specifically responsive to the saved query")
        else:
            trust_level = "low"
            why = "Related but scope-limited; unlikely to directly answer the query on its own."
        text = f"This source is only a weak match: {scope_reason}"
    elif topical_relevance == "off_topic":
        scope_reason = (
            weak_match_reason or "It is authoritative or retrievable, but it does not directly answer the saved query."
        )
        if is_authoritative:
            trust_level = "low_authoritative_but_weak"
            why = "Authoritative source that does not address the saved query."
            cautions.append("authoritative but off-target for the saved query")
        else:
            trust_level = "low"
            why = "Off-topic for the saved query; direct reading is unlikely to help."
        text = f"This source is off-topic for the saved query: {scope_reason}"
    else:
        trust_level = "medium"
        why = "Source relevance is unclassified; inspect directly before relying on it."
        text = "Inspect this source directly before citing specific claims."

    entries.append(
        {
            "recommendation": text,
            "trustLevel": trust_level,
            "whyRecommended": why,
            "cautions": list(cautions),
        }
    )
    canonical_url = str(source.get("canonicalUrl") or source.get("retrievedUrl") or "").strip()
    if canonical_url:
        entries.append(
            {
                "recommendation": f"Open the canonical source: {canonical_url}",
                "trustLevel": trust_level,
                "whyRecommended": "Direct link to the canonical copy of this source.",
                "cautions": [],
            }
        )
    provider = str(source.get("provider") or "")
    if tool_profile != "guided" and provider == "govinfo":
        entries.append(
            {
                "recommendation": "Use get_cfr_text for authoritative CFR follow-through.",
                "trustLevel": "high",
                "whyRecommended": "GovInfo-backed CFR text is the authoritative regulatory source.",
                "cautions": [],
            }
        )
    elif tool_profile != "guided" and provider == "federal_register":
        entries.append(
            {
                "recommendation": "Use get_federal_register_document to read the full Federal Register item.",
                "trustLevel": "high",
                "whyRecommended": "The Federal Register document is the primary rulemaking source.",
                "cautions": [],
            }
        )
    elif tool_profile != "guided" and provider == "ecos":
        entries.append(
            {
                "recommendation": "Use get_document_text_ecos for the full ECOS document text when available.",
                "trustLevel": "high",
                "whyRecommended": "ECOS primary documents provide species-specific regulatory context.",
                "cautions": [],
            }
        )
    elif tool_profile != "guided" and provider in {"semantic_scholar", "openalex", "arxiv", "core", "scholarapi"}:
        entries.append(
            {
                "recommendation": (
                    "Use expert paper-detail tools if you need the full provider payload or citation expansion."
                ),
                "trustLevel": "medium",
                "whyRecommended": "Expert paper-detail tools expose provider-specific metadata and citation graphs.",
                "cautions": [],
            }
        )
    if tool_profile == "guided":
        entries.append(
            {
                "recommendation": (
                    "Use inspect_source to compare provenance, scope, and access signals before citing this source."
                ),
                "trustLevel": trust_level,
                "whyRecommended": "inspect_source is the guided way to audit a single source's provenance.",
                "cautions": [],
            }
        )
    return entries[:3]


_LEGACY_GUIDED_FIELDS = {
    "verifiedFindings",
    "sources",
    "unverifiedLeads",
    "coverage",
}

_FOLLOW_UP_COMPACT_FIELDS = {
    "searchSessionId",
    "answerStatus",
    "answer",
    "unsupportedAsks",
    "followUpQuestions",
    "evidenceGaps",
    "nextActions",
    "resultStatus",
    "answerability",
    "sessionResolution",
    "abstentionDetails",
    "executionProvenance",
    "inputNormalization",
    "machineFailure",
    "evidenceUsePlan",
    "providerUsed",
    "degradationReason",
    "agentHints",
    "confidenceSignals",
}

_RESEARCH_COMPACT_FIELDS = {
    "intent",
    "status",
    "searchSessionId",
    "summary",
    "evidenceGaps",
    "nextActions",
    "clarification",
    "resultStatus",
    "answerability",
    "routingSummary",
    "resultState",
    "abstentionDetails",
    "executionProvenance",
    "inputNormalization",
    "machineFailure",
}

_COMPACT_NULL_OK_FIELDS = {"answer", "searchSessionId"}

# Fields that must always be preserved (even as empty/None) on compact follow-ups
# for grounded answers so agents retain the minimum actionable contract.
_FOLLOW_UP_COMPACT_GROUNDED_PRESERVE = {
    "searchSessionId",
    "answerStatus",
    "answer",
    "structuredSourceIds",
    "evidenceGaps",
    "nextActions",
    "resultMeaning",
    "resultState",
    "answerability",
    "topRecommendation",
    "selectedEvidenceIds",
    "selectedLeadIds",
    "agentHints",
    "sessionResolution",
    "inputNormalization",
    "executionProvenance",
    "unsupportedAsks",
    "followUpQuestions",
    "abstentionDetails",
    "coverage",
    "sourcesSuppressed",
    "legacyFieldsIncluded",
    "responseMode",
}

# Fields intentionally dropped from compact follow-up responses (beyond legacy).
_FOLLOW_UP_COMPACT_DROP = {
    "failureSummary",
    "confidenceSignals",
    "trustSummary",
    "telemetry",
    "evidenceUsePlan",
    "providerUsed",
    "degradationReason",
    "classificationProvenance",
    "degradedClassification",
}


def _compact_coverage_payload(coverage: Any) -> dict[str, Any] | None:
    """Collapse a coverage/CoverageSummary dict to compact, agent-usable fields."""

    if not isinstance(coverage, dict):
        return None
    search_mode = coverage.get("searchMode")
    total = coverage.get("totalSources")
    by_access = coverage.get("byAccessStatus")
    result: dict[str, Any] = {}
    if isinstance(search_mode, str) and search_mode.strip():
        result["searchMode"] = search_mode.strip()
    if isinstance(total, int):
        result["totalSources"] = total
    if isinstance(by_access, dict):
        result["byAccessStatus"] = by_access
    return result or None


def _is_empty_for_compact(value: Any) -> bool:
    """True when a value should be omitted from a compact payload."""

    if value is None:
        return True
    if isinstance(value, (list, dict, str, tuple, set)) and len(value) == 0:
        return True
    return False


def _compact_record_identifiers(records: Any) -> list[str]:
    """Collect source-like identifiers from a list of source/evidence payloads."""

    if not isinstance(records, list):
        return []
    identifiers: list[str] = []
    for entry in records:
        if isinstance(entry, str):
            identifier = entry.strip()
        elif isinstance(entry, dict):
            identifier = str(
                entry.get("sourceId") or entry.get("sourceAlias") or entry.get("evidenceId") or entry.get("id") or ""
            ).strip()
        else:
            identifier = ""
        if identifier:
            identifiers.append(identifier)
    return identifiers


def _dedupe_compact_identifiers(identifiers: list[str]) -> list[str]:
    """Deduplicate identifiers while preserving order."""

    seen: set[str] = set()
    deduped: list[str] = []
    for identifier in identifiers:
        if identifier in seen:
            continue
        seen.add(identifier)
        deduped.append(identifier)
    return deduped


def _apply_follow_up_response_mode(
    response: dict[str, Any],
    *,
    response_mode: str,
    include_legacy_fields: bool,
) -> dict[str, Any]:
    """Apply response_mode shaping to a follow-up response dict.

    Compact: drop legacy (verifiedFindings/likelyUnverified) unless opted in,
    derive structuredSourceIds from structuredSources, collapse coverage, and
    omit None/empty fields. Preserves abstentionDetails and topRecommendation.
    Standard: exclude None and empty fields (but keep legacy).
    Debug: no filtering.
    """

    if response_mode == "debug":
        shaped = dict(response)
        shaped["responseMode"] = "debug"
        shaped.setdefault(
            "legacyFieldsIncluded",
            any(key in shaped for key in _LEGACY_GUIDED_FIELDS),
        )
        return shaped

    shaped = dict(response)

    if response_mode == "compact":
        suppressed_count = 0
        dropped_source_payload = False
        kept_identifiers = _dedupe_compact_identifiers(
            [
                str(identifier).strip()
                for identifier in (
                    list(shaped.get("selectedEvidenceIds") or []) + list(shaped.get("selectedLeadIds") or [])
                )
                if str(identifier).strip()
            ]
        )
        if not include_legacy_fields:
            for key in ("verifiedFindings", "unverifiedLeads", "likelyUnverified"):
                value = shaped.pop(key, None)
                if isinstance(value, list):
                    suppressed_count += len(value)
        evidence = shaped.pop("evidence", None)
        if isinstance(evidence, list) and evidence:
            dropped_source_payload = True
        sources = shaped.pop("sources", None)
        if isinstance(sources, list) and sources:
            dropped_source_payload = True
        structured_sources = shaped.pop("structuredSources", None)
        if isinstance(structured_sources, list):
            ids = _dedupe_compact_identifiers(_compact_record_identifiers(structured_sources))
            if ids and not kept_identifiers:
                shaped["structuredSourceIds"] = ids
            if ids:
                dropped_source_payload = True
        coverage = shaped.get("coverage")
        collapsed = _compact_coverage_payload(coverage)
        if collapsed is not None:
            compact_coverage = dict(collapsed)
            if compact_coverage:
                shaped["coverage"] = compact_coverage
            else:
                shaped.pop("coverage", None)
        elif coverage is None or coverage == {}:
            shaped.pop("coverage", None)
        for drop_key in _FOLLOW_UP_COMPACT_DROP:
            shaped.pop(drop_key, None)
        shaped["legacyFieldsIncluded"] = bool(include_legacy_fields) and any(
            key in shaped for key in ("verifiedFindings", "unverifiedLeads", "likelyUnverified")
        )
        if suppressed_count > 0:
            shaped["sourcesSuppressed"] = suppressed_count
        elif dropped_source_payload:
            shaped["sourcesSuppressed"] = True
        shaped["responseMode"] = "compact"

    keep_keys = set(shaped.keys())
    for key in list(shaped.keys()):
        if key in _COMPACT_NULL_OK_FIELDS:
            continue
        if key not in keep_keys:
            continue
        if _is_empty_for_compact(shaped[key]):
            shaped.pop(key, None)

    if response_mode != "compact":
        shaped.setdefault(
            "legacyFieldsIncluded",
            any(key in shaped for key in _LEGACY_GUIDED_FIELDS),
        )
        shaped["responseMode"] = response_mode
    return shaped


def _apply_inspect_source_compaction(response: dict[str, Any]) -> dict[str, Any]:
    """Trim inspect_source payloads to the inspected source plus concise context."""

    shaped = dict(response)
    source = shaped.get("source")
    source_identifier = (
        str(source.get("sourceId") or source.get("sourceAlias") or "").strip() if isinstance(source, dict) else ""
    )
    evidence = shaped.get("evidence")
    if isinstance(evidence, list):
        filtered_evidence = [
            entry
            for entry in evidence
            if isinstance(entry, dict)
            and str(entry.get("evidenceId") or entry.get("sourceId") or entry.get("sourceAlias") or "").strip()
            == source_identifier
        ]
        if filtered_evidence:
            shaped["evidence"] = filtered_evidence[:1]
        else:
            shaped.pop("evidence", None)
    leads = shaped.get("leads")
    if isinstance(leads, list):
        shaped.pop("leads", None)
    coverage_summary = shaped.get("coverageSummary")
    collapsed_coverage = _compact_coverage_payload(coverage_summary)
    if collapsed_coverage is not None:
        shaped["coverageSummary"] = collapsed_coverage
    else:
        shaped.pop("coverageSummary", None)
    for key in list(shaped.keys()):
        if key in _COMPACT_NULL_OK_FIELDS:
            continue
        if _is_empty_for_compact(shaped[key]):
            shaped.pop(key, None)
    return shaped


def _append_deterministic_fallback_gap(
    evidence_gaps: list[str],
    *,
    strategy_metadata: dict[str, Any] | None,
) -> list[str]:
    metadata = strategy_metadata or {}
    configured = _guided_normalize_whitespace(metadata.get("configuredSmartProvider"))
    active = _guided_normalize_whitespace(metadata.get("activeSmartProvider"))
    if active != "deterministic" or configured in {None, "deterministic"}:
        return evidence_gaps
    if any(
        "deterministic fallback" in str(gap).lower() or "deterministic_synthesis_fallback" in str(gap).lower()
        for gap in evidence_gaps
    ):
        return evidence_gaps
    return [*evidence_gaps, f"deterministic fallback: smart provider '{configured}' was unavailable for this run."]


_LLM_ANSWERABILITY_MAP = {
    "answered": "grounded",
    "abstained": "insufficient",
    "insufficient_evidence": "insufficient",
}


async def dispatch_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    client: Any,
    core_client: Any,
    openalex_client: Any,
    scholarapi_client: Any,
    arxiv_client: Any,
    enable_core: bool,
    enable_semantic_scholar: bool,
    enable_openalex: bool,
    enable_scholarapi: bool,
    enable_arxiv: bool,
    serpapi_client: Any = None,
    enable_serpapi: bool = False,
    crossref_client: Any = None,
    unpaywall_client: Any = None,
    ecos_client: Any = None,
    federal_register_client: Any = None,
    govinfo_client: Any = None,
    enable_crossref: bool = True,
    enable_unpaywall: bool = True,
    enable_ecos: bool = True,
    enable_federal_register: bool = True,
    enable_govinfo_cfr: bool = True,
    enrichment_service: PaperEnrichmentService | None = None,
    provider_order: list[SearchProvider] | None = None,
    provider_registry: Any = None,
    workspace_registry: Any = None,
    agentic_runtime: Any = None,
    transport_mode: str = "stdio",
    tool_profile: str = "guided",
    hide_disabled_tools: bool = False,
    session_ttl_seconds: int | None = None,
    embeddings_enabled: bool | None = None,
    guided_research_latency_profile: str = "deep",
    guided_follow_up_latency_profile: str = "deep",
    guided_allow_paid_providers: bool = True,
    guided_escalation_enabled: bool = True,
    guided_escalation_max_passes: int = 2,
    guided_escalation_allow_paid_providers: bool = True,
    ctx: Any = None,
    allow_elicitation: bool = True,
) -> dict[str, Any]:
    """Dispatch one MCP tool call to the correct backend implementation."""
    resolved_enrichment_service = enrichment_service or PaperEnrichmentService(
        crossref_client=crossref_client,
        unpaywall_client=unpaywall_client,
        openalex_client=openalex_client,
        enable_crossref=enable_crossref,
        enable_unpaywall=enable_unpaywall,
        enable_openalex=enable_openalex,
        provider_registry=provider_registry,
    )

    # Phase 2 Step 2: collect every dependency into a single frozen
    # ``DispatchContext`` bag. The inner ``_dispatch_internal`` forwarder (and,
    # in later phases, extracted branch helpers) pass this single object
    # around instead of re-spelling the 40-kwarg list on every call.
    dispatch_ctx: DispatchContext = build_dispatch_context(
        client=client,
        core_client=core_client,
        openalex_client=openalex_client,
        scholarapi_client=scholarapi_client,
        arxiv_client=arxiv_client,
        enable_core=enable_core,
        enable_semantic_scholar=enable_semantic_scholar,
        enable_openalex=enable_openalex,
        enable_scholarapi=enable_scholarapi,
        enable_arxiv=enable_arxiv,
        serpapi_client=serpapi_client,
        enable_serpapi=enable_serpapi,
        crossref_client=crossref_client,
        unpaywall_client=unpaywall_client,
        ecos_client=ecos_client,
        federal_register_client=federal_register_client,
        govinfo_client=govinfo_client,
        enable_crossref=enable_crossref,
        enable_unpaywall=enable_unpaywall,
        enable_ecos=enable_ecos,
        enable_federal_register=enable_federal_register,
        enable_govinfo_cfr=enable_govinfo_cfr,
        enrichment_service=resolved_enrichment_service,
        provider_order=provider_order,
        provider_registry=provider_registry,
        workspace_registry=workspace_registry,
        agentic_runtime=agentic_runtime,
        transport_mode=transport_mode,
        tool_profile=tool_profile,
        hide_disabled_tools=hide_disabled_tools,
        session_ttl_seconds=session_ttl_seconds,
        embeddings_enabled=embeddings_enabled,
        guided_research_latency_profile=guided_research_latency_profile,
        guided_follow_up_latency_profile=guided_follow_up_latency_profile,
        guided_allow_paid_providers=guided_allow_paid_providers,
        guided_escalation_enabled=guided_escalation_enabled,
        guided_escalation_max_passes=guided_escalation_max_passes,
        guided_escalation_allow_paid_providers=guided_escalation_allow_paid_providers,
        ctx=ctx,
        allow_elicitation=allow_elicitation,
    )

    async def _dispatch_internal(tool_name: str, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        return await dispatch_tool(tool_name, tool_arguments, **dispatch_ctx.as_kwargs())

    if name == "get_runtime_status":
        cast(GetRuntimeStatusArgs, TOOL_INPUT_MODELS[name].model_validate(arguments))
        diagnostics = _build_provider_diagnostics_snapshot(
            include_recent_outcomes=False,
            provider_order=provider_order,
            provider_registry=provider_registry,
            agentic_runtime=agentic_runtime,
            transport_mode=transport_mode,
            tool_profile=tool_profile,
            hide_disabled_tools=hide_disabled_tools,
            session_ttl_seconds=session_ttl_seconds,
            embeddings_enabled=embeddings_enabled,
            guided_research_latency_profile=guided_research_latency_profile,
            guided_follow_up_latency_profile=guided_follow_up_latency_profile,
            guided_allow_paid_providers=guided_allow_paid_providers,
            guided_escalation_enabled=guided_escalation_enabled,
            guided_escalation_max_passes=guided_escalation_max_passes,
            guided_escalation_allow_paid_providers=guided_escalation_allow_paid_providers,
            enable_core=enable_core,
            enable_semantic_scholar=enable_semantic_scholar,
            enable_openalex=enable_openalex,
            enable_arxiv=enable_arxiv,
            enable_serpapi=enable_serpapi,
            enable_scholarapi=enable_scholarapi,
            enable_crossref=enable_crossref,
            enable_unpaywall=enable_unpaywall,
            enable_ecos=enable_ecos,
            enable_federal_register=enable_federal_register,
            enable_govinfo_cfr=enable_govinfo_cfr,
            ecos_client=ecos_client,
            serpapi_client=serpapi_client,
            scholarapi_client=scholarapi_client,
        )
        return {
            "status": "ok",
            "runtimeSummary": diagnostics["runtimeSummary"],
            "providerOrder": diagnostics.get("providerOrder") or [],
            "providers": diagnostics.get("providers") or [],
            "warnings": diagnostics["runtimeSummary"].get("warnings") or [],
        }

    if name == "resolve_reference":
        resolve_args = cast(ResolveReferenceArgs, TOOL_INPUT_MODELS[name].model_validate(arguments))
        parsed = parse_citation(resolve_args.reference)
        raw = await _dispatch_internal("resolve_citation", {"citation": resolve_args.reference})
        best_match = raw.get("bestMatch")
        alternatives = list(raw.get("alternatives") or [])
        resolution_confidence = str(raw.get("resolutionConfidence") or "low")
        resolution_state = str(raw.get("knownItemResolutionState") or "").strip().lower()
        if resolution_state == "needs_disambiguation" and resolution_confidence == "high":
            resolution_confidence = "medium"
        resolution_type = "title_fragment"
        if parsed.looks_like_regulatory:
            resolution_type = "regulatory_reference"
        elif looks_like_paper_identifier(resolve_args.reference):
            resolution_type = "paper_identifier"
        elif looks_like_citation_query(resolve_args.reference):
            resolution_type = "citation_repair"
        status = "no_match"
        if parsed.looks_like_regulatory and best_match is None:
            status = "regulatory_primary_source"
        elif resolution_state == "needs_disambiguation":
            status = "needs_disambiguation"
        elif best_match is not None:
            if resolution_type == "paper_identifier" or resolution_state == "resolved_exact":
                status = "resolved"
            elif alternatives:
                status = "multiple_candidates"
            else:
                status = "needs_disambiguation"
        elif alternatives:
            status = "multiple_candidates"
        next_actions: list[str] = []
        if status == "regulatory_primary_source":
            next_actions = [
                "Use research for a full trust-graded regulatory pass.",
                "If the reference points to an exact CFR citation, keep that citation in the next research query.",
                "Inspect returned sources before treating the regulatory text as current and settled.",
            ]
        elif status == "needs_disambiguation":
            next_actions = (
                [
                    "The title matched but author, year, or venue conflicts make this result unsafe to cite directly.",
                    "Use research with the title plus author or year to find the correct paper.",
                    "Review the conflicting fields in bestMatch before treating this as a confirmed citation.",
                ]
                if best_match is not None
                else [
                    "The current clues are too sparse or conflicting to cite a single paper safely.",
                    "Add an author, year, venue, DOI, or a more exact title fragment, then rerun resolve_reference.",
                    (
                        "Use research if you want the server to explore nearby candidates "
                        "with a broader trust-graded pass."
                    ),
                ]
            )
        elif status == "resolved":
            next_actions = [
                "Citation resolved against one leading candidate; inspect the metadata before citing it directly.",
                "Use research with the resolved title, DOI, or identifier to gather broader context.",
                "Inspect the resolved source metadata before citing or expanding it.",
            ]
        elif status == "multiple_candidates":
            next_actions = (
                [
                    "Multiple plausible matches were returned — compare bestMatch against alternatives before citing.",
                    "Use research with an added author, year, venue, or more specific title fragment to disambiguate.",
                    "If the best match is good enough, inspect its source record before relying on it.",
                ]
                if best_match is not None
                else [
                    "Multiple plausible matches were returned — review the alternatives before citing any one of them.",
                    "Use research with an added author, year, venue, or more specific title fragment to disambiguate.",
                    "Inspect the strongest alternative before relying on it.",
                ]
            )
        else:
            next_actions = [
                "Try adding an author, year, venue, DOI, or a more exact title fragment.",
                "Use research when you want a broader trust-graded search instead of one exact resolution.",
            ]
        return {
            "resolutionType": resolution_type,
            "status": status,
            "bestMatch": best_match,
            "alternatives": alternatives,
            "resolutionConfidence": resolution_confidence,
            "knownItemResolutionState": resolution_state or None,
            "nextActions": next_actions,
            "searchSessionId": raw.get("searchSessionId"),
        }

    if name == "research":
        normalized_research_arguments, research_normalization = _guided_normalize_research_arguments(arguments)
        research_args = cast(ResearchArgs, TOOL_INPUT_MODELS[name].model_validate(normalized_research_arguments))
        intent = "discovery"
        # Hard identifiers (DOI, arXiv, URL) are unambiguous — always short-circuit to reference resolution.
        # For heuristic-only signals (title-like, year-bearing citation patterns), prefer the planner's judgment
        # when a runtime is available: it can reason about intent with context that static matching cannot.
        # The heuristic path is preserved as a fallback when no planner is configured.
        if looks_like_paper_identifier(research_args.query):
            intent = "known_item"
        elif agentic_runtime is None and _guided_is_known_item_query(research_args.query):
            intent = "known_item"
        elif _guided_is_mixed_intent_query(research_args.query, research_args.focus):
            intent = "mixed"
        elif detect_regulatory_intent(research_args.query, research_args.focus):
            intent = "regulatory"

        clarification = _guided_underspecified_reference_clarification(
            query=research_args.query,
            focus=research_args.focus,
        )
        if clarification is not None:
            evidence_gaps = [
                "The query looks like an underspecified citation or title fragment, so guided research refused "
                "to guess a likely source from weak clues alone."
            ]
            trust_summary = _guided_trust_summary([], evidence_gaps)
            failure_summary = _guided_failure_summary(
                failure_summary={
                    "outcome": "needs_clarification",
                    "whatFailed": "guided_research_reference_preflight",
                    "whatStillWorked": (
                        "The server recognized a vague reference fragment and returned bounded clarification "
                        "instead of surfacing likely-but-unrelated sources."
                    ),
                    "fallbackAttempted": False,
                    "fallbackMode": None,
                    "primaryPathFailureReason": "underspecified_reference_fragment",
                    "completenessImpact": evidence_gaps[0],
                    "recommendedNextAction": "resolve_reference",
                },
                status="needs_disambiguation",
                sources=[],
                evidence_gaps=evidence_gaps,
            )
            contract_fields = await _guided_contract_fields(
                query=research_args.query,
                intent=intent,
                status="needs_disambiguation",
                sources=[],
                unverified_leads=[],
                evidence_gaps=evidence_gaps,
                coverage_summary=None,
                strategy_metadata={
                    "intent": intent,
                    "intentRationale": (
                        "Guided research detected an underspecified citation-like fragment and chose clarification "
                        "over speculative retrieval."
                    ),
                    "routingConfidence": "low",
                    "querySpecificity": "low",
                    "ambiguityLevel": "high",
                },
            )
            response = cast(
                dict[str, Any],
                {
                    "intent": intent,
                    "status": "needs_disambiguation",
                    "searchSessionId": None,
                    "summary": _guided_summary(
                        intent,
                        "needs_disambiguation",
                        [],
                        [],
                        routing_summary=cast(dict[str, Any] | None, contract_fields.get("routingSummary")),
                    ),
                    "verifiedFindings": [],
                    "sources": [],
                    "unverifiedLeads": [],
                    "evidenceGaps": evidence_gaps,
                    "trustSummary": trust_summary,
                    "coverage": None,
                    "failureSummary": failure_summary,
                    "resultMeaning": _guided_result_meaning(
                        status="needs_disambiguation",
                        verified_findings=[],
                        evidence_gaps=evidence_gaps,
                        coverage=None,
                        failure_summary=failure_summary,
                        source_count=0,
                    ),
                    "nextActions": [
                        "Add an exact title, author surname, agency, venue, or tighter year clue and rerun research.",
                        "Use resolve_reference if you can provide a cleaner citation fragment, DOI, arXiv id, or URL.",
                        (
                            "Use get_runtime_status if behavior differs across environments and you need the active "
                            "runtime truth."
                        ),
                    ],
                    "clarification": clarification,
                    "resultState": _guided_result_state(
                        status="needs_disambiguation",
                        sources=[],
                        evidence_gaps=evidence_gaps,
                        search_session_id=None,
                    ),
                    "executionProvenance": _guided_execution_provenance_payload(
                        execution_mode="guided_research_preflight_clarification",
                        answer_source="research",
                        passes_run=0,
                    ),
                    "inputNormalization": _guided_normalization_payload(research_normalization),
                },
            )
            response.update(contract_fields)
            abstention_details = _guided_abstention_details_payload(
                status="needs_disambiguation",
                sources=[],
                evidence_gaps=evidence_gaps,
                trust_summary=trust_summary,
            )
            if abstention_details is not None:
                response["abstentionDetails"] = abstention_details
            return _guided_finalize_response(tool_name="research", response=response)

        if intent == "known_item":
            resolved = await _dispatch_internal("resolve_reference", {"reference": research_args.query})
            paper = resolved.get("bestMatch", {}).get("paper") if isinstance(resolved.get("bestMatch"), dict) else None
            sources = (
                [_guided_source_record_from_paper(research_args.query, paper, index=1)]
                if isinstance(paper, dict)
                else []
            )
            verified_findings = _guided_findings_from_sources(sources)
            unverified_leads = _guided_unverified_leads_from_sources(sources)
            evidence_gaps = []
            status = "succeeded" if paper is not None else ("partial" if resolved.get("alternatives") else "abstained")
            trust_summary = _guided_trust_summary(sources, evidence_gaps)
            failure_summary = _guided_failure_summary(
                failure_summary=None,
                status=status,
                sources=sources,
                evidence_gaps=evidence_gaps,
                all_sources_off_topic=_guided_sources_all_off_topic(sources),
            )
            response = cast(
                dict[str, Any],
                {
                    "intent": intent,
                    "status": status,
                    "searchSessionId": resolved.get("searchSessionId"),
                    "summary": (
                        f"Resolved the request to {paper.get('title')}."
                        if isinstance(paper, dict)
                        else "The reference could not be resolved confidently."
                    ),
                    "verifiedFindings": verified_findings,
                    "sources": sources,
                    "unverifiedLeads": unverified_leads,
                    "evidenceGaps": evidence_gaps,
                    "trustSummary": trust_summary,
                    "coverage": None,
                    "failureSummary": failure_summary,
                    "resultMeaning": _guided_result_meaning(
                        status=status,
                        verified_findings=verified_findings,
                        evidence_gaps=evidence_gaps,
                        coverage=None,
                        failure_summary=failure_summary,
                        source_count=len(sources),
                        all_sources_off_topic=_guided_sources_all_off_topic(sources),
                    ),
                    "nextActions": _guided_next_actions(
                        search_session_id=cast(str | None, resolved.get("searchSessionId")),
                        status=status,
                        has_sources=bool(sources),
                        all_sources_off_topic=_guided_sources_all_off_topic(sources),
                    ),
                    "clarification": None,
                    "resultState": _guided_result_state(
                        status=status,
                        sources=sources,
                        evidence_gaps=evidence_gaps,
                        search_session_id=cast(str | None, resolved.get("searchSessionId")),
                    ),
                    "executionProvenance": _guided_execution_provenance_payload(
                        execution_mode="reference_resolution",
                        answer_source="resolve_reference",
                        passes_run=1,
                    ),
                    "inputNormalization": _guided_normalization_payload(research_normalization),
                },
            )
            response.update(
                await _guided_contract_fields(
                    query=research_args.query,
                    intent=intent,
                    status=status,
                    sources=sources,
                    unverified_leads=unverified_leads,
                    evidence_gaps=evidence_gaps,
                    coverage_summary=None,
                    strategy_metadata=None,
                )
            )
            abstention_details = _guided_abstention_details_payload(
                status=status,
                sources=sources,
                evidence_gaps=evidence_gaps,
                trust_summary=trust_summary,
            )
            if abstention_details is not None:
                response["abstentionDetails"] = abstention_details
            return _guided_finalize_response(tool_name="research", response=response)

        if agentic_runtime is not None:
            initial_provider_budget = _guided_provider_budget_payload(
                allow_paid_providers=guided_allow_paid_providers,
            )
            smart_request = {
                "query": research_args.query,
                "limit": research_args.limit,
                "year": research_args.year,
                "venue": research_args.venue,
                "focus": research_args.focus,
                "latencyProfile": guided_research_latency_profile,
                "providerBudget": initial_provider_budget,
            }
            adequacy_bundle = (
                agentic_runtime._provider_bundle_for_profile(guided_research_latency_profile)
                if hasattr(agentic_runtime, "_provider_bundle_for_profile")
                else None
            )
            smart_runs: list[dict[str, Any]] = []
            pass_modes: list[str] = []
            escalation_attempted = False
            escalation_reason: str | None = None
            review_pass_reason: str | None = None

            async def _run_guided_smart_pass(pass_mode: str, provider_budget: dict[str, Any]) -> None:
                await _run_guided_smart_pass_with_overrides(pass_mode, provider_budget, {})

            async def _run_guided_smart_pass_with_overrides(
                pass_mode: str,
                provider_budget: dict[str, Any],
                overrides: dict[str, Any],
            ) -> None:
                smart = await _dispatch_internal(
                    "search_papers_smart",
                    {
                        **smart_request,
                        **overrides,
                        "mode": pass_mode,
                        "providerBudget": provider_budget,
                    },
                )
                if not isinstance(smart, dict):
                    raise ValueError("search_papers_smart returned a non-object payload.")
                smart_runs.append(smart)
                pass_modes.append(pass_mode)

            async def _summarize_guided_smart_runs() -> dict[str, Any]:
                sources = _guided_dedupe_source_records(
                    [
                        _guided_source_record_from_structured_source(source, index=index)
                        for smart in smart_runs
                        for index, source in enumerate(smart.get("structuredSources") or [], start=1)
                        if isinstance(source, dict)
                    ]
                )
                verified_findings = _guided_findings_from_sources(sources)
                evidence_gaps = list(
                    dict.fromkeys(
                        str(gap).strip()
                        for smart in smart_runs
                        for gap in (smart.get("evidenceGaps") or [])
                        if str(gap).strip()
                    )
                )
                unverified_leads = _guided_dedupe_source_records(
                    [
                        _guided_source_record_from_structured_source(source, index=index)
                        for smart in smart_runs
                        for index, source in enumerate(smart.get("candidateLeads") or [], start=1)
                        if isinstance(source, dict)
                    ]
                ) or _guided_unverified_leads_from_sources(sources)
                merged_coverage = _guided_merge_coverage_summaries(
                    *(cast(dict[str, Any] | None, smart.get("coverageSummary")) for smart in smart_runs)
                )
                merged_failure_summary = _guided_merge_failure_summaries(
                    *(cast(dict[str, Any] | None, smart.get("failureSummary")) for smart in smart_runs)
                )
                clarification = next(
                    (
                        smart.get("clarification")
                        for smart in smart_runs
                        if isinstance(smart.get("clarification"), dict)
                    ),
                    None,
                )
                derived_intent = (
                    "mixed"
                    if intent == "mixed" or ("regulatory" in pass_modes and "review" in pass_modes)
                    else str(smart_runs[0].get("strategyMetadata", {}).get("intent") or intent)
                )
                status_intent = (
                    "regulatory"
                    if derived_intent == "mixed" and any(source.get("isPrimarySource") for source in sources)
                    else derived_intent
                )
                status, adequacy_reason = await _guided_research_status(
                    query=research_args.query,
                    intent=status_intent,
                    sources=sources,
                    findings=verified_findings,
                    unverified_leads_count=len(unverified_leads),
                    coverage_summary=merged_coverage,
                    failure_summary=merged_failure_summary,
                    clarification=cast(dict[str, Any] | None, clarification),
                    provider_bundle=adequacy_bundle,
                )
                return {
                    "sources": sources,
                    "verifiedFindings": verified_findings,
                    "evidenceGaps": evidence_gaps,
                    "unverifiedLeads": unverified_leads,
                    "coverage": merged_coverage,
                    "failureSummary": merged_failure_summary,
                    "clarification": clarification,
                    "derivedIntent": derived_intent,
                    "status": status,
                    "adequacyReason": adequacy_reason,
                }

            try:
                if intent in {"mixed", "regulatory"}:
                    await _run_guided_smart_pass("regulatory", initial_provider_budget)
                    should_run_review, review_pass_reason = _guided_should_add_review_pass(
                        initial_intent=intent,
                        query=research_args.query,
                        focus=research_args.focus,
                        primary_smart=smart_runs[0],
                        pass_modes=pass_modes,
                    )
                    if should_run_review:
                        await _run_guided_smart_pass_with_overrides(
                            "review",
                            initial_provider_budget,
                            _guided_review_pass_overrides(
                                query=research_args.query,
                                focus=research_args.focus,
                                primary_smart=smart_runs[0],
                            ),
                        )
                else:
                    await _run_guided_smart_pass(
                        "regulatory" if intent == "regulatory" else "auto",
                        initial_provider_budget,
                    )
                smart_summary = await _summarize_guided_smart_runs()
                if guided_escalation_enabled and _guided_should_escalate_research(
                    intent=intent,
                    status=str(smart_summary["status"]),
                    sources=cast(list[dict[str, Any]], smart_summary["sources"]),
                    verified_findings=cast(list[dict[str, Any]], smart_summary["verifiedFindings"]),
                    clarification=cast(dict[str, Any] | None, smart_summary["clarification"]),
                    pass_modes=pass_modes,
                    max_passes=guided_escalation_max_passes,
                ):
                    escalation_attempted = True
                    escalation_reason = "no_trustworthy_sources_after_initial_pass"
                    await _run_guided_smart_pass(
                        "review",
                        _guided_provider_budget_payload(
                            allow_paid_providers=guided_escalation_allow_paid_providers,
                        ),
                    )
                    smart_summary = await _summarize_guided_smart_runs()
            except Exception as error:
                return _guided_machine_failure_payload(
                    search_session_id=None,
                    error=error,
                    normalization=research_normalization,
                    execution_provenance=_guided_execution_provenance_payload(
                        execution_mode="guided_research",
                        latency_profile_applied=guided_research_latency_profile,
                        allow_paid_providers=guided_allow_paid_providers,
                        provider_budget_applied=initial_provider_budget,
                        escalation_attempted=escalation_attempted,
                        escalation_reason=escalation_reason,
                        passes_run=len(pass_modes),
                        pass_modes=pass_modes,
                    ),
                )

            sources = cast(list[dict[str, Any]], smart_summary["sources"])
            verified_findings = cast(list[dict[str, Any]], smart_summary["verifiedFindings"])
            evidence_gaps = cast(list[str], smart_summary["evidenceGaps"])
            unverified_leads = cast(list[dict[str, Any]], smart_summary["unverifiedLeads"])
            merged_coverage = cast(dict[str, Any] | None, smart_summary["coverage"])
            merged_failure_summary = cast(dict[str, Any] | None, smart_summary["failureSummary"])
            derived_intent = str(smart_summary["derivedIntent"])
            status = str(smart_summary["status"])
            adequacy_reason = str(smart_summary.get("adequacyReason") or "").strip() or None

            # Supplement academic smart-pass results with Federal Register primary-source documents
            # for regulatory-intent queries.  Only runs when the FR client is available and enabled.
            if intent in {"mixed", "regulatory"} and enable_federal_register and federal_register_client is not None:
                try:
                    fr_response = await federal_register_client.search_documents(
                        query=research_args.query,
                        limit=5,
                    )
                    fr_docs = fr_response.data if hasattr(fr_response, "data") else []
                    fr_sources = _guided_sources_from_fr_documents(research_args.query, fr_docs)
                    if fr_sources:
                        sources = _guided_dedupe_source_records(sources + fr_sources)
                        verified_findings = _guided_findings_from_sources(sources)
                        _status_intent = "regulatory" if any(s.get("isPrimarySource") for s in sources) else intent
                        status, adequacy_reason = await _guided_research_status(
                            query=research_args.query,
                            intent=_status_intent,
                            sources=sources,
                            findings=verified_findings,
                            unverified_leads_count=len(unverified_leads),
                            coverage_summary=merged_coverage,
                            failure_summary=merged_failure_summary,
                            clarification=cast(dict[str, Any] | None, smart_summary.get("clarification")),
                            provider_bundle=adequacy_bundle,
                        )
                except Exception as error:
                    logger.warning(
                        "Federal Register supplementation failed during guided research; "
                        "continuing without FR sources: %s",
                        error,
                    )

            failure_summary = _guided_failure_summary(
                failure_summary=merged_failure_summary,
                status=status,
                sources=sources,
                evidence_gaps=evidence_gaps,
                all_sources_off_topic=_guided_sources_all_off_topic(sources),
            )
            primary_smart = smart_runs[0]
            search_session_id = next(
                (
                    cast(str | None, smart.get("searchSessionId"))
                    for smart in smart_runs
                    if smart.get("searchSessionId")
                ),
                None,
            )
            strategy_metadata = _guided_strategy_metadata_from_runs(smart_runs)
            trust_summary = _guided_trust_summary(
                sources,
                evidence_gaps,
                subject_chain_gaps=cast(
                    list[str] | None,
                    (strategy_metadata or {}).get("subjectChainGaps"),
                ),
            )
            if review_pass_reason:
                strategy_metadata["reviewPassReason"] = review_pass_reason
            evidence_gaps = _append_deterministic_fallback_gap(
                evidence_gaps,
                strategy_metadata=strategy_metadata,
            )
            regulatory_timeline = next(
                (
                    smart.get("regulatoryTimeline")
                    for smart in smart_runs
                    if smart.get("regulatoryTimeline") is not None
                ),
                primary_smart.get("regulatoryTimeline"),
            )
            evidence_gaps = await _guided_generate_evidence_gaps(
                query=research_args.query,
                intent=derived_intent,
                sources=sources,
                existing_evidence_gaps=evidence_gaps,
                coverage_summary=merged_coverage,
                strategy_metadata=strategy_metadata,
                timeline=cast(dict[str, Any] | None, regulatory_timeline),
                provider_bundle=adequacy_bundle,
            )
            contract_fields = await _guided_contract_fields(
                query=research_args.query,
                intent=derived_intent,
                status=status,
                sources=sources,
                unverified_leads=unverified_leads,
                evidence_gaps=evidence_gaps,
                coverage_summary=merged_coverage,
                strategy_metadata=strategy_metadata,
                timeline=cast(dict[str, Any] | None, regulatory_timeline),
                pass_modes=pass_modes,
                review_pass_reason=review_pass_reason,
            )
            response = {
                "intent": derived_intent,
                "status": status,
                "searchSessionId": search_session_id,
                "summary": _guided_summary(
                    derived_intent,
                    status,
                    verified_findings,
                    sources,
                    routing_summary=cast(dict[str, Any] | None, contract_fields.get("routingSummary")),
                    pass_modes=pass_modes,
                ),
                "verifiedFindings": verified_findings,
                "sources": sources,
                "unverifiedLeads": unverified_leads,
                "evidenceGaps": evidence_gaps,
                "trustSummary": trust_summary,
                "coverage": merged_coverage,
                "failureSummary": failure_summary,
                "resultMeaning": _guided_result_meaning(
                    status=status,
                    verified_findings=verified_findings,
                    evidence_gaps=evidence_gaps,
                    coverage=merged_coverage,
                    failure_summary=failure_summary,
                    source_count=len(sources),
                    all_sources_off_topic=_guided_sources_all_off_topic(sources),
                ),
                "nextActions": _guided_next_actions(
                    search_session_id=search_session_id,
                    status=status,
                    has_sources=bool(sources),
                    all_sources_off_topic=_guided_sources_all_off_topic(sources),
                ),
                "clarification": smart_summary.get("clarification"),
                "resultState": _guided_result_state(
                    status=status,
                    sources=sources,
                    evidence_gaps=evidence_gaps,
                    search_session_id=search_session_id,
                ),
                "executionProvenance": _guided_execution_provenance_payload(
                    execution_mode=("guided_hybrid_research" if len(pass_modes) > 1 else "guided_research"),
                    latency_profile_applied=guided_research_latency_profile,
                    allow_paid_providers=(
                        guided_allow_paid_providers or (escalation_attempted and guided_escalation_allow_paid_providers)
                    ),
                    provider_budget_applied=_guided_provider_budget_payload(
                        allow_paid_providers=(
                            guided_allow_paid_providers
                            or (escalation_attempted and guided_escalation_allow_paid_providers)
                        ),
                    ),
                    strategy_metadata=strategy_metadata,
                    escalation_attempted=escalation_attempted,
                    escalation_reason=escalation_reason,
                    passes_run=len(pass_modes),
                    pass_modes=pass_modes,
                ),
                "inputNormalization": _guided_normalization_payload(research_normalization),
            }
            response.update(contract_fields)
            abstention_details = _guided_abstention_details_payload(
                status=status,
                sources=sources,
                evidence_gaps=evidence_gaps,
                trust_summary=trust_summary,
            )
            if abstention_details is not None:
                response["abstentionDetails"] = abstention_details
            return _guided_finalize_response(tool_name="research", response=response)

        raw = await _dispatch_internal(
            "search_papers",
            {
                "query": research_args.query,
                "limit": research_args.limit,
                "year": research_args.year,
                "venue": ([research_args.venue] if research_args.venue else None),
            },
        )
        sources = [
            _guided_source_record_from_paper(research_args.query, paper, index=index)
            for index, paper in enumerate(raw.get("data") or [], start=1)
            if isinstance(paper, dict)
        ]
        verified_findings = _guided_findings_from_sources(sources)
        evidence_gaps = []
        unverified_leads = _guided_unverified_leads_from_sources(sources)
        trust_summary = _guided_trust_summary(sources, evidence_gaps)
        status, adequacy_reason = await _guided_research_status(
            query=research_args.query,
            intent=intent,
            sources=sources,
            findings=verified_findings,
            unverified_leads_count=len(unverified_leads),
            coverage_summary=cast(dict[str, Any] | None, raw.get("coverageSummary")),
            failure_summary=cast(dict[str, Any] | None, raw.get("failureSummary")),
            clarification=None,
            provider_bundle=None,
        )
        evidence_gaps = await _guided_generate_evidence_gaps(
            query=research_args.query,
            intent=intent,
            sources=sources,
            existing_evidence_gaps=evidence_gaps,
            coverage_summary=cast(dict[str, Any] | None, raw.get("coverageSummary")),
            strategy_metadata=None,
            timeline=None,
            provider_bundle=None,
        )
        failure_summary = _guided_failure_summary(
            failure_summary=cast(dict[str, Any] | None, raw.get("failureSummary")),
            status=status,
            sources=sources,
            evidence_gaps=evidence_gaps,
            all_sources_off_topic=_guided_sources_all_off_topic(sources),
        )
        response = cast(
            dict[str, Any],
            {
                "intent": intent,
                "status": status,
                "searchSessionId": raw.get("searchSessionId"),
                "summary": _guided_summary(intent, status, verified_findings, sources),
                "verifiedFindings": verified_findings,
                "sources": sources,
                "unverifiedLeads": unverified_leads,
                "evidenceGaps": evidence_gaps,
                "trustSummary": trust_summary,
                "coverage": raw.get("coverageSummary"),
                "failureSummary": failure_summary,
                "resultMeaning": _guided_result_meaning(
                    status=status,
                    verified_findings=verified_findings,
                    evidence_gaps=evidence_gaps,
                    coverage=cast(dict[str, Any] | None, raw.get("coverageSummary")),
                    failure_summary=failure_summary,
                    source_count=len(sources),
                    all_sources_off_topic=_guided_sources_all_off_topic(sources),
                ),
                "nextActions": _guided_next_actions(
                    search_session_id=cast(str | None, raw.get("searchSessionId")),
                    status=status,
                    has_sources=bool(sources),
                    all_sources_off_topic=_guided_sources_all_off_topic(sources),
                ),
                "clarification": None,
                "resultState": _guided_result_state(
                    status=status,
                    sources=sources,
                    evidence_gaps=evidence_gaps,
                    search_session_id=cast(str | None, raw.get("searchSessionId")),
                ),
                "executionProvenance": _guided_execution_provenance_payload(
                    execution_mode="guided_raw_broker_fallback",
                    answer_source="search_papers",
                    passes_run=1,
                ),
                "inputNormalization": _guided_normalization_payload(research_normalization),
            },
        )
        response.update(
            await _guided_contract_fields(
                query=research_args.query,
                intent=intent,
                status=status,
                sources=sources,
                unverified_leads=unverified_leads,
                evidence_gaps=evidence_gaps,
                coverage_summary=cast(dict[str, Any] | None, raw.get("coverageSummary")),
                strategy_metadata=None,
            )
        )
        abstention_details = _guided_abstention_details_payload(
            status=status,
            sources=sources,
            evidence_gaps=evidence_gaps,
            trust_summary=trust_summary,
        )
        if abstention_details is not None:
            response["abstentionDetails"] = abstention_details
        return _guided_finalize_response(tool_name="research", response=response)

    if name == "follow_up_research":
        normalized_follow_up_arguments, follow_up_normalization = _guided_normalize_follow_up_arguments(
            arguments,
            workspace_registry=workspace_registry,
        )
        session_resolution = _guided_follow_up_session_resolution(
            arguments=arguments,
            normalized_arguments=normalized_follow_up_arguments,
            normalization=follow_up_normalization,
            workspace_registry=workspace_registry,
        )
        follow_up_args = cast(
            FollowUpResearchArgs,
            TOOL_INPUT_MODELS[name].model_validate(normalized_follow_up_arguments),
        )
        session_state = _guided_session_state(
            workspace_registry=workspace_registry,
            search_session_id=follow_up_args.search_session_id,
        )
        session_answer = await _answer_follow_up_from_session_state(
            question=follow_up_args.question,
            session_state=session_state,
            response_mode=_guided_follow_up_response_mode(follow_up_args.question, {}),
        )
        if session_answer is not None:
            session_answer["inputNormalization"] = _guided_normalization_payload(follow_up_normalization)
            session_answer["sessionResolution"] = session_resolution
            return _guided_finalize_response(
                tool_name="follow_up_research",
                response=session_answer,
                response_mode=follow_up_args.response_mode,
                include_legacy_fields=follow_up_args.include_legacy_fields,
            )
        if not follow_up_args.search_session_id:
            evidence_gaps = ["A unique saved search session could not be identified for this follow-up question."]
            trust_summary = _guided_trust_summary([], evidence_gaps)
            failure_summary = _guided_failure_summary(
                failure_summary={
                    "outcome": "needs_clarification",
                    "whatFailed": "follow_up_session_inference",
                    "whatStillWorked": (
                        "The server preserved a safe failure state instead of binding to the wrong session."
                    ),
                    "fallbackAttempted": False,
                    "fallbackMode": None,
                    "primaryPathFailureReason": "ambiguous_or_missing_search_session",
                    "completenessImpact": evidence_gaps[0],
                    "recommendedNextAction": "research",
                },
                status="partial",
                sources=[],
                evidence_gaps=evidence_gaps,
            )
            response = {
                "searchSessionId": None,
                "answerStatus": "insufficient_evidence",
                "answer": None,
                "evidence": [],
                "unsupportedAsks": [follow_up_args.question],
                "followUpQuestions": [],
                "sources": [],
                "unverifiedLeads": [],
                "verifiedFindings": [],
                "evidenceGaps": evidence_gaps,
                "trustSummary": trust_summary,
                "coverage": None,
                "failureSummary": failure_summary,
                "resultMeaning": _guided_result_meaning(
                    status="partial",
                    verified_findings=[],
                    evidence_gaps=evidence_gaps,
                    coverage=None,
                    failure_summary=failure_summary,
                    source_count=0,
                ),
                "nextActions": [
                    "Provide an explicit searchSessionId from a prior research result.",
                    "Run research again if you need a fresh grounded session to follow up on.",
                ],
                "resultState": _guided_result_state(
                    status="partial",
                    sources=[],
                    evidence_gaps=evidence_gaps,
                    search_session_id=None,
                ),
                "sessionResolution": session_resolution,
                "executionProvenance": _guided_execution_provenance_payload(
                    execution_mode="guided_follow_up",
                    answer_source="none",
                    latency_profile_applied=guided_follow_up_latency_profile,
                    passes_run=0,
                ),
                "inputNormalization": _guided_normalization_payload(follow_up_normalization),
            }
            abstention_details = _guided_abstention_details_payload(
                status="insufficient_evidence",
                sources=[],
                evidence_gaps=evidence_gaps,
                trust_summary=trust_summary,
            )
            if abstention_details is not None:
                response["abstentionDetails"] = abstention_details
            return _guided_finalize_response(
                tool_name="follow_up_research",
                response=response,
                response_mode=follow_up_args.response_mode,
                include_legacy_fields=follow_up_args.include_legacy_fields,
            )
        if agentic_runtime is None:
            evidence_gaps = [follow_up_args.question]
            trust_summary = _guided_trust_summary([], evidence_gaps)
            saved_session_has_sources = False
            saved_session_all_off_topic = False
            if workspace_registry is not None and follow_up_args.search_session_id:
                try:
                    record = workspace_registry.get(follow_up_args.search_session_id)
                    if record is not None:
                        saved_session_has_sources, saved_session_all_off_topic = _guided_saved_session_topicality(
                            _guided_record_source_candidates(record)
                        )
                except Exception:
                    saved_session_has_sources = False
                    saved_session_all_off_topic = False
            saved_session_inspectable = saved_session_has_sources and not saved_session_all_off_topic
            # Sixth rubber-duck pass (finding 1): align failureSummary with the
            # two-bool saved-session signal already feeding resultState so the
            # recommendation does not contradict bestNextInternalAction.
            if saved_session_inspectable:
                what_still_worked = "The saved search session can still be inspected source by source."
                recommended_next_action = "inspect_source"
            elif saved_session_has_sources:
                what_still_worked = (
                    "The saved session exists but every stored candidate was classified off_topic, "
                    "so a fresh research call is the productive recovery path."
                )
                recommended_next_action = "research"
            else:
                what_still_worked = (
                    "No saved grounded evidence is available; run research again to rebuild the session."
                )
                recommended_next_action = "research"
            failure_summary = _guided_failure_summary(
                failure_summary={
                    "outcome": "total_failure",
                    "whatFailed": "Grounded follow-up requires the smart runtime to be enabled.",
                    "whatStillWorked": what_still_worked,
                    "fallbackAttempted": False,
                    "fallbackMode": None,
                    "primaryPathFailureReason": "smart_runtime_unavailable",
                    "completenessImpact": "No grounded synthesis was attempted.",
                    "recommendedNextAction": recommended_next_action,
                },
                status="partial",
                sources=[],
                evidence_gaps=evidence_gaps,
            )
            response = {
                "searchSessionId": follow_up_args.search_session_id,
                "answerStatus": "insufficient_evidence",
                "answer": None,
                "evidence": [],
                "unsupportedAsks": [follow_up_args.question],
                "followUpQuestions": [],
                "sources": [],
                "unverifiedLeads": [],
                "verifiedFindings": [],
                "evidenceGaps": evidence_gaps,
                "trustSummary": trust_summary,
                "coverage": None,
                "failureSummary": failure_summary,
                "resultMeaning": _guided_result_meaning(
                    status="partial",
                    verified_findings=[],
                    evidence_gaps=evidence_gaps,
                    coverage=None,
                    failure_summary=failure_summary,
                    source_count=0,
                ),
                "nextActions": _guided_next_actions(
                    search_session_id=follow_up_args.search_session_id,
                    status="partial",
                    has_sources=False,
                    saved_session_inspectable=saved_session_inspectable,
                ),
                "resultState": _guided_result_state(
                    status="partial",
                    sources=[],
                    evidence_gaps=evidence_gaps,
                    search_session_id=follow_up_args.search_session_id,
                    saved_session_has_sources=saved_session_has_sources,
                    saved_session_all_off_topic=saved_session_all_off_topic,
                ),
                "sessionResolution": session_resolution,
                "executionProvenance": _guided_execution_provenance_payload(
                    execution_mode="guided_follow_up",
                    answer_source="smart_runtime_unavailable",
                    latency_profile_applied=guided_follow_up_latency_profile,
                    passes_run=0,
                ),
                "inputNormalization": _guided_normalization_payload(follow_up_normalization),
            }
            abstention_details = _guided_abstention_details_payload(
                status="insufficient_evidence",
                sources=[],
                evidence_gaps=evidence_gaps,
                trust_summary=trust_summary,
            )
            if abstention_details is not None:
                response["abstentionDetails"] = abstention_details
            return _guided_finalize_response(
                tool_name="follow_up_research",
                response=response,
                response_mode=follow_up_args.response_mode,
                include_legacy_fields=follow_up_args.include_legacy_fields,
            )
        session_strategy_metadata: dict[str, Any] = {}
        if workspace_registry is not None:
            try:
                session_record = workspace_registry.get(follow_up_args.search_session_id)
                metadata = getattr(session_record, "metadata", {})
                if isinstance(metadata, dict) and isinstance(metadata.get("strategyMetadata"), dict):
                    session_strategy_metadata = cast(dict[str, Any], metadata.get("strategyMetadata"))
            except Exception:
                session_strategy_metadata = {}
        follow_up_strategy_metadata = _guided_live_strategy_metadata(
            agentic_runtime=agentic_runtime,
            strategy_metadata=session_strategy_metadata,
            latency_profile=guided_follow_up_latency_profile,
        )
        try:
            follow_up_response_mode = _guided_follow_up_response_mode(
                follow_up_args.question,
                session_strategy_metadata,
            )
            follow_up_answer_mode = _guided_follow_up_answer_mode(
                follow_up_args.question,
                session_strategy_metadata,
            )
            ask = await _dispatch_internal(
                "ask_result_set",
                {
                    "searchSessionId": follow_up_args.search_session_id,
                    "question": follow_up_args.question,
                    "answerMode": follow_up_answer_mode,
                    "latencyProfile": guided_follow_up_latency_profile,
                },
            )
            if not isinstance(ask, dict):
                raise ValueError("ask_result_set returned a non-object payload.")
        except Exception as error:
            failure_saved_has_sources = False
            failure_saved_all_off_topic = False
            if workspace_registry is not None and follow_up_args.search_session_id:
                try:
                    failure_record = workspace_registry.get(follow_up_args.search_session_id)
                    if failure_record is not None:
                        failure_saved_has_sources, failure_saved_all_off_topic = _guided_saved_session_topicality(
                            _guided_record_source_candidates(failure_record)
                        )
                except Exception:
                    failure_saved_has_sources = False
                    failure_saved_all_off_topic = False
            machine_failure = _guided_machine_failure_payload(
                search_session_id=follow_up_args.search_session_id,
                error=error,
                normalization=follow_up_normalization,
                execution_provenance=_guided_execution_provenance_payload(
                    execution_mode="guided_follow_up",
                    answer_source="ask_result_set",
                    latency_profile_applied=guided_follow_up_latency_profile,
                    strategy_metadata=follow_up_strategy_metadata,
                    passes_run=1,
                ),
                saved_session_has_sources=failure_saved_has_sources,
                saved_session_all_off_topic=failure_saved_all_off_topic,
            )
            response = {
                "searchSessionId": follow_up_args.search_session_id,
                "answerStatus": "insufficient_evidence",
                "answer": None,
                "evidence": [],
                "unsupportedAsks": [follow_up_args.question],
                "followUpQuestions": [],
                "verifiedFindings": [],
                "sources": [],
                "unverifiedLeads": [],
                "evidenceGaps": machine_failure.get("evidenceGaps") or [],
                "trustSummary": machine_failure.get("trustSummary") or _guided_trust_summary([], []),
                "coverage": machine_failure.get("coverage"),
                "failureSummary": machine_failure.get("failureSummary"),
                "resultMeaning": machine_failure.get("resultMeaning"),
                "nextActions": machine_failure.get("nextActions")
                or _guided_next_actions(
                    search_session_id=follow_up_args.search_session_id,
                    status="partial",
                    has_sources=False,
                ),
                "resultState": machine_failure.get("resultState")
                or _guided_result_state(
                    status="partial",
                    sources=[],
                    evidence_gaps=list(machine_failure.get("evidenceGaps") or []),
                    search_session_id=follow_up_args.search_session_id,
                ),
                "machineFailure": machine_failure.get("machineFailure"),
                "sessionResolution": session_resolution,
                "executionProvenance": machine_failure.get("executionProvenance")
                or _guided_execution_provenance_payload(
                    execution_mode="guided_follow_up",
                    answer_source="ask_result_set",
                    latency_profile_applied=guided_follow_up_latency_profile,
                    strategy_metadata=follow_up_strategy_metadata,
                    passes_run=1,
                ),
                "inputNormalization": machine_failure.get("inputNormalization")
                or _guided_normalization_payload(follow_up_normalization),
            }
            abstention_details = _guided_abstention_details_payload(
                status="insufficient_evidence",
                sources=[],
                evidence_gaps=cast(list[str], machine_failure.get("evidenceGaps") or []),
                trust_summary=cast(
                    dict[str, Any], machine_failure.get("trustSummary") or _guided_trust_summary([], [])
                ),
            )
            if abstention_details is not None:
                response["abstentionDetails"] = abstention_details
            return _guided_finalize_response(
                tool_name="follow_up_research",
                response=response,
                response_mode=follow_up_args.response_mode,
                include_legacy_fields=follow_up_args.include_legacy_fields,
            )
        raw_selected_evidence_ids = [
            str(identifier).strip() for identifier in (ask.get("selectedEvidenceIds") or []) if str(identifier).strip()
        ]
        raw_selected_lead_ids = [
            str(identifier).strip() for identifier in (ask.get("selectedLeadIds") or []) if str(identifier).strip()
        ]
        sources = [
            _guided_source_record_from_structured_source(source, index=index)
            for index, source in enumerate(ask.get("structuredSources") or [], start=1)
            if isinstance(source, dict)
        ]
        evidence_gaps = list(ask.get("evidenceGaps") or [])
        unverified_leads = [
            _guided_source_record_from_structured_source(source, index=index)
            for index, source in enumerate(ask.get("candidateLeads") or [], start=1)
            if isinstance(source, dict)
        ] or _guided_unverified_leads_from_sources(sources)
        session_sources = [source for source in (session_state or {}).get("sources") or [] if isinstance(source, dict)]
        session_leads = [lead for lead in (session_state or {}).get("unverifiedLeads") or [] if isinstance(lead, dict)]
        if session_sources:
            sources = _guided_enrich_records_from_saved_session(sources, session_sources)
            sources = _guided_append_selected_saved_records(
                sources,
                session_sources,
                raw_selected_evidence_ids,
            )
        if session_leads:
            unverified_leads = _guided_enrich_records_from_saved_session(unverified_leads, session_leads)
            unverified_leads = _guided_append_selected_saved_records(
                unverified_leads,
                session_leads,
                raw_selected_lead_ids,
            )
        visible_source_ids = {
            str(source.get("sourceId") or source.get("sourceAlias") or "").strip()
            for source in sources
            if str(source.get("sourceId") or source.get("sourceAlias") or "").strip()
        }
        visible_lead_ids = {
            str(lead.get("sourceId") or lead.get("sourceAlias") or "").strip()
            for lead in unverified_leads
            if str(lead.get("sourceId") or lead.get("sourceAlias") or "").strip()
        }
        selected_evidence_ids = [
            identifier for identifier in raw_selected_evidence_ids if identifier in visible_source_ids
        ]
        selected_lead_ids = [identifier for identifier in raw_selected_lead_ids if identifier in visible_lead_ids]
        follow_up_coverage_summary = _guided_source_coverage_summary(
            sources=sources,
            leads=unverified_leads,
            base_coverage=cast(dict[str, Any] | None, ask.get("coverageSummary")),
        )
        answer_status = str(ask.get("answerStatus") or "answered")
        answer_text = ask.get("answer")
        answer_is_responsive = answer_status != "answered" or _guided_metadata_answer_is_responsive(
            question=follow_up_args.question,
            answer_text=answer_text,
            sources=sources,
            leads=unverified_leads,
            selected_evidence_ids=selected_evidence_ids,
            selected_lead_ids=selected_lead_ids,
        )
        if answer_status == "answered" and (not _guided_is_usable_answer_text(answer_text) or not answer_is_responsive):
            if session_answer is not None:
                session_answer["inputNormalization"] = _guided_normalization_payload(follow_up_normalization)
                session_answer["sessionResolution"] = session_resolution
                return _guided_finalize_response(
                    tool_name="follow_up_research",
                    response=session_answer,
                    response_mode=follow_up_args.response_mode,
                    include_legacy_fields=follow_up_args.include_legacy_fields,
                )
            answer_status = "insufficient_evidence"
            answer_text = None
        guided_status = "partial" if answer_status == "insufficient_evidence" else answer_status
        verified_findings = _guided_findings_from_sources(sources)
        trust_summary = _guided_trust_summary(
            sources,
            evidence_gaps,
            classification_provenance=cast(dict[str, Any] | None, ask.get("classificationProvenance")),
            subject_chain_gaps=cast(
                list[str] | None,
                (follow_up_strategy_metadata or {}).get("subjectChainGaps"),
            ),
        )
        # Seventh rubber-duck pass (finding 1): when the current ask_result_set
        # response has no structuredSources but the saved search session still
        # holds inspectable candidates, failureSummary/nextActions/resultState
        # must route to inspect_source instead of research — matching the
        # no-runtime and exception branches and bestNextInternalAction.
        # Ninth rubber-duck pass (finding 4): also honor the saved-session
        # signal when the current response HAS sources but every one is
        # off_topic. Without this, an all-off-topic current pool strands the
        # saved session and routes to research even though inspect_source can
        # still reach on-topic saved candidates.
        follow_up_all_off_topic = _guided_sources_all_off_topic(sources)
        follow_up_saved_has_sources = False
        follow_up_saved_all_off_topic = False
        if (
            (not sources or follow_up_all_off_topic)
            and workspace_registry is not None
            and follow_up_args.search_session_id
        ):
            try:
                _follow_up_saved_record = workspace_registry.get(follow_up_args.search_session_id)
                if _follow_up_saved_record is not None:
                    follow_up_saved_has_sources, follow_up_saved_all_off_topic = _guided_saved_session_topicality(
                        _guided_record_source_candidates(_follow_up_saved_record)
                    )
            except Exception:
                follow_up_saved_has_sources = False
                follow_up_saved_all_off_topic = False
        follow_up_saved_inspectable = follow_up_saved_has_sources and not follow_up_saved_all_off_topic
        failure_summary = _guided_failure_summary(
            failure_summary=cast(dict[str, Any] | None, ask.get("failureSummary")),
            status=guided_status,
            sources=sources,
            evidence_gaps=evidence_gaps,
            all_sources_off_topic=follow_up_all_off_topic,
        )
        # When current sources are empty/all-off-topic but the saved session is
        # still inspectable, prefer inspect_source over the default "research"
        # recommendation so the summary agrees with resultState routing.
        if (
            follow_up_saved_inspectable
            and (not sources or follow_up_all_off_topic)
            and failure_summary.get("recommendedNextAction") == "research"
        ):
            failure_summary["recommendedNextAction"] = "inspect_source"
        response = cast(
            dict[str, Any],
            {
                "searchSessionId": follow_up_args.search_session_id,
                "answerStatus": answer_status,
                "answer": answer_text,
                "providerUsed": ask.get("providerUsed"),
                "degradationReason": ask.get("degradationReason"),
                "evidenceUsePlan": ask.get("evidenceUsePlan"),
                "evidence": ask.get("evidence") or [],
                "unsupportedAsks": ask.get("unsupportedAsks") or [],
                "followUpQuestions": ask.get("followUpQuestions") or [],
                "verifiedFindings": verified_findings,
                "sources": sources,
                "unverifiedLeads": unverified_leads,
                "evidenceGaps": evidence_gaps,
                "trustSummary": trust_summary,
                "coverage": follow_up_coverage_summary,
                "failureSummary": failure_summary,
                "resultMeaning": _guided_result_meaning(
                    status=guided_status,
                    verified_findings=verified_findings,
                    evidence_gaps=evidence_gaps,
                    coverage=follow_up_coverage_summary,
                    failure_summary=failure_summary,
                    source_count=len(sources),
                    all_sources_off_topic=follow_up_all_off_topic,
                ),
                "nextActions": _guided_next_actions(
                    search_session_id=follow_up_args.search_session_id,
                    status=guided_status,
                    has_sources=bool(sources),
                    saved_session_inspectable=follow_up_saved_inspectable,
                    all_sources_off_topic=follow_up_all_off_topic,
                ),
                "resultState": _guided_result_state(
                    status=guided_status,
                    sources=sources,
                    evidence_gaps=evidence_gaps,
                    search_session_id=follow_up_args.search_session_id,
                    saved_session_has_sources=follow_up_saved_has_sources,
                    saved_session_all_off_topic=follow_up_saved_all_off_topic,
                ),
                "sessionResolution": session_resolution,
                "executionProvenance": _guided_execution_provenance_payload(
                    execution_mode="guided_follow_up",
                    answer_source="ask_result_set",
                    latency_profile_applied=guided_follow_up_latency_profile,
                    strategy_metadata=follow_up_strategy_metadata,
                    passes_run=1,
                ),
                "inputNormalization": _guided_normalization_payload(follow_up_normalization),
                "confidenceSignals": _guided_confidence_signals(
                    status=guided_status,
                    sources=sources,
                    evidence_gaps=evidence_gaps,
                    degradation_reason=cast(str | None, ask.get("degradationReason")),
                    synthesis_mode=(
                        "session_introspection"
                        if follow_up_response_mode in {"metadata", "relevance_triage"}
                        else "grounded_follow_up"
                    ),
                    evidence_use_plan_applied=bool(ask.get("evidenceUsePlan")),
                    subject_chain_gaps=cast(
                        list[str] | None,
                        (follow_up_strategy_metadata or {}).get("subjectChainGaps"),
                    ),
                ),
            },
        )
        if ask.get("agentHints") is not None:
            response["agentHints"] = ask.get("agentHints")
        response.update(
            await _guided_contract_fields(
                query=follow_up_args.question,
                intent=str(
                    (session_strategy_metadata or {}).get("intent")
                    or (session_state or {}).get("intent")
                    or "discovery"
                ),
                status=guided_status,
                sources=sources,
                unverified_leads=unverified_leads,
                evidence_gaps=evidence_gaps,
                coverage_summary=follow_up_coverage_summary,
                strategy_metadata=follow_up_strategy_metadata,
            )
        )
        response["confidenceSignals"] = _guided_confidence_signals(
            status=guided_status,
            sources=sources,
            evidence_gaps=evidence_gaps,
            degradation_reason=cast(str | None, ask.get("degradationReason")),
            synthesis_mode=(
                "session_introspection"
                if follow_up_response_mode in {"metadata", "relevance_triage"}
                else "grounded_follow_up"
            ),
            evidence_use_plan_applied=bool(ask.get("evidenceUsePlan")),
            subject_chain_gaps=cast(
                list[str] | None,
                (follow_up_strategy_metadata or {}).get("subjectChainGaps"),
            ),
        )
        response["selectedEvidenceIds"] = selected_evidence_ids
        response["selectedLeadIds"] = selected_lead_ids
        if ask.get("topRecommendation") is not None:
            top_recommendation = ask.get("topRecommendation")
            if isinstance(top_recommendation, dict):
                recommendation_source_id = str(top_recommendation.get("sourceId") or "").strip()
                if recommendation_source_id and recommendation_source_id in visible_source_ids:
                    response["topRecommendation"] = top_recommendation
            else:
                response["topRecommendation"] = top_recommendation
        if ask.get("structuredSources") is not None:
            response["structuredSources"] = ask.get("structuredSources")
        if answer_status == "insufficient_evidence":
            response["resultMeaning"] = _guided_result_meaning(
                status="partial",
                verified_findings=verified_findings,
                evidence_gaps=evidence_gaps,
                coverage=follow_up_coverage_summary,
                failure_summary=failure_summary,
                source_count=len(sources),
                all_sources_off_topic=follow_up_all_off_topic,
            )
            response["resultState"] = _guided_result_state(
                status="partial",
                sources=sources,
                evidence_gaps=evidence_gaps,
                search_session_id=follow_up_args.search_session_id,
                saved_session_has_sources=follow_up_saved_has_sources,
                saved_session_all_off_topic=follow_up_saved_all_off_topic,
            )
            response["resultStatus"] = "partial"
            response["answerability"] = classify_answerability(
                status="partial",
                evidence=cast(list[dict[str, Any]], response.get("evidence") or []),
                leads=unverified_leads,
                evidence_gaps=evidence_gaps,
                answer_text="",
                evidence_quality_profile=str(
                    cast(dict[str, Any], response.get("confidenceSignals") or {}).get("evidenceQualityProfile") or "low"
                ),
            )
            abstention_details = _guided_abstention_details_payload(
                status="insufficient_evidence",
                sources=sources,
                evidence_gaps=evidence_gaps,
                trust_summary=trust_summary,
            )
            if abstention_details is not None:
                response["abstentionDetails"] = abstention_details
            return _guided_finalize_response(
                tool_name="follow_up_research",
                response=response,
                response_mode=follow_up_args.response_mode,
                include_legacy_fields=follow_up_args.include_legacy_fields,
            )
        if answer_status != "answered" and session_answer is not None:
            session_answer["inputNormalization"] = _guided_normalization_payload(follow_up_normalization)
            session_answer["sessionResolution"] = session_resolution
            return _guided_finalize_response(
                tool_name="follow_up_research",
                response=session_answer,
                response_mode=follow_up_args.response_mode,
                include_legacy_fields=follow_up_args.include_legacy_fields,
            )
        abstention_details = _guided_abstention_details_payload(
            status=answer_status,
            sources=sources,
            evidence_gaps=evidence_gaps,
            trust_summary=trust_summary,
        )
        if abstention_details is not None:
            response["abstentionDetails"] = abstention_details
        return _guided_finalize_response(
            tool_name="follow_up_research",
            response=response,
            response_mode=follow_up_args.response_mode,
            include_legacy_fields=follow_up_args.include_legacy_fields,
        )

    if name == "inspect_source":
        normalized_inspect_arguments, inspect_normalization = _guided_normalize_inspect_arguments(
            arguments,
            workspace_registry=workspace_registry,
        )
        session_resolution = _guided_inspect_session_resolution(
            arguments=arguments,
            normalized_arguments=normalized_inspect_arguments,
            normalization=inspect_normalization,
            workspace_registry=workspace_registry,
        )
        inspect_args = cast(InspectSourceArgs, TOOL_INPUT_MODELS[name].model_validate(normalized_inspect_arguments))
        if not inspect_args.search_session_id:
            evidence_gaps = [
                "inspect_source could not infer a unique searchSessionId. "
                "Provide an explicit searchSessionId from a prior research result."
            ]
            trust_summary = _guided_trust_summary([], evidence_gaps)
            failure_summary = _guided_failure_summary(
                failure_summary={
                    "outcome": "needs_clarification",
                    "whatFailed": "inspect_source_session_inference",
                    "whatStillWorked": "The server returned candidate sessions instead of selecting the wrong one.",
                    "fallbackAttempted": False,
                    "fallbackMode": None,
                    "primaryPathFailureReason": "ambiguous_or_missing_search_session",
                    "completenessImpact": evidence_gaps[0],
                    "recommendedNextAction": "research",
                },
                status="needs_disambiguation",
                sources=[],
                evidence_gaps=evidence_gaps,
            )
            response = {
                "searchSessionId": None,
                "source": None,
                "directReadRecommendations": [],
                "nextActions": [
                    "Provide an explicit searchSessionId from a prior research result.",
                    "Run research again if you need a fresh grounded session to inspect.",
                ],
                "failureSummary": failure_summary,
                "resultMeaning": _guided_result_meaning(
                    status="needs_disambiguation",
                    verified_findings=[],
                    evidence_gaps=evidence_gaps,
                    coverage=None,
                    failure_summary=failure_summary,
                    source_count=0,
                ),
                "resultState": _guided_result_state(
                    status="needs_disambiguation",
                    sources=[],
                    evidence_gaps=evidence_gaps,
                    search_session_id=None,
                ),
                "sessionResolution": session_resolution,
                "sourceResolution": _guided_source_resolution_payload(
                    requested_source_id=inspect_args.source_id,
                    resolved_source_id=None,
                    match_type="missing_session_id",
                ),
                "executionProvenance": _guided_execution_provenance_payload(
                    execution_mode="guided_source_inspection",
                    answer_source="saved_session_source",
                    passes_run=0,
                ),
                "inputNormalization": _guided_normalization_payload(inspect_normalization),
            }
            abstention_details = _guided_abstention_details_payload(
                status="needs_disambiguation",
                sources=[],
                evidence_gaps=evidence_gaps,
                trust_summary=trust_summary,
            )
            if abstention_details is not None:
                response["abstentionDetails"] = abstention_details
            return response
        source, match_type = _find_record_source_with_resolution(
            workspace_registry=workspace_registry,
            search_session_id=inspect_args.search_session_id,
            source_id=inspect_args.source_id,
        )
        if source is None:
            available_ids: list[str] = []
            saved_candidates: list[dict[str, Any]] = []
            if workspace_registry is not None and inspect_args.search_session_id:
                try:
                    record = workspace_registry.get(inspect_args.search_session_id)
                    saved_candidates = _guided_record_source_candidates(record)
                    available_ids = [
                        str(candidate.get("sourceId") or "").strip()
                        for candidate in saved_candidates
                        if str(candidate.get("sourceId") or "").strip()
                    ][:8]
                except Exception:
                    available_ids = []
                    saved_candidates = []
            saved_has_sources, saved_all_off_topic = _guided_saved_session_topicality(saved_candidates)
            saved_inspectable = (
                saved_has_sources
                and not saved_all_off_topic
                and any(_candidate_is_inspectable(candidate) for candidate in saved_candidates)
            )
            evidence_gaps = [
                "Could not find sourceId "
                f"{inspect_args.source_id!r} in searchSessionId "
                f"{inspect_args.search_session_id!r}."
            ]
            trust_summary = _guided_trust_summary([], evidence_gaps)
            # Sixth rubber-duck pass (finding 1): when the saved session has no
            # inspectable candidates (empty or all off_topic), recommend
            # research instead of inspect_source so the failure summary agrees
            # with resultState.bestNextInternalAction.
            if saved_inspectable:
                recommended_next_action = "inspect_source"
                what_still_worked = "The server returned available source IDs for explicit retry."
                next_actions = [
                    "Provide an exact sourceId from the saved session.",
                    "Use inspect_source again after choosing one of the available source IDs.",
                ]
            else:
                recommended_next_action = "research"
                what_still_worked = (
                    "The saved session has no inspectable on-topic candidates; "
                    "run research again to rebuild grounded evidence."
                )
                next_actions = [
                    "Use research to rebuild a grounded session with on-topic sources before inspecting.",
                ]
            failure_summary = _guided_failure_summary(
                failure_summary={
                    "outcome": "needs_clarification",
                    "whatFailed": "inspect_source_source_resolution",
                    "whatStillWorked": what_still_worked,
                    "fallbackAttempted": False,
                    "fallbackMode": None,
                    "primaryPathFailureReason": match_type,
                    "completenessImpact": evidence_gaps[0],
                    "recommendedNextAction": recommended_next_action,
                },
                status="needs_disambiguation",
                sources=[],
                evidence_gaps=evidence_gaps,
            )
            response = {
                "searchSessionId": inspect_args.search_session_id,
                "source": None,
                "directReadRecommendations": [],
                "nextActions": next_actions,
                "failureSummary": failure_summary,
                "resultMeaning": _guided_result_meaning(
                    status="needs_disambiguation",
                    verified_findings=[],
                    evidence_gaps=evidence_gaps,
                    coverage=None,
                    failure_summary=failure_summary,
                    source_count=0,
                ),
                "resultState": _guided_result_state(
                    status="needs_disambiguation",
                    sources=[],
                    evidence_gaps=evidence_gaps,
                    search_session_id=inspect_args.search_session_id,
                    saved_session_has_sources=saved_has_sources,
                    saved_session_all_off_topic=saved_all_off_topic,
                    saved_session_inspectable_override=saved_inspectable,
                ),
                "sessionResolution": session_resolution,
                "sourceResolution": _guided_source_resolution_payload(
                    requested_source_id=inspect_args.source_id,
                    resolved_source_id=None,
                    match_type=match_type,
                    available_source_ids=available_ids,
                    available_candidates=saved_candidates[:8],
                    candidates_have_inspectable=saved_inspectable,
                ),
                "executionProvenance": _guided_execution_provenance_payload(
                    execution_mode="guided_source_inspection",
                    answer_source="saved_session_source",
                    passes_run=0,
                ),
                "inputNormalization": _guided_normalization_payload(inspect_normalization),
            }
            abstention_details = _guided_abstention_details_payload(
                status="needs_disambiguation",
                sources=[],
                evidence_gaps=evidence_gaps,
                trust_summary=trust_summary,
            )
            if abstention_details is not None:
                response["abstentionDetails"] = abstention_details
            return response
        inspect_session_state = _guided_session_state(
            workspace_registry=workspace_registry,
            search_session_id=inspect_args.search_session_id,
        )
        inspect_strategy_metadata = cast(dict[str, Any] | None, (inspect_session_state or {}).get("strategyMetadata"))
        why_classified_weak = _compose_why_classified_weak_match(source, strategy_metadata=inspect_strategy_metadata)
        inspect_response: dict[str, Any] = {
            "searchSessionId": inspect_args.search_session_id,
            "source": source,
            "evidenceId": source.get("sourceId"),
            "selectedEvidenceIds": [str(source.get("sourceId") or "").strip()],
            "directReadRecommendations": _direct_read_recommendations(source, tool_profile=tool_profile),
            "directReadRecommendationDetails": _direct_read_recommendation_details(source, tool_profile=tool_profile),
            "confidenceSignals": _guided_confidence_signals(
                status="succeeded",
                sources=[source],
                evidence_gaps=[],
                synthesis_mode="source_audit",
                source=source,
                subject_chain_gaps=cast(
                    list[str] | None,
                    (inspect_strategy_metadata or {}).get("subjectChainGaps"),
                ),
            ),
            "nextActions": _guided_next_actions(
                search_session_id=inspect_args.search_session_id,
                status="succeeded",
                has_sources=True,
                calling_tool="inspect_source",
            ),
            "sessionResolution": session_resolution,
            "sourceResolution": _guided_source_resolution_payload(
                requested_source_id=inspect_args.source_id,
                resolved_source_id=cast(str | None, source.get("sourceId")),
                match_type=match_type,
            ),
            "resultState": _guided_result_state(
                status="succeeded",
                sources=[source],
                evidence_gaps=[],
                search_session_id=inspect_args.search_session_id,
            ),
            "executionProvenance": _guided_execution_provenance_payload(
                execution_mode="guided_source_inspection",
                answer_source="saved_session_source",
                passes_run=0,
            ),
            **(
                await _guided_contract_fields(
                    query=str((inspect_session_state or {}).get("query") or ""),
                    intent=str((inspect_session_state or {}).get("intent") or "discovery"),
                    status="succeeded",
                    sources=[source],
                    unverified_leads=[],
                    evidence_gaps=[],
                    coverage_summary=cast(dict[str, Any] | None, (inspect_session_state or {}).get("coverage")),
                    strategy_metadata=cast(
                        dict[str, Any] | None, (inspect_session_state or {}).get("strategyMetadata")
                    ),
                    timeline=cast(dict[str, Any] | None, (inspect_session_state or {}).get("timeline")),
                )
            ),
            "inputNormalization": _guided_normalization_payload(inspect_normalization),
        }
        if why_classified_weak:
            inspect_response["whyClassifiedAsWeakMatch"] = why_classified_weak
        return _apply_inspect_source_compaction(inspect_response)

    if name == "search_papers_smart":
        return await _dispatch_search_papers_smart(dispatch_ctx, arguments)

    if name == "ask_result_set":
        return await _dispatch_ask_result_set(dispatch_ctx, arguments)

    if name == "map_research_landscape":
        return await _dispatch_map_research_landscape(dispatch_ctx, arguments)

    if name == "expand_research_graph":
        return await _dispatch_expand_research_graph(dispatch_ctx, arguments)

    if name == "get_provider_diagnostics":
        validated_payload = TOOL_INPUT_MODELS[name].model_validate(arguments)
        args_dict = validated_payload.model_dump(by_alias=False)
        return _build_provider_diagnostics_snapshot(
            include_recent_outcomes=bool(args_dict.get("include_recent_outcomes", True)),
            provider_order=provider_order,
            provider_registry=provider_registry,
            agentic_runtime=agentic_runtime,
            transport_mode=transport_mode,
            tool_profile=tool_profile,
            hide_disabled_tools=hide_disabled_tools,
            session_ttl_seconds=session_ttl_seconds,
            embeddings_enabled=embeddings_enabled,
            guided_research_latency_profile=guided_research_latency_profile,
            guided_follow_up_latency_profile=guided_follow_up_latency_profile,
            guided_allow_paid_providers=guided_allow_paid_providers,
            guided_escalation_enabled=guided_escalation_enabled,
            guided_escalation_max_passes=guided_escalation_max_passes,
            guided_escalation_allow_paid_providers=guided_escalation_allow_paid_providers,
            enable_core=enable_core,
            enable_semantic_scholar=enable_semantic_scholar,
            enable_openalex=enable_openalex,
            enable_arxiv=enable_arxiv,
            enable_serpapi=enable_serpapi,
            enable_scholarapi=enable_scholarapi,
            enable_crossref=enable_crossref,
            enable_unpaywall=enable_unpaywall,
            enable_ecos=enable_ecos,
            enable_federal_register=enable_federal_register,
            enable_govinfo_cfr=enable_govinfo_cfr,
            ecos_client=ecos_client,
            serpapi_client=serpapi_client,
            scholarapi_client=scholarapi_client,
        )

    if name == "search_species_ecos":
        return await _dispatch_search_species_ecos(dispatch_ctx, arguments)

    if name == "get_species_profile_ecos":
        return await _dispatch_get_species_profile_ecos(dispatch_ctx, arguments)

    if name == "list_species_documents_ecos":
        return await _dispatch_list_species_documents_ecos(dispatch_ctx, arguments)

    if name == "get_document_text_ecos":
        return await _dispatch_get_document_text_ecos(dispatch_ctx, arguments)

    if name == "search_federal_register":
        return await _dispatch_search_federal_register(dispatch_ctx, arguments)

    if name == "get_federal_register_document":
        return await _dispatch_get_federal_register_document(dispatch_ctx, arguments)

    if name == "get_cfr_text":
        return await _dispatch_get_cfr_text(dispatch_ctx, arguments)

    if name == "get_paper_metadata_crossref":
        return await _dispatch_get_paper_metadata_crossref(dispatch_ctx, arguments)

    if name == "get_paper_open_access_unpaywall":
        return await _dispatch_get_paper_open_access_unpaywall(dispatch_ctx, arguments)

    if name == "enrich_paper":
        return await _dispatch_enrich_paper(dispatch_ctx, arguments)

    if name == "search_papers":
        return await _dispatch_search_papers(dispatch_ctx, arguments)

    if name == "search_papers_match":
        return await _dispatch_search_papers_match(dispatch_ctx, arguments)

    if name == "get_paper_details":
        return await _dispatch_get_paper_details(dispatch_ctx, arguments)

    if name == "resolve_citation":
        return await _dispatch_resolve_citation(dispatch_ctx, arguments)

    if name == "paper_autocomplete_openalex":
        return await _dispatch_paper_autocomplete_openalex(dispatch_ctx, arguments)

    if name == "search_papers_openalex":
        return await _dispatch_search_papers_openalex(dispatch_ctx, arguments)

    if name == "search_papers_scholarapi":
        return await _dispatch_search_papers_scholarapi(dispatch_ctx, arguments)

    if name == "list_papers_scholarapi":
        return await _dispatch_list_papers_scholarapi(dispatch_ctx, arguments)

    if name == "get_paper_text_scholarapi":
        return await _dispatch_get_paper_text_scholarapi(dispatch_ctx, arguments)

    if name == "get_paper_texts_scholarapi":
        return await _dispatch_get_paper_texts_scholarapi(dispatch_ctx, arguments)

    if name == "get_paper_pdf_scholarapi":
        return await _dispatch_get_paper_pdf_scholarapi(dispatch_ctx, arguments)

    if name == "search_entities_openalex":
        return await _dispatch_search_entities_openalex(dispatch_ctx, arguments)

    if name == "search_papers_openalex_by_entity":
        return await _dispatch_search_papers_openalex_by_entity(dispatch_ctx, arguments)

    if name == "search_papers_openalex_bulk":
        return await _dispatch_search_papers_openalex_bulk(dispatch_ctx, arguments)

    if name == "search_papers_serpapi_cited_by":
        return await _dispatch_search_papers_serpapi_cited_by(dispatch_ctx, arguments)

    if name == "search_papers_serpapi_versions":
        return await _dispatch_search_papers_serpapi_versions(dispatch_ctx, arguments)

    if name == "get_author_profile_serpapi":
        return await _dispatch_get_author_profile_serpapi(dispatch_ctx, arguments)

    if name == "get_author_articles_serpapi":
        return await _dispatch_get_author_articles_serpapi(dispatch_ctx, arguments)

    if name == "get_serpapi_account_status":
        return await _dispatch_get_serpapi_account_status(dispatch_ctx, arguments)

    if name in PROVIDER_SEARCH_TOOLS:
        return await _dispatch_provider_search_tool(dispatch_ctx, arguments, name=name)

    if name == "get_paper_citation_formats":
        return await _dispatch_get_paper_citation_formats(dispatch_ctx, arguments)

    if name == "search_papers_bulk":
        return await _dispatch_search_papers_bulk(dispatch_ctx, arguments)

    if not enable_openalex and name.endswith("_openalex"):
        raise ValueError(
            f"{name} requires OpenAlex, which is disabled. Set PAPER_CHASER_ENABLE_OPENALEX=true to use this tool."
        )

    if name == "get_paper_details_openalex":
        return await _dispatch_get_paper_details_openalex(dispatch_ctx, arguments)

    if name == "get_paper_citations_openalex":
        return await _dispatch_get_paper_citations_openalex(dispatch_ctx, arguments)

    if name == "get_paper_references_openalex":
        return await _dispatch_get_paper_references_openalex(dispatch_ctx, arguments)

    if name == "search_authors_openalex":
        return await _dispatch_search_authors_openalex(dispatch_ctx, arguments)

    if name == "get_author_info_openalex":
        return await _dispatch_get_author_info_openalex(dispatch_ctx, arguments)

    if name == "get_author_papers_openalex":
        return await _dispatch_get_author_papers_openalex(dispatch_ctx, arguments)

    return await _dispatch_non_search_tool(dispatch_ctx, arguments, name=name)


def _finalize_tool_result(
    tool_name: str,
    arguments: dict[str, Any],
    result: Any,
    *,
    workspace_registry: Any,
) -> dict[str, Any]:
    """Add compatibility metadata to raw tool responses."""
    if not isinstance(result, dict):
        return dump_jsonable(result)
    if (
        tool_name in SMART_TOOLS
        or tool_name
        in {
            "get_provider_diagnostics",
            "get_serpapi_account_status",
        }
        or workspace_registry is None
    ):
        return result
    return augment_tool_result(
        tool_name=tool_name,
        arguments=arguments,
        result=result,
        workspace_registry=workspace_registry,
    )


async def _maybe_elicit_and_retry(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
    client: Any,
    core_client: Any,
    openalex_client: Any,
    arxiv_client: Any,
    serpapi_client: Any,
    scholarapi_client: Any,
    crossref_client: Any,
    unpaywall_client: Any,
    ecos_client: Any,
    federal_register_client: Any,
    govinfo_client: Any,
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
    provider_order: list[SearchProvider] | None,
    provider_registry: Any,
    workspace_registry: Any,
    enrichment_service: PaperEnrichmentService | None,
    agentic_runtime: Any,
    ctx: Any,
    allow_elicitation: bool,
) -> dict[str, Any] | None:
    """Use a single bounded elicitation to refine ambiguous raw-tool requests."""
    if not allow_elicitation:
        return None
    if tool_name not in {"search_papers", "search_papers_match", "search_authors"}:
        return None
    if ctx is None or not _client_supports_extension(ctx, "elicitation"):
        return None

    clarification = build_clarification(tool_name, arguments, result)
    if clarification is None:
        return None

    try:
        elicitation = await ctx.elicit(
            _elicitation_message(tool_name, clarification),
            str,
        )
    except Exception:
        return None

    if getattr(elicitation, "action", None) != "accept":
        return None

    refinement = str(getattr(elicitation, "data", "") or "").strip()
    if not refinement:
        return None

    if tool_name == "search_papers_match" and looks_like_paper_identifier(refinement):
        resolved = await client.get_paper_details(
            paper_id=refinement,
            fields=arguments.get("fields"),
        )
        resolved_result = dict(dump_jsonable(resolved))
        if arguments.get("includeEnrichment") and enrichment_service is not None:
            enrichment_source = await hydrate_paper_for_enrichment(
                resolved_result,
                detail_client=client,
            )
            enriched_payload = await enrichment_service.enrich_paper_payload(
                enrichment_source,
                query=resolved_result.get("title"),
            )
            resolved_result = attach_enrichments_to_paper_payload(
                resolved_result,
                enriched_paper=enriched_payload,
            )
        resolved_result["matchFound"] = True
        resolved_result["matchStrategy"] = "elicited_identifier"
        resolved_result["normalizedQuery"] = refinement
        return _finalize_tool_result(
            "search_papers_match",
            {
                "query": refinement,
                "fields": arguments.get("fields"),
                "includeEnrichment": arguments.get("includeEnrichment", False),
            },
            resolved_result,
            workspace_registry=workspace_registry,
        )

    retry_arguments = dict(arguments)
    retry_arguments["query"] = _refined_query(
        original_query=str(arguments.get("query") or ""),
        refinement=refinement,
    )
    return await dispatch_tool(
        tool_name,
        retry_arguments,
        client=client,
        core_client=core_client,
        openalex_client=openalex_client,
        arxiv_client=arxiv_client,
        scholarapi_client=scholarapi_client,
        enable_core=enable_core,
        enable_semantic_scholar=enable_semantic_scholar,
        enable_openalex=enable_openalex,
        enable_arxiv=enable_arxiv,
        serpapi_client=serpapi_client,
        enable_serpapi=enable_serpapi,
        enable_scholarapi=enable_scholarapi,
        crossref_client=crossref_client,
        unpaywall_client=unpaywall_client,
        ecos_client=ecos_client,
        federal_register_client=federal_register_client,
        govinfo_client=govinfo_client,
        enable_crossref=enable_crossref,
        enable_unpaywall=enable_unpaywall,
        enable_ecos=enable_ecos,
        enable_federal_register=enable_federal_register,
        enable_govinfo_cfr=enable_govinfo_cfr,
        enrichment_service=enrichment_service,
        provider_order=provider_order,
        provider_registry=provider_registry,
        workspace_registry=workspace_registry,
        agentic_runtime=agentic_runtime,
        ctx=ctx,
        allow_elicitation=False,
    )


def _client_supports_extension(ctx: Any, extension_id: str) -> bool:
    try:
        return bool(ctx.client_supports_extension(extension_id))
    except Exception:
        return False


def _elicitation_message(tool_name: str, clarification: Any) -> str:
    options = getattr(clarification, "options", None) or []
    option_text = f" Options: {', '.join(options)}." if options else ""
    if tool_name == "search_papers":
        return (
            "This concept query is broad. Reply with one short focus to refine it "
            "(for example: method focus, application focus, recent work only)."
            f"{option_text}"
        )
    if tool_name == "search_authors":
        return (
            "Several authors may match this name. Reply with one short affiliation, "
            "coauthor, venue, or topic clue to narrow the author search."
            f"{option_text}"
        )
    return (
        "This title-only lookup needs one tighter clue. Reply with a refined title "
        "fragment or paste a DOI, arXiv ID, or URL."
        f"{option_text}"
    )


def _refined_query(*, original_query: str, refinement: str) -> str:
    normalized_original = original_query.strip()
    normalized_refinement = refinement.strip()
    if not normalized_original:
        return normalized_refinement
    if not normalized_refinement:
        return normalized_original
    if normalized_refinement.lower() in normalized_original.lower():
        return normalized_original
    return f"{normalized_original} {normalized_refinement}".strip()


# ---------------------------------------------------------------------------
# Phase 2 Track C amendment 3: branch entrypoints accepting ``DispatchContext``.
# ---------------------------------------------------------------------------
#
# These four module-level async functions are the Phase 3 seam for relocating
# ``dispatch_tool``'s largest branches into sibling submodules. Each entrypoint
# accepts the frozen ``DispatchContext`` bag plus the already-normalized
# ``arguments`` dict, so Phase 3 can move a branch body without rewiring any
# caller or re-spelling the ~40 dispatch kwargs.
#
# * ``_dispatch_search_papers_smart`` and ``_dispatch_ask_result_set`` are fully
#   extracted today: ``dispatch_tool`` delegates to them directly. Their bodies
#   are small and touch only ``ctx.agentic_runtime`` plus ``ctx.ctx`` (the MCP
#   request-scoped Context), so the migration was risk-free.
# * ``_dispatch_research`` and ``_dispatch_follow_up_research`` are Phase 3
#   preparation shims: they lock the ``(ctx, arguments)`` calling convention
#   into the module surface so new callers can adopt it now, but the inline
#   branch inside ``dispatch_tool`` still owns the logic. Phase 3 will move the
#   body into these entrypoints (and then into sibling submodules) one at a
#   time. Delegation routes through ``dispatch_tool(name, arguments,
#   **ctx.as_kwargs())``, which falls into the inline branch and returns —
#   there is no recursion because the inline branch never calls the shim back.


async def _dispatch_research(
    ctx: DispatchContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Entrypoint for the ``research`` guided tool.

    Phase 2 Track C amendment 3 pins the ``(ctx, arguments)`` calling
    convention for this branch so Phase 3 can relocate the 685-line body into
    a sibling submodule without rewiring any caller. The body still lives
    inline in :func:`dispatch_tool` today; this shim delegates back via
    ``dispatch_tool(name, arguments, **ctx.as_kwargs())``, which lands in the
    inline branch and returns. There is no recursion because the inline
    branch never calls this shim.
    """
    return await dispatch_tool("research", arguments, **ctx.as_kwargs())


async def _dispatch_follow_up_research(
    ctx: DispatchContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Entrypoint for the ``follow_up_research`` guided tool.

    Same Phase 3 prep pattern as :func:`_dispatch_research`: pins the
    ``(ctx, arguments)`` convention on the module surface and delegates to
    the inline ``dispatch_tool`` branch until Phase 3 moves the body.
    """
    return await dispatch_tool("follow_up_research", arguments, **ctx.as_kwargs())


# ---------------------------------------------------------------------------
# Phase 3 re-exports: symbols relocated to `dispatch/guided/` submodules.
# ---------------------------------------------------------------------------
#
# Each `from .guided.<submodule> import ...` line below preserves the
# pre-Phase-3 dispatch seam: callers (including the facade allowlist in
# `paper_chaser_mcp/dispatch/__init__.py` and tests that still target
# `paper_chaser_mcp.dispatch._core.<name>`) keep the same import path.
# Moving a symbol into a guided submodule requires appending its name here so
# `_core._guided_*` lookups resolve through the re-import.
from .expert.enrichment import (  # noqa: E402,F401 — Phase 4 re-export seam
    _dispatch_enrich_paper,
    _dispatch_get_paper_metadata_crossref,
    _dispatch_get_paper_open_access_unpaywall,
    _dispatch_non_search_tool,
    _dispatch_provider_search_tool,
)
from .expert.openalex import (  # noqa: E402,F401 — Phase 4 re-export seam
    _dispatch_get_author_info_openalex,
    _dispatch_get_author_papers_openalex,
    _dispatch_get_paper_citations_openalex,
    _dispatch_get_paper_details_openalex,
    _dispatch_get_paper_references_openalex,
    _dispatch_paper_autocomplete_openalex,
    _dispatch_search_authors_openalex,
    _dispatch_search_entities_openalex,
    _dispatch_search_papers_openalex,
    _dispatch_search_papers_openalex_bulk,
    _dispatch_search_papers_openalex_by_entity,
)
from .expert.raw import (  # noqa: E402,F401 — Phase 4 re-export seam
    _dispatch_get_paper_details,
    _dispatch_resolve_citation,
    _dispatch_search_papers,
    _dispatch_search_papers_bulk,
    _dispatch_search_papers_match,
)
from .expert.regulatory import (  # noqa: E402,F401 — Phase 4 re-export seam
    _dispatch_get_cfr_text,
    _dispatch_get_document_text_ecos,
    _dispatch_get_federal_register_document,
    _dispatch_get_species_profile_ecos,
    _dispatch_list_species_documents_ecos,
    _dispatch_search_federal_register,
    _dispatch_search_species_ecos,
)
from .expert.scholarapi import (  # noqa: E402,F401 — Phase 4 re-export seam
    _dispatch_get_paper_pdf_scholarapi,
    _dispatch_get_paper_text_scholarapi,
    _dispatch_get_paper_texts_scholarapi,
    _dispatch_list_papers_scholarapi,
    _dispatch_search_papers_scholarapi,
)
from .expert.serpapi import (  # noqa: E402,F401 — Phase 4 re-export seam
    _dispatch_get_author_articles_serpapi,
    _dispatch_get_author_profile_serpapi,
    _dispatch_get_paper_citation_formats,
    _dispatch_get_serpapi_account_status,
    _dispatch_search_papers_serpapi_cited_by,
    _dispatch_search_papers_serpapi_versions,
)
from .guided.citations import (  # noqa: E402,F401 — Phase 3 re-export seam
    _assign_verification_status,
    _guided_citation_from_paper,
    _guided_citation_from_structured_source,
    _guided_journal_or_publisher,
    _guided_normalize_access_axes,
    _guided_normalize_verification_status,
    _guided_open_access_route,
    _guided_year_text,
)
from .guided.findings import (  # noqa: E402,F401 — Phase 3 re-export seam
    _guided_findings_from_sources,
    _guided_unverified_leads_from_sources,
)
from .guided.follow_up import (  # noqa: E402,F401 -- Phase 3 re-export seam
    _answer_follow_up_from_session_state,
    _guided_follow_up_answer_mode,
    _guided_follow_up_introspection_facets,
    _guided_follow_up_response_mode,
    _guided_is_usable_answer_text,
    _guided_metadata_answer_is_responsive,
    _guided_relevance_triage_answers,
    _guided_requested_metadata_facets,
    _guided_source_metadata_answers,
)
from .guided.inspect_source import (  # noqa: E402,F401 — Phase 3 re-export seam
    _guided_append_selected_saved_records,
    _guided_compact_source_candidate,
    _guided_extract_question,
    _guided_extract_source_reference_from_question,
    _guided_select_follow_up_source,
    _guided_source_resolution_payload,
)
from .guided.research import (  # noqa: E402,F401 — Phase 3 re-export seam
    _guided_normalization_payload,
    _guided_normalize_follow_up_arguments,
    _guided_normalize_inspect_arguments,
    _guided_normalize_research_arguments,
)
from .guided.resolve_reference import (  # noqa: E402,F401 — Phase 3 re-export seam
    _guided_note_repair,
    _guided_underspecified_reference_clarification,
)
from .guided.response import (  # noqa: E402,F401 -- Phase 3 re-export seam
    _guided_compact_response_if_needed,
    _guided_contract_fields,
    _guided_finalize_response,
)
from .guided.sessions import (  # noqa: E402,F401 — Phase 3 re-export seam
    _GUIDED_RECOVERABLE_SESSION_TOOLS,
    _guided_active_session_ids,
    _guided_candidate_records,
    _guided_enrich_records_from_saved_session,
    _guided_extract_search_session_id,
    _guided_follow_up_session_resolution,
    _guided_infer_single_session_id,
    _guided_inspect_session_resolution,
    _guided_latest_compatible_session_id,
    _guided_resolve_session_id_for_source,
    _guided_saved_session_topicality,
    _guided_session_candidates,
    _guided_session_exists,
    _guided_session_findings,
    _guided_session_state,
    _guided_unique_compatible_session_id,
)
from .guided.sources import (  # noqa: E402,F401 — Phase 3 re-export seam
    _guided_dedupe_source_records,
    _guided_extract_source_id,
    _guided_merge_source_record_sets,
    _guided_merge_source_records,
    _guided_source_coverage_summary,
    _guided_source_id,
    _guided_source_identity,
    _guided_source_matches_reference,
    _guided_source_record_from_paper,
    _guided_source_record_from_structured_source,
    _guided_source_records_share_surface,
    _guided_sources_from_fr_documents,
)
from .guided.strategy_metadata import (  # noqa: E402,F401 — Phase 3 re-export seam
    _guided_abstention_details_payload,
    _guided_execution_provenance_payload,
    _guided_is_agency_guidance_query,
    _guided_is_known_item_query,
    _guided_is_mixed_intent_query,
    _guided_live_strategy_metadata,
    _guided_mentions_literature,
    _guided_merge_coverage_summaries,
    _guided_merge_failure_summaries,
    _guided_provider_budget_payload,
    _guided_reference_signal_words,
    _guided_review_pass_overrides,
    _guided_should_add_review_pass,
    _guided_should_escalate_research,
    _guided_strategy_metadata_from_runs,
)
from .guided.trust import (  # noqa: E402,F401 — Phase 3 re-export seam
    _guided_best_next_internal_action,
    _guided_confidence_signals,
    _guided_deterministic_evidence_gaps,
    _guided_deterministic_fallback_used,
    _guided_failure_summary,
    _guided_follow_up_status,
    _guided_generate_evidence_gaps,
    _guided_machine_failure_payload,
    _guided_missing_evidence_type,
    _guided_next_actions,
    _guided_partial_recovery_possible,
    _guided_record_source_candidates,
    _guided_research_status,
    _guided_result_meaning,
    _guided_result_state,
    _guided_sources_all_off_topic,
    _guided_summary,
    _guided_trust_summary,
)
from .smart.ask import _dispatch_ask_result_set  # noqa: E402,F401 — Phase 4 re-export seam
from .smart.graph import _dispatch_expand_research_graph  # noqa: E402,F401 — Phase 4 re-export seam
from .smart.landscape import _dispatch_map_research_landscape  # noqa: E402,F401 — Phase 4 re-export seam
from .smart.search import _dispatch_search_papers_smart  # noqa: E402,F401 — Phase 4 re-export seam
