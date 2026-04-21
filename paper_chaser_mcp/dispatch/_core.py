"""Dispatch helpers for MCP tool routing."""

import logging
import re
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any, Callable, Literal, cast

from ..agentic.planner import (
    detect_literature_intent,
    detect_regulatory_intent,
    looks_like_exact_title,
)
from ..agentic.provider_helpers import generate_evidence_gaps_without_llm
from ..citation_repair import looks_like_citation_query, looks_like_paper_identifier, parse_citation, resolve_citation
from ..clients.scholarapi import (
    ScholarApiError,
    ScholarApiKeyMissingError,
    ScholarApiQuotaError,
    ScholarApiUpstreamError,
)
from ..clients.serpapi import SerpApiKeyMissingError
from ..compat import augment_tool_result, build_clarification
from ..enrichment import (
    PaperEnrichmentService,
    attach_enrichments_to_paper_payload,
    hydrate_paper_for_enrichment,
)
from ..guided_semantic import (
    build_evidence_records,
    build_follow_up_decision,
    build_routing_decision,
    classify_answerability,
    explicit_source_reference,
    strip_null_fields,
)
from ..identifiers import resolve_doi_from_paper_payload
from ..models import TOOL_INPUT_MODELS, CitationFormatsResponse, RuntimeSummary, dump_jsonable
from ..models.common import (
    AbstentionDetails,
    CitationFormat,
    ConfidenceSignals,
    ExportLink,
    GuidedExecutionProvenance,
    GuidedResultState,
    InputNormalization,
    MachineFailure,
    NormalizationRepair,
    RuntimeHealthStatus,
    SessionCandidate,
    SessionResolution,
    SourceResolution,
)
from ..models.tools import (
    AskResultSetArgs,
    BasicSearchPapersArgs,
    EcosSpeciesLookupArgs,
    ExpandResearchGraphArgs,
    FollowUpResearchArgs,
    GetCfrTextArgs,
    GetCitationFormatsArgs,
    GetDocumentTextEcosArgs,
    GetFederalRegisterDocumentArgs,
    GetRuntimeStatusArgs,
    InspectSourceArgs,
    ListSpeciesDocumentsEcosArgs,
    MapResearchLandscapeArgs,
    PaperLookupArgs,
    PaperMatchArgs,
    ResearchArgs,
    ResolveCitationArgs,
    ResolveReferenceArgs,
    SearchFederalRegisterArgs,
    SearchPapersArgs,
    SearchProvider,
    SearchSpeciesEcosArgs,
    SmartSearchPapersArgs,
)
from ..provider_runtime import ProviderOutcomeEnvelope, ProviderStatusBucket, policy_for_provider
from ..search import search_papers_with_fallback
from ..utils.cursor import (
    OFFSET_TOOLS,
    PROVIDER,
    SUPPORTED_VERSIONS,
    compute_context_hash,
    cursor_from_token,
    decode_bulk_cursor,
)
from .context import DispatchContext, build_dispatch_context
from .normalization import (
    _guided_normalize_citation_surface,
    _guided_normalize_source_locator,
    _guided_normalize_whitespace,
    _guided_normalize_year_hint,
    _guided_strip_research_prefix,
)
from .paging import _cursor_to_offset, _encode_next_cursor
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


def _guided_source_id(candidate: dict[str, Any], *, fallback_prefix: str, index: int) -> str:
    for key in (
        "sourceId",
        "evidenceId",
        "paperId",
        "canonicalId",
        "recommendedExpansionId",
        "citationText",
        "canonicalUrl",
        "url",
    ):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    title = str(candidate.get("title") or "").strip()
    if title:
        return title
    return f"{fallback_prefix}-{index}"


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
def _guided_source_record_from_structured_source(source: dict[str, Any], *, index: int) -> dict[str, Any]:
    _source_type = source.get("sourceType") or "unknown"
    _access_status, _full_text_url_found, _full_text_observed, _body_text_embedded, _qa_readable_text = (
        _guided_normalize_access_axes(source)
    )
    _default_verification = _guided_normalize_verification_status(
        source,
        source_type=str(_source_type),
        full_text_url_found=_full_text_url_found,
        body_text_embedded=_body_text_embedded,
    )
    topical_relevance = source.get("topicalRelevance") or "weak_match"
    weak_match_reason = str(source.get("whyClassifiedAsWeakMatch") or "").strip() or None
    if weak_match_reason is None and topical_relevance in {"weak_match", "off_topic"}:
        weak_match_reason = str(source.get("note") or source.get("whyNotVerified") or "").strip() or None
    normalized_open_access_source = {
        **source,
        "accessStatus": _access_status,
        "fullTextUrlFound": _full_text_url_found,
        "fullTextObserved": _full_text_observed,
    }
    return {
        "sourceId": _guided_source_id(source, fallback_prefix="source", index=index),
        "title": source.get("title"),
        "provider": source.get("provider"),
        "sourceType": _source_type,
        "verificationStatus": _default_verification,
        "accessStatus": _access_status,
        "topicalRelevance": topical_relevance,
        "confidence": source.get("confidence") or "medium",
        "isPrimarySource": bool(source.get("isPrimarySource")),
        "canonicalUrl": source.get("canonicalUrl"),
        "retrievedUrl": source.get("retrievedUrl"),
        "fullTextUrlFound": _full_text_url_found,
        "fullTextObserved": _full_text_observed,
        "bodyTextEmbedded": _body_text_embedded,
        "qaReadableText": _qa_readable_text,
        "abstractObserved": bool(source.get("abstractObserved")),
        "openAccessRoute": _guided_open_access_route(normalized_open_access_source),
        "citationText": source.get("citationText"),
        "citation": _guided_citation_from_structured_source(source),
        "date": source.get("date"),
        "note": source.get("note"),
        "whyClassifiedAsWeakMatch": weak_match_reason,
    }


def _guided_source_record_from_paper(query: str, paper: dict[str, Any], *, index: int) -> dict[str, Any]:
    canonical_url = paper.get("canonicalUrl") or paper.get("url") or paper.get("pdfUrl")
    source_type = paper.get("sourceType") or "scholarly_article"
    doi, _ = resolve_doi_from_paper_payload(paper)
    _access_status, _full_text_url_found, _full_text_observed, _body_text_embedded, _qa_readable_text = (
        _guided_normalize_access_axes(paper)
    )
    verification_status = _guided_normalize_verification_status(
        {**paper, "doi": doi},
        source_type=str(source_type),
        full_text_url_found=_full_text_url_found,
        body_text_embedded=_body_text_embedded,
    )
    # Backward-compatible default: a scholarly article with basic descriptive
    # metadata (title + at least one author OR a venue) should remain
    # ``verified_metadata`` rather than silently drop to ``unverified`` just
    # because it lacks a DOI. ``unverified`` is reserved for records truly
    # missing descriptive metadata. (ws-dispatch-contract-trust / finding #3.)
    if (
        verification_status == "unverified"
        and str(source_type) == "scholarly_article"
        and str(paper.get("title") or "").strip()
    ):
        _authors = paper.get("authors") or []
        _has_author = False
        if isinstance(_authors, list):
            for _author in _authors:
                if isinstance(_author, str) and _author.strip():
                    _has_author = True
                    break
                if isinstance(_author, dict) and str(_author.get("name") or "").strip():
                    _has_author = True
                    break
        _has_venue = bool(str(paper.get("venue") or "").strip())
        if _has_author or _has_venue:
            verification_status = "verified_metadata"
    # Access-status split (P0-2): an explicit accessStatus on the input wins;
    # otherwise derive from the strongest signal available. ``body_text_embedded``
    # implies inline body content; a URL-only hit becomes ``url_verified``
    # (distinct from the deprecated ``full_text_verified`` which used to be
    # emitted for any URL-found regulatory or scholarly record).
    topical_relevance = _paper_topical_relevance(query, paper)
    confidence = paper.get("confidence") or ("high" if topical_relevance == "on_topic" else "medium")
    weak_match_reason = str(paper.get("whyClassifiedAsWeakMatch") or "").strip() or None
    if weak_match_reason is None and topical_relevance in {"weak_match", "off_topic"}:
        weak_match_reason = str(paper.get("note") or paper.get("venue") or "").strip() or None
    normalized_open_access_paper = {
        **paper,
        "accessStatus": _access_status,
        "fullTextUrlFound": _full_text_url_found,
        "fullTextObserved": _full_text_observed,
        "canonicalUrl": canonical_url,
        "retrievedUrl": paper.get("retrievedUrl") or canonical_url,
    }
    return strip_null_fields(
        {
            "sourceId": _guided_source_id(paper, fallback_prefix="paper", index=index),
            "title": paper.get("title"),
            "provider": paper.get("source"),
            "sourceType": source_type,
            "verificationStatus": verification_status,
            "accessStatus": _access_status,
            "topicalRelevance": topical_relevance,
            "confidence": confidence,
            "isPrimarySource": bool(paper.get("isPrimarySource")),
            "canonicalUrl": canonical_url,
            "retrievedUrl": paper.get("retrievedUrl") or canonical_url,
            "fullTextUrlFound": _full_text_url_found,
            "fullTextObserved": _full_text_observed,
            "bodyTextEmbedded": _body_text_embedded,
            "qaReadableText": _qa_readable_text,
            "abstractObserved": bool(paper.get("abstractObserved")),
            "openAccessRoute": _guided_open_access_route(normalized_open_access_paper),
            "citationText": str(paper.get("canonicalId") or paper.get("paperId") or "") or None,
            "citation": _guided_citation_from_paper(paper, canonical_url),
            "date": paper.get("publicationDate") or paper.get("year"),
            "note": paper.get("note") or paper.get("venue"),
            "whyClassifiedAsWeakMatch": weak_match_reason,
        }
    )


def _guided_sources_from_fr_documents(query: str, documents: list[Any]) -> list[dict[str, Any]]:
    """Convert FederalRegisterDocument objects into guided source records."""
    sources: list[dict[str, Any]] = []
    for index, doc in enumerate(documents or [], start=1):
        # Support both attribute-style (Pydantic model) and dict-style access
        def _get(attr: str, default: Any = None) -> Any:
            if isinstance(doc, dict):
                return doc.get(attr, default)
            return getattr(doc, attr, default)

        title = _get("title")
        if not title:
            continue
        html_url = _get("htmlUrl") or _get("bodyHtmlUrl")
        pdf_url = _get("pdfUrl")
        canonical_url = html_url or _get("govInfoLink") or pdf_url
        doc_number = _get("documentNumber")
        doc_type = str(_get("documentType") or "").strip()
        pub_date = _get("publicationDate")
        citation = _get("citation")
        agencies_raw = _get("agencies") or []
        agency_names = [
            str(getattr(a, "name", None) or (a.get("name") if isinstance(a, dict) else "") or "").strip()
            for a in agencies_raw
        ]
        agency_str = "; ".join(n for n in agency_names if n) or None
        cfr_refs = _get("cfrReferences") or []
        abstract = _get("abstract")

        # Build citation text from FR citation or document number
        citation_text = citation or (f"Fed. Reg. No. {doc_number}" if doc_number else None)

        # FR documents are authoritative primary sources, but topical
        # relevance must still be *computed* from the query — the FR search
        # API sometimes returns adjacent/off-topic rules that should be
        # flagged for escalation rather than folded into evidence as on-topic.
        source_type = "federal_register_rule" if "rule" in doc_type.lower() else "regulatory_document"
        note_parts = [agency_str] if agency_str else []
        if cfr_refs:
            note_parts.append("CFR: " + ", ".join(str(r) for r in cfr_refs[:3]))

        fr_source_candidate = {
            "title": title,
            "abstract": abstract,
            "venue": "Federal Register",
            "note": "; ".join(note_parts) if note_parts else None,
        }
        topical_relevance = compute_topical_relevance(query, fr_source_candidate)

        sources.append(
            {
                "sourceId": f"fr-{doc_number}" if doc_number else f"fr-source-{index}",
                "title": title,
                "provider": "federal_register",
                "sourceType": source_type,
                # P0-2: URL-found alone does not imply body-text-embedded.
                # FR docs here are URL-discovered metadata; body must be
                # fetched separately to earn ``verified_primary_source``.
                "verificationStatus": "verified_metadata",
                "accessStatus": ("url_verified" if html_url else ("pdf_available" if pdf_url else "access_unverified")),
                "topicalRelevance": topical_relevance,
                "confidence": "high" if topical_relevance == "on_topic" else "medium",
                "isPrimarySource": True,
                "canonicalUrl": canonical_url,
                "retrievedUrl": canonical_url,
                "fullTextUrlFound": bool(html_url),
                "bodyTextEmbedded": False,
                "qaReadableText": False,
                "abstractObserved": bool(abstract),
                "openAccessRoute": "open_access" if canonical_url else None,
                "citationText": citation_text,
                "citation": {
                    "title": title,
                    "authors": agency_names or [],
                    "year": pub_date[:4] if isinstance(pub_date, str) and len(pub_date) >= 4 else None,
                    "venue": "Federal Register",
                    "canonicalId": citation_text,
                    "citationText": citation_text,
                },
                "date": pub_date,
                "note": "; ".join(note_parts) if note_parts else None,
            }
        )
    return sources


def _guided_findings_from_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for source in sources:
        if source.get("topicalRelevance") != "on_topic":
            continue
        verification_status = str(source.get("verificationStatus") or "")
        if verification_status not in {"verified_primary_source", "verified_metadata"}:
            continue
        claim = str(source.get("title") or source.get("note") or source.get("sourceId") or "").strip()
        if not claim:
            continue
        findings.append(
            {
                "claim": claim,
                "supportingSourceIds": [source["sourceId"]],
                "trustLevel": "verified",
            }
        )
    return findings


def _guided_unverified_leads_from_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    leads: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sources:
        if source.get("topicalRelevance") == "on_topic" and source.get("verificationStatus") in {
            "verified_primary_source",
            "verified_metadata",
        }:
            continue
        source_id = str(source.get("sourceId") or "").strip()
        if source_id and source_id in seen:
            continue
        if source_id:
            seen.add(source_id)
        leads.append(source)
    return leads[:6]


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


def _guided_note_repair(
    repairs: list[dict[str, str]],
    *,
    field: str,
    original: Any,
    normalized: Any,
    reason: str,
) -> None:
    if original == normalized:
        return
    repairs.append(
        {
            "field": field,
            "from": str(original if original is not None else ""),
            "to": str(normalized if normalized is not None else ""),
            "reason": reason,
        }
    )


def _guided_session_exists(
    *,
    workspace_registry: Any,
    search_session_id: str | None,
) -> bool:
    if workspace_registry is None:
        return False
    normalized_id = _guided_normalize_whitespace(search_session_id)
    if not normalized_id:
        return False
    try:
        workspace_registry.get(normalized_id)
    except Exception:
        return False
    return True


def _guided_active_session_ids(workspace_registry: Any) -> list[str]:
    if workspace_registry is None:
        return []
    records = getattr(workspace_registry, "_records", None)
    if not isinstance(records, dict):
        return []
    active: list[tuple[float, str]] = []
    now = time.time()
    for session_id, record in records.items():
        if not isinstance(session_id, str) or not session_id.strip():
            continue
        if record is None:
            continue
        is_expired = getattr(record, "is_expired", None)
        expired = False
        if callable(is_expired):
            try:
                expired = bool(is_expired(now))
            except Exception as exc:
                logger.debug("Failed to evaluate session expiration for %s: %s", session_id, exc)
                expired = True
        if expired:
            continue
        created_at = float(getattr(record, "created_at", 0.0) or 0.0)
        active.append((created_at, session_id))
    active.sort(key=lambda item: item[0], reverse=True)
    return [session_id for _, session_id in active]


_GUIDED_RECOVERABLE_SESSION_TOOLS = {"research", "search_papers_smart"}


def _guided_candidate_records(
    workspace_registry: Any,
    *,
    require_sources: bool = False,
) -> list[Any]:
    if workspace_registry is None:
        return []
    active_records = getattr(workspace_registry, "active_records", None)
    records: list[Any] = []
    if callable(active_records):
        try:
            active = active_records(source_tools=_GUIDED_RECOVERABLE_SESSION_TOOLS)
            records = list(cast(Any, active)) if active is not None else []
        except Exception as exc:
            logger.debug("Failed to read active workspace records: %s", exc)
            records = []
    if not records:
        for session_id in _guided_active_session_ids(workspace_registry):
            record = None
            try:
                record = workspace_registry.get(session_id)
            except Exception as exc:
                logger.debug("Failed to load workspace record %s: %s", session_id, exc)
            if str(getattr(record, "source_tool", "") or "") not in _GUIDED_RECOVERABLE_SESSION_TOOLS:
                continue
            records.append(record)
    if require_sources:
        records = [record for record in records if _guided_record_source_candidates(record)]
    return records


def _guided_latest_compatible_session_id(
    workspace_registry: Any,
    *,
    require_sources: bool = False,
) -> str | None:
    records = _guided_candidate_records(workspace_registry, require_sources=require_sources)
    if not records:
        return None
    return str(getattr(records[0], "search_session_id", "") or "") or None


def _guided_unique_compatible_session_id(
    workspace_registry: Any,
    *,
    require_sources: bool = False,
) -> str | None:
    records = _guided_candidate_records(workspace_registry, require_sources=require_sources)
    if len(records) != 1:
        return None
    return str(getattr(records[0], "search_session_id", "") or "") or None


def _guided_resolve_session_id_for_source(
    workspace_registry: Any,
    source_id: str | None,
) -> tuple[str | None, str | None]:
    normalized_source_id = _guided_normalize_whitespace(source_id)
    if not normalized_source_id:
        return _guided_unique_compatible_session_id(workspace_registry, require_sources=True), None

    matched_records: list[tuple[str | None, str | None]] = []
    for record in _guided_candidate_records(workspace_registry, require_sources=True):
        resolved, match_type = _find_record_source_with_resolution(
            workspace_registry=workspace_registry,
            search_session_id=str(getattr(record, "search_session_id", "") or ""),
            source_id=normalized_source_id,
        )
        if resolved is not None:
            matched_records.append((str(getattr(record, "search_session_id", "") or "") or None, match_type))
    if len(matched_records) == 1:
        return matched_records[0]
    return _guided_unique_compatible_session_id(workspace_registry, require_sources=True), None


def _guided_infer_single_session_id(workspace_registry: Any) -> str | None:
    return _guided_unique_compatible_session_id(workspace_registry)


def _guided_extract_search_session_id(arguments: dict[str, Any]) -> Any:
    return next(
        (
            arguments.get(key)
            for key in (
                "searchSessionId",
                "search_session_id",
                "sessionId",
                "session_id",
                "session",
            )
            if arguments.get(key) is not None
        ),
        None,
    )


def _guided_extract_source_id(arguments: dict[str, Any]) -> Any:
    return next(
        (
            arguments.get(key)
            for key in ("sourceId", "source_id", "evidenceId", "evidence_id", "source", "sourceRef", "leadId", "id")
            if arguments.get(key) is not None
        ),
        None,
    )


def _guided_extract_question(arguments: dict[str, Any]) -> Any:
    return next(
        (arguments.get(key) for key in ("question", "prompt", "query") if arguments.get(key) is not None),
        None,
    )


def _guided_session_candidates(
    workspace_registry: Any,
    *,
    require_sources: bool = False,
    limit: int = 5,
) -> list[SessionCandidate]:
    now = time.time()
    candidates: list[SessionCandidate] = []
    for record in _guided_candidate_records(workspace_registry, require_sources=require_sources)[:limit]:
        payload = record.payload if isinstance(record.payload, dict) else {}
        sources = _guided_record_source_candidates(record)
        query = str(getattr(record, "query", None) or payload.get("query") or "").strip() or None
        summary = (
            str(payload.get("summary") or (sources[0].get("title") if sources else "") or query or "").strip() or None
        )
        age_seconds = max(0, int(now - float(getattr(record, "created_at", 0.0) or 0.0)))
        candidate = SessionCandidate(
            searchSessionId=str(getattr(record, "search_session_id", "") or ""),
            sourceTool=str(getattr(record, "source_tool", "") or "unknown"),
            query=query,
            summary=summary,
            ageSeconds=age_seconds,
            sourceCount=len(sources),
        )
        candidates.append(candidate)
    return candidates


def _guided_follow_up_session_resolution(
    *,
    arguments: dict[str, Any],
    normalized_arguments: dict[str, Any],
    normalization: dict[str, Any],
    workspace_registry: Any,
) -> dict[str, Any]:
    requested = _guided_normalize_whitespace(_guided_extract_search_session_id(arguments))
    resolved = _guided_normalize_whitespace(normalized_arguments.get("searchSessionId"))
    candidates = _guided_session_candidates(workspace_registry)
    if requested and resolved and requested == resolved:
        mode = "provided_explicitly"
        visible_candidates: list[SessionCandidate] = []
    elif requested and resolved:
        mode = "repaired_to_unique_active_session"
        visible_candidates = []
    elif not requested and resolved:
        mode = "inferred_single_active_session"
        visible_candidates = []
    elif len(candidates) > 1:
        mode = "ambiguous"
        visible_candidates = candidates
    elif requested:
        mode = "session_unavailable"
        visible_candidates = candidates
    else:
        mode = "missing"
        visible_candidates = candidates
    resolution = SessionResolution(
        requestedSearchSessionId=requested,
        resolvedSearchSessionId=resolved,
        resolutionMode=mode,
        warnings=list(normalization.get("warnings") or []),
        candidates=visible_candidates,
    )
    return resolution.model_dump(by_alias=True, exclude_none=True)


def _guided_inspect_session_resolution(
    *,
    arguments: dict[str, Any],
    normalized_arguments: dict[str, Any],
    normalization: dict[str, Any],
    workspace_registry: Any,
) -> dict[str, Any]:
    requested = _guided_normalize_whitespace(_guided_extract_search_session_id(arguments))
    resolved = _guided_normalize_whitespace(normalized_arguments.get("searchSessionId"))
    normalized_source_id = _guided_normalize_whitespace(normalized_arguments.get("sourceId"))
    source_inferred_session_id, _ = _guided_resolve_session_id_for_source(workspace_registry, normalized_source_id)
    candidates = _guided_session_candidates(workspace_registry, require_sources=True)
    if requested and resolved and requested == resolved:
        mode = "provided_explicitly"
        visible_candidates: list[SessionCandidate] = []
    elif requested and resolved and source_inferred_session_id and resolved == source_inferred_session_id:
        mode = "repaired_to_source_bearing_session"
        visible_candidates = []
    elif requested and resolved:
        mode = "repaired_to_unique_active_session"
        visible_candidates = []
    elif not requested and resolved and source_inferred_session_id and resolved == source_inferred_session_id:
        mode = "inferred_source_bearing_session"
        visible_candidates = []
    elif not requested and resolved:
        mode = "inferred_single_active_session"
        visible_candidates = []
    elif len(candidates) > 1:
        mode = "ambiguous"
        visible_candidates = candidates
    elif requested:
        mode = "session_unavailable"
        visible_candidates = candidates
    else:
        mode = "missing"
        visible_candidates = candidates
    resolution = SessionResolution(
        requestedSearchSessionId=requested,
        resolvedSearchSessionId=resolved,
        resolutionMode=mode,
        warnings=list(normalization.get("warnings") or []),
        candidates=visible_candidates,
    )
    return resolution.model_dump(by_alias=True, exclude_none=True)


def _guided_compact_source_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Project a saved-session source candidate to its disambiguation-critical fields.

    Includes inspectability-signaling fields (canonicalUrl / retrievedUrl /
    fullTextUrlFound / abstractObserved) so agents can verify a candidate
    would be inspectable without a round trip.
    """
    keep_keys = (
        "sourceId",
        "title",
        "topicalRelevance",
        "canonicalUrl",
        "retrievedUrl",
        "fullTextUrlFound",
        "abstractObserved",
        "confidence",
        "accessStatus",
        "verificationStatus",
        "publicationYear",
    )
    projected: dict[str, Any] = {}
    for key in keep_keys:
        value = candidate.get(key)
        if value not in (None, "", [], {}):
            projected[key] = value
    return projected


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


def _guided_source_resolution_payload(
    *,
    requested_source_id: str | None,
    resolved_source_id: str | None,
    match_type: str | None,
    available_source_ids: list[str] | None = None,
    available_candidates: list[dict[str, Any]] | None = None,
    candidates_have_inspectable: bool | None = None,
) -> dict[str, Any]:
    compact_candidates: list[dict[str, Any]] = []
    if available_candidates:
        for candidate in available_candidates:
            if not isinstance(candidate, dict):
                continue
            projection = _guided_compact_source_candidate(candidate)
            if projection.get("sourceId"):
                compact_candidates.append(projection)
    if candidates_have_inspectable is None:
        candidates_have_inspectable = any(_candidate_is_inspectable(candidate) for candidate in compact_candidates)
    resolution = SourceResolution(
        requestedSourceId=_guided_normalize_whitespace(requested_source_id),
        resolvedSourceId=_guided_normalize_whitespace(resolved_source_id),
        matchType=match_type,
        availableSourceIds=available_source_ids or [],
        availableSourceCandidates=compact_candidates,
        candidatesHaveInspectable=bool(candidates_have_inspectable) if compact_candidates else False,
    )
    return resolution.model_dump(by_alias=True, exclude_none=True)


def _guided_execution_provenance_payload(
    *,
    execution_mode: str,
    answer_source: str | None = None,
    latency_profile_applied: str | None = None,
    allow_paid_providers: bool | None = None,
    provider_budget_applied: dict[str, Any] | None = None,
    strategy_metadata: dict[str, Any] | None = None,
    escalation_attempted: bool = False,
    escalation_reason: str | None = None,
    passes_run: int = 0,
    pass_modes: list[str] | None = None,
) -> dict[str, Any]:
    metadata = strategy_metadata if isinstance(strategy_metadata, dict) else {}
    configured_provider = _guided_normalize_whitespace(metadata.get("configuredSmartProvider")) or None
    active_provider = _guided_normalize_whitespace(metadata.get("activeSmartProvider")) or None
    latency_profile = latency_profile_applied or _guided_normalize_whitespace(metadata.get("latencyProfile")) or None
    budget_payload = provider_budget_applied or cast(dict[str, Any], metadata.get("providerBudgetApplied") or {})
    deterministic_fallback_used = bool(
        active_provider == "deterministic" and configured_provider not in {None, "deterministic"}
    )
    provenance = GuidedExecutionProvenance(
        executionMode=execution_mode,
        answerSource=answer_source,
        serverPolicyApplied=GUIDED_POLICY_NAME,
        latencyProfileApplied=latency_profile,
        allowPaidProviders=allow_paid_providers,
        providerBudgetApplied=budget_payload,
        configuredSmartProvider=configured_provider,
        activeSmartProvider=active_provider,
        deterministicFallbackUsed=deterministic_fallback_used,
        escalationAttempted=escalation_attempted,
        escalationReason=escalation_reason,
        passesRun=passes_run,
        passModes=pass_modes or [],
    )
    return provenance.model_dump(by_alias=True, exclude_none=True)


def _guided_live_strategy_metadata(
    *,
    agentic_runtime: Any,
    strategy_metadata: dict[str, Any] | None = None,
    latency_profile: str | None = None,
) -> dict[str, Any]:
    merged = dict(strategy_metadata or {})
    provider_bundle = getattr(agentic_runtime, "_provider_bundle", None)
    if provider_bundle is not None and hasattr(provider_bundle, "selection_metadata"):
        try:
            selection = provider_bundle.selection_metadata()
            if isinstance(selection, dict):
                merged.update(selection)
        except Exception as exc:
            merged.setdefault("selectionMetadataError", type(exc).__name__)
    if latency_profile and not merged.get("latencyProfile"):
        merged["latencyProfile"] = latency_profile
    return merged


def _guided_abstention_details_payload(
    *,
    status: str,
    sources: list[dict[str, Any]],
    evidence_gaps: list[str],
    trust_summary: dict[str, Any],
) -> dict[str, Any] | None:
    if status not in {"abstained", "needs_disambiguation", "insufficient_evidence", "partial"}:
        return None
    category = _guided_missing_evidence_type(status=status, evidence_gaps=evidence_gaps, sources=sources)
    all_sources_off_topic = _guided_sources_all_off_topic(sources)
    # R11 finding 1: when every source is off_topic, agent-facing hints must
    # not tell the caller to inspect the returned sources — they aren't useful.
    # Force the off_topic_only category so the hints route to tighten-query.
    if sources and all_sources_off_topic and category != "off_topic_only":
        category = "off_topic_only"
    weak_match_count = int(trust_summary.get("weakMatchCount") or 0)
    off_topic_count = int(trust_summary.get("offTopicCount") or 0)
    on_topic_source_count = int(trust_summary.get("onTopicSourceCount") or 0)
    if status == "partial" and category == "coverage_gap":
        if weak_match_count and weak_match_count >= max(on_topic_source_count, 1):
            category = "weak_topical_match"
        elif on_topic_source_count and on_topic_source_count < 3:
            category = "narrow_evidence_pool"
    if category == "anchor_missing":
        refinement_hints = ["Add a specific title, DOI, species name, agency, venue, or year range."]
    elif category == "off_topic_only":
        refinement_hints = ["Tighten the query to the exact topic or anchored subject you need."]
    elif category == "provider_gap":
        refinement_hints = [
            "Retry later or compare get_runtime_status if provider behavior differs across environments.",
        ]
    elif category == "weak_topical_match":
        refinement_hints = [
            "Add a year range or venue to reduce weak topical matches.",
            "Specify the exact species (common or scientific name), agency, or concept you care about.",
            "Try resolve_reference if you have a DOI, arXiv id, URL, or full citation.",
        ]
    elif category == "narrow_evidence_pool":
        refinement_hints = [
            "Broaden the query with synonyms or a wider year range to recover more evidence.",
            "Run follow_up_research on the saved session to reuse the grounded sources you already have.",
            "Try resolve_reference if you have a DOI, arXiv id, URL, or full citation.",
        ]
    elif status == "partial" and sources:
        refinement_hints = [
            "Inspect the returned sources before treating the result as settled.",
            "Add a specific anchor (title, DOI, species, agency, venue, or year range) and retry research.",
        ]
    elif sources:
        refinement_hints = ["Inspect the returned sources before treating the result as settled."]
    else:
        refinement_hints = ["Narrow the request so the server can recover a stronger initial anchor."]
    effective_inspectable_count = 0 if all_sources_off_topic else len(sources)
    can_inspect = bool(sources) and not all_sources_off_topic
    details = AbstentionDetails(
        category=category,
        reason=(
            evidence_gaps[0] if evidence_gaps else "The current evidence was not strong enough to ground an answer."
        ),
        inspectableSourceCount=effective_inspectable_count,
        onTopicSourceCount=on_topic_source_count,
        weakMatchCount=weak_match_count,
        offTopicCount=off_topic_count,
        canInspectSources=can_inspect,
        refinementHints=refinement_hints,
    )
    return details.model_dump(by_alias=True, exclude_none=True)


def _guided_provider_budget_payload(*, allow_paid_providers: bool) -> dict[str, Any]:
    return {"allowPaidProviders": bool(allow_paid_providers)}


def _guided_strategy_metadata_from_runs(smart_runs: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    merged_lists: dict[str, list[str]] = {
        "providerPlan": [],
        "providersUsed": [],
        "secondaryIntents": [],
        "queryVariantsTried": [],
        "retrievalHypotheses": [],
    }
    confidence_rank = {"high": 3, "medium": 2, "low": 1}
    specificity_rank = {"high": 3, "medium": 2, "low": 1}
    ambiguity_rank = {"low": 1, "medium": 2, "high": 3}
    routing_confidences: list[str] = []
    intent_confidences: list[str] = []
    query_specificities: list[str] = []
    ambiguity_levels: list[str] = []

    for smart in smart_runs:
        metadata = smart.get("strategyMetadata")
        if not isinstance(metadata, dict):
            continue
        for field in (
            "intent",
            "intentRationale",
            "anchorType",
            "anchoredSubject",
            "configuredSmartProvider",
            "activeSmartProvider",
            "providerBudgetApplied",
            "latencyProfile",
            # Phase 4/5 planner classification signals — pass through the first
            # non-empty value so downstream routingSummary serialization can
            # expose them to agents.
            "intentFamily",
            "regulatoryIntent",
            "subjectCard",
        ):
            value = metadata.get(field)
            if value not in (None, "", [], {}) and field not in merged:
                merged[field] = value
        for field in merged_lists:
            for item in metadata.get(field) or []:
                text = str(item).strip()
                if text and text not in merged_lists[field]:
                    merged_lists[field].append(text)
        for item in metadata.get("subjectChainGaps") or []:
            text = str(item).strip()
            if text:
                subject_chain_gaps = merged.setdefault("subjectChainGaps", [])
                if isinstance(subject_chain_gaps, list) and text not in subject_chain_gaps:
                    subject_chain_gaps.append(text)

        intent_confidence = str(metadata.get("intentConfidence") or "").strip()
        if intent_confidence in confidence_rank:
            intent_confidences.append(intent_confidence)
        routing_confidence = str(metadata.get("routingConfidence") or "").strip()
        if routing_confidence in confidence_rank:
            routing_confidences.append(routing_confidence)
        query_specificity = str(metadata.get("querySpecificity") or "").strip()
        if query_specificity in specificity_rank:
            query_specificities.append(query_specificity)
        ambiguity_level = str(metadata.get("ambiguityLevel") or "").strip()
        if ambiguity_level in ambiguity_rank:
            ambiguity_levels.append(ambiguity_level)

    for field, values in merged_lists.items():
        if values:
            merged[field] = values
    if intent_confidences:
        merged["intentConfidence"] = min(intent_confidences, key=lambda value: confidence_rank[value])
    if routing_confidences:
        merged["routingConfidence"] = min(routing_confidences, key=lambda value: confidence_rank[value])
    if query_specificities:
        merged["querySpecificity"] = min(query_specificities, key=lambda value: specificity_rank[value])
    if ambiguity_levels:
        merged["ambiguityLevel"] = max(ambiguity_levels, key=lambda value: ambiguity_rank[value])
    return merged


def _guided_should_add_review_pass(
    *,
    initial_intent: str,
    query: str,
    focus: str | None,
    primary_smart: dict[str, Any],
    pass_modes: list[str],
) -> tuple[bool, str | None]:
    if "review" in pass_modes:
        return False, None
    if initial_intent == "mixed":
        return True, "mixed_intent_query"
    if initial_intent != "regulatory":
        return False, None
    if _guided_is_agency_guidance_query(query):
        return False, None

    metadata = cast(
        dict[str, Any],
        primary_smart.get("strategyMetadata") if isinstance(primary_smart.get("strategyMetadata"), dict) else {},
    )
    secondary_intents = {str(item).strip() for item in metadata.get("secondaryIntents") or [] if str(item).strip()}
    query_specificity = str(metadata.get("querySpecificity") or "").strip()
    ambiguity_level = str(metadata.get("ambiguityLevel") or "").strip()
    retrieval_hypotheses = [
        str(item).strip() for item in metadata.get("retrievalHypotheses") or [] if str(item).strip()
    ]
    # LLM-first: when the planner classifies the query as hybrid regulatory+literature
    # (either from the LLM or its deterministic fallback), honor that signal before any
    # keyword heuristic -- but only when an independent query-side literature cue
    # corroborates the hybrid label. Planners occasionally hallucinate
    # ``hybrid_regulatory_plus_literature`` for regulation-only asks (e.g.
    # "what does EPA require for stormwater discharges?"); trusting that label
    # in isolation causes the guided workflow to tack on an unnecessary review
    # pass. Corroboration is present when the query mentions literature-shaped
    # terms, the planner's secondaryIntents include review/literature, or
    # retrievalHypotheses carry explicit cross-domain literature cues.
    regulatory_intent = str(metadata.get("regulatoryIntent") or "").strip()
    if regulatory_intent == "hybrid_regulatory_plus_literature":
        has_literature_corroboration = (
            _guided_mentions_literature(query, focus)
            or "review" in secondary_intents
            or "literature" in secondary_intents
            or any(
                any(
                    marker in hypothesis.lower()
                    for marker in (
                        "literature",
                        "peer-review",
                        "peer review",
                        "peer-reviewed",
                        "systematic review",
                        "meta-analysis",
                        "hybrid_policy_science",
                    )
                )
                for hypothesis in retrieval_hypotheses
            )
        )
        if has_literature_corroboration:
            return True, "planner_hybrid_regulatory_plus_literature"
    if "review" in secondary_intents:
        return True, "review_secondary_intent_detected"
    if detect_literature_intent(query, focus) and (query_specificity == "low" or ambiguity_level in {"medium", "high"}):
        return True, "broad_regulatory_query_with_literature_signal"
    if ambiguity_level == "high" and retrieval_hypotheses:
        return True, "ambiguous_regulatory_query_with_retrieval_hypotheses"
    return False, None


def _guided_review_pass_overrides(
    *,
    query: str,
    focus: str | None,
    primary_smart: dict[str, Any],
) -> dict[str, Any]:
    metadata = cast(
        dict[str, Any],
        primary_smart.get("strategyMetadata") if isinstance(primary_smart.get("strategyMetadata"), dict) else {},
    )
    anchored_subject = _guided_normalize_whitespace(metadata.get("anchoredSubject"))
    existing_focus = _guided_normalize_whitespace(focus)
    if not anchored_subject:
        return {}

    lowered_query = _guided_normalize_whitespace(query).lower()
    lowered_focus = existing_focus.lower()
    if anchored_subject.lower() in lowered_query or anchored_subject.lower() in lowered_focus:
        return {}
    merged_focus = " ".join(part for part in [existing_focus, anchored_subject] if part)
    return {"focus": merged_focus} if merged_focus else {}


def _guided_is_agency_guidance_query(query: str) -> bool:
    normalized = _guided_normalize_whitespace(query).lower()
    if "guidance" not in normalized:
        return False
    return any(marker in normalized for marker in ("agency", "epa", "fda", "guidance for industry"))


def _guided_should_escalate_research(
    *,
    intent: str,
    status: str,
    sources: list[dict[str, Any]],
    verified_findings: list[dict[str, Any]],
    clarification: dict[str, Any] | None,
    pass_modes: list[str],
    max_passes: int,
) -> bool:
    if clarification is not None:
        return False
    if len(pass_modes) >= max_passes:
        return False
    if intent in {"known_item", "mixed", "regulatory"}:
        return False
    if verified_findings:
        return False
    if sources:
        return False
    return status in {"abstained", "partial"} and "review" not in pass_modes


def _guided_normalize_research_arguments(arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized_args = dict(arguments)
    repairs: list[dict[str, str]] = []
    warnings: list[str] = []

    raw_query = _guided_normalize_whitespace(arguments.get("query"))
    normalized_query = _guided_normalize_citation_surface(_guided_strip_research_prefix(raw_query))
    if not normalized_query:
        raw_focus = _guided_normalize_whitespace(arguments.get("focus"))
        if raw_focus:
            normalized_query = _guided_normalize_citation_surface(raw_focus)
            warnings.append("query was empty; reused normalized focus as the research query.")
    normalized_args["query"] = normalized_query or raw_query
    _guided_note_repair(
        repairs,
        field="query",
        original=arguments.get("query"),
        normalized=normalized_args["query"],
        reason="query_normalization",
    )

    normalized_focus = _guided_normalize_citation_surface(_guided_normalize_whitespace(arguments.get("focus")))
    normalized_args["focus"] = normalized_focus or None
    _guided_note_repair(
        repairs,
        field="focus",
        original=arguments.get("focus"),
        normalized=normalized_args["focus"],
        reason="focus_normalization",
    )

    normalized_venue = _guided_normalize_whitespace(arguments.get("venue")) or None
    normalized_args["venue"] = normalized_venue
    _guided_note_repair(
        repairs,
        field="venue",
        original=arguments.get("venue"),
        normalized=normalized_venue,
        reason="venue_normalization",
    )

    normalized_year = _guided_normalize_year_hint(arguments.get("year"))
    normalized_args["year"] = normalized_year
    _guided_note_repair(
        repairs,
        field="year",
        original=arguments.get("year"),
        normalized=normalized_year,
        reason="year_normalization",
    )

    normalization = {
        "normalizedQuery": normalized_args.get("query"),
        "normalizedFocus": normalized_args.get("focus"),
        "normalizedVenue": normalized_args.get("venue"),
        "normalizedYear": normalized_args.get("year"),
        "repairs": repairs,
        "warnings": warnings,
    }
    return normalized_args, normalization


def _guided_normalize_follow_up_arguments(
    arguments: dict[str, Any],
    *,
    workspace_registry: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized_args = dict(arguments)
    for alias in ("search_session_id", "sessionId", "session_id", "session", "prompt", "query"):
        normalized_args.pop(alias, None)
    repairs: list[dict[str, str]] = []
    warnings: list[str] = []

    raw_search_session_id = _guided_extract_search_session_id(arguments)
    normalized_search_session_id: str | None = _guided_normalize_whitespace(raw_search_session_id)
    if normalized_search_session_id and not _guided_session_exists(
        workspace_registry=workspace_registry,
        search_session_id=normalized_search_session_id,
    ):
        inferred_id = _guided_infer_single_session_id(workspace_registry)
        if inferred_id is not None:
            warnings.append(
                "searchSessionId "
                f"'{normalized_search_session_id}' was unavailable; using active session '{inferred_id}'."
            )
            normalized_search_session_id = inferred_id
        else:
            warnings.append(
                f"searchSessionId '{normalized_search_session_id}' was unavailable and could not be repaired safely."
            )
            normalized_search_session_id = None
    if not normalized_search_session_id:
        inferred_id = _guided_infer_single_session_id(workspace_registry)
        if inferred_id is not None:
            normalized_search_session_id = inferred_id
            warnings.append(f"searchSessionId was missing; inferred active session '{inferred_id}'.")
        elif len(_guided_candidate_records(workspace_registry)) > 1:
            warnings.append(
                "searchSessionId was missing and multiple active sessions exist; provide an explicit searchSessionId."
            )
    normalized_args["searchSessionId"] = normalized_search_session_id
    _guided_note_repair(
        repairs,
        field="searchSessionId",
        original=raw_search_session_id,
        normalized=normalized_search_session_id,
        reason="session_id_normalization",
    )

    raw_question = _guided_extract_question(arguments)
    normalized_question = _guided_normalize_whitespace(raw_question)
    normalized_args["question"] = normalized_question
    _guided_note_repair(
        repairs,
        field="question",
        original=raw_question,
        normalized=normalized_question,
        reason="question_normalization",
    )
    if not normalized_question:
        warnings.append("question was empty after normalization; follow-up quality may be limited.")

    normalization = {
        "normalizedSearchSessionId": normalized_args.get("searchSessionId"),
        "normalizedQuestion": normalized_args.get("question"),
        "repairs": repairs,
        "warnings": warnings,
    }
    return normalized_args, normalization


def _guided_normalize_inspect_arguments(
    arguments: dict[str, Any],
    *,
    workspace_registry: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized_args = dict(arguments)
    for alias in (
        "search_session_id",
        "sessionId",
        "session_id",
        "session",
        "source_id",
        "source",
        "sourceRef",
        "evidenceId",
        "evidence_id",
        "leadId",
        "lead_id",
        "id",
    ):
        normalized_args.pop(alias, None)
    repairs: list[dict[str, str]] = []
    warnings: list[str] = []

    raw_source_id = _guided_extract_source_id(arguments)
    normalized_source_id = _guided_normalize_whitespace(raw_source_id)

    raw_search_session_id = _guided_extract_search_session_id(arguments)
    normalized_search_session_id: str | None = _guided_normalize_whitespace(raw_search_session_id)
    if normalized_search_session_id and not _guided_session_exists(
        workspace_registry=workspace_registry,
        search_session_id=normalized_search_session_id,
    ):
        inferred_id = _guided_infer_single_session_id(workspace_registry)
        if inferred_id is not None:
            warnings.append(
                "searchSessionId "
                f"'{normalized_search_session_id}' was unavailable; using active session '{inferred_id}'."
            )
            normalized_search_session_id = inferred_id
        else:
            warnings.append(
                f"searchSessionId '{normalized_search_session_id}' was unavailable and could not be repaired safely."
            )
            normalized_search_session_id = None
    if not normalized_search_session_id:
        inferred_id = _guided_infer_single_session_id(workspace_registry)
        if inferred_id is not None:
            normalized_search_session_id = inferred_id
            warnings.append(f"searchSessionId was missing; inferred active session '{inferred_id}'.")
        elif len(_guided_candidate_records(workspace_registry)) > 1:
            warnings.append(
                "searchSessionId was missing and multiple active sessions exist; provide an explicit searchSessionId."
            )
    normalized_args["searchSessionId"] = normalized_search_session_id
    _guided_note_repair(
        repairs,
        field="searchSessionId",
        original=raw_search_session_id,
        normalized=normalized_search_session_id,
        reason="session_id_normalization",
    )

    if not normalized_source_id and normalized_search_session_id and workspace_registry is not None:
        record = None
        try:
            record = workspace_registry.get(normalized_search_session_id)
        except Exception as exc:
            logger.debug(
                "Failed to load workspace record %s while inferring sourceId: %s",
                normalized_search_session_id,
                exc,
            )
        if record is not None:
            candidates = _guided_record_source_candidates(record)
            if len(candidates) == 1:
                normalized_source_id = _guided_normalize_whitespace(candidates[0].get("sourceId"))
                warnings.append(f"sourceId was missing; inferred the only inspectable source '{normalized_source_id}'.")
    normalized_args["sourceId"] = normalized_source_id
    _guided_note_repair(
        repairs,
        field="sourceId",
        original=raw_source_id,
        normalized=normalized_source_id,
        reason="source_id_normalization",
    )
    if not normalized_source_id:
        warnings.append("sourceId was empty after normalization; source inspection may fail.")

    normalization = {
        "normalizedSearchSessionId": normalized_args.get("searchSessionId"),
        "normalizedSourceId": normalized_args.get("sourceId"),
        "repairs": repairs,
        "warnings": warnings,
    }
    return normalized_args, normalization


def _guided_normalization_payload(normalization: dict[str, Any]) -> dict[str, Any] | None:
    repairs = [repair for repair in normalization.get("repairs") or [] if isinstance(repair, dict)]
    warnings = [warning for warning in normalization.get("warnings") or [] if isinstance(warning, str) and warning]
    if not repairs and not warnings:
        return None
    payload = InputNormalization.model_validate(
        {
            **normalization,
            "repairs": [
                NormalizationRepair.model_validate(repair).model_dump(by_alias=True, exclude_none=True)
                for repair in repairs
            ],
            "warnings": warnings,
        }
    )
    return payload.model_dump(by_alias=True, exclude_none=True)


def _guided_is_known_item_query(query: str) -> bool:
    return looks_like_paper_identifier(query) or looks_like_citation_query(query) or looks_like_exact_title(query)


def _guided_mentions_literature(query: str, focus: str | None = None) -> bool:
    normalized = " ".join(part for part in [query, focus or ""] if part).lower()
    if not normalized:
        return False
    if any(term in normalized for term in _GUIDED_LITERATURE_TERMS):
        return True
    if "scholarship" in normalized:
        return True
    return bool(re.search(r"\b(?:doi|systematic review|meta-analysis|peer-reviewed|scientific reports?)\b", normalized))


def _guided_is_mixed_intent_query(
    query: str,
    focus: str | None = None,
    *,
    planner_regulatory_intent: str | None = None,
) -> bool:
    # LLM-first: when the planner has already classified this query as
    # ``hybrid_regulatory_plus_literature`` (either from the LLM or from the
    # planner's deterministic fallback), trust that signal directly and skip the
    # keyword heuristic. Callers without planner context (the current default at
    # the top of guided research dispatch, before any smart pass has run) fall
    # back to the deterministic regulatory + literature keyword check below.
    if planner_regulatory_intent == "hybrid_regulatory_plus_literature":
        return True
    return detect_regulatory_intent(query, focus) and _guided_mentions_literature(query, focus)


def _guided_reference_signal_words(candidate: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9'/-]*", candidate.lower())
    return [
        word
        for word in words
        if word not in _GUIDED_REFERENCE_GENERIC_CANDIDATE_WORDS
        and word not in _GUIDED_REFERENCE_UNCERTAINTY_MARKERS
        and word not in {"a", "an", "and", "for", "in", "of", "on", "the", "to", "with"}
    ]


def _guided_underspecified_reference_clarification(
    *,
    query: str,
    focus: str | None,
) -> dict[str, Any] | None:
    combined = _guided_normalize_whitespace(" ".join(part for part in [query, focus or ""] if part))
    if not combined or looks_like_paper_identifier(combined):
        return None
    parsed = parse_citation(combined)
    if parsed.identifier:
        return None

    citation_like = bool(
        parsed.year is not None or looks_like_citation_query(combined) or looks_like_exact_title(combined)
    )
    if not citation_like:
        return None

    if parsed.author_surnames or parsed.venue_hints:
        return None

    strongest_candidate_words = max(
        (len(_guided_reference_signal_words(candidate)) for candidate in parsed.title_candidates),
        default=0,
    )
    weak_anchor = strongest_candidate_words <= 4
    uncertainty_hits = sum(
        1
        for marker in _GUIDED_REFERENCE_UNCERTAINTY_MARKERS
        if re.search(rf"\b{re.escape(marker)}\b", combined, re.IGNORECASE)
    )
    if not weak_anchor or (uncertainty_hits == 0 and not parsed.looks_like_non_paper):
        return None

    if parsed.looks_like_non_paper or detect_regulatory_intent(query, focus):
        return {
            "reason": "underspecified_reference_fragment",
            "question": (
                "This looks like a vague reference fragment and may point to either a paper or a policy-style "
                "document. Add an exact title, one author surname, an agency or venue, or confirm which type "
                "of source you want before the server guesses."
            ),
            "options": [
                "add exact title",
                "add author surname",
                "add agency or venue",
                "paper vs policy source",
            ],
            "canProceedWithoutAnswer": True,
        }
    return {
        "reason": "underspecified_reference_fragment",
        "question": (
            "This looks like a vague paper/reference fragment. Add an exact title, one author surname, or a "
            "venue/year clue before guided research infers a likely paper from weak hints."
        ),
        "options": [
            "add exact title",
            "add author surname",
            "add venue or year",
            "use resolve_reference",
        ],
        "canProceedWithoutAnswer": True,
    }


def _guided_dedupe_source_records(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for source in sources:
        key = (
            str(source.get("sourceId") or "").strip(),
            str(source.get("canonicalUrl") or "").strip(),
            str(source.get("title") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def _guided_source_matches_reference(candidate: dict[str, Any], reference: Any) -> bool:
    normalized_reference = _guided_normalize_whitespace(reference)
    if not normalized_reference:
        return False
    lowered_reference = normalized_reference.lower()
    normalized_locator = _guided_normalize_source_locator(normalized_reference)
    for value in (
        candidate.get("sourceId"),
        candidate.get("sourceAlias"),
        candidate.get("citationText"),
        candidate.get("canonicalUrl"),
        candidate.get("retrievedUrl"),
        candidate.get("title"),
    ):
        normalized_candidate = _guided_normalize_whitespace(value)
        if not normalized_candidate:
            continue
        if normalized_candidate.lower() == lowered_reference:
            return True
        if normalized_locator and _guided_normalize_source_locator(normalized_candidate) == normalized_locator:
            return True
    return False


def _guided_source_records_share_surface(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_titles = {
        _guided_normalize_whitespace(left.get("title")).lower(),
        _guided_normalize_whitespace(left.get("citationText")).lower(),
    } - {""}
    right_titles = {
        _guided_normalize_whitespace(right.get("title")).lower(),
        _guided_normalize_whitespace(right.get("citationText")).lower(),
    } - {""}
    if left_titles and right_titles and left_titles & right_titles:
        return True

    left_locators = {
        _guided_normalize_source_locator(left.get("canonicalUrl")),
        _guided_normalize_source_locator(left.get("retrievedUrl")),
    } - {""}
    right_locators = {
        _guided_normalize_source_locator(right.get("canonicalUrl")),
        _guided_normalize_source_locator(right.get("retrievedUrl")),
    } - {""}
    return bool(left_locators and right_locators and left_locators & right_locators)


def _guided_source_identity(source: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(source.get("sourceId") or "").strip(),
        str(source.get("canonicalUrl") or "").strip(),
        str(source.get("title") or "").strip().lower(),
    )


def _guided_merge_source_records(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary)
    for key, value in secondary.items():
        if key not in merged or merged[key] in (None, "", [], {}):
            merged[key] = value
            continue
        if isinstance(value, bool) and value and not bool(merged[key]):
            merged[key] = True
    access_status, full_text_url_found, full_text_observed, body_text_embedded, qa_readable_text = (
        _guided_normalize_access_axes(merged)
    )
    source_type = str(merged.get("sourceType") or "unknown")
    merged["accessStatus"] = access_status
    merged["fullTextUrlFound"] = full_text_url_found
    merged["fullTextObserved"] = full_text_observed
    merged["bodyTextEmbedded"] = body_text_embedded
    merged["qaReadableText"] = qa_readable_text
    merged["verificationStatus"] = _guided_normalize_verification_status(
        merged,
        source_type=source_type,
        full_text_url_found=full_text_url_found,
        body_text_embedded=body_text_embedded,
    )
    return merged


def _guided_merge_source_record_sets(*record_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    ordered_keys: list[tuple[str, str, str]] = []
    for record_set in record_sets:
        for record in record_set:
            canonical_record = _guided_merge_source_records({}, record)
            key = _guided_source_identity(canonical_record)
            if key not in merged_by_key:
                merged_by_key[key] = canonical_record
                ordered_keys.append(key)
            else:
                merged_by_key[key] = _guided_merge_source_records(merged_by_key[key], canonical_record)
    return [merged_by_key[key] for key in ordered_keys]


def _guided_source_coverage_summary(
    *,
    sources: list[dict[str, Any]],
    leads: list[dict[str, Any]],
    base_coverage: dict[str, Any] | None,
) -> dict[str, Any] | None:
    summary = dict(base_coverage or {})
    visible_records = [record for record in [*sources, *leads] if isinstance(record, dict)]
    if not summary and not visible_records:
        return None
    summary["totalSources"] = len(visible_records)
    by_access_status: dict[str, int] = {}
    for record in visible_records:
        access_status = str(record.get("accessStatus") or "access_unverified").strip() or "access_unverified"
        by_access_status[access_status] = by_access_status.get(access_status, 0) + 1
    summary["byAccessStatus"] = by_access_status
    return summary


def _guided_merge_coverage_summaries(*coverages: dict[str, Any] | None) -> dict[str, Any] | None:
    usable = [coverage for coverage in coverages if isinstance(coverage, dict)]
    if not usable:
        return None
    if len(usable) == 1:
        return dict(usable[0])

    def _merge_list(field: str) -> list[Any]:
        merged: list[Any] = []
        seen_values: set[str] = set()
        for coverage in usable:
            for item in coverage.get(field) or []:
                marker = repr(item)
                if marker in seen_values:
                    continue
                seen_values.add(marker)
                merged.append(item)
        return merged

    likely_completeness = "unknown"
    for candidate, normalized_value in (
        ("incomplete", "incomplete"),
        ("partial", "partial"),
        ("likely_complete", "likely_complete"),
        ("complete", "likely_complete"),
        ("unknown", "unknown"),
        ("none", "unknown"),
    ):
        if any(str(coverage.get("likelyCompleteness") or "") == candidate for coverage in usable):
            likely_completeness = normalized_value
            break

    providers_attempted = _merge_list("providersAttempted")
    providers_succeeded = _merge_list("providersSucceeded")
    succeeded_markers = {repr(item) for item in providers_succeeded}
    providers_failed = [item for item in _merge_list("providersFailed") if repr(item) not in succeeded_markers]
    failed_markers = {repr(item) for item in providers_failed}
    providers_zero_results = [
        item
        for item in _merge_list("providersZeroResults")
        if repr(item) not in succeeded_markers and repr(item) not in failed_markers
    ]

    merged: dict[str, Any] = {
        "providersAttempted": providers_attempted,
        "providersSucceeded": providers_succeeded,
        "providersFailed": providers_failed,
        "providersZeroResults": providers_zero_results,
        "likelyCompleteness": likely_completeness,
        "searchMode": "guided_hybrid_research",
        "retrievalNotes": _merge_list("retrievalNotes"),
    }
    primary_document_coverage = usable[0].get("primaryDocumentCoverage")
    for coverage in usable:
        if coverage.get("primaryDocumentCoverage"):
            primary_document_coverage = coverage.get("primaryDocumentCoverage")
            break
    if primary_document_coverage is not None:
        merged["primaryDocumentCoverage"] = primary_document_coverage
    merged["summaryLine"] = (
        f"{len(merged['providersAttempted'])} provider(s) searched across blended literature and regulatory passes, "
        f"{len(merged['providersFailed'])} failed, {len(merged['providersZeroResults'])} returned zero results, "
        f"likely completeness: {likely_completeness}."
    )
    return merged


def _guided_merge_failure_summaries(*summaries: dict[str, Any] | None) -> dict[str, Any] | None:
    usable = [summary for summary in summaries if isinstance(summary, dict)]
    if not usable:
        return None
    if len(usable) == 1:
        return dict(usable[0])
    what_failed = (
        "; ".join(
            str(summary.get("whatFailed") or "").strip()
            for summary in usable
            if str(summary.get("whatFailed") or "").strip()
        )
        or None
    )
    what_still_worked_parts = [
        str(summary.get("whatStillWorked") or "").strip()
        for summary in usable
        if str(summary.get("whatStillWorked") or "").strip()
    ]
    completeness_parts = [
        str(summary.get("completenessImpact") or "").strip()
        for summary in usable
        if str(summary.get("completenessImpact") or "").strip()
    ]
    return {
        "outcome": "fallback_success" if any(summary.get("fallbackAttempted") for summary in usable) else "no_failure",
        "whatFailed": what_failed,
        "whatStillWorked": " ".join(dict.fromkeys(what_still_worked_parts)) or None,
        "fallbackAttempted": any(bool(summary.get("fallbackAttempted")) for summary in usable),
        "fallbackMode": "guided_hybrid_research",
        "primaryPathFailureReason": "; ".join(
            str(summary.get("primaryPathFailureReason") or "").strip()
            for summary in usable
            if str(summary.get("primaryPathFailureReason") or "").strip()
        )
        or None,
        "completenessImpact": " ".join(dict.fromkeys(completeness_parts)) or None,
        "recommendedNextAction": "review_partial_results",
    }
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


def _guided_trust_summary(
    sources: list[dict[str, Any]],
    evidence_gaps: list[str],
    *,
    classification_provenance: dict[str, Any] | None = None,
    subject_chain_gaps: list[str] | None = None,
) -> dict[str, Any]:
    # verifiedPrimarySourceCount now reflects the true primary-source bucket:
    # records marked ``isPrimarySource=True`` whose verification status is
    # either ``verified_primary_source`` or ``verified_metadata``. This fixes
    # the prior miscount that treated any ``verified_primary_source`` record
    # as primary regardless of the ``isPrimarySource`` flag (P0-3 item 2).
    verified_primary_source_count = sum(
        1
        for source in sources
        if source.get("isPrimarySource") is True
        and source.get("verificationStatus") in {"verified_primary_source", "verified_metadata"}
    )
    # Narrower full-text signal: isPrimarySource=True AND full-text verified.
    full_text_verified_primary_source_count = sum(
        1
        for source in sources
        if source.get("isPrimarySource") is True and source.get("verificationStatus") == "verified_primary_source"
    )
    # Retain the broader status-only counts for the combined verifiedSourceCount
    # total (keeps the overall denominator stable for downstream clients).
    status_verified_primary_source_count = sum(
        1 for source in sources if source.get("verificationStatus") == "verified_primary_source"
    )
    verified_metadata_source_count = sum(
        1 for source in sources if source.get("verificationStatus") == "verified_metadata"
    )
    on_topic_source_count = sum(1 for source in sources if source.get("topicalRelevance") == "on_topic")
    weak_match_reasons = [
        str(source.get("whyClassifiedAsWeakMatch") or source.get("note") or "").strip()
        for source in sources
        if source.get("topicalRelevance") == "weak_match"
        and str(source.get("whyClassifiedAsWeakMatch") or source.get("note") or "").strip()
    ]
    off_topic_reasons = [
        str(source.get("whyClassifiedAsWeakMatch") or source.get("note") or "").strip()
        for source in sources
        if source.get("topicalRelevance") == "off_topic"
        and str(source.get("whyClassifiedAsWeakMatch") or source.get("note") or "").strip()
    ]
    weak_match_rationales = [
        str(source.get("classificationRationale") or "").strip()
        for source in sources
        if source.get("topicalRelevance") == "weak_match" and str(source.get("classificationRationale") or "").strip()
    ]
    off_topic_rationales = [
        str(source.get("classificationRationale") or "").strip()
        for source in sources
        if source.get("topicalRelevance") == "off_topic" and str(source.get("classificationRationale") or "").strip()
    ]
    if verified_primary_source_count > 0 and on_topic_source_count > 0:
        strength_explanation = "Verified primary sources provide direct on-topic support."
    elif verified_metadata_source_count > 0 and on_topic_source_count > 0:
        strength_explanation = (
            "On-topic support is present, but some records remain metadata-verified rather than full-text verified."
        )
    elif weak_match_reasons:
        strength_explanation = (
            "The saved evidence is related, but the strongest remaining records are still scope-limited."
        )
    elif off_topic_reasons:
        strength_explanation = "Available sources are mostly off-topic for the saved query."
    else:
        strength_explanation = "No strong verified support was recorded for the saved query."
    authoritative_but_weak = _authoritative_but_weak_source_ids(sources)
    authoritative_but_weak_count = len(authoritative_but_weak)
    strong_on_topic_count = on_topic_source_count
    weak_match_bucket_count = sum(1 for source in sources if source.get("topicalRelevance") == "weak_match")
    off_topic_bucket_count = sum(1 for source in sources if source.get("topicalRelevance") == "off_topic")
    breakdown_fragments: list[str] = []
    if strong_on_topic_count:
        noun = "source" if strong_on_topic_count == 1 else "sources"
        breakdown_fragments.append(f"{strong_on_topic_count} strong on-topic verified {noun}")
    if authoritative_but_weak_count:
        noun = "source" if authoritative_but_weak_count == 1 else "sources"
        breakdown_fragments.append(f"{authoritative_but_weak_count} authoritative but weak-match {noun}")
        remaining_weak = max(weak_match_bucket_count - authoritative_but_weak_count, 0)
        if remaining_weak:
            noun = "lead" if remaining_weak == 1 else "leads"
            breakdown_fragments.append(f"{remaining_weak} other weak-match {noun}")
    elif weak_match_bucket_count:
        noun = "lead" if weak_match_bucket_count == 1 else "leads"
        breakdown_fragments.append(f"{weak_match_bucket_count} weak-match {noun}")
    if off_topic_bucket_count:
        noun = "lead" if off_topic_bucket_count == 1 else "leads"
        breakdown_fragments.append(f"{off_topic_bucket_count} off-target {noun}")
    if breakdown_fragments:
        strength_explanation = f"{strength_explanation} Breakdown: {', '.join(breakdown_fragments)}."
    summary = {
        "verifiedSourceCount": status_verified_primary_source_count + verified_metadata_source_count,
        "verifiedPrimarySourceCount": verified_primary_source_count,
        "fullTextVerifiedPrimarySourceCount": full_text_verified_primary_source_count,
        "verifiedMetadataSourceCount": verified_metadata_source_count,
        "onTopicSourceCount": on_topic_source_count,
        "weakMatchCount": sum(1 for source in sources if source.get("topicalRelevance") == "weak_match"),
        "offTopicCount": sum(1 for source in sources if source.get("topicalRelevance") == "off_topic"),
        "evidenceGapCount": len(evidence_gaps),
        "rationaleByBucket": {
            "weakMatch": weak_match_reasons[:3],
            "offTopic": off_topic_reasons[:3],
        },
        "classificationRationaleByBucket": {
            "weakMatch": weak_match_rationales[:3],
            "offTopic": off_topic_rationales[:3],
        },
        "authoritativeButWeak": authoritative_but_weak,
        "strengthExplanation": strength_explanation,
    }
    if authoritative_but_weak_count:
        # Missed-escalation prose note (P0-3 item 5): call out that these
        # authoritative records are NOT grounded evidence and subject-chain
        # grounding may be weak, so agents do not silently fold them in.
        noun = "source" if authoritative_but_weak_count == 1 else "sources"
        summary["authoritativeButWeakNote"] = (
            f"{authoritative_but_weak_count} authoritative {noun} found but not topically responsive "
            "(missed-escalation bucket); subject-chain grounding may be weak — do not treat as "
            "grounded evidence without a disambiguation or primary-source follow-up."
        )
    top_rationale: str | None = None
    if off_topic_rationales:
        top_rationale = f"Off-topic example: {off_topic_rationales[0]}"
    elif weak_match_rationales:
        top_rationale = f"Weak-match example: {weak_match_rationales[0]}"
    if top_rationale:
        summary["trustRationale"] = f"{strength_explanation} {top_rationale}".strip()[:280]
    else:
        summary["trustRationale"] = strength_explanation
    if classification_provenance and classification_provenance.get("total"):
        summary["classificationProvenance"] = classification_provenance
        summary["degradedClassification"] = bool(classification_provenance.get("degradedClassification"))
    # ws-dispatch-contract-trust (finding #5): surface planner subject-chain
    # gaps in machine-readable trust signals, not just the prose rationale
    # emitted by ``_compose_why_classified_weak_match``. Clients reading
    # ``trustSummary``/``confidenceSignals`` previously missed the same reason
    # shown in the human sentence.
    _subject_chain_gaps = [str(item).strip() for item in subject_chain_gaps or [] if str(item).strip()]
    if _subject_chain_gaps:
        summary["subjectChainGaps"] = _subject_chain_gaps
    return summary


def _guided_confidence_signals(
    *,
    status: str,
    sources: list[dict[str, Any]],
    evidence_gaps: list[str],
    degradation_reason: str | None = None,
    synthesis_mode: str | None = None,
    source: dict[str, Any] | None = None,
    evidence_use_plan_applied: bool = False,
    subject_chain_gaps: list[str] | None = None,
) -> dict[str, Any]:
    verified_on_topic_primary = sum(
        1
        for item in sources
        if item.get("topicalRelevance") == "on_topic" and item.get("verificationStatus") == "verified_primary_source"
    )
    verified_on_topic = sum(
        1
        for item in sources
        if item.get("topicalRelevance") == "on_topic"
        and item.get("verificationStatus") in {"verified_primary_source", "verified_metadata"}
    )
    if verified_on_topic_primary > 0 or verified_on_topic >= 3:
        evidence_quality_profile = "high"
    elif verified_on_topic > 0:
        evidence_quality_profile = "medium"
    else:
        evidence_quality_profile = "low"

    trust_revision_reason = degradation_reason or (evidence_gaps[0] if evidence_gaps else None)
    # ws-dispatch-contract-trust (finding #5): if no other revision reason
    # exists, surface the first subject-chain gap so the machine-readable
    # signal doesn't go empty when the only trust deficit is a planner gap.
    _subject_chain_gaps = [str(item).strip() for item in subject_chain_gaps or [] if str(item).strip()]
    if trust_revision_reason is None and _subject_chain_gaps:
        trust_revision_reason = _subject_chain_gaps[0]
    fallback_explanation = None
    if degradation_reason == "deterministic_synthesis_fallback":
        fallback_explanation = (
            "A deterministic fallback answered the follow-up because model-backed synthesis was unavailable."
        )
    if synthesis_mode is None:
        if status in {"answered", "succeeded"} and degradation_reason is None:
            synthesis_mode = "grounded"
        elif status in {"answered", "succeeded", "partial"}:
            synthesis_mode = "limited"
        else:
            synthesis_mode = "insufficient"

    source_scope_label = None
    source_scope_reason = None
    if source is None and len(sources) == 1:
        source = sources[0]
    if source is not None:
        topical_relevance = str(source.get("topicalRelevance") or "").strip()
        source_scope_reason = str(source.get("whyClassifiedAsWeakMatch") or source.get("note") or "").strip() or None
        if topical_relevance == "on_topic":
            source_scope_label = "directly_responsive"
        elif topical_relevance == "weak_match" and source.get("verificationStatus") == "verified_primary_source":
            source_scope_label = "authoritative_but_scope_limited"
        elif topical_relevance == "weak_match":
            source_scope_label = "related_but_incomplete"
        elif topical_relevance == "off_topic":
            source_scope_label = "off_topic"

    result = ConfidenceSignals(
        evidenceQualityProfile=cast(Any, evidence_quality_profile),
        synthesisMode=synthesis_mode,
        trustRevisionReason=trust_revision_reason,
        evidenceUsePlanApplied=evidence_use_plan_applied,
        fallbackExplanation=fallback_explanation,
        sourceScopeLabel=source_scope_label,
        sourceScopeReason=source_scope_reason,
    ).model_dump(by_alias=True, exclude_none=True)

    # Workstream C (ws-trust-ux-deepen): additive detail fields that expose the
    # richer WS-C enums without breaking the existing ``evidenceQualityProfile``
    # / ``synthesisMode`` contract.
    result["evidenceProfileDetail"] = _evidence_quality_detail(sources)
    result["synthesisPath"] = _synthesis_path(
        status=status,
        sources=sources,
        evidence_gaps=evidence_gaps,
        synthesis_mode=synthesis_mode,
    )
    narrative = _trust_revision_narrative(
        sources=sources,
        evidence_gaps=evidence_gaps,
        degradation_reason=degradation_reason,
    )
    if narrative:
        result["trustRevisionNarrative"] = narrative
    # ws-dispatch-contract-trust (finding #5): expose subject-chain gaps as a
    # first-class additive field so clients reading ``confidenceSignals`` get
    # the same reason already surfaced in prose by
    # ``_compose_why_classified_weak_match``.
    if _subject_chain_gaps:
        result["subjectChainGaps"] = _subject_chain_gaps
    return result


def _guided_sources_all_off_topic(sources: list[dict[str, Any]] | None) -> bool:
    """Return True when ``sources`` is non-empty and every entry is ``off_topic``.

    Seventh rubber-duck pass (finding 2): shared predicate used by
    ``_guided_failure_summary`` / ``_guided_next_actions`` / ``_guided_result_state``
    so their cross-field routing stays consistent when the current response
    contains only off-topic sources.
    """

    items = [source for source in (sources or []) if isinstance(source, dict)]
    if not items:
        return False
    return all(str(source.get("topicalRelevance") or "").strip().lower() == "off_topic" for source in items)


def _guided_failure_summary(
    *,
    failure_summary: dict[str, Any] | None,
    status: str,
    sources: list[dict[str, Any]],
    evidence_gaps: list[str],
    all_sources_off_topic: bool = False,
) -> dict[str, Any]:
    if failure_summary is not None:
        summary = dict(failure_summary)
    else:
        summary = {}
    outcome = str(summary.get("outcome") or "").strip()
    if not outcome:
        if status == "abstained" and not sources:
            outcome = "partial_success"
        elif status == "abstained":
            outcome = "no_failure"
        elif summary.get("fallbackAttempted"):
            outcome = "fallback_success"
        else:
            outcome = "no_failure"
    # Seventh rubber-duck pass (finding 2): when every current source is
    # off_topic, inspect_source cannot rescue the result — route to research
    # so the default recommendation agrees with _guided_result_state's
    # all-off-topic routing.
    effective_has_inspectable = bool(sources) and not all_sources_off_topic
    recommended_next_action = summary.get("recommendedNextAction")
    if not recommended_next_action:
        recommended_next_action = "inspect_source" if effective_has_inspectable else "research"
    completeness_impact = summary.get("completenessImpact")
    if not completeness_impact and evidence_gaps:
        completeness_impact = evidence_gaps[0]
    what_still_worked = summary.get("whatStillWorked")
    if not what_still_worked:
        if effective_has_inspectable:
            what_still_worked = "The guided run still returned inspectable sources."
        elif sources:
            # Ninth rubber-duck pass (finding 2): an all-off-topic pool is not
            # "inspectable" in the routing sense — do not claim otherwise.
            what_still_worked = "The guided run returned sources, but all were off-topic for the question."
        else:
            what_still_worked = (
                "No provider failures were recorded, but the evidence was not strong enough to ground a result."
            )
    fallback_attempted = bool(summary.get("fallbackAttempted"))
    fallback_mode = summary.get("fallbackMode")
    # Invariant: if fallbackMode is non-null, fallbackAttempted must be True
    if fallback_mode is not None and not fallback_attempted:
        fallback_attempted = True
    return {
        "outcome": outcome,
        "whatFailed": summary.get("whatFailed"),
        "whatStillWorked": what_still_worked,
        "fallbackAttempted": fallback_attempted,
        "fallbackMode": fallback_mode,
        "primaryPathFailureReason": summary.get("primaryPathFailureReason"),
        "completenessImpact": completeness_impact,
        "recommendedNextAction": recommended_next_action,
    }


def _guided_result_meaning(
    *,
    status: str,
    verified_findings: list[dict[str, Any]],
    evidence_gaps: list[str],
    coverage: dict[str, Any] | None,
    failure_summary: dict[str, Any],
    source_count: int = 0,
    all_sources_off_topic: bool = False,
) -> str:
    if verified_findings:
        return f"This result contains {len(verified_findings)} verified finding(s) grounded in the returned sources."
    if status == "partial":
        if source_count <= 0 or all_sources_off_topic:
            if all_sources_off_topic:
                return (
                    "This result returned sources, but all were off-topic for the query. "
                    "Tighten the anchor (exact title, DOI, species, agency, venue, or year range) and rerun research."
                )
            return (
                "This result is currently metadata-only and did not include inspectable sources. "
                "Use follow_up_research or rerun research with a tighter anchor."
            )
        return "This result found some relevant evidence, but the trust or coverage state is still incomplete."
    if status == "needs_disambiguation":
        return "This result needs a stronger anchor before the server can produce a grounded answer."
    if status == "abstained":
        return "This result did not find sufficiently trustworthy evidence to support a grounded answer."
    if failure_summary.get("outcome") not in {None, "no_failure"}:
        return "This result reflects degraded retrieval and should be treated as a partial recovery path."
    summary_line = str((coverage or {}).get("summaryLine") or "").strip()
    return summary_line or "This result should be reviewed source by source before relying on it."


def _guided_deterministic_fallback_used(provider_bundle: Any | None) -> bool:
    if provider_bundle is None or not hasattr(provider_bundle, "selection_metadata"):
        return False
    try:
        selection = provider_bundle.selection_metadata()
    except Exception:
        return False
    configured = str(selection.get("configuredSmartProvider") or "").strip()
    active = str(selection.get("activeSmartProvider") or "").strip()
    return bool(configured and configured != "deterministic" and active == "deterministic")


def _guided_partial_recovery_possible(
    *,
    coverage_summary: dict[str, Any] | None,
    failure_summary: dict[str, Any] | None,
) -> bool:
    coverage = coverage_summary or {}
    failed = [str(provider).strip() for provider in (coverage.get("providersFailed") or []) if str(provider).strip()]
    if failed:
        return True
    failure = failure_summary or {}
    return bool(
        failure.get("fallbackAttempted") or failure.get("fallbackMode") or failure.get("primaryPathFailureReason")
    )


async def _guided_research_status(
    *,
    query: str,
    intent: str,
    sources: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    unverified_leads_count: int,
    coverage_summary: dict[str, Any] | None,
    failure_summary: dict[str, Any] | None,
    clarification: dict[str, Any] | None,
    provider_bundle: Any | None = None,
) -> tuple[str, str | None]:
    if clarification is not None:
        return "needs_disambiguation", None
    if intent == "known_item" and findings:
        return "succeeded", None
    if intent == "regulatory":
        primary_document_coverage = cast(
            dict[str, Any] | None,
            (coverage_summary or {}).get("primaryDocumentCoverage"),
        )
        primary_sources = [
            source
            for source in sources
            if source.get("topicalRelevance") == "on_topic" and bool(source.get("isPrimarySource"))
        ]
        if primary_document_coverage is not None and primary_document_coverage.get("currentTextRequested"):
            if primary_document_coverage.get("currentTextSatisfied"):
                return ("partial" if failure_summary is not None else "succeeded"), None
            if primary_sources:
                return "partial", None
            if unverified_leads_count > 0:
                return "partial", None
            return "abstained", None
        if primary_sources:
            return ("partial" if failure_summary is not None else "succeeded"), None
        if unverified_leads_count > 0:
            return "partial", None
        return ("needs_disambiguation" if sources else "abstained"), None
    if len(findings) >= 2:
        base_status = "partial" if failure_summary is not None else "succeeded"
    else:
        on_topic_verified = sum(
            1
            for source in sources
            if source.get("topicalRelevance") == "on_topic"
            and source.get("verificationStatus") in {"verified_primary_source", "verified_metadata"}
        )
        if on_topic_verified >= 5:
            base_status = "partial" if failure_summary is not None else "succeeded"
        elif sources:
            base_status = "partial"
        elif unverified_leads_count > 0:
            base_status = "partial"
        else:
            base_status = "abstained"

    if (
        base_status == "abstained"
        and _guided_deterministic_fallback_used(provider_bundle)
        and _guided_partial_recovery_possible(
            coverage_summary=coverage_summary,
            failure_summary=failure_summary,
        )
    ):
        return (
            "partial",
            (
                "Configured smart provider was unavailable, so guided research stayed on "
                "deterministic fallback while retrieval remained incomplete."
            ),
        )

    adequacy_reason: str | None = None
    verified_sources = [
        source
        for source in sources
        if source.get("topicalRelevance") == "on_topic"
        and source.get("verificationStatus") in {"verified_primary_source", "verified_metadata"}
    ]
    if (
        base_status == "partial"
        and provider_bundle is not None
        and verified_sources
        and intent not in {"known_item", "regulatory"}
    ):
        try:
            adequacy = await provider_bundle.aassess_result_adequacy(
                query=query,
                intent=intent,
                verified_sources=verified_sources,
                evidence_gaps=[],
            )
            adequacy_label = str(adequacy.get("adequacy") or "partial")
            adequacy_reason = str(adequacy.get("reason") or "").strip() or None
            if adequacy_label == "succeeded":
                base_status = "succeeded"
            elif adequacy_label == "insufficient":
                base_status = "abstained"
        except Exception:
            adequacy_reason = None
    return base_status, adequacy_reason


def _guided_deterministic_evidence_gaps(
    *,
    query: str,
    intent: str,
    sources: list[dict[str, Any]],
    existing_evidence_gaps: list[str],
    retrieval_hypotheses: list[str],
    coverage_summary: dict[str, Any] | None,
    timeline: dict[str, Any] | None,
    anchor_type: str | None,
) -> list[str]:
    return generate_evidence_gaps_without_llm(
        query=query,
        intent=intent,
        sources=sources,
        evidence_gaps=existing_evidence_gaps,
        retrieval_hypotheses=retrieval_hypotheses,
        coverage_summary=coverage_summary,
        timeline=timeline,
        anchor_type=anchor_type,
    )


async def _guided_generate_evidence_gaps(
    *,
    query: str,
    intent: str,
    sources: list[dict[str, Any]],
    existing_evidence_gaps: list[str],
    coverage_summary: dict[str, Any] | None,
    strategy_metadata: dict[str, Any] | None,
    timeline: dict[str, Any] | None,
    provider_bundle: Any | None,
) -> list[str]:
    metadata = strategy_metadata or {}
    retrieval_hypotheses = [
        str(item).strip() for item in metadata.get("retrievalHypotheses") or [] if str(item).strip()
    ]
    anchor_type = str(metadata.get("anchorType") or "").strip() or None
    deterministic_gaps = _guided_deterministic_evidence_gaps(
        query=query,
        intent=intent,
        sources=sources,
        existing_evidence_gaps=existing_evidence_gaps,
        retrieval_hypotheses=retrieval_hypotheses,
        coverage_summary=coverage_summary,
        timeline=timeline,
        anchor_type=anchor_type,
    )
    if provider_bundle is None or not hasattr(provider_bundle, "agenerate_evidence_gaps"):
        return deterministic_gaps
    try:
        model_gaps = await provider_bundle.agenerate_evidence_gaps(
            query=query,
            intent=intent,
            sources=sources,
            evidence_gaps=existing_evidence_gaps,
            retrieval_hypotheses=retrieval_hypotheses,
            coverage_summary=coverage_summary,
            timeline=timeline,
            anchor_type=anchor_type,
        )
        cleaned_model_gaps = [str(gap).strip() for gap in model_gaps or [] if str(gap).strip()]
        if cleaned_model_gaps:
            return cleaned_model_gaps
    except Exception:
        logger.debug("Guided evidence-gap generation failed; using deterministic fallback.")
    return deterministic_gaps


def _guided_machine_failure_payload(
    *,
    search_session_id: str | None,
    error: Exception,
    normalization: dict[str, Any] | None = None,
    execution_provenance: dict[str, Any] | None = None,
    saved_session_has_sources: bool = False,
    saved_session_all_off_topic: bool = False,
) -> dict[str, Any]:
    evidence_gaps = ["Smart runtime returned an invalid or unstructured result payload, so guided output was degraded."]
    # Fifth rubber-duck pass (finding 2): when a saved session is still
    # inspectable, recommend inspect_source rather than research so the
    # failure payload agrees with the result-state routing used elsewhere.
    saved_session_inspectable = saved_session_has_sources and not saved_session_all_off_topic
    recommended_next_action = "inspect_source" if saved_session_inspectable else "research"
    failure_summary = _guided_failure_summary(
        failure_summary={
            "outcome": "total_failure",
            "whatFailed": "smart_runtime_structural_failure",
            "whatStillWorked": "The guided wrapper recovered and returned a machine-readable failure state.",
            "fallbackAttempted": False,
            "fallbackMode": None,
            "primaryPathFailureReason": str(type(error).__name__),
            "completenessImpact": evidence_gaps[0],
            "recommendedNextAction": recommended_next_action,
        },
        status="partial",
        sources=[],
        evidence_gaps=evidence_gaps,
    )
    payload: dict[str, Any] = {
        "intent": "discovery",
        "status": "partial",
        "searchSessionId": search_session_id,
        "summary": "Smart retrieval failed structurally; the server returned a safe machine-readable failure state.",
        "verifiedFindings": [],
        "sources": [],
        "unverifiedLeads": [],
        "evidenceGaps": evidence_gaps,
        "trustSummary": _guided_trust_summary([], evidence_gaps),
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
            search_session_id=search_session_id,
            status="partial",
            has_sources=False,
            saved_session_inspectable=saved_session_inspectable,
        ),
        "resultState": _guided_result_state(
            status="partial",
            sources=[],
            evidence_gaps=evidence_gaps,
            search_session_id=search_session_id,
            saved_session_has_sources=saved_session_has_sources,
            saved_session_all_off_topic=saved_session_all_off_topic,
        ),
        "machineFailure": MachineFailure(
            category="smart_runtime_structural_failure",
            errorType=type(error).__name__,
            error=str(error),
            retryable=True,
            bestNextInternalAction=recommended_next_action,
        ).model_dump(by_alias=True, exclude_none=True),
    }
    if execution_provenance is not None:
        payload["executionProvenance"] = execution_provenance
    normalization_payload = _guided_normalization_payload(normalization or {})
    if normalization_payload is not None:
        payload["inputNormalization"] = normalization_payload
    return payload


def _guided_summary(
    intent: str,
    status: str,
    findings: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    *,
    routing_summary: dict[str, Any] | None = None,
    pass_modes: list[str] | None = None,
) -> str:
    all_sources_off_topic = _guided_sources_all_off_topic(sources)
    if findings:
        top_claim = str(findings[0].get("claim") or "").strip()
        additional_count = max(len(findings) - 1, 0)
        if additional_count:
            summary = f"Top result: {top_claim}. Verified support includes {additional_count} additional source(s)."
        else:
            summary = f"Top result: {top_claim}."
    elif sources and all_sources_off_topic:
        # R12 finding: when every source is off_topic, do NOT tell callers to
        # "inspect the source" — that contradicts research-next routing.
        summary = (
            "The search returned sources, but every candidate was off-topic for the request. "
            "Tighten the anchor (exact title, DOI, species, agency, venue, or year range) and rerun research."
        )
    elif sources:
        top_title = str(sources[0].get("title") or sources[0].get("sourceId") or "").strip()
        if top_title:
            summary = (
                f"Top result: {top_title}. Evidence is still partial or mixed, "
                "so inspect the source before relying on it."
            )
        else:
            summary = (
                "The search found some source leads, but the evidence stayed too weak, off-topic, or incomplete "
                "for a grounded summary."
            )
    elif status == "needs_disambiguation":
        summary = "The request needs a more specific anchor before the system can build a grounded result."
    else:
        summary = "No sufficiently trustworthy evidence was found for a grounded result."

    notes: list[str] = []
    routing = routing_summary if isinstance(routing_summary, dict) else {}
    query_specificity = str(routing.get("querySpecificity") or "").strip()
    ambiguity_level = str(routing.get("ambiguityLevel") or "").strip()
    hypotheses = [str(item).strip() for item in routing.get("retrievalHypotheses") or [] if str(item).strip()]
    if hypotheses:
        hypothesis_label = "hypothesis" if len(hypotheses) == 1 else "hypotheses"
        notes.append(
            "The query was broad or ambiguous, so the server explored "
            f"{len(hypotheses)} bounded retrieval {hypothesis_label}."
        )
    elif query_specificity == "low" or ambiguity_level in {"medium", "high"}:
        notes.append("The query stayed broad or ambiguous, so the server blended nearby retrieval routes.")
    if pass_modes and "regulatory" in pass_modes and "review" in pass_modes:
        notes.append("This result blends regulatory and literature passes.")
    if notes:
        summary = f"{summary} {' '.join(dict.fromkeys(notes))}"
    return summary


def _guided_next_actions(
    *,
    search_session_id: str | None,
    status: str,
    has_sources: bool,
    calling_tool: str | None = None,
    saved_session_inspectable: bool = False,
    all_sources_off_topic: bool = False,
) -> list[str]:
    actions: list[str] = []
    # Sixth rubber-duck pass (finding 2): even when the current response has no
    # sources, a saved session can still hold inspectable candidates. Surface
    # inspect_source in nextActions so it agrees with the failure-summary and
    # machine-failure bestNextInternalAction routing.
    # Seventh rubber-duck pass (finding 2): when the current response has
    # sources but every one is off_topic, inspect_source is not productive —
    # treat the current response as empty for inspect-routing purposes so
    # nextActions agrees with bestNextInternalAction ("research").
    effective_has_inspectable = has_sources and not all_sources_off_topic
    inspect_relevant = effective_has_inspectable or saved_session_inspectable
    if search_session_id and inspect_relevant and calling_tool != "inspect_source":
        actions.append(
            f"Use inspect_source with searchSessionId='{search_session_id}' and one sourceId to inspect evidence."
        )
    if search_session_id:
        actions.append(
            f"Use follow_up_research with searchSessionId='{search_session_id}' to ask one grounded follow-up question."
        )
    if status in {"abstained", "needs_disambiguation"}:
        actions.append("Narrow the request with a specific title, DOI, species name, agency, venue, or year range.")
    if status == "partial":
        actions.append("Refine the request to reduce evidence gaps before treating the result as settled.")
    actions.append(
        "Use get_runtime_status if behavior differs across environments and you need the active runtime truth."
    )
    return actions[:4]


def _guided_missing_evidence_type(
    *,
    status: str,
    evidence_gaps: list[str],
    sources: list[dict[str, Any]],
) -> str:
    if status in {"succeeded", "answered"}:
        return "none"
    joined_gaps = " ".join(str(gap).lower() for gap in evidence_gaps)
    if "off-topic" in joined_gaps:
        return "off_topic_only"
    if any(marker in joined_gaps for marker in ("clarif", "anchor", "disambiguation")):
        return "anchor_missing"
    if any(marker in joined_gaps for marker in ("provider", "timeout", "failed", "error")):
        return "provider_gap"
    if not sources:
        return "no_sources"
    return "coverage_gap"


def _guided_saved_session_topicality(
    candidates: list[dict[str, Any]] | None,
) -> tuple[bool, bool]:
    """Return ``(has_sources, all_off_topic)`` for saved-session candidates.

    Fifth rubber-duck pass: routing the empty-current-response path to
    ``inspect_source`` must distinguish a saved session that still holds
    on-topic/weak-match evidence from one where every stored candidate has
    already been classified ``off_topic``. The latter should fall through to
    ``research`` like the current-response all-off-topic path does.
    """

    items = [c for c in (candidates or []) if isinstance(c, dict)]
    has_sources = bool(items)
    all_off_topic = has_sources and all(
        str(c.get("topicalRelevance") or "").strip().lower() == "off_topic" for c in items
    )
    return has_sources, all_off_topic


def _guided_best_next_internal_action(
    *,
    status: str,
    has_sources: bool,
    search_session_id: str | None,
    saved_session_has_sources: bool = False,
    saved_session_all_off_topic: bool = False,
    all_sources_off_topic: bool = False,
) -> str:
    normalized_status = str(status or "").strip().lower()
    weak_statuses = {
        "abstained",
        "needs_disambiguation",
        "failed",
        "insufficient_evidence",
        "partial",
    }
    # When every returned source is off-topic, inspect_source cannot rescue the
    # result; the agent should refine the query instead of chasing irrelevant
    # evidence.
    if has_sources and search_session_id and not all_sources_off_topic:
        return "inspect_source"
    # The CURRENT response may have no sources (e.g., smart runtime unavailable
    # or inspect_source retry with a wrong sourceId), yet the SAVED SESSION may
    # still be inspectable. In that case prefer inspect_source over research so
    # resultState agrees with the failureSummary's recommended retry. If every
    # saved candidate is off_topic, fall through to research instead so the
    # 9ee3168 "all off-topic → research" guarantee still holds.
    if not has_sources and saved_session_has_sources and not saved_session_all_off_topic and search_session_id:
        return "inspect_source"
    # Without inspectable sources, neither inspect_source nor another follow_up_research
    # over the same (empty) session can progress: keep the guidance aligned with the
    # failureSummary's recommended retry instead of looping the agent.
    if not has_sources and normalized_status in weak_statuses:
        return "research"
    if all_sources_off_topic:
        return "research"
    if search_session_id:
        return "follow_up_research"
    if normalized_status in weak_statuses:
        return "research"
    return "research"


def _guided_result_state(
    *,
    status: str,
    sources: list[dict[str, Any]],
    evidence_gaps: list[str],
    search_session_id: str | None,
    saved_session_has_sources: bool = False,
    saved_session_all_off_topic: bool = False,
    saved_session_inspectable_override: bool | None = None,
) -> dict[str, Any]:
    has_sources = bool(sources)
    all_sources_off_topic = has_sources and all(
        str(source.get("topicalRelevance") or "").strip().lower() == "off_topic"
        for source in sources
        if isinstance(source, dict)
    )
    normalized_status = str(status or "").strip() or "unknown"
    if normalized_status in {"succeeded", "answered"} and has_sources and not all_sources_off_topic:
        groundedness = "grounded"
    elif normalized_status == "answered":
        groundedness = "partial"
    elif normalized_status in {"partial", "insufficient_evidence"}:
        groundedness = "partial"
    elif normalized_status in {"abstained", "needs_disambiguation", "failed"}:
        groundedness = "insufficient_evidence"
    else:
        groundedness = "unknown"
    # Ninth rubber-duck pass (finding 3): when the status claims success but
    # every returned source is off_topic, the result is not actually grounded
    # in on-topic evidence. Downgrade groundedness to "insufficient_evidence"
    # so it agrees with bestNextInternalAction="research".
    if all_sources_off_topic and normalized_status in {"succeeded", "answered"}:
        groundedness = "insufficient_evidence"
    # Fifth rubber-duck pass (finding 3): hasInspectableSources must agree with
    # bestNextInternalAction. When the current response is empty but the saved
    # session still carries on_topic/weak_match evidence, the saved candidates
    # remain reachable via inspect_source, so the flag has to reflect that.
    saved_session_inspectable = (
        saved_session_inspectable_override
        if saved_session_inspectable_override is not None
        else (saved_session_has_sources and not saved_session_all_off_topic)
    )
    current_inspectable = has_sources and not all_sources_off_topic
    inspectable_sources = current_inspectable or saved_session_inspectable
    # Ninth rubber-duck pass (finding 1): canAnswerFollowUp must reflect whether
    # inspectable evidence is actually reachable. An all-off-topic current pool
    # cannot ground a follow-up, while a saved-session-inspectable case can.
    missing_evidence_type = _guided_missing_evidence_type(
        status=normalized_status,
        evidence_gaps=evidence_gaps,
        sources=sources,
    )
    # Ninth rubber-duck pass (finding 3): normalize missingEvidenceType for the
    # all-off-topic success/answered case so it reflects the off-topic gap
    # instead of advertising "none".
    if all_sources_off_topic and missing_evidence_type == "none":
        missing_evidence_type = "off_topic_only"
    state = GuidedResultState(
        status=normalized_status,
        groundedness=groundedness,
        hasInspectableSources=inspectable_sources,
        canAnswerFollowUp=bool(search_session_id) and inspectable_sources,
        bestNextInternalAction=_guided_best_next_internal_action(
            status=normalized_status,
            has_sources=has_sources,
            search_session_id=search_session_id,
            saved_session_has_sources=saved_session_has_sources,
            saved_session_all_off_topic=saved_session_all_off_topic,
            all_sources_off_topic=all_sources_off_topic,
        ),
        missingEvidenceType=missing_evidence_type,
    )
    return state.model_dump(by_alias=True, exclude_none=True)


def _guided_record_source_candidates(record: Any) -> list[dict[str, Any]]:
    payload = record.payload if isinstance(record.payload, dict) else {}
    has_explicit_source_payload = any(
        isinstance(payload.get(key), list) and bool(payload.get(key))
        for key in ("evidence", "sources", "structuredSources", "leads", "candidateLeads", "unverifiedLeads")
    )
    payload_sources = [source for source in payload.get("sources") or [] if isinstance(source, dict)]
    structured_sources = [
        _guided_source_record_from_structured_source(source, index=index)
        for index, source in enumerate(payload.get("structuredSources") or [], start=1)
        if isinstance(source, dict)
    ]
    evidence_sources = [
        _guided_source_record_from_structured_source(source, index=index)
        for index, source in enumerate(payload.get("evidence") or [], start=1)
        if isinstance(source, dict)
    ]
    lead_sources = [
        _guided_source_record_from_structured_source(source, index=index)
        for key in ("leads", "candidateLeads", "unverifiedLeads")
        for index, source in enumerate(payload.get(key) or [], start=1)
        if isinstance(source, dict)
    ]
    query = str(record.query or payload.get("query") or "")
    explicit_candidates = _guided_dedupe_source_records(
        _guided_merge_source_record_sets(
            payload_sources,
            structured_sources,
            evidence_sources,
            lead_sources,
        )
    )
    paper_sources = [
        _guided_source_record_from_paper(query, paper, index=index)
        for index, paper in enumerate(getattr(record, "papers", []) or [], start=1)
        if isinstance(paper, dict)
    ]
    if not paper_sources:
        return explicit_candidates
    if not has_explicit_source_payload:
        return _guided_dedupe_source_records(_guided_merge_source_record_sets(explicit_candidates, paper_sources))

    augmented_candidates = list(explicit_candidates)
    for paper_source in paper_sources:
        paper_source_id = str(paper_source.get("sourceId") or "").strip()
        if not paper_source_id:
            continue
        if any(_guided_source_matches_reference(candidate, paper_source_id) for candidate in augmented_candidates):
            continue
        merged_candidate = paper_source
        for candidate in explicit_candidates:
            if _guided_source_records_share_surface(candidate, paper_source):
                merged_candidate = _guided_merge_source_records(paper_source, candidate)
                break
        augmented_candidates.append(merged_candidate)
    return _guided_dedupe_source_records(augmented_candidates)


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


def _guided_follow_up_status(status: str | None) -> str:
    normalized = str(status or "").strip()
    if normalized in {"succeeded", "partial", "needs_disambiguation", "abstained", "failed"}:
        return normalized
    if normalized == "answered":
        return "succeeded"
    if normalized == "insufficient_evidence":
        return "partial"
    return "partial"


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


def _guided_compact_response_if_needed(*, tool_name: str, response: dict[str, Any]) -> dict[str, Any]:
    compact_fields: set[str] | None = None
    if tool_name == "follow_up_research":
        answer_status = str(response.get("answerStatus") or "").strip()
        if answer_status == "insufficient_evidence":
            compact_fields = _FOLLOW_UP_COMPACT_FIELDS
    elif tool_name == "research":
        status = str(response.get("status") or response.get("resultStatus") or "").strip()
        if status == "abstained":
            compact_fields = _RESEARCH_COMPACT_FIELDS

    if compact_fields is None:
        return response

    compacted = {key: value for key, value in response.items() if key in compact_fields}
    for key in list(compacted):
        if key in _COMPACT_NULL_OK_FIELDS:
            continue
        value = compacted[key]
        if value is None or value == [] or value == {}:
            compacted.pop(key, None)
    if any(key in response for key in ("evidence", "leads", *sorted(_LEGACY_GUIDED_FIELDS))):
        compacted["sourcesSuppressed"] = True
    compacted["legacyFieldsIncluded"] = False
    return compacted


def _guided_finalize_response(
    *,
    tool_name: str,
    response: dict[str, Any],
    response_mode: str = "standard",
    include_legacy_fields: bool = False,
) -> dict[str, Any]:
    finalized = dict(response)
    for key in ("verifiedFindings", "unverifiedLeads"):
        if finalized.get(key) == []:
            finalized.pop(key, None)
    finalized = _guided_compact_response_if_needed(tool_name=tool_name, response=finalized)
    if tool_name == "follow_up_research" and "sourcesSuppressed" not in finalized:
        finalized = _apply_follow_up_response_mode(
            finalized,
            response_mode=response_mode,
            include_legacy_fields=include_legacy_fields,
        )
    if "legacyFieldsIncluded" not in finalized:
        finalized["legacyFieldsIncluded"] = any(key in finalized for key in _LEGACY_GUIDED_FIELDS)
    return finalized


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


async def _guided_contract_fields(
    *,
    query: str,
    intent: str,
    status: str,
    sources: list[dict[str, Any]],
    unverified_leads: list[dict[str, Any]],
    evidence_gaps: list[str],
    coverage_summary: dict[str, Any] | None,
    strategy_metadata: dict[str, Any] | None,
    timeline: dict[str, Any] | None = None,
    pass_modes: list[str] | None = None,
    review_pass_reason: str | None = None,
    answer_text: str = "",
    provider_bundle: Any | None = None,
) -> dict[str, Any]:
    evidence, leads = build_evidence_records(sources=sources, leads=unverified_leads)
    routing_summary = build_routing_decision(
        query=query,
        intent=intent,
        strategy_metadata=strategy_metadata,
        coverage_summary=coverage_summary,
    ).model_dump(by_alias=True)
    result_status = _guided_follow_up_status(status)

    # P0-1 Fix #5: compute the evidence quality profile (same derivation as
    # ``_guided_confidence_signals``) so the answerability ladder and the
    # confidence signals agree. When the profile is ``low`` the ladder must
    # not advertise ``grounded``.
    _verified_on_topic_primary = sum(
        1
        for item in sources
        if item.get("topicalRelevance") == "on_topic" and item.get("verificationStatus") == "verified_primary_source"
    )
    _verified_on_topic = sum(
        1
        for item in sources
        if item.get("topicalRelevance") == "on_topic"
        and item.get("verificationStatus") in {"verified_primary_source", "verified_metadata"}
    )
    if _verified_on_topic_primary > 0 or _verified_on_topic >= 3:
        _evidence_quality_profile = "high"
    elif _verified_on_topic > 0:
        _evidence_quality_profile = "medium"
    else:
        _evidence_quality_profile = "low"

    answerability = classify_answerability(
        status=status,
        evidence=evidence,
        leads=leads,
        evidence_gaps=evidence_gaps,
        answer_text=answer_text,
        evidence_quality_profile=_evidence_quality_profile,
    )

    if provider_bundle is not None and answer_text and answerability == "grounded":
        try:
            validation = await provider_bundle.avalidate_answer_status(
                query=query,
                answer_text=answer_text,
                evidence_count=len(evidence),
            )
            if validation is not None:
                llm_mapped = _LLM_ANSWERABILITY_MAP.get(validation.classification)
                if llm_mapped and llm_mapped != answerability:
                    answerability = llm_mapped
        except Exception:
            logger.debug("LLM answer status validation failed; using deterministic result.")

    metadata_for_routing = strategy_metadata or {}
    subject_card_payload: dict[str, Any] | None = None
    raw_subject_card = metadata_for_routing.get("subjectCard")
    if isinstance(raw_subject_card, dict) and raw_subject_card:
        subject_card_payload = dict(raw_subject_card)
    elif raw_subject_card is not None and hasattr(raw_subject_card, "model_dump"):
        try:
            dumped = raw_subject_card.model_dump(by_alias=True, exclude_none=True)  # type: ignore[union-attr]
        except Exception:  # pragma: no cover - defensive
            dumped = {}
        if dumped:
            subject_card_payload = dumped
    subject_chain_gaps_payload = [
        str(item).strip() for item in metadata_for_routing.get("subjectChainGaps") or [] if str(item).strip()
    ]
    intent_family_payload = str(metadata_for_routing.get("intentFamily") or "").strip() or None
    regulatory_intent_payload = str(metadata_for_routing.get("regulatoryIntent") or "").strip() or None

    return {
        "resultStatus": result_status,
        "answerability": answerability,
        "routingSummary": {
            "intent": routing_summary["intent"],
            "decisionConfidence": routing_summary["confidence"],
            "rationale": routing_summary["rationale"],
            "anchorType": ((routing_summary.get("anchor") or {}).get("anchorType")),
            "anchorValue": ((routing_summary.get("anchor") or {}).get("anchorValue")),
            "querySpecificity": routing_summary.get("querySpecificity"),
            "ambiguityLevel": routing_summary.get("ambiguityLevel"),
            "secondaryIntents": list(routing_summary.get("secondaryIntents") or []),
            "retrievalHypotheses": list(routing_summary.get("retrievalHypotheses") or []),
            "providerPlan": list((routing_summary.get("providerPlan") or {}).get("providers") or []),
            "requiredPrimarySources": list(((routing_summary.get("anchor") or {}).get("requiredPrimarySources") or [])),
            "successCriteria": list(((routing_summary.get("anchor") or {}).get("successCriteria") or [])),
            "passModes": list(pass_modes or []),
            "reviewPassReason": review_pass_reason,
            "providersAttempted": list((coverage_summary or {}).get("providersAttempted") or []),
            "providersMatched": list((coverage_summary or {}).get("providersSucceeded") or []),
            "providersFailed": list((coverage_summary or {}).get("providersFailed") or []),
            "providersNotAttempted": [
                provider
                for provider in list((routing_summary.get("providerPlan") or {}).get("providers") or [])
                if provider not in list((coverage_summary or {}).get("providersAttempted") or [])
            ],
            "whyPartial": evidence_gaps[0] if evidence_gaps and result_status != "succeeded" else None,
            # Phase 4/5 planner classification signals surfaced for agents
            # (additive — existing consumers that only read retrievalHypotheses
            # continue to work unchanged).
            "intentFamily": intent_family_payload,
            "regulatoryIntent": regulatory_intent_payload,
            "subjectCard": subject_card_payload,
            "subjectChainGaps": subject_chain_gaps_payload,
        },
        "coverageSummary": coverage_summary,
        "evidence": evidence,
        "leads": leads,
        "confidenceSignals": _guided_confidence_signals(
            status=result_status,
            sources=sources,
            evidence_gaps=evidence_gaps,
            subject_chain_gaps=subject_chain_gaps_payload,
        ),
        "timeline": timeline,
    }


def _guided_session_findings(payload: dict[str, Any], sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for finding in payload.get("verifiedFindings") or []:
        if isinstance(finding, dict):
            claim = str(finding.get("claim") or "").strip()
            if claim:
                findings.append(
                    {
                        "claim": claim,
                        "supportingSourceIds": list(finding.get("supportingSourceIds") or []),
                        "trustLevel": str(finding.get("trustLevel") or "verified"),
                    }
                )
            continue
        if isinstance(finding, str) and finding.strip():
            supporting_source_ids = [
                str(source.get("sourceId") or "")
                for source in sources
                if finding.strip() in str(source.get("title") or source.get("note") or source.get("sourceId") or "")
            ]
            findings.append(
                {
                    "claim": finding.strip(),
                    "supportingSourceIds": [source_id for source_id in supporting_source_ids if source_id][:1],
                    "trustLevel": "verified",
                }
            )
    return findings or _guided_findings_from_sources(sources)


def _guided_session_state(
    *,
    workspace_registry: Any,
    search_session_id: str | None,
) -> dict[str, Any] | None:
    if workspace_registry is None:
        return None
    normalized_search_session_id = _guided_normalize_whitespace(search_session_id)
    if not normalized_search_session_id:
        return None
    try:
        record = workspace_registry.get(normalized_search_session_id)
    except Exception:
        return None
    payload = record.payload if isinstance(record.payload, dict) else {}
    has_explicit_source_payload = any(
        isinstance(payload.get(key), list) and bool(payload.get(key))
        for key in ("evidence", "sources", "structuredSources", "leads", "candidateLeads", "unverifiedLeads")
    )
    payload_sources = [source for source in payload.get("sources") or [] if isinstance(source, dict)]
    structured_sources = [
        _guided_source_record_from_structured_source(source, index=index)
        for index, source in enumerate(payload.get("structuredSources") or [], start=1)
        if isinstance(source, dict)
    ]
    evidence_sources = [
        _guided_source_record_from_structured_source(source, index=index)
        for index, source in enumerate(payload.get("evidence") or [], start=1)
        if isinstance(source, dict)
    ]
    query = str(record.query or payload.get("query") or "")
    paper_sources = (
        [
            _guided_source_record_from_paper(query, paper, index=index)
            for index, paper in enumerate(record.papers, start=1)
            if isinstance(paper, dict)
        ]
        if not has_explicit_source_payload
        else []
    )
    sources = _guided_merge_source_record_sets(payload_sources, structured_sources, evidence_sources, paper_sources)
    lead_sources = [
        _guided_source_record_from_structured_source(source, index=index)
        for key in ("unverifiedLeads", "leads", "candidateLeads")
        for index, source in enumerate(payload.get(key) or [], start=1)
        if isinstance(source, dict)
    ]
    unverified_leads = _guided_merge_source_record_sets(lead_sources) or _guided_unverified_leads_from_sources(sources)
    verified_findings = _guided_session_findings(payload, sources)
    evidence_gaps = list(payload.get("evidenceGaps") or [])
    coverage = cast(dict[str, Any] | None, payload.get("coverage") or payload.get("coverageSummary"))
    status = _guided_follow_up_status(payload.get("status") or payload.get("answerStatus"))
    failure_summary = _guided_failure_summary(
        failure_summary=cast(dict[str, Any] | None, payload.get("failureSummary")),
        status=status,
        sources=sources,
        evidence_gaps=evidence_gaps,
        all_sources_off_topic=_guided_sources_all_off_topic(sources),
    )
    # Forward ``subjectChainGaps`` recorded on the original
    # ``strategyMetadata`` so that the rebuilt ``trustSummary`` surfaces the
    # same regulatory subject-chain gap explanation seen on the initial
    # research turn. Without this, saved-session follow-ups silently lose the
    # "subject chain incomplete" trust signal.
    strategy_metadata_payload = payload.get("strategyMetadata")
    subject_chain_gaps = (
        [str(item).strip() for item in strategy_metadata_payload.get("subjectChainGaps") or [] if str(item).strip()]
        if isinstance(strategy_metadata_payload, dict)
        else []
    )
    return {
        "searchSessionId": record.search_session_id,
        "query": str(record.query or payload.get("query") or ""),
        "intent": str(payload.get("intent") or payload.get("strategyMetadata", {}).get("intent") or "discovery"),
        "status": status,
        "sources": sources,
        "unverifiedLeads": unverified_leads,
        "verifiedFindings": verified_findings,
        "evidenceGaps": evidence_gaps,
        "trustSummary": _guided_trust_summary(
            [*sources, *unverified_leads],
            evidence_gaps,
            subject_chain_gaps=subject_chain_gaps or None,
        ),
        "coverage": _guided_source_coverage_summary(
            sources=sources,
            leads=unverified_leads,
            base_coverage=coverage,
        ),
        "failureSummary": failure_summary,
        "resultMeaning": payload.get("resultMeaning")
        or _guided_result_meaning(
            status=status,
            verified_findings=verified_findings,
            evidence_gaps=evidence_gaps,
            coverage=coverage,
            failure_summary=failure_summary,
            source_count=len(sources),
            all_sources_off_topic=_guided_sources_all_off_topic(sources),
        ),
        "nextActions": payload.get("nextActions")
        or _guided_next_actions(
            search_session_id=record.search_session_id,
            status=status,
            has_sources=bool(sources),
            all_sources_off_topic=_guided_sources_all_off_topic(sources),
        ),
        "strategyMetadata": cast(
            dict[str, Any] | None,
            payload.get("strategyMetadata") or record.metadata.get("strategyMetadata"),
        ),
        "routingSummary": cast(
            dict[str, Any] | None,
            payload.get("routingSummary") or record.metadata.get("routingSummary"),
        ),
        "timeline": cast(dict[str, Any] | None, payload.get("timeline") or payload.get("regulatoryTimeline")),
        "resultState": payload.get("resultState")
        or _guided_result_state(
            status=status,
            sources=sources,
            evidence_gaps=evidence_gaps,
            search_session_id=record.search_session_id,
        ),
    }


def _guided_follow_up_introspection_facets(question: str) -> set[str]:
    text = question.lower()
    facets: set[str] = set()
    if any(
        marker in text
        for marker in (
            "provider",
            "providers",
            "coverage",
            "searched",
            "search mode",
            "completeness",
            "attempted",
            "succeeded",
            "failed",
            "zero results",
            "zero-result",
            "zero result",
        )
    ):
        facets.add("coverage")
    if any(
        marker in text
        for marker in (
            "evidence gap",
            "evidence gaps",
            "what prevented",
            "blocking gap",
            "missing evidence",
            "what was missing",
            "why abstained",
            "why partial",
            "why incomplete",
            "prevented a grounded",
            "prevented grounded",
        )
    ):
        facets.add("evidence_gaps")
    if any(
        marker in text
        for marker in (
            "fallback",
            "what failed",
            "failure",
            "still worked",
            "degraded",
        )
    ):
        facets.add("failure_summary")
    if any(
        marker in text
        for marker in (
            "verified finding",
            "verified findings",
            "strongest verified",
            "strongest finding",
            "main finding",
            "best finding",
            "top finding",
            "trusted finding",
        )
    ):
        facets.add("verified_findings")
    if any(
        marker in text
        for marker in (
            "trust summary",
            "on-topic",
            "off-topic",
            "weak match",
            "how many verified",
            "how many on-topic",
            "how many sources",
            "trust state",
        )
    ):
        facets.add("trust_summary")
    if any(
        marker in text
        for marker in (
            "which of these",
            "which are relevant",
            "which are actual",
            "which returned items",
            "which are off-target",
            "which are off-topic",
            "which are off target",
            "classify these",
            "relevant versus off-target",
            "relevant vs off-target",
            "actual guidance",
            "weak match",
            "weak matches",
        )
    ):
        facets.add("relevance_triage")
    if any(
        marker in text
        for marker in (
            "what does this result mean",
            "result meaning",
            "what status",
            "why was this result",
        )
    ):
        facets.add("result_meaning")
    if not facets and any(
        marker in text
        for marker in (
            "which sources",
            "what sources",
            "which source",
            "which documents",
            "what documents",
            "what records",
        )
    ):
        facets.add("source_overview")
    if explicit_source_reference(question):
        facets.add("specific_source")
    return facets


def _guided_extract_source_reference_from_question(question: str) -> str | None:
    return explicit_source_reference(question)


def _guided_is_usable_answer_text(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return bool(re.search(r"[A-Za-z0-9]", text))


def _guided_select_follow_up_source(question: str, sources: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not sources:
        return None
    if len(sources) == 1:
        return sources[0]

    source_reference = _guided_extract_source_reference_from_question(question)
    if source_reference:
        lowered_reference = source_reference.lower()
        for source in sources:
            if lowered_reference == _guided_normalize_whitespace(source.get("sourceId")).lower():
                return source
            if lowered_reference == _guided_normalize_whitespace(source.get("sourceAlias")).lower():
                return source
    return None


def _guided_source_metadata_answers(question: str, sources: list[dict[str, Any]]) -> list[str]:
    source = _guided_select_follow_up_source(question, sources)
    if source is None:
        return []

    normalized_question = _guided_normalize_whitespace(question).lower()
    is_source_overview_question = any(
        marker in normalized_question
        for marker in ("which sources", "what sources", "which records", "what records", "which documents")
    )
    raw_citation = source.get("citation")
    citation: dict[str, Any] = cast(dict[str, Any], raw_citation) if isinstance(raw_citation, dict) else {}
    answers: list[str] = []

    if any(marker in normalized_question for marker in ("author", "authors", "who wrote", "written by")):
        raw_authors = citation.get("authors")
        authors: list[Any] = raw_authors if isinstance(raw_authors, list) else []
        author_names = [str(author).strip() for author in authors if str(author).strip()]
        if author_names:
            answers.append("Authors listed for this source: " + ", ".join(author_names) + ".")

    if any(marker in normalized_question for marker in ("venue", "journal", "publisher", "published in")):
        venue = str(citation.get("journalOrPublisher") or "").strip()
        if venue:
            answers.append(f"Venue listed for this source: {venue}.")

    if any(marker in normalized_question for marker in ("doi", "identifier")):
        doi = str(citation.get("doi") or "").strip()
        if doi:
            answers.append(f"DOI listed for this source: {doi}.")

    if any(marker in normalized_question for marker in ("year", "publication year", "published")):
        year = str(citation.get("year") or source.get("date") or "").strip()
        if year:
            answers.append(f"Publication year listed for this source: {year}.")

    if not is_source_overview_question and any(
        marker in normalized_question for marker in ("title", "which paper", "which source", "what paper")
    ):
        title = str(source.get("title") or source.get("sourceId") or "").strip()
        if title:
            answers.append(f"Matched source title: {title}.")

    return answers


def _guided_relevance_triage_answers(
    *,
    session_state: dict[str, Any],
    follow_up_decision: Any,
) -> list[str]:
    sources = [source for source in session_state.get("sources") or [] if isinstance(source, dict)]
    leads = [lead for lead in session_state.get("unverifiedLeads") or [] if isinstance(lead, dict)]
    selected_evidence_ids = set(follow_up_decision.selected_evidence_ids)
    selected_lead_ids = set(follow_up_decision.selected_lead_ids)

    if selected_evidence_ids:
        sources = [
            source
            for source in sources
            if str(source.get("sourceId") or source.get("sourceAlias") or "").strip() in selected_evidence_ids
        ]
    if selected_lead_ids:
        leads = [
            lead
            for lead in leads
            if str(lead.get("sourceId") or lead.get("sourceAlias") or "").strip() in selected_lead_ids
        ]

    strong: list[str] = []
    weak: list[str] = []
    off_target: list[str] = []
    for candidate in sources + leads:
        title = str(candidate.get("title") or candidate.get("sourceId") or "").strip()
        if not title:
            continue
        provider = str(candidate.get("provider") or "unknown provider")
        detail = f"{title} ({provider})"
        topical_relevance = str(candidate.get("topicalRelevance") or "weak_match")
        verification_status = str(candidate.get("verificationStatus") or "unverified")
        is_primary = bool(candidate.get("isPrimarySource"))
        if (
            topical_relevance == "on_topic"
            and verification_status in {"verified_primary_source", "verified_metadata"}
            and is_primary
        ):
            strong.append(detail)
        elif topical_relevance == "off_topic":
            off_target.append(detail)
        else:
            weak.append(detail)

    answers: list[str] = []
    if strong:
        answers.append("Strong on-topic guidance records: " + "; ".join(strong[:3]) + ".")
    if weak:
        answers.append("Related but weaker or less certain records: " + "; ".join(weak[:3]) + ".")
    if off_target:
        answers.append("Off-target records kept only as leads: " + "; ".join(off_target[:3]) + ".")
    if not answers:
        answers.append("The saved session did not contain enough source detail to classify relevance confidently.")
    return answers


def _guided_enrich_records_from_saved_session(
    current_records: list[dict[str, Any]],
    saved_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for record in current_records:
        best_match: dict[str, Any] | None = None
        for saved in saved_records:
            if any(
                _guided_source_matches_reference(saved, reference)
                for reference in (
                    record.get("sourceId"),
                    record.get("sourceAlias"),
                    record.get("citationText"),
                    record.get("canonicalUrl"),
                    record.get("retrievedUrl"),
                )
                if _guided_normalize_whitespace(reference)
            ):
                best_match = saved
                break
            if _guided_source_records_share_surface(saved, record):
                best_match = saved
                break
        enriched.append(_guided_merge_source_records(best_match, record) if best_match is not None else record)
    return _guided_dedupe_source_records(enriched)


def _guided_append_selected_saved_records(
    current_records: list[dict[str, Any]],
    saved_records: list[dict[str, Any]],
    selected_ids: list[Any],
) -> list[dict[str, Any]]:
    augmented = list(current_records)
    for selected_id in selected_ids:
        normalized_selected_id = _guided_normalize_whitespace(selected_id)
        if not normalized_selected_id:
            continue
        if any(_guided_source_matches_reference(record, normalized_selected_id) for record in augmented):
            continue
        matched_saved = next(
            (saved for saved in saved_records if _guided_source_matches_reference(saved, normalized_selected_id)),
            None,
        )
        if matched_saved is not None:
            augmented.append(matched_saved)
    return _guided_dedupe_source_records(augmented)


def _guided_requested_metadata_facets(question: str) -> set[str]:
    lowered = _guided_normalize_whitespace(question).lower()
    facets: set[str] = set()
    if any(marker in lowered for marker in ("author", "authors", "who wrote", "written by")):
        facets.add("authors")
    if any(marker in lowered for marker in ("venue", "journal", "publisher", "published in")):
        facets.add("venue")
    if any(marker in lowered for marker in ("doi", "identifier")):
        facets.add("identifier")
    if any(marker in lowered for marker in ("publication year", "what year", "year published", "published in")):
        facets.add("year")
    if any(
        marker in lowered
        for marker in (
            "what records",
            "what sources",
            "which documents",
            "what documents",
            "which paper",
            "which source",
        )
    ):
        facets.add("inventory")
    return facets


def _guided_metadata_answer_is_responsive(
    *,
    question: str,
    answer_text: Any,
    sources: list[dict[str, Any]],
    leads: list[dict[str, Any]],
    selected_evidence_ids: list[Any],
    selected_lead_ids: list[Any],
) -> bool:
    requested_facets = _guided_requested_metadata_facets(question)
    if not requested_facets:
        return True

    answer_lower = _guided_normalize_whitespace(answer_text).lower()
    if not answer_lower:
        return False

    selected_source_ids = {
        _guided_normalize_whitespace(identifier)
        for identifier in selected_evidence_ids
        if _guided_normalize_whitespace(identifier)
    }
    selected_lead_ids_set = {
        _guided_normalize_whitespace(identifier)
        for identifier in selected_lead_ids
        if _guided_normalize_whitespace(identifier)
    }
    candidate_records = [
        record
        for record in sources
        if _guided_normalize_whitespace(record.get("sourceId") or record.get("sourceAlias")) in selected_source_ids
    ] + [
        record
        for record in leads
        if _guided_normalize_whitespace(record.get("sourceId") or record.get("sourceAlias")) in selected_lead_ids_set
    ]
    if not candidate_records and len(sources) == 1 and requested_facets - {"inventory"}:
        candidate_records = sources[:1]
    if not candidate_records:
        return False

    citation_payloads = [
        citation for citation in (record.get("citation") for record in candidate_records) if isinstance(citation, dict)
    ]

    def _contains_any(values: list[str]) -> bool:
        return any(value.lower() in answer_lower for value in values if value)

    if "authors" in requested_facets:
        author_values = [
            _guided_normalize_whitespace(author.get("name") if isinstance(author, dict) else author)
            for citation in citation_payloads
            for author in (citation.get("authors") or [])
        ]
        if not author_values or not _contains_any(author_values):
            return False
    if "venue" in requested_facets:
        venue_values = [
            _guided_normalize_whitespace(citation.get("journalOrPublisher")) for citation in citation_payloads
        ]
        if not venue_values or not _contains_any(venue_values):
            return False
    if "identifier" in requested_facets:
        doi_values = [_guided_normalize_whitespace(citation.get("doi")) for citation in citation_payloads]
        if not doi_values or not _contains_any(doi_values):
            return False
    if "year" in requested_facets:
        year_values = []
        for record in candidate_records:
            citation = record.get("citation")
            citation_dict = citation if isinstance(citation, dict) else {}
            year_values.append(_guided_normalize_whitespace(citation_dict.get("year") or record.get("date")))
        if not year_values or not _contains_any(year_values):
            return False
    if "inventory" in requested_facets:
        inventory_values = [
            _guided_normalize_whitespace(record.get("title") or record.get("sourceId") or record.get("sourceAlias"))
            for record in candidate_records
        ]
        if not inventory_values or not _contains_any(inventory_values):
            return False
    return True


def _guided_follow_up_response_mode(question: str, session_strategy_metadata: dict[str, Any]) -> str:
    lowered = question.lower()
    facets = _guided_follow_up_introspection_facets(question)
    if "relevance_triage" in facets:
        return "relevance_triage"
    if any(marker in lowered for marker in ("compare", "versus", "vs", "tradeoff", "tradeoffs")):
        return "comparison"
    if facets or any(
        marker in lowered
        for marker in (
            "author",
            "authors",
            "who wrote",
            "written by",
            "venue",
            "journal",
            "publisher",
            "published in",
            "doi",
            "identifier",
            "publication year",
            "what year",
            "what venue",
            "what records",
            "what sources",
            "which documents",
        )
    ):
        return "metadata"
    if any(marker in lowered for marker in ("mechanism", "pathway", "causal", "how does")):
        return "mechanism_summary"
    if any(
        marker in lowered for marker in ("regulatory history", "timeline", "rulemaking", "listing", "critical habitat")
    ):
        return "regulatory_chain"
    if any(marker in lowered for marker in ("trade-off", "tradeoff", "tradeoffs", "practical implications")):
        return "intervention_tradeoff"
    if any(
        marker in lowered
        for marker in (
            "limitation",
            "limitations",
            "validation",
            "validated",
            "operationally useful",
            "most useful",
            "practical",
            "implementation",
        )
    ):
        return "evidence_planning"
    follow_up_mode = str(session_strategy_metadata.get("followUpMode") or "").strip().lower()
    if follow_up_mode == "comparison":
        return "comparison"
    if follow_up_mode == "claim_check":
        return "evidence_planning"
    return "metadata" if explicit_source_reference(question) else "evidence_planning"


def _guided_follow_up_answer_mode(question: str, session_strategy_metadata: dict[str, Any]) -> str:
    response_mode = _guided_follow_up_response_mode(question, session_strategy_metadata)
    if response_mode == "comparison":
        return "comparison"
    if response_mode in {"mechanism_summary", "regulatory_chain", "intervention_tradeoff"}:
        return "claim_check"
    return "qa"


async def _answer_follow_up_from_session_state(
    *,
    question: str,
    session_state: dict[str, Any] | None,
    response_mode: str,
) -> dict[str, Any] | None:
    if session_state is None:
        return None
    if response_mode not in {"metadata", "relevance_triage"}:
        return None
    answer_parts: list[str] = []
    coverage = cast(dict[str, Any] | None, session_state.get("coverage")) or {}
    failure_summary = cast(dict[str, Any] | None, session_state.get("failureSummary")) or {}
    evidence_gaps = list(session_state.get("evidenceGaps") or [])
    verified_findings = [
        finding for finding in session_state.get("verifiedFindings") or [] if isinstance(finding, dict)
    ]
    sources = [source for source in session_state.get("sources") or [] if isinstance(source, dict)]
    trust_summary = cast(dict[str, Any] | None, session_state.get("trustSummary")) or {}
    facets = _guided_follow_up_introspection_facets(question)
    follow_up_decision = build_follow_up_decision(
        question=question,
        session_state=session_state,
        facets=facets,
    )
    metadata_answers = _guided_source_metadata_answers(question, sources)
    if not facets and not metadata_answers and not follow_up_decision.answer_from_session:
        return None

    answer_parts.extend(metadata_answers)

    if "coverage" in facets:
        attempted = list(coverage.get("providersAttempted") or [])
        succeeded = list(coverage.get("providersSucceeded") or [])
        failed = list(coverage.get("providersFailed") or [])
        zero_results = list(coverage.get("providersZeroResults") or [])
        likely_completeness = str(coverage.get("likelyCompleteness") or "unknown")
        sentences: list[str] = []
        if attempted:
            sentences.append(f"Providers searched were {', '.join(attempted)}.")
        else:
            sentences.append("No provider-attempt summary was saved for this session.")
        if failed:
            sentences.append(f"Failed providers: {', '.join(failed)}.")
        else:
            sentences.append("No provider failures were recorded.")
        if zero_results:
            sentences.append(f"Zero-result providers: {', '.join(zero_results)}.")
        if succeeded:
            sentences.append(f"Successful providers: {', '.join(succeeded)}.")
        if coverage.get("searchMode"):
            sentences.append(f"Search mode was {coverage['searchMode']}.")
        sentences.append(f"Likely completeness was {likely_completeness}.")
        answer_parts.append(" ".join(sentences))

    if "evidence_gaps" in facets:
        if evidence_gaps:
            if "specific" in question.lower() or len(evidence_gaps) == 1:
                answer_parts.append(f"The main evidence gap was: {evidence_gaps[0]}")
            else:
                answer_parts.append("Key evidence gaps were: " + "; ".join(evidence_gaps[:3]) + ".")
        else:
            answer_parts.append("No explicit evidence gaps were recorded in the saved session.")

    if "failure_summary" in facets:
        outcome = str(failure_summary.get("outcome") or "no_failure")
        what_failed = str(failure_summary.get("whatFailed") or "").strip()
        what_still_worked = str(failure_summary.get("whatStillWorked") or "").strip()
        completeness_impact = str(failure_summary.get("completenessImpact") or "").strip()
        fallback_attempted = bool(failure_summary.get("fallbackAttempted"))
        summary_sentences = [f"Failure outcome was {outcome}."]
        if what_failed:
            summary_sentences.append(what_failed)
        if what_still_worked:
            summary_sentences.append(what_still_worked)
        if fallback_attempted:
            fallback_mode = str(failure_summary.get("fallbackMode") or "fallback")
            summary_sentences.append(f"Fallback was attempted via {fallback_mode}.")
        if completeness_impact:
            summary_sentences.append(completeness_impact)
        answer_parts.append(" ".join(summary_sentences))

    if "verified_findings" in facets:
        if verified_findings:
            strongest_finding = str(verified_findings[0].get("claim") or "").strip()
            if strongest_finding:
                answer_parts.append(f"The strongest verified finding in the saved session was: {strongest_finding}.")
            if len(verified_findings) > 1:
                remaining_findings = [
                    str(finding.get("claim") or "").strip()
                    for finding in verified_findings[1:3]
                    if str(finding.get("claim") or "").strip()
                ]
                if remaining_findings:
                    answer_parts.append("Other verified findings included: " + "; ".join(remaining_findings) + ".")
        else:
            answer_parts.append("No verified findings were recorded in the saved session.")

    if "trust_summary" in facets:
        answer_parts.append(
            "Trust summary: "
            f"{int(trust_summary.get('verifiedSourceCount') or 0)} verified source(s), "
            f"{int(trust_summary.get('onTopicSourceCount') or 0)} on-topic, "
            f"{int(trust_summary.get('weakMatchCount') or 0)} weak match, and "
            f"{int(trust_summary.get('offTopicCount') or 0)} off-topic."
        )

    if "result_meaning" in facets:
        result_meaning = str(session_state.get("resultMeaning") or "").strip()
        if result_meaning:
            answer_parts.append(result_meaning)

    if "source_overview" in facets:
        selected_evidence_ids = set(follow_up_decision.selected_evidence_ids)
        selected_sources = [
            source
            for source in sources
            if str(source.get("sourceId") or source.get("sourceAlias") or "").strip() in selected_evidence_ids
        ] or sources
        if selected_sources:
            source_titles = [
                str(source.get("title") or source.get("sourceId") or "").strip()
                for source in selected_sources[:3]
                if str(source.get("title") or source.get("sourceId") or "").strip()
            ]
            if source_titles:
                answer_parts.append("Saved sources included: " + "; ".join(source_titles) + ".")
        else:
            unverified_leads = [lead for lead in session_state.get("unverifiedLeads") or [] if isinstance(lead, dict)]
            selected_lead_ids = set(follow_up_decision.selected_lead_ids)
            selected_leads = [
                lead
                for lead in unverified_leads
                if str(lead.get("sourceId") or lead.get("sourceAlias") or "").strip() in selected_lead_ids
            ] or unverified_leads
            lead_titles = [
                str(lead.get("title") or lead.get("sourceId") or "").strip()
                for lead in selected_leads[:3]
                if str(lead.get("title") or lead.get("sourceId") or "").strip()
            ]
            if lead_titles:
                answer_parts.append("Saved source leads included: " + "; ".join(lead_titles) + ".")
            else:
                answer_parts.append("No saved sources were available for this session.")

    if "relevance_triage" in facets:
        answer_parts.extend(
            _guided_relevance_triage_answers(
                session_state=session_state,
                follow_up_decision=follow_up_decision,
            )
        )

    if "specific_source" in facets:
        source_reference = _guided_extract_source_reference_from_question(question)
        if source_reference:
            source = None
            match_type = "unresolved"
            lower_reference = source_reference.lower()
            session_leads = [lead for lead in session_state.get("unverifiedLeads") or [] if isinstance(lead, dict)]
            source_matches = [
                candidate
                for candidate in sources + session_leads
                if (
                    lower_reference == _guided_normalize_whitespace(candidate.get("sourceId")).lower()
                    or lower_reference == _guided_normalize_whitespace(candidate.get("sourceAlias")).lower()
                    or lower_reference == _guided_normalize_whitespace(candidate.get("citationText")).lower()
                    or lower_reference == _guided_normalize_whitespace(candidate.get("canonicalUrl")).lower()
                    or lower_reference == _guided_normalize_whitespace(candidate.get("retrievedUrl")).lower()
                )
            ]
            if len(source_matches) == 1:
                source = source_matches[0]
                match_type = "session_local_match"
            if source is not None:
                source_title = str(source.get("title") or source.get("sourceId") or "requested source")
                source_provider = str(source.get("provider") or "unknown provider")
                source_relevance = str(source.get("topicalRelevance") or "unknown relevance")
                answer_parts.append(
                    f"Source {source_title} ({source_provider}) was matched via {match_type} "
                    f"with relevance {source_relevance}."
                )
            else:
                answer_parts.append(
                    f"No saved source matched '{source_reference}' in this session. "
                    "Use inspect_source with an exact sourceId for direct inspection."
                )

    answer_parts = [part.strip() for part in answer_parts if _guided_is_usable_answer_text(part)]
    if not answer_parts:
        return None

    # Filter sources to only those referenced in the follow-up answer
    # to avoid re-serializing the entire session source set (payload efficiency).
    _referenced_ids = set(follow_up_decision.selected_evidence_ids + follow_up_decision.selected_lead_ids)
    if _referenced_ids:
        _filtered_sources = [
            s for s in sources if str(s.get("sourceId") or s.get("sourceAlias") or "").strip() in _referenced_ids
        ]
    else:
        _filtered_sources = sources[:3]  # Fallback: cap at 3 when no explicit selection
    _filtered_leads = (
        [
            lead
            for lead in (session_state.get("unverifiedLeads") or [])
            if isinstance(lead, dict)
            and str(lead.get("sourceId") or lead.get("sourceAlias") or "").strip() in _referenced_ids
        ]
        if _referenced_ids
        else []
    )
    _filtered_sources = _guided_dedupe_source_records(_filtered_sources)
    _filtered_leads = _guided_dedupe_source_records(_filtered_leads)
    _filtered_findings = [
        f
        for f in (session_state.get("verifiedFindings") or [])
        if isinstance(f, dict)
        and str(f.get("sourceId") or f.get("claim") or "").strip()
        in {s.get("title") or s.get("sourceId") for s in _filtered_sources}
    ]

    # Ensure trustSummary.authoritativeButWeak is always present (possibly
    # empty) for shape consistency across research / follow_up_research /
    # inspect_source synthesis paths.
    _session_trust_summary = _guided_trust_summary(
        [*_filtered_sources, *_filtered_leads],
        list(session_state.get("evidenceGaps") or []),
    )
    _session_trust_summary.setdefault("authoritativeButWeak", [])
    _session_coverage = _guided_source_coverage_summary(
        sources=_filtered_sources,
        leads=_filtered_leads,
        base_coverage=cast(dict[str, Any] | None, session_state.get("coverage")),
    )

    # P0-1 Fix #2: gate ``answered`` on the saved trust summary. When the
    # session has no on-topic or no verified sources the introspection fast
    # path has nothing grounded to lean on, so fall back to
    # ``insufficient_evidence`` instead of synthesising a confident answer.
    def _coerce_positive_int(value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 0
        return parsed if parsed > 0 else 0

    _on_topic_count = _coerce_positive_int(_session_trust_summary.get("onTopicSourceCount"))
    _verified_count = _coerce_positive_int(_session_trust_summary.get("verifiedSourceCount"))
    # P0-1 Fix #2: block the introspection fast path from advertising
    # ``answered`` only when the answer rests entirely on topical claims
    # pulled from a weak pool. Metadata-style questions (coverage,
    # evidenceGaps, failureSummary, resultMeaning, trustSummary,
    # verifiedFindings) are always legitimately answerable from saved
    # session structure — they are never topical claims about the domain.
    # Likewise, answers grounded in explicitly selected evidence / lead ids
    # remain legitimate introspection. Only when the answer is built solely
    # from ``follow_up_decision.answer_from_session`` *and* the pool reports
    # zero on-topic / verified sources *and* the decision did not pin any
    # ids do we downgrade to ``insufficient_evidence``.
    _has_meta_backing = bool(facets) or bool(metadata_answers)
    _has_strong_session_evidence = (
        _has_meta_backing
        or _on_topic_count >= 1
        or _verified_count >= 1
        or bool(follow_up_decision.selected_evidence_ids)
        or bool(follow_up_decision.selected_lead_ids)
    )
    _answer_status_value: Literal["answered", "insufficient_evidence"] = (
        "answered" if _has_strong_session_evidence else "insufficient_evidence"
    )
    _answer_text = " ".join(answer_parts) if _has_strong_session_evidence else ""
    _selected_evidence_ids = follow_up_decision.selected_evidence_ids if _has_strong_session_evidence else []
    _selected_lead_ids = follow_up_decision.selected_lead_ids if _has_strong_session_evidence else []

    return {
        "searchSessionId": session_state["searchSessionId"],
        "answerStatus": _answer_status_value,
        "answer": _answer_text,
        "evidence": [],
        "selectedEvidenceIds": _selected_evidence_ids,
        "selectedLeadIds": _selected_lead_ids,
        "unsupportedAsks": [] if _has_strong_session_evidence else [question],
        "followUpQuestions": [],
        "verifiedFindings": _filtered_findings,
        "sources": _filtered_sources,
        "unverifiedLeads": _filtered_leads,
        "evidenceGaps": session_state["evidenceGaps"],
        "trustSummary": _session_trust_summary,
        "coverage": _session_coverage,
        "failureSummary": session_state["failureSummary"],
        "resultMeaning": session_state["resultMeaning"],
        "nextActions": session_state["nextActions"],
        "resultState": _guided_result_state(
            status=_answer_status_value,
            sources=sources,
            evidence_gaps=evidence_gaps,
            search_session_id=str(session_state.get("searchSessionId") or ""),
        ),
        **(
            await _guided_contract_fields(
                query=str(session_state.get("query") or ""),
                intent=str(session_state.get("intent") or "discovery"),
                status=_answer_status_value,
                sources=sources,
                unverified_leads=cast(list[dict[str, Any]], session_state.get("unverifiedLeads") or []),
                evidence_gaps=evidence_gaps,
                coverage_summary=coverage,
                strategy_metadata=cast(dict[str, Any] | None, session_state.get("strategyMetadata")),
                timeline=cast(dict[str, Any] | None, session_state.get("timeline")),
            )
        ),
        "executionProvenance": _guided_execution_provenance_payload(
            execution_mode="session_introspection",
            answer_source="saved_session_metadata",
            passes_run=0,
        ),
        "confidenceSignals": _guided_confidence_signals(
            status=_answer_status_value,
            sources=sources,
            evidence_gaps=evidence_gaps,
            synthesis_mode="session_introspection",
            subject_chain_gaps=cast(
                list[str] | None,
                (session_state.get("strategyMetadata") or {}).get("subjectChainGaps")
                if isinstance(session_state.get("strategyMetadata"), dict)
                else None,
            ),
        ),
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
        landscape_args = cast(
            MapResearchLandscapeArgs,
            TOOL_INPUT_MODELS[name].model_validate(arguments),
        )
        if agentic_runtime is None:
            return {
                "error": "FEATURE_NOT_CONFIGURED",
                "message": ("map_research_landscape requires the agentic runtime to be enabled."),
                "fallbackTools": [
                    "search_papers",
                    "search_papers_bulk",
                    "get_paper_citations",
                ],
            }
        return await agentic_runtime.map_research_landscape(
            search_session_id=landscape_args.search_session_id,
            max_themes=landscape_args.max_themes,
            latency_profile=landscape_args.latency_profile,
            ctx=ctx,
        )

    if name == "expand_research_graph":
        graph_args = cast(
            ExpandResearchGraphArgs,
            TOOL_INPUT_MODELS[name].model_validate(arguments),
        )
        if agentic_runtime is None:
            return {
                "error": "FEATURE_NOT_CONFIGURED",
                "message": ("expand_research_graph requires the agentic runtime to be enabled."),
                "fallbackTools": [
                    "get_paper_citations",
                    "get_paper_references",
                    "get_paper_authors",
                ],
            }
        return await agentic_runtime.expand_research_graph(
            seed_paper_ids=graph_args.seed_paper_ids,
            seed_search_session_id=graph_args.seed_search_session_id,
            direction=graph_args.direction,
            hops=graph_args.hops,
            per_seed_limit=graph_args.per_seed_limit,
            latency_profile=graph_args.latency_profile,
            ctx=ctx,
        )

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
        if not enable_ecos or ecos_client is None:
            raise ValueError(
                "search_species_ecos requires ECOS, which is disabled. "
                "Set PAPER_CHASER_ENABLE_ECOS=true to use this tool."
            )
        ecos_args = cast(
            SearchSpeciesEcosArgs,
            TOOL_INPUT_MODELS[name].model_validate(arguments),
        )
        result = await ecos_client.search_species(
            query=ecos_args.query,
            limit=ecos_args.limit,
            match_mode=ecos_args.match_mode,
        )
        return _finalize_tool_result(
            name,
            arguments,
            result,
            workspace_registry=workspace_registry,
        )

    if name == "get_species_profile_ecos":
        if not enable_ecos or ecos_client is None:
            raise ValueError(
                "get_species_profile_ecos requires ECOS, which is disabled. "
                "Set PAPER_CHASER_ENABLE_ECOS=true to use this tool."
            )
        species_lookup_args = cast(
            EcosSpeciesLookupArgs,
            TOOL_INPUT_MODELS[name].model_validate(arguments),
        )
        result = await ecos_client.get_species_profile(
            species_id=species_lookup_args.species_id,
        )
        return _finalize_tool_result(
            name,
            arguments,
            result,
            workspace_registry=workspace_registry,
        )

    if name == "list_species_documents_ecos":
        if not enable_ecos or ecos_client is None:
            raise ValueError(
                "list_species_documents_ecos requires ECOS, which is disabled. "
                "Set PAPER_CHASER_ENABLE_ECOS=true to use this tool."
            )
        document_list_args = cast(
            ListSpeciesDocumentsEcosArgs,
            TOOL_INPUT_MODELS[name].model_validate(arguments),
        )
        result = await ecos_client.list_species_documents(
            species_id=document_list_args.species_id,
            document_kinds=document_list_args.document_kinds,
        )
        return _finalize_tool_result(
            name,
            arguments,
            result,
            workspace_registry=workspace_registry,
        )

    if name == "get_document_text_ecos":
        if not enable_ecos or ecos_client is None:
            raise ValueError(
                "get_document_text_ecos requires ECOS, which is disabled. "
                "Set PAPER_CHASER_ENABLE_ECOS=true to use this tool."
            )
        document_text_args = cast(
            GetDocumentTextEcosArgs,
            TOOL_INPUT_MODELS[name].model_validate(arguments),
        )
        result = await ecos_client.get_document_text(url=document_text_args.url)
        return _finalize_tool_result(
            name,
            arguments,
            result,
            workspace_registry=workspace_registry,
        )

    if name == "search_federal_register":
        if not enable_federal_register or federal_register_client is None:
            raise ValueError(
                "search_federal_register requires Federal Register support, which is disabled. "
                "Set PAPER_CHASER_ENABLE_FEDERAL_REGISTER=true to use this tool."
            )
        fr_args = cast(
            SearchFederalRegisterArgs,
            TOOL_INPUT_MODELS[name].model_validate(arguments),
        )
        result = await federal_register_client.search_documents(
            query=fr_args.query,
            limit=fr_args.limit,
            agencies=fr_args.agencies,
            document_types=fr_args.document_types,
            publication_date_from=fr_args.publication_date_from,
            publication_date_to=fr_args.publication_date_to,
            cfr_citation=fr_args.cfr_citation,
            cfr_title=fr_args.cfr_title,
            cfr_part=fr_args.cfr_part,
            document_number=fr_args.document_number,
        )
        return _finalize_tool_result(
            name,
            arguments,
            result,
            workspace_registry=workspace_registry,
        )

    if name == "get_federal_register_document":
        if govinfo_client is None:
            raise ValueError("get_federal_register_document requires GovInfo client initialization.")
        document_args = cast(
            GetFederalRegisterDocumentArgs,
            TOOL_INPUT_MODELS[name].model_validate(arguments),
        )
        result = await govinfo_client.get_federal_register_document(
            identifier=document_args.identifier,
            federal_register_client=federal_register_client if enable_federal_register else None,
        )
        return _finalize_tool_result(
            name,
            arguments,
            result,
            workspace_registry=workspace_registry,
        )

    if name == "get_cfr_text":
        if not enable_govinfo_cfr or govinfo_client is None:
            raise ValueError(
                "get_cfr_text requires GovInfo CFR support, which is disabled. "
                "Set PAPER_CHASER_ENABLE_GOVINFO_CFR=true to use this tool."
            )
        cfr_args = cast(
            GetCfrTextArgs,
            TOOL_INPUT_MODELS[name].model_validate(arguments),
        )
        result = await govinfo_client.get_cfr_text(
            title_number=cfr_args.title_number,
            part_number=cfr_args.part_number,
            section_number=cfr_args.section_number,
            revision_year=cfr_args.revision_year,
            effective_date=cfr_args.effective_date,
        )
        return _finalize_tool_result(
            name,
            arguments,
            result,
            workspace_registry=workspace_registry,
        )

    if name == "get_paper_metadata_crossref":
        crossref_result = await resolved_enrichment_service.get_crossref_metadata(
            **TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
        )
        return _finalize_tool_result(
            name,
            arguments,
            dump_jsonable(crossref_result),
            workspace_registry=workspace_registry,
        )

    if name == "get_paper_open_access_unpaywall":
        unpaywall_result = await resolved_enrichment_service.get_unpaywall_open_access(
            **TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
        )
        return _finalize_tool_result(
            name,
            arguments,
            dump_jsonable(unpaywall_result),
            workspace_registry=workspace_registry,
        )

    if name == "enrich_paper":
        enrichment_result = await resolved_enrichment_service.enrich_paper(
            **TOOL_INPUT_MODELS[name].model_validate(arguments).model_dump(by_alias=False)
        )
        return _finalize_tool_result(
            name,
            arguments,
            dump_jsonable(enrichment_result),
            workspace_registry=workspace_registry,
        )

    if name == "search_papers":
        search_args = cast(
            SearchPapersArgs,
            TOOL_INPUT_MODELS[name].model_validate(arguments),
        )
        search_result = await search_papers_with_fallback(
            query=search_args.query,
            limit=search_args.limit,
            year=search_args.year,
            fields=search_args.fields,
            venue=search_args.venue,
            publication_date_or_year=search_args.publication_date_or_year,
            fields_of_study=search_args.fields_of_study,
            publication_types=search_args.publication_types,
            open_access_pdf=search_args.open_access_pdf,
            min_citation_count=search_args.min_citation_count,
            enable_core=enable_core,
            enable_semantic_scholar=enable_semantic_scholar,
            enable_arxiv=enable_arxiv,
            enable_serpapi=enable_serpapi,
            enable_scholarapi=enable_scholarapi,
            preferred_provider=search_args.preferred_provider,
            provider_order=search_args.provider_order or provider_order,
            core_client=core_client,
            semantic_client=client,
            arxiv_client=arxiv_client,
            serpapi_client=serpapi_client,
            scholarapi_client=scholarapi_client,
            provider_registry=provider_registry,
            allow_default_hedging=(search_args.preferred_provider is None and search_args.provider_order is None),
        )
        elicited = await _maybe_elicit_and_retry(
            tool_name=name,
            arguments=arguments,
            result=search_result,
            client=client,
            core_client=core_client,
            openalex_client=openalex_client,
            arxiv_client=arxiv_client,
            serpapi_client=serpapi_client,
            scholarapi_client=scholarapi_client,
            crossref_client=crossref_client,
            unpaywall_client=unpaywall_client,
            ecos_client=ecos_client,
            federal_register_client=federal_register_client,
            govinfo_client=govinfo_client,
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
            provider_order=provider_order,
            provider_registry=provider_registry,
            workspace_registry=workspace_registry,
            enrichment_service=resolved_enrichment_service,
            agentic_runtime=agentic_runtime,
            ctx=ctx,
            allow_elicitation=allow_elicitation,
        )
        if elicited is not None:
            return elicited
        return _finalize_tool_result(
            name,
            arguments,
            search_result,
            workspace_registry=workspace_registry,
        )

    if name == "search_papers_match":
        match_args = cast(
            PaperMatchArgs,
            TOOL_INPUT_MODELS[name].model_validate(arguments),
        )
        serialized = dump_jsonable(
            await client.search_papers_match(
                query=match_args.query,
                fields=match_args.fields,
                openalex_client=openalex_client,
                enable_openalex=enable_openalex,
                crossref_client=crossref_client,
                enable_crossref=enable_crossref,
            )
        )
        if (
            match_args.include_enrichment
            and isinstance(serialized, dict)
            and serialized.get("matchFound", True) is not False
        ):
            enrichment_source = await hydrate_paper_for_enrichment(
                serialized,
                detail_client=client,
            )
            enriched_payload = await resolved_enrichment_service.enrich_paper_payload(
                enrichment_source,
                query=serialized.get("title") or match_args.query,
            )
            serialized = attach_enrichments_to_paper_payload(
                serialized,
                enriched_paper=enriched_payload,
            )
        elicited = await _maybe_elicit_and_retry(
            tool_name=name,
            arguments=arguments,
            result=serialized,
            client=client,
            core_client=core_client,
            openalex_client=openalex_client,
            arxiv_client=arxiv_client,
            serpapi_client=serpapi_client,
            scholarapi_client=scholarapi_client,
            crossref_client=crossref_client,
            unpaywall_client=unpaywall_client,
            ecos_client=ecos_client,
            federal_register_client=federal_register_client,
            govinfo_client=govinfo_client,
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
            provider_order=provider_order,
            provider_registry=provider_registry,
            workspace_registry=workspace_registry,
            enrichment_service=resolved_enrichment_service,
            agentic_runtime=agentic_runtime,
            ctx=ctx,
            allow_elicitation=allow_elicitation,
        )
        if elicited is not None:
            return elicited
        return _finalize_tool_result(
            name,
            arguments,
            serialized,
            workspace_registry=workspace_registry,
        )

    if name == "get_paper_details":
        paper_lookup_args = cast(
            PaperLookupArgs,
            TOOL_INPUT_MODELS[name].model_validate(arguments),
        )
        serialized = dump_jsonable(
            await client.get_paper_details(
                paper_id=paper_lookup_args.paper_id,
                fields=paper_lookup_args.fields,
            )
        )
        if paper_lookup_args.include_enrichment and isinstance(serialized, dict):
            enrichment_source = await hydrate_paper_for_enrichment(
                serialized,
                detail_client=client,
            )
            enriched_payload = await resolved_enrichment_service.enrich_paper_payload(
                enrichment_source,
                query=serialized.get("title"),
            )
            serialized = attach_enrichments_to_paper_payload(
                serialized,
                enriched_paper=enriched_payload,
            )
        return _finalize_tool_result(
            name,
            arguments,
            serialized,
            workspace_registry=workspace_registry,
        )

    if name == "resolve_citation":
        citation_args = cast(
            ResolveCitationArgs,
            TOOL_INPUT_MODELS[name].model_validate(arguments),
        )
        result = await resolve_citation(
            citation=citation_args.citation,
            max_candidates=citation_args.max_candidates,
            client=client,
            enable_core=enable_core,
            enable_semantic_scholar=enable_semantic_scholar,
            enable_openalex=enable_openalex,
            enable_arxiv=enable_arxiv,
            enable_serpapi=enable_serpapi,
            core_client=core_client,
            openalex_client=openalex_client,
            arxiv_client=arxiv_client,
            serpapi_client=serpapi_client,
            title_hint=citation_args.title_hint,
            author_hint=citation_args.author_hint,
            year_hint=citation_args.year_hint,
            venue_hint=citation_args.venue_hint,
            doi_hint=citation_args.doi_hint,
            include_enrichment=citation_args.include_enrichment,
            enrichment_service=resolved_enrichment_service,
        )
        return _finalize_tool_result(
            name,
            arguments,
            result,
            workspace_registry=workspace_registry,
        )

    if name == "paper_autocomplete_openalex":
        if not enable_openalex:
            raise ValueError(
                "paper_autocomplete_openalex requires OpenAlex, which is disabled. "
                "Set PAPER_CHASER_ENABLE_OPENALEX=true to use this tool."
            )
        validated_payload = TOOL_INPUT_MODELS[name].model_validate(arguments)
        args_dict = validated_payload.model_dump(by_alias=False)
        result = await openalex_client.paper_autocomplete(
            query=args_dict["query"],
            limit=args_dict.get("limit", 10),
        )
        return _finalize_tool_result(
            name,
            arguments,
            result,
            workspace_registry=workspace_registry,
        )

    if name == "search_papers_openalex":
        if not enable_openalex:
            raise ValueError(
                "search_papers_openalex requires OpenAlex, which is disabled. "
                "Set PAPER_CHASER_ENABLE_OPENALEX=true to use this tool."
            )
        validated_payload = TOOL_INPUT_MODELS[name].model_validate(arguments)
        args_dict = validated_payload.model_dump(by_alias=False)
        result = await openalex_client.search(
            query=args_dict["query"],
            limit=args_dict.get("limit", 10),
            year=args_dict.get("year"),
        )
        return _finalize_tool_result(
            name,
            arguments,
            result,
            workspace_registry=workspace_registry,
        )

    if name == "search_papers_scholarapi":
        if not enable_scholarapi:
            raise ValueError(
                "search_papers_scholarapi requires ScholarAPI, which is not enabled. "
                "Set PAPER_CHASER_ENABLE_SCHOLARAPI=true and provide SCHOLARAPI_API_KEY."
            )
        validated_payload = TOOL_INPUT_MODELS[name].model_validate(arguments)
        args_dict = validated_payload.model_dump(by_alias=False)
        ctx_hash = compute_context_hash(name, args_dict)
        result = await _call_explicit_scholarapi_tool(
            operation=lambda: scholarapi_client.search(
                query=args_dict["query"],
                limit=args_dict.get("limit", 10),
                cursor=_cursor_to_bulk_token(
                    args_dict.get("cursor"),
                    tool=name,
                    context_hash=ctx_hash,
                    expected_provider="scholarapi",
                ),
                indexed_after=args_dict.get("indexed_after"),
                indexed_before=args_dict.get("indexed_before"),
                published_after=args_dict.get("published_after"),
                published_before=args_dict.get("published_before"),
                has_text=args_dict.get("has_text"),
                has_pdf=args_dict.get("has_pdf"),
            ),
            endpoint="search",
            provider_registry=provider_registry,
            request_id=f"tool-{name}",
        )
        serialized = dump_jsonable(result)
        serialized = _encode_next_bulk_cursor(
            serialized,
            name,
            context_hash=ctx_hash,
            provider="scholarapi",
        )
        return _finalize_tool_result(
            name,
            arguments,
            serialized,
            workspace_registry=workspace_registry,
        )

    if name == "list_papers_scholarapi":
        if not enable_scholarapi:
            raise ValueError(
                "list_papers_scholarapi requires ScholarAPI, which is not enabled. "
                "Set PAPER_CHASER_ENABLE_SCHOLARAPI=true and provide SCHOLARAPI_API_KEY."
            )
        validated_payload = TOOL_INPUT_MODELS[name].model_validate(arguments)
        args_dict = validated_payload.model_dump(by_alias=False)
        ctx_hash = compute_context_hash(name, args_dict)
        result = await _call_explicit_scholarapi_tool(
            operation=lambda: scholarapi_client.list_papers(
                query=args_dict.get("query"),
                limit=args_dict.get("limit", 100),
                cursor=_cursor_to_bulk_token(
                    args_dict.get("cursor"),
                    tool=name,
                    context_hash=ctx_hash,
                    expected_provider="scholarapi",
                ),
                indexed_after=args_dict.get("indexed_after"),
                indexed_before=args_dict.get("indexed_before"),
                published_after=args_dict.get("published_after"),
                published_before=args_dict.get("published_before"),
                has_text=args_dict.get("has_text"),
                has_pdf=args_dict.get("has_pdf"),
            ),
            endpoint="list",
            provider_registry=provider_registry,
            request_id=f"tool-{name}",
        )
        serialized = dump_jsonable(result)
        serialized.setdefault("retrievalNote", SCHOLARAPI_LIST_RETRIEVAL_NOTE)
        serialized = _encode_next_bulk_cursor(
            serialized,
            name,
            context_hash=ctx_hash,
            provider="scholarapi",
        )
        return _finalize_tool_result(
            name,
            arguments,
            serialized,
            workspace_registry=workspace_registry,
        )

    if name == "get_paper_text_scholarapi":
        if not enable_scholarapi:
            raise ValueError(
                "get_paper_text_scholarapi requires ScholarAPI, which is not enabled. "
                "Set PAPER_CHASER_ENABLE_SCHOLARAPI=true and provide SCHOLARAPI_API_KEY."
            )
        validated_payload = TOOL_INPUT_MODELS[name].model_validate(arguments)
        args_dict = validated_payload.model_dump(by_alias=False)
        result = await _call_explicit_scholarapi_tool(
            operation=lambda: scholarapi_client.get_text(paper_id=args_dict["paper_id"]),
            endpoint="text",
            provider_registry=provider_registry,
            request_id=f"tool-{name}",
        )
        return _finalize_tool_result(
            name,
            arguments,
            result,
            workspace_registry=workspace_registry,
        )

    if name == "get_paper_texts_scholarapi":
        if not enable_scholarapi:
            raise ValueError(
                "get_paper_texts_scholarapi requires ScholarAPI, which is not enabled. "
                "Set PAPER_CHASER_ENABLE_SCHOLARAPI=true and provide SCHOLARAPI_API_KEY."
            )
        validated_payload = TOOL_INPUT_MODELS[name].model_validate(arguments)
        args_dict = validated_payload.model_dump(by_alias=False)
        result = await _call_explicit_scholarapi_tool(
            operation=lambda: scholarapi_client.get_texts(paper_ids=args_dict["paper_ids"]),
            endpoint="texts",
            provider_registry=provider_registry,
            request_id=f"tool-{name}",
        )
        return _finalize_tool_result(
            name,
            arguments,
            result,
            workspace_registry=workspace_registry,
        )

    if name == "get_paper_pdf_scholarapi":
        if not enable_scholarapi:
            raise ValueError(
                "get_paper_pdf_scholarapi requires ScholarAPI, which is not enabled. "
                "Set PAPER_CHASER_ENABLE_SCHOLARAPI=true and provide SCHOLARAPI_API_KEY."
            )
        validated_payload = TOOL_INPUT_MODELS[name].model_validate(arguments)
        args_dict = validated_payload.model_dump(by_alias=False)
        result = await _call_explicit_scholarapi_tool(
            operation=lambda: scholarapi_client.get_pdf(paper_id=args_dict["paper_id"]),
            endpoint="pdf",
            provider_registry=provider_registry,
            request_id=f"tool-{name}",
        )
        return _finalize_tool_result(
            name,
            arguments,
            result,
            workspace_registry=workspace_registry,
        )

    if name == "search_entities_openalex":
        if not enable_openalex:
            raise ValueError(
                "search_entities_openalex requires OpenAlex, which is disabled. "
                "Set PAPER_CHASER_ENABLE_OPENALEX=true to use this tool."
            )
        validated_payload = TOOL_INPUT_MODELS[name].model_validate(arguments)
        args_dict = validated_payload.model_dump(by_alias=False)
        ctx_hash = compute_context_hash(name, args_dict)
        result = await openalex_client.search_entities(
            entity_type=args_dict["entity_type"],
            query=args_dict["query"],
            limit=args_dict.get("limit", 10),
            cursor=_cursor_to_bulk_token(
                args_dict.get("cursor"),
                tool=name,
                context_hash=ctx_hash,
                expected_provider="openalex",
            ),
        )
        serialized = dump_jsonable(result)
        serialized = _encode_next_bulk_cursor(
            serialized,
            name,
            context_hash=ctx_hash,
            provider="openalex",
        )
        return _finalize_tool_result(
            name,
            arguments,
            serialized,
            workspace_registry=workspace_registry,
        )

    if name == "search_papers_openalex_by_entity":
        if not enable_openalex:
            raise ValueError(
                "search_papers_openalex_by_entity requires OpenAlex, "
                "which is disabled. "
                "Set PAPER_CHASER_ENABLE_OPENALEX=true to use this tool."
            )
        validated_payload = TOOL_INPUT_MODELS[name].model_validate(arguments)
        args_dict = validated_payload.model_dump(by_alias=False)
        ctx_hash = compute_context_hash(name, args_dict)
        result = await openalex_client.search_works_by_entity(
            entity_type=args_dict["entity_type"],
            entity_id=args_dict["entity_id"],
            limit=args_dict.get("limit", 100),
            cursor=_cursor_to_bulk_token(
                args_dict.get("cursor"),
                tool=name,
                context_hash=ctx_hash,
                expected_provider="openalex",
            ),
            year=args_dict.get("year"),
        )
        serialized = dump_jsonable(result)
        serialized = _encode_next_bulk_cursor(
            serialized,
            name,
            context_hash=ctx_hash,
            provider="openalex",
        )
        return _finalize_tool_result(
            name,
            arguments,
            serialized,
            workspace_registry=workspace_registry,
        )

    if name == "search_papers_openalex_bulk":
        if not enable_openalex:
            raise ValueError(
                "search_papers_openalex_bulk requires OpenAlex, which is disabled. "
                "Set PAPER_CHASER_ENABLE_OPENALEX=true to use this tool."
            )
        validated_payload = TOOL_INPUT_MODELS[name].model_validate(arguments)
        args_dict = validated_payload.model_dump(by_alias=False)
        ctx_hash = compute_context_hash(name, args_dict)
        result = await openalex_client.search_bulk(
            query=args_dict["query"],
            limit=args_dict.get("limit", 100),
            cursor=_cursor_to_bulk_token(
                args_dict.get("cursor"),
                tool=name,
                context_hash=ctx_hash,
                expected_provider="openalex",
            ),
            year=args_dict.get("year"),
        )
        serialized = dump_jsonable(result)
        serialized = _encode_next_bulk_cursor(
            serialized,
            name,
            context_hash=ctx_hash,
            provider="openalex",
        )
        return _finalize_tool_result(
            name,
            arguments,
            serialized,
            workspace_registry=workspace_registry,
        )

    if name == "search_papers_serpapi_cited_by":
        if not enable_serpapi:
            raise ValueError(
                "search_papers_serpapi_cited_by requires SerpApi, "
                "which is not enabled. "
                "Set PAPER_CHASER_ENABLE_SERPAPI=true and provide SERPAPI_API_KEY."
            )
        validated_payload = TOOL_INPUT_MODELS[name].model_validate(arguments)
        args_dict = validated_payload.model_dump(by_alias=False)
        ctx_hash = compute_context_hash(name, args_dict)
        result = await serpapi_client.search_cited_by(
            cites_id=args_dict["cites_id"],
            query=args_dict.get("query"),
            limit=args_dict.get("limit", 10),
            start=_cursor_to_offset(
                args_dict.get("cursor"),
                name,
                context_hash=ctx_hash,
                expected_provider="serpapi_google_scholar",
            )
            or 0,
            year=args_dict.get("year"),
        )
        serialized = dump_jsonable(result)
        serialized = _encode_next_cursor(
            serialized,
            name,
            context_hash=ctx_hash,
            provider="serpapi_google_scholar",
        )
        return _finalize_tool_result(
            name,
            arguments,
            serialized,
            workspace_registry=workspace_registry,
        )

    if name == "search_papers_serpapi_versions":
        if not enable_serpapi:
            raise ValueError(
                "search_papers_serpapi_versions requires SerpApi, "
                "which is not enabled. "
                "Set PAPER_CHASER_ENABLE_SERPAPI=true and provide SERPAPI_API_KEY."
            )
        validated_payload = TOOL_INPUT_MODELS[name].model_validate(arguments)
        args_dict = validated_payload.model_dump(by_alias=False)
        ctx_hash = compute_context_hash(name, args_dict)
        result = await serpapi_client.search_versions(
            cluster_id=args_dict["cluster_id"],
            limit=args_dict.get("limit", 10),
            start=_cursor_to_offset(
                args_dict.get("cursor"),
                name,
                context_hash=ctx_hash,
                expected_provider="serpapi_google_scholar",
            )
            or 0,
        )
        serialized = dump_jsonable(result)
        serialized = _encode_next_cursor(
            serialized,
            name,
            context_hash=ctx_hash,
            provider="serpapi_google_scholar",
        )
        return _finalize_tool_result(
            name,
            arguments,
            serialized,
            workspace_registry=workspace_registry,
        )

    if name == "get_author_profile_serpapi":
        if not enable_serpapi:
            raise ValueError(
                "get_author_profile_serpapi requires SerpApi, which is not enabled. "
                "Set PAPER_CHASER_ENABLE_SERPAPI=true and provide SERPAPI_API_KEY."
            )
        validated_payload = TOOL_INPUT_MODELS[name].model_validate(arguments)
        args_dict = validated_payload.model_dump(by_alias=False)
        result = await serpapi_client.get_author_profile(
            author_id=args_dict["author_id"],
        )
        return _finalize_tool_result(
            name,
            arguments,
            result,
            workspace_registry=workspace_registry,
        )

    if name == "get_author_articles_serpapi":
        if not enable_serpapi:
            raise ValueError(
                "get_author_articles_serpapi requires SerpApi, which is not enabled. "
                "Set PAPER_CHASER_ENABLE_SERPAPI=true and provide SERPAPI_API_KEY."
            )
        validated_payload = TOOL_INPUT_MODELS[name].model_validate(arguments)
        args_dict = validated_payload.model_dump(by_alias=False)
        ctx_hash = compute_context_hash(name, args_dict)
        result = await serpapi_client.get_author_articles(
            author_id=args_dict["author_id"],
            limit=args_dict.get("limit", 10),
            start=_cursor_to_offset(
                args_dict.get("cursor"),
                name,
                context_hash=ctx_hash,
                expected_provider="serpapi_google_scholar",
            )
            or 0,
            sort=args_dict.get("sort"),
        )
        serialized = dump_jsonable(result)
        serialized = _encode_next_cursor(
            serialized,
            name,
            context_hash=ctx_hash,
            provider="serpapi_google_scholar",
        )
        return _finalize_tool_result(
            name,
            arguments,
            serialized,
            workspace_registry=workspace_registry,
        )

    if name == "get_serpapi_account_status":
        if not enable_serpapi:
            raise ValueError(
                "get_serpapi_account_status requires SerpApi, which is not enabled. "
                "Set PAPER_CHASER_ENABLE_SERPAPI=true and provide SERPAPI_API_KEY."
            )
        result = await serpapi_client.get_account_status()
        return _finalize_tool_result(
            name,
            arguments,
            result,
            workspace_registry=workspace_registry,
        )

    if name in PROVIDER_SEARCH_TOOLS:
        provider_arguments = cast(
            BasicSearchPapersArgs,
            TOOL_INPUT_MODELS[name].model_validate(arguments),
        )
        provider_search_result = await search_papers_with_fallback(
            query=provider_arguments.query,
            limit=provider_arguments.limit,
            year=provider_arguments.year,
            fields=getattr(provider_arguments, "fields", None),
            venue=getattr(provider_arguments, "venue", None),
            publication_date_or_year=getattr(provider_arguments, "publication_date_or_year", None),
            fields_of_study=getattr(provider_arguments, "fields_of_study", None),
            publication_types=getattr(provider_arguments, "publication_types", None),
            open_access_pdf=getattr(provider_arguments, "open_access_pdf", None),
            min_citation_count=getattr(provider_arguments, "min_citation_count", None),
            enable_core=enable_core,
            enable_semantic_scholar=enable_semantic_scholar,
            enable_arxiv=enable_arxiv,
            enable_serpapi=enable_serpapi,
            enable_scholarapi=enable_scholarapi,
            provider_order=[PROVIDER_SEARCH_TOOLS[name]],
            core_client=core_client,
            semantic_client=client,
            arxiv_client=arxiv_client,
            serpapi_client=serpapi_client,
            scholarapi_client=scholarapi_client,
            provider_registry=provider_registry,
            allow_default_hedging=False,
        )
        return _finalize_tool_result(
            name,
            arguments,
            provider_search_result,
            workspace_registry=workspace_registry,
        )

    if name == "get_paper_citation_formats":
        validated_cf = cast(
            GetCitationFormatsArgs,
            TOOL_INPUT_MODELS[name].model_validate(arguments),
        )
        if not enable_serpapi:
            raise ValueError(
                "get_paper_citation_formats requires SerpApi, which is not enabled. "
                "Set PAPER_CHASER_ENABLE_SERPAPI=true and provide SERPAPI_API_KEY "
                "to use this tool. SerpApi is a paid service — see "
                "https://serpapi.com for details."
            )
        if serpapi_client is None:
            raise ValueError(
                "SerpApi client is not available. Set PAPER_CHASER_ENABLE_SERPAPI=true and SERPAPI_API_KEY."
            )
        try:
            raw = await serpapi_client.get_citation_formats(
                result_id=validated_cf.result_id,
            )
        except SerpApiKeyMissingError:
            raise
        raw_citations = raw.get("citations") or []
        raw_links = raw.get("links") or []
        citation_response = CitationFormatsResponse(
            result_id=validated_cf.result_id,
            citations=[
                CitationFormat(
                    title=str(c.get("title") or ""),
                    snippet=str(c.get("snippet") or ""),
                )
                for c in raw_citations
                if isinstance(c, dict)
            ],
            export_links=[
                ExportLink(
                    name=str(lnk.get("name") or ""),
                    link=str(lnk.get("link") or ""),
                )
                for lnk in raw_links
                if isinstance(lnk, dict)
            ],
        )
        return _finalize_tool_result(
            name,
            arguments,
            dump_jsonable(citation_response),
            workspace_registry=workspace_registry,
        )

    if name == "search_papers_bulk":
        validated_payload = TOOL_INPUT_MODELS[name].model_validate(arguments)
        args_dict = validated_payload.model_dump(by_alias=False)
        ctx_hash = compute_context_hash(name, args_dict)
        method = getattr(client, "search_papers_bulk")
        result = await method(
            query=args_dict["query"],
            fields=args_dict.get("fields"),
            token=_cursor_to_bulk_token(
                args_dict.get("cursor"),
                tool=name,
                context_hash=ctx_hash,
            ),
            sort=args_dict.get("sort"),
            limit=args_dict.get("limit", 100),
            year=args_dict.get("year"),
            publication_date_or_year=args_dict.get("publication_date_or_year"),
            fields_of_study=args_dict.get("fields_of_study"),
            publication_types=args_dict.get("publication_types"),
            open_access_pdf=args_dict.get("open_access_pdf"),
            min_citation_count=args_dict.get("min_citation_count"),
        )
        serialized = dump_jsonable(result)
        serialized = _encode_next_bulk_cursor(serialized, name, context_hash=ctx_hash)
        sort_param = args_dict.get("sort")
        if sort_param:
            retrieval_note = (
                f"Results are sorted by '{sort_param}' (Semantic Scholar "
                "/paper/search/bulk). Bulk retrieval is exhaustive corpus "
                "collection — results are ordered by the specified sort, not "
                "by relevance to the query. This is a different contract from "
                "search_papers. Use pagination.nextCursor to continue."
            )
        else:
            retrieval_note = (
                "ORDERING: search_papers_bulk uses exhaustive corpus traversal "
                "with an internal ordering that is NOT relevance-ranked. This is "
                "NOT 'page 2' of search_papers — the ranking semantics differ and "
                "results may appear unrelated to the discovery page. For "
                "relevance-ranked results use search_papers or "
                "search_papers_semantic_scholar. For citation-ranked bulk "
                "retrieval pass sort='citationCount:desc'. Use "
                "pagination.nextCursor to continue this bulk stream."
            )
        serialized.setdefault("retrievalNote", retrieval_note)
        return _finalize_tool_result(
            name,
            arguments,
            serialized,
            workspace_registry=workspace_registry,
        )

    if not enable_openalex and name.endswith("_openalex"):
        raise ValueError(
            f"{name} requires OpenAlex, which is disabled. Set PAPER_CHASER_ENABLE_OPENALEX=true to use this tool."
        )

    if name == "get_paper_details_openalex":
        validated_payload = TOOL_INPUT_MODELS[name].model_validate(arguments)
        args_dict = validated_payload.model_dump(by_alias=False)
        result = await openalex_client.get_paper_details(paper_id=args_dict["paper_id"])
        return _finalize_tool_result(
            name,
            arguments,
            result,
            workspace_registry=workspace_registry,
        )

    if name == "get_paper_citations_openalex":
        validated_payload = TOOL_INPUT_MODELS[name].model_validate(arguments)
        args_dict = validated_payload.model_dump(by_alias=False)
        ctx_hash = compute_context_hash(name, args_dict)
        result = await openalex_client.get_paper_citations(
            paper_id=args_dict["paper_id"],
            limit=args_dict.get("limit", 100),
            cursor=_cursor_to_bulk_token(
                args_dict.get("cursor"),
                tool=name,
                context_hash=ctx_hash,
                expected_provider="openalex",
            ),
        )
        serialized = dump_jsonable(result)
        serialized = _encode_next_bulk_cursor(
            serialized,
            name,
            context_hash=ctx_hash,
            provider="openalex",
        )
        return _finalize_tool_result(
            name,
            arguments,
            serialized,
            workspace_registry=workspace_registry,
        )

    if name == "get_paper_references_openalex":
        validated_payload = TOOL_INPUT_MODELS[name].model_validate(arguments)
        args_dict = validated_payload.model_dump(by_alias=False)
        ctx_hash = compute_context_hash(name, args_dict)
        result = await openalex_client.get_paper_references(
            paper_id=args_dict["paper_id"],
            limit=args_dict.get("limit", 100),
            offset=_cursor_to_offset(
                args_dict.get("cursor"),
                name,
                context_hash=ctx_hash,
                expected_provider="openalex",
            )
            or 0,
        )
        serialized = dump_jsonable(result)
        serialized = _encode_next_cursor(
            serialized,
            name,
            context_hash=ctx_hash,
            provider="openalex",
        )
        return _finalize_tool_result(
            name,
            arguments,
            serialized,
            workspace_registry=workspace_registry,
        )

    if name == "search_authors_openalex":
        validated_payload = TOOL_INPUT_MODELS[name].model_validate(arguments)
        args_dict = validated_payload.model_dump(by_alias=False)
        ctx_hash = compute_context_hash(name, args_dict)
        result = await openalex_client.search_authors(
            query=args_dict["query"],
            limit=args_dict.get("limit", 10),
            cursor=_cursor_to_bulk_token(
                args_dict.get("cursor"),
                tool=name,
                context_hash=ctx_hash,
                expected_provider="openalex",
            ),
        )
        serialized = dump_jsonable(result)
        serialized = _encode_next_bulk_cursor(
            serialized,
            name,
            context_hash=ctx_hash,
            provider="openalex",
        )
        return _finalize_tool_result(
            name,
            arguments,
            serialized,
            workspace_registry=workspace_registry,
        )

    if name == "get_author_info_openalex":
        validated_payload = TOOL_INPUT_MODELS[name].model_validate(arguments)
        args_dict = validated_payload.model_dump(by_alias=False)
        result = await openalex_client.get_author_info(author_id=args_dict["author_id"])
        return _finalize_tool_result(
            name,
            arguments,
            result,
            workspace_registry=workspace_registry,
        )

    if name == "get_author_papers_openalex":
        validated_payload = TOOL_INPUT_MODELS[name].model_validate(arguments)
        args_dict = validated_payload.model_dump(by_alias=False)
        ctx_hash = compute_context_hash(name, args_dict)
        result = await openalex_client.get_author_papers(
            author_id=args_dict["author_id"],
            limit=args_dict.get("limit", 100),
            cursor=_cursor_to_bulk_token(
                args_dict.get("cursor"),
                tool=name,
                context_hash=ctx_hash,
                expected_provider="openalex",
            ),
            year=args_dict.get("year"),
        )
        serialized = dump_jsonable(result)
        serialized = _encode_next_bulk_cursor(
            serialized,
            name,
            context_hash=ctx_hash,
            provider="openalex",
        )
        return _finalize_tool_result(
            name,
            arguments,
            serialized,
            workspace_registry=workspace_registry,
        )

    try:
        method_name, build_args = NON_SEARCH_TOOL_HANDLERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown tool: {name}") from exc

    validated_payload = TOOL_INPUT_MODELS[name].model_validate(arguments)
    args_dict = validated_payload.model_dump(by_alias=False)
    ctx_hash = compute_context_hash(name, args_dict) if name in OFFSET_TOOLS else None
    method = getattr(client, method_name)
    result = await method(**build_args(args_dict))
    serialized = dump_jsonable(result)
    if name == "search_snippets" and isinstance(serialized, dict):
        serialized = await _maybe_fallback_snippet_search(
            serialized=serialized,
            args_dict=args_dict,
            client=client,
        )
    if name in OFFSET_TOOLS:
        serialized = _encode_next_cursor(serialized, name, context_hash=ctx_hash)
    elicited = await _maybe_elicit_and_retry(
        tool_name=name,
        arguments=arguments,
        result=serialized,
        client=client,
        core_client=core_client,
        openalex_client=openalex_client,
        arxiv_client=arxiv_client,
        serpapi_client=serpapi_client,
        scholarapi_client=scholarapi_client,
        crossref_client=crossref_client,
        unpaywall_client=unpaywall_client,
        ecos_client=ecos_client,
        federal_register_client=federal_register_client,
        govinfo_client=govinfo_client,
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
        provider_order=provider_order,
        provider_registry=provider_registry,
        workspace_registry=workspace_registry,
        enrichment_service=resolved_enrichment_service,
        agentic_runtime=agentic_runtime,
        ctx=ctx,
        allow_elicitation=allow_elicitation,
    )
    if elicited is not None:
        return elicited
    return _finalize_tool_result(
        name,
        arguments,
        serialized,
        workspace_registry=workspace_registry,
    )


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


async def _dispatch_search_papers_smart(
    ctx: DispatchContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Entrypoint for the ``search_papers_smart`` tool.

    Extracted from ``dispatch_tool`` in Phase 2 Track C amendment 3. The
    body touches only ``ctx.agentic_runtime`` and the per-call ``ctx.ctx``
    MCP context, so no additional rewiring was required.
    """
    smart_args = cast(
        SmartSearchPapersArgs,
        TOOL_INPUT_MODELS["search_papers_smart"].model_validate(arguments),
    )
    if ctx.agentic_runtime is None:
        return {
            "error": "FEATURE_NOT_CONFIGURED",
            "message": ("search_papers_smart is not available because the agentic runtime was not initialized."),
            "fallbackTools": [
                "search_papers",
                "search_papers_bulk",
                "search_papers_match",
            ],
        }
    return await ctx.agentic_runtime.search_papers_smart(
        query=smart_args.query,
        limit=smart_args.limit,
        search_session_id=smart_args.search_session_id,
        mode=smart_args.mode,
        year=smart_args.year,
        venue=smart_args.venue,
        focus=smart_args.focus,
        latency_profile=smart_args.latency_profile,
        provider_budget=(
            smart_args.provider_budget.model_dump(by_alias=False) if smart_args.provider_budget is not None else None
        ),
        include_enrichment=smart_args.include_enrichment,
        ctx=ctx.ctx,
    )


async def _dispatch_ask_result_set(
    ctx: DispatchContext,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Entrypoint for the ``ask_result_set`` tool.

    Extracted from ``dispatch_tool`` in Phase 2 Track C amendment 3.
    """
    ask_args = cast(
        AskResultSetArgs,
        TOOL_INPUT_MODELS["ask_result_set"].model_validate(arguments),
    )
    if ctx.agentic_runtime is None:
        return {
            "error": "FEATURE_NOT_CONFIGURED",
            "message": "ask_result_set requires the agentic runtime to be enabled.",
            "fallbackTools": [
                "search_papers",
                "get_paper_details",
                "get_paper_citations",
            ],
        }
    return await ctx.agentic_runtime.ask_result_set(
        search_session_id=ask_args.search_session_id,
        question=ask_args.question,
        top_k=ask_args.top_k,
        answer_mode=ask_args.answer_mode,
        latency_profile=ask_args.latency_profile,
        ctx=ctx.ctx,
    )


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
