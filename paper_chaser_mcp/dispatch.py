"""Dispatch helpers for MCP tool routing."""

import logging
import re
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any, Callable, cast

from .agentic.planner import (
    detect_literature_intent,
    detect_regulatory_intent,
    looks_like_exact_title,
    query_facets,
    query_terms,
)
from .citation_repair import looks_like_citation_query, looks_like_paper_identifier, parse_citation, resolve_citation
from .clients.scholarapi import (
    ScholarApiError,
    ScholarApiKeyMissingError,
    ScholarApiQuotaError,
    ScholarApiUpstreamError,
)
from .clients.serpapi import SerpApiKeyMissingError
from .compat import augment_tool_result, build_clarification
from .enrichment import (
    PaperEnrichmentService,
    attach_enrichments_to_paper_payload,
    hydrate_paper_for_enrichment,
)
from .guided_semantic import (
    build_evidence_records,
    build_follow_up_decision,
    build_routing_decision,
    classify_answerability,
    explicit_source_reference,
)
from .identifiers import resolve_doi_from_paper_payload
from .models import TOOL_INPUT_MODELS, CitationFormatsResponse, RuntimeSummary, dump_jsonable
from .models.common import (
    AbstentionDetails,
    CitationFormat,
    ExportLink,
    GuidedExecutionProvenance,
    GuidedResultState,
    InputNormalization,
    MachineFailure,
    NormalizationRepair,
    SessionCandidate,
    SessionResolution,
    SourceResolution,
)
from .models.tools import (
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
from .provider_runtime import ProviderOutcomeEnvelope, ProviderStatusBucket, policy_for_provider
from .search import search_papers_with_fallback
from .utils.cursor import (
    OFFSET_TOOLS,
    PROVIDER,
    SUPPORTED_VERSIONS,
    compute_context_hash,
    cursor_from_offset,
    cursor_from_token,
    decode_bulk_cursor,
    decode_cursor,
    is_legacy_offset,
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


def _cursor_to_offset(
    cursor: str | None,
    tool: str | None = None,
    context_hash: str | None = None,
    expected_provider: str = PROVIDER,
) -> int | None:
    """Decode an opaque pagination cursor to an integer offset.

    Accepts both structured server-issued cursors (URL-safe base64 JSON) and
    legacy plain integer strings for backward compatibility.

    Returns ``None`` when *cursor* is ``None`` (start from the beginning).

    When *tool* is provided, structured cursors are validated to ensure they
    were issued by the same tool.  When *context_hash* is also provided, the
    cursor's embedded hash is compared against the current request's context,
    so a cursor from a different query on the same tool is rejected.

    In production all dispatch handlers pass *tool* and *context_hash*
    explicitly.  Passing ``tool=None`` skips cross-tool validation; this is
    intentional only for the ``cursor=None`` early-return path.

    Raises ``ValueError`` for stale, mistyped, corrupted, cross-tool, or
    cross-query cursors, unsupported versions, unknown providers, and negative
    integer offsets.
    """
    if cursor is None:
        return None
    if is_legacy_offset(cursor):
        offset = int(cursor)
        if offset < 0:
            raise ValueError(
                f"Invalid pagination cursor {cursor!r}: offset must be non-negative. "
                "code=INVALID_CURSOR. "
                f"{CURSOR_REUSE_HINT} Restart the request without a cursor."
            )
        return offset
    # Structured cursor: decode and validate
    try:
        state = decode_cursor(cursor)
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
    if tool is not None and state.tool != tool:
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
    return state.offset


def _encode_next_cursor(
    result: dict[str, Any],
    tool: str,
    context_hash: str | None = None,
    provider: str = PROVIDER,
) -> dict[str, Any]:
    """Re-encode a plain integer ``nextCursor`` in *result* as a structured cursor.

    Operates on the serialized dict returned by ``dump_jsonable``.  The
    ``pagination`` key is present on all offset-backed tool responses.

    The *context_hash* is embedded into the new cursor so that future requests
    can validate that the cursor belongs to the same query stream.

    If ``pagination.nextCursor`` is already a structured (non-integer) cursor or
    is ``None``, the result is returned unchanged.
    """
    pagination = result.get("pagination")
    if not isinstance(pagination, dict):
        return result
    raw_cursor = pagination.get("nextCursor")
    if raw_cursor is None:
        return result
    if is_legacy_offset(raw_cursor):
        pagination["nextCursor"] = cursor_from_offset(
            tool,
            int(raw_cursor),
            context_hash=context_hash,
            provider=provider,
        )
    return result


def _snippet_fallback_query(query: str) -> str:
    normalized = " ".join(str(query or "").strip().strip("\"'").split())
    tokens = re.findall(r"[A-Za-z0-9]{3,}", normalized)
    return " ".join(tokens[:10]) if tokens else normalized


def _snippet_fallback_results(
    degraded_payload: dict[str, Any],
    papers_payload: dict[str, Any],
) -> dict[str, Any]:
    fallback_items: list[dict[str, Any]] = []
    for index, paper in enumerate((papers_payload.get("data") or []), start=1):
        if not isinstance(paper, dict):
            continue
        snippet_text = str(paper.get("abstract") or paper.get("title") or "").strip()
        if not snippet_text:
            continue
        fallback_items.append(
            {
                "score": round(max(0.0, 1.0 - ((index - 1) * 0.05)), 6),
                "snippet": {
                    "text": snippet_text[:400],
                    "snippetKind": "fallback_paper_match",
                    "section": "abstract" if paper.get("abstract") else "title",
                },
                "paper": {
                    "paperId": paper.get("paperId"),
                    "title": paper.get("title"),
                    "year": paper.get("year"),
                    "url": paper.get("url"),
                },
            }
        )
    if not fallback_items:
        return degraded_payload

    payload = dict(degraded_payload)
    payload["data"] = fallback_items
    payload["fallbackUsed"] = "search_papers"
    payload["message"] = (
        "Semantic Scholar snippet search could not serve this query, so the "
        "server returned best-effort paper matches from search_papers instead."
    )
    return payload


async def _maybe_fallback_snippet_search(
    *,
    serialized: dict[str, Any],
    args_dict: dict[str, Any],
    client: Any,
) -> dict[str, Any]:
    if serialized.get("degraded") is not True or serialized.get("data"):
        return serialized
    fallback_query = _snippet_fallback_query(str(args_dict.get("query") or ""))
    if not fallback_query:
        return serialized
    try:
        fallback_payload = dump_jsonable(
            await client.search_papers(
                query=fallback_query,
                limit=args_dict.get("limit", 10),
                fields=["paperId", "title", "year", "url", "abstract"],
                year=args_dict.get("year"),
                publication_date_or_year=args_dict.get("publication_date_or_year"),
                fields_of_study=args_dict.get("fields_of_study"),
                min_citation_count=args_dict.get("min_citation_count"),
                venue=[args_dict["venue"]] if args_dict.get("venue") else None,
            )
        )
    except Exception:
        return serialized
    return _snippet_fallback_results(serialized, fallback_payload)


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


def _smart_runtime_provider_state(agentic_runtime: Any) -> tuple[dict[str, bool], list[str], str | None, str | None]:
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
    if agentic_runtime is not None and hasattr(agentic_runtime, "smart_provider_diagnostics"):
        smart_provider_enabled, smart_provider_order = agentic_runtime.smart_provider_diagnostics()
    provider_bundle = getattr(agentic_runtime, "_provider_bundle", None)
    if provider_bundle is not None and hasattr(provider_bundle, "selection_metadata"):
        selection = provider_bundle.selection_metadata()
        configured_value = selection.get("configuredSmartProvider")
        active_value = selection.get("activeSmartProvider")
        configured_smart_provider = str(configured_value) if configured_value else None
        active_smart_provider = str(active_value) if active_value else None
    return smart_provider_enabled, smart_provider_order, configured_smart_provider, active_smart_provider


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
    active_provider_set = sorted([provider for provider, enabled in enabled_state.items() if enabled])
    disabled_provider_set = sorted([provider for provider, enabled in enabled_state.items() if not enabled])
    runtime_warnings: list[str] = []
    enabled_raw_providers = [provider for provider in (provider_order or []) if provider in active_provider_set]
    if len(enabled_raw_providers) <= 1:
        runtime_warnings.append(
            "The effective broker order is very narrow, so no-result responses "
            "may reflect limited provider coverage rather than absence of evidence."
        )
    if not enable_serpapi and serpapi_client is not None:
        runtime_warnings.append(
            "SerpApi client state is present but PAPER_CHASER_ENABLE_SERPAPI is "
            "false, so paid recall recovery is disabled."
        )
    if not enable_scholarapi and scholarapi_client is not None:
        runtime_warnings.append(
            "ScholarAPI client state is present but PAPER_CHASER_ENABLE_"
            "SCHOLARAPI is false, so ScholarAPI discovery and full-text paths "
            "are inactive."
        )
    if hide_disabled_tools:
        runtime_warnings.append(
            "Disabled tools are hidden from list_tools output, which can make capability gaps harder to diagnose."
        )
    if transport_mode == "stdio":
        runtime_warnings.append(
            "The current runtime is stdio, so HTTP deployment settings do not affect this invocation path."
        )
    if getattr(ecos_client, "verify_tls", True) is False:
        runtime_warnings.append(
            "ECOS TLS verification is disabled. This should only be a temporary troubleshooting state."
        )
    if tool_profile == "guided" and hide_disabled_tools:
        runtime_warnings.append(
            "Guided profile is active while expert tools are hidden, so escalation paths are intentionally unavailable."
        )
    if configured_smart_provider == "huggingface":
        runtime_warnings.append(
            "Hugging Face is configured as a chat-only smart provider in this repo; embeddings stay disabled."
        )
    if configured_smart_provider == "openrouter":
        runtime_warnings.append(
            "OpenRouter is configured as a chat-only smart provider in this repo; embeddings stay disabled."
        )
    runtime_summary = RuntimeSummary(
        effectiveProfile=tool_profile,
        transportMode=transport_mode,
        smartLayerEnabled=agentic_runtime is not None,
        activeProviderSet=active_provider_set,
        disabledProviderSet=disabled_provider_set,
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
    )
    if provider_registry is None:
        return {
            "generatedAt": None,
            "providerOrder": provider_list_order,
            "providers": [],
            "runtimeSummary": runtime_summary.model_dump(by_alias=True, exclude_none=True),
        }

    snapshot = provider_registry.snapshot(
        enabled=enabled_state,
        provider_order=provider_list_order,
    )
    if not include_recent_outcomes:
        for provider in snapshot.get("providers", []):
            if isinstance(provider, dict):
                provider["recentOutcomes"] = []
    snapshot["runtimeSummary"] = runtime_summary.model_dump(by_alias=True, exclude_none=True)
    return snapshot


def _tokenize_relevance_text(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", value.lower()))


def _facet_match(tokens: set[str], facet: str) -> bool:
    facet_tokens = _tokenize_relevance_text(facet)
    return bool(facet_tokens) and facet_tokens.issubset(tokens)


def _topical_relevance_from_signals(
    *,
    query_similarity: float,
    title_facet_coverage: float,
    title_anchor_coverage: float,
    query_facet_coverage: float,
    query_anchor_coverage: float,
) -> str:
    title_has_anchor = title_facet_coverage > 0 or title_anchor_coverage > 0
    body_has_anchor = query_facet_coverage > 0 or query_anchor_coverage > 0
    has_facet_signal = title_facet_coverage > 0 or query_facet_coverage > 0
    # Require a multi-token phrase match (facet) for the standard threshold, or a
    # strict majority of query terms when no phrase match exists.  A single-token
    # title hit with low similarity is a weak signal, not grounded evidence.
    if title_has_anchor and ((has_facet_signal and query_similarity >= 0.25) or query_similarity > 0.5):
        return "on_topic"
    if query_similarity < 0.12 or not (title_has_anchor or body_has_anchor):
        return "off_topic"
    return "weak_match"


def _paper_topical_relevance(query: str, paper: dict[str, Any]) -> str:
    facets = query_facets(query)
    terms = query_terms(query)
    title_tokens = _tokenize_relevance_text(str(paper.get("title") or ""))
    paper_tokens = _tokenize_relevance_text(
        " ".join(
            part
            for part in [
                str(paper.get("title") or ""),
                str(paper.get("abstract") or ""),
                str(paper.get("venue") or ""),
            ]
            if part
        )
    )
    matched_terms = [term for term in terms if term in paper_tokens]
    matched_title_terms = [term for term in terms if term in title_tokens]
    matched_facets = [facet for facet in facets if _facet_match(paper_tokens, facet)]
    matched_title_facets = [facet for facet in facets if _facet_match(title_tokens, facet)]
    term_coverage = len(matched_terms) / len(terms) if terms else 0.0
    title_term_coverage = len(matched_title_terms) / len(terms) if terms else 0.0
    query_similarity = max(term_coverage, title_term_coverage)
    return _topical_relevance_from_signals(
        query_similarity=query_similarity,
        title_facet_coverage=(len(matched_title_facets) / len(facets) if facets else 0.0),
        title_anchor_coverage=title_term_coverage,
        query_facet_coverage=(len(matched_facets) / len(facets) if facets else 0.0),
        query_anchor_coverage=term_coverage,
    )


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


def _guided_source_record_from_structured_source(source: dict[str, Any], *, index: int) -> dict[str, Any]:
    return {
        "sourceId": _guided_source_id(source, fallback_prefix="source", index=index),
        "sourceAlias": f"source-{index}",
        "title": source.get("title"),
        "provider": source.get("provider"),
        "sourceType": source.get("sourceType") or "unknown",
        "verificationStatus": source.get("verificationStatus") or "unverified",
        "accessStatus": source.get("accessStatus") or "access_unverified",
        "topicalRelevance": source.get("topicalRelevance") or "weak_match",
        "confidence": source.get("confidence") or "medium",
        "isPrimarySource": bool(source.get("isPrimarySource")),
        "canonicalUrl": source.get("canonicalUrl"),
        "retrievedUrl": source.get("retrievedUrl"),
        "fullTextObserved": bool(source.get("fullTextObserved")),
        "abstractObserved": bool(source.get("abstractObserved")),
        "openAccessRoute": _guided_open_access_route(source),
        "citationText": source.get("citationText"),
        "citation": _guided_citation_from_structured_source(source),
        "date": source.get("date"),
        "note": source.get("note"),
    }


def _guided_source_record_from_paper(query: str, paper: dict[str, Any], *, index: int) -> dict[str, Any]:
    canonical_url = paper.get("canonicalUrl") or paper.get("url") or paper.get("pdfUrl")
    source_type = paper.get("sourceType") or "scholarly_article"
    verification_status = paper.get("verificationStatus") or "verified_metadata"
    access_status = paper.get("accessStatus") or (
        "full_text_verified" if paper.get("fullTextObserved") else "access_unverified"
    )
    topical_relevance = _paper_topical_relevance(query, paper)
    confidence = paper.get("confidence") or ("high" if topical_relevance == "on_topic" else "medium")
    return {
        "sourceId": _guided_source_id(paper, fallback_prefix="paper", index=index),
        "sourceAlias": f"source-{index}",
        "title": paper.get("title"),
        "provider": paper.get("source"),
        "sourceType": source_type,
        "verificationStatus": verification_status,
        "accessStatus": access_status,
        "topicalRelevance": topical_relevance,
        "confidence": confidence,
        "isPrimarySource": bool(paper.get("isPrimarySource")),
        "canonicalUrl": canonical_url,
        "retrievedUrl": paper.get("retrievedUrl") or canonical_url,
        "fullTextObserved": bool(paper.get("fullTextObserved")),
        "abstractObserved": bool(paper.get("abstractObserved")),
        "openAccessRoute": _guided_open_access_route(paper),
        "citationText": str(paper.get("canonicalId") or paper.get("paperId") or "") or None,
        "citation": _guided_citation_from_paper(paper, canonical_url),
        "date": paper.get("publicationDate") or paper.get("year"),
        "note": paper.get("venue"),
    }


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


_GUIDED_QUERY_PREFIX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^\s*(?:please\s+|kindly\s+)?(?:help\s+me\s+)?(?:find|search(?:\s+for)?|look\s+up|research|summarize|show)\s+"
        r"(?:papers?|literature|studies|evidence|sources?|information)\s+(?:about|on|for)\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:please\s+|kindly\s+)?(?:help\s+me\s+)?(?:find|search(?:\s+for)?|look\s+up|research|summarize|show)\s+",
        re.IGNORECASE,
    ),
)

_GUIDED_CFR_SECTION_RE = re.compile(
    r"\b(?P<title>\d{1,2})\s*c\.?\s*f\.?\s*r\.?\s*(?P<part>\d{1,4})\s*(?:[.\-:/]\s*|\s+)(?P<section>\d{1,4})\b",
    re.IGNORECASE,
)
_GUIDED_CFR_PART_RE = re.compile(
    r"\b(?P<title>\d{1,2})\s*c\.?\s*f\.?\s*r\.?\s*part\s*(?P<part>\d{1,4})\b",
    re.IGNORECASE,
)
_GUIDED_FR_CITATION_RE = re.compile(
    r"\b(?P<volume>\d+)\s*f\.?\s*r\.?\s*(?P<page>\d+)\b",
    re.IGNORECASE,
)
_GUIDED_YEAR_RANGE_RE = re.compile(r"\b(?P<start>(?:19|20)\d{2})\s*[-:/]\s*(?P<end>(?:19|20)\d{2})\b")
_GUIDED_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _guided_normalize_whitespace(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _guided_normalize_source_locator(value: Any) -> str:
    normalized = _guided_normalize_whitespace(value).lower().rstrip("/")
    if not normalized:
        return ""
    normalized = re.sub(r"^https?://", "", normalized)
    normalized = re.sub(r"^www\.", "", normalized)
    return normalized


def _guided_strip_research_prefix(query: str) -> str:
    stripped = query
    for pattern in _GUIDED_QUERY_PREFIX_PATTERNS:
        candidate = pattern.sub("", stripped, count=1).strip()
        if candidate and candidate != stripped:
            return candidate
    return stripped


def _guided_normalize_citation_surface(text: str) -> str:
    normalized = text
    normalized = _GUIDED_CFR_SECTION_RE.sub(r"\g<title> CFR \g<part>.\g<section>", normalized)
    normalized = _GUIDED_CFR_PART_RE.sub(r"\g<title> CFR Part \g<part>", normalized)
    normalized = _GUIDED_FR_CITATION_RE.sub(r"\g<volume> FR \g<page>", normalized)
    return normalized


def _guided_normalize_year_hint(value: Any) -> str | None:
    text = _guided_normalize_whitespace(value)
    if not text:
        return None
    range_match = _GUIDED_YEAR_RANGE_RE.search(text)
    if range_match:
        start = range_match.group("start")
        end = range_match.group("end")
        return f"{start}:{end}" if start <= end else f"{end}:{start}"
    years = _GUIDED_YEAR_RE.findall(text)
    if len(years) >= 2:
        start, end = years[0], years[1]
        return f"{start}:{end}" if start <= end else f"{end}:{start}"
    if years:
        return years[0]
    return text


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


def _guided_source_resolution_payload(
    *,
    requested_source_id: str | None,
    resolved_source_id: str | None,
    match_type: str | None,
    available_source_ids: list[str] | None = None,
) -> dict[str, Any]:
    resolution = SourceResolution(
        requestedSourceId=_guided_normalize_whitespace(requested_source_id),
        resolvedSourceId=_guided_normalize_whitespace(resolved_source_id),
        matchType=match_type,
        availableSourceIds=available_source_ids or [],
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
    if status not in {"abstained", "needs_disambiguation", "insufficient_evidence"}:
        return None
    category = _guided_missing_evidence_type(status=status, evidence_gaps=evidence_gaps, sources=sources)
    if category == "anchor_missing":
        refinement_hints = ["Add a specific title, DOI, species name, agency, venue, or year range."]
    elif category == "off_topic_only":
        refinement_hints = ["Tighten the query to the exact topic or anchored subject you need."]
    elif category == "provider_gap":
        refinement_hints = [
            "Retry later or compare get_runtime_status if provider behavior differs across environments.",
        ]
    elif sources:
        refinement_hints = ["Inspect the returned sources before treating the result as settled."]
    else:
        refinement_hints = ["Narrow the request so the server can recover a stronger initial anchor."]
    details = AbstentionDetails(
        category=category,
        reason=(
            evidence_gaps[0] if evidence_gaps else "The current evidence was not strong enough to ground an answer."
        ),
        inspectableSourceCount=len(sources),
        onTopicSourceCount=int(trust_summary.get("onTopicSourceCount") or 0),
        weakMatchCount=int(trust_summary.get("weakMatchCount") or 0),
        offTopicCount=int(trust_summary.get("offTopicCount") or 0),
        canInspectSources=bool(sources),
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
        ):
            value = metadata.get(field)
            if value not in (None, "", [], {}) and field not in merged:
                merged[field] = value
        for field in merged_lists:
            for item in metadata.get(field) or []:
                text = str(item).strip()
                if text and text not in merged_lists[field]:
                    merged_lists[field].append(text)

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
        inferred_id, _ = _guided_resolve_session_id_for_source(workspace_registry, normalized_source_id)
        if inferred_id is not None:
            warnings.append(
                "searchSessionId "
                f"'{normalized_search_session_id}' was unavailable; using source-bearing session '{inferred_id}'."
            )
            normalized_search_session_id = inferred_id
        else:
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
        inferred_id, _ = _guided_resolve_session_id_for_source(workspace_registry, normalized_source_id)
        if inferred_id is not None:
            normalized_search_session_id = inferred_id
            warnings.append(f"searchSessionId was missing; inferred source-bearing session '{inferred_id}'.")
        else:
            inferred_id = _guided_infer_single_session_id(workspace_registry)
        if inferred_id is not None and not normalized_search_session_id:
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

    if not normalized_search_session_id:
        inferred_session_id, _ = _guided_resolve_session_id_for_source(workspace_registry, normalized_source_id)
        if inferred_session_id is not None:
            normalized_search_session_id = inferred_session_id
            normalized_args["searchSessionId"] = normalized_search_session_id
            warnings.append(
                f"searchSessionId was missing; inferred source-bearing session '{normalized_search_session_id}'."
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
    return bool(re.search(r"\b(?:doi|systematic review|meta-analysis|peer-reviewed|scientific reports?)\b", normalized))


def _guided_is_mixed_intent_query(query: str, focus: str | None = None) -> bool:
    return detect_regulatory_intent(query, focus) and _guided_mentions_literature(query, focus)


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

    merged: dict[str, Any] = {
        "providersAttempted": _merge_list("providersAttempted"),
        "providersSucceeded": _merge_list("providersSucceeded"),
        "providersFailed": _merge_list("providersFailed"),
        "providersZeroResults": _merge_list("providersZeroResults"),
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


def _guided_open_access_route(source: dict[str, Any]) -> str:
    explicit = str(source.get("openAccessRoute") or "").strip()
    if explicit:
        return explicit
    source_type = str(source.get("sourceType") or "")
    access_status = str(source.get("accessStatus") or "")
    canonical_url = str(source.get("canonicalUrl") or "").strip().lower()
    retrieved_url = str(source.get("retrievedUrl") or "").strip().lower()
    provider = str(source.get("provider") or source.get("source") or "").strip().lower()
    if "sci-hub" in retrieved_url:
        return "mirror_only"
    if access_status == "oa_verified" and canonical_url.startswith("https://doi.org/"):
        return "canonical_open_access"
    if provider in {"arxiv", "core", "openalex"} or source_type == "repository_record":
        return "repository_open_access"
    if access_status in {"full_text_verified", "oa_verified", "oa_uncertain", "abstract_only"}:
        return "non_oa_or_unconfirmed"
    return "unknown"


def _guided_citation_from_structured_source(source: dict[str, Any]) -> dict[str, Any] | None:
    citation = source.get("citation")
    if isinstance(citation, dict):
        return citation
    citation_text = str(source.get("citationText") or source.get("citation") or "").strip() or None
    title = str(source.get("title") or citation_text or "").strip() or None
    url = str(source.get("canonicalUrl") or source.get("retrievedUrl") or "").strip() or None
    year = _guided_year_text(source.get("date"))
    if not any([title, url, year, citation_text]):
        return None
    return {
        "authors": [],
        "year": year,
        "title": title,
        "journalOrPublisher": _guided_journal_or_publisher(source),
        "doi": None,
        "url": url,
        "sourceType": source.get("sourceType") or "unknown",
        "confidence": source.get("confidence") or "medium",
    }


def _guided_citation_from_paper(paper: dict[str, Any], canonical_url: str | None) -> dict[str, Any] | None:
    doi, _ = resolve_doi_from_paper_payload(paper)
    authors = [
        str(author.get("name") or "").strip()
        for author in (paper.get("authors") or [])
        if isinstance(author, dict) and str(author.get("name") or "").strip()
    ]
    year = _guided_year_text(paper.get("publicationDate") or paper.get("year"))
    journal_or_publisher = _guided_journal_or_publisher(paper)
    if not any([authors, year, paper.get("title"), journal_or_publisher, doi, canonical_url]):
        return None
    return {
        "authors": authors,
        "year": year,
        "title": paper.get("title"),
        "journalOrPublisher": journal_or_publisher,
        "doi": doi,
        "url": canonical_url,
        "sourceType": paper.get("sourceType") or "unknown",
        "confidence": paper.get("confidence") or "medium",
    }


def _guided_year_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return match.group(0) if match else None


def _guided_journal_or_publisher(payload: dict[str, Any]) -> str | None:
    enrichments = payload.get("enrichments")
    if isinstance(enrichments, dict):
        crossref = enrichments.get("crossref")
        if isinstance(crossref, dict):
            publisher = str(crossref.get("publisher") or "").strip()
            if publisher:
                return publisher
    venue = str(payload.get("venue") or "").strip()
    if venue:
        return venue
    provider = str(payload.get("provider") or payload.get("source") or "").strip()
    return provider or None


def _guided_trust_summary(sources: list[dict[str, Any]], evidence_gaps: list[str]) -> dict[str, Any]:
    return {
        "verifiedSourceCount": sum(
            1
            for source in sources
            if source.get("verificationStatus") in {"verified_primary_source", "verified_metadata"}
        ),
        "onTopicSourceCount": sum(1 for source in sources if source.get("topicalRelevance") == "on_topic"),
        "weakMatchCount": sum(1 for source in sources if source.get("topicalRelevance") == "weak_match"),
        "offTopicCount": sum(1 for source in sources if source.get("topicalRelevance") == "off_topic"),
        "evidenceGapCount": len(evidence_gaps),
    }


def _guided_failure_summary(
    *,
    failure_summary: dict[str, Any] | None,
    status: str,
    sources: list[dict[str, Any]],
    evidence_gaps: list[str],
) -> dict[str, Any]:
    if failure_summary is not None:
        summary = dict(failure_summary)
    else:
        summary = {}
    outcome = str(summary.get("outcome") or "").strip()
    if not outcome:
        if status == "abstained":
            outcome = "no_failure"
        elif summary.get("fallbackAttempted"):
            outcome = "fallback_success"
        else:
            outcome = "no_failure"
    recommended_next_action = summary.get("recommendedNextAction")
    if not recommended_next_action:
        recommended_next_action = "inspect_source" if sources else "research"
    completeness_impact = summary.get("completenessImpact")
    if not completeness_impact and evidence_gaps:
        completeness_impact = evidence_gaps[0]
    what_still_worked = summary.get("whatStillWorked")
    if not what_still_worked:
        what_still_worked = (
            "The guided run still returned inspectable sources."
            if sources
            else "No provider failures were recorded, but the evidence was not strong enough to ground a result."
        )
    return {
        "outcome": outcome,
        "whatFailed": summary.get("whatFailed"),
        "whatStillWorked": what_still_worked,
        "fallbackAttempted": bool(summary.get("fallbackAttempted")),
        "fallbackMode": summary.get("fallbackMode"),
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
) -> str:
    if verified_findings:
        return f"This result contains {len(verified_findings)} verified finding(s) grounded in the returned sources."
    if status == "partial":
        if source_count <= 0:
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


def _guided_research_status(
    *,
    intent: str,
    sources: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    unverified_leads_count: int,
    coverage_summary: dict[str, Any] | None,
    failure_summary: dict[str, Any] | None,
    clarification: dict[str, Any] | None,
) -> str:
    if clarification is not None:
        return "needs_disambiguation"
    if intent == "known_item" and findings:
        return "succeeded"
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
                return "partial" if failure_summary is not None else "succeeded"
            if primary_sources:
                return "partial"
            if unverified_leads_count > 0:
                return "partial"
            return "abstained"
        if primary_sources:
            return "partial" if failure_summary is not None else "succeeded"
        if unverified_leads_count > 0:
            return "partial"
        return "needs_disambiguation" if sources else "abstained"
    if len(findings) >= 2:
        return "partial" if failure_summary is not None else "succeeded"
    if sources:
        return "partial"
    if unverified_leads_count > 0:
        return "partial"
    return "abstained"


def _guided_machine_failure_payload(
    *,
    search_session_id: str | None,
    error: Exception,
    normalization: dict[str, Any] | None = None,
    execution_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence_gaps = ["Smart runtime returned an invalid or unstructured result payload, so guided output was degraded."]
    failure_summary = _guided_failure_summary(
        failure_summary={
            "outcome": "total_failure",
            "whatFailed": "smart_runtime_structural_failure",
            "whatStillWorked": "The guided wrapper recovered and returned a machine-readable failure state.",
            "fallbackAttempted": False,
            "fallbackMode": None,
            "primaryPathFailureReason": str(type(error).__name__),
            "completenessImpact": evidence_gaps[0],
            "recommendedNextAction": "research",
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
        ),
        "resultState": _guided_result_state(
            status="partial",
            sources=[],
            evidence_gaps=evidence_gaps,
            search_session_id=search_session_id,
        ),
        "machineFailure": MachineFailure(
            category="smart_runtime_structural_failure",
            errorType=type(error).__name__,
            error=str(error),
            retryable=True,
            bestNextInternalAction="research",
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
    if findings:
        top_claim = str(findings[0].get("claim") or "").strip()
        additional_count = max(len(findings) - 1, 0)
        if additional_count:
            summary = f"Top result: {top_claim}. Verified support includes {additional_count} additional source(s)."
        else:
            summary = f"Top result: {top_claim}."
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
) -> list[str]:
    actions: list[str] = []
    if search_session_id and has_sources:
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


def _guided_best_next_internal_action(
    *,
    status: str,
    has_sources: bool,
    search_session_id: str | None,
) -> str:
    if has_sources and search_session_id:
        return "inspect_source"
    if search_session_id:
        return "follow_up_research"
    if status in {"abstained", "needs_disambiguation", "failed"}:
        return "research"
    return "research"


def _guided_result_state(
    *,
    status: str,
    sources: list[dict[str, Any]],
    evidence_gaps: list[str],
    search_session_id: str | None,
) -> dict[str, Any]:
    has_sources = bool(sources)
    normalized_status = str(status or "").strip() or "unknown"
    if normalized_status in {"succeeded", "answered"} and has_sources:
        groundedness = "grounded"
    elif normalized_status == "answered":
        groundedness = "partial"
    elif normalized_status in {"partial", "insufficient_evidence"}:
        groundedness = "partial"
    elif normalized_status in {"abstained", "needs_disambiguation", "failed"}:
        groundedness = "insufficient_evidence"
    else:
        groundedness = "unknown"
    state = GuidedResultState(
        status=normalized_status,
        groundedness=groundedness,
        hasInspectableSources=has_sources,
        canAnswerFollowUp=bool(search_session_id),
        bestNextInternalAction=_guided_best_next_internal_action(
            status=normalized_status,
            has_sources=has_sources,
            search_session_id=search_session_id,
        ),
        missingEvidenceType=_guided_missing_evidence_type(
            status=normalized_status,
            evidence_gaps=evidence_gaps,
            sources=sources,
        ),
    )
    return state.model_dump(by_alias=True, exclude_none=True)


def _guided_record_source_candidates(record: Any) -> list[dict[str, Any]]:
    payload = record.payload if isinstance(record.payload, dict) else {}
    collected_sources: list[dict[str, Any]] = []

    for index, source in enumerate(payload.get("evidence") or [], start=1):
        if isinstance(source, dict):
            collected_sources.append(_guided_source_record_from_structured_source(source, index=index))

    for source in payload.get("sources") or []:
        if isinstance(source, dict):
            collected_sources.append(source)

    for key in ("structuredSources", "leads", "candidateLeads", "unverifiedLeads"):
        for index, source in enumerate(payload.get(key) or [], start=1):
            if isinstance(source, dict):
                collected_sources.append(_guided_source_record_from_structured_source(source, index=index))

    query = str(record.query or payload.get("query") or "")
    paper_sources = [
        _guided_source_record_from_paper(query, paper, index=index)
        for index, paper in enumerate(getattr(record, "papers", []) or [], start=1)
        if isinstance(paper, dict)
    ]
    collected_sources.extend(paper_sources)
    return _guided_dedupe_source_records(collected_sources)


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
    recommendations: list[str] = []
    canonical_url = str(source.get("canonicalUrl") or source.get("retrievedUrl") or "").strip()
    if canonical_url:
        recommendations.append(f"Open the canonical source: {canonical_url}")
    provider = str(source.get("provider") or "")
    if tool_profile != "guided" and provider == "govinfo":
        recommendations.append("Use get_cfr_text for authoritative CFR follow-through.")
    elif tool_profile != "guided" and provider == "federal_register":
        recommendations.append("Use get_federal_register_document to read the full Federal Register item.")
    elif tool_profile != "guided" and provider == "ecos":
        recommendations.append("Use get_document_text_ecos for the full ECOS document text when available.")
    elif tool_profile != "guided" and provider in {"semantic_scholar", "openalex", "arxiv", "core", "scholarapi"}:
        recommendations.append(
            "Use expert paper-detail tools if you need the full provider payload or citation expansion."
        )
    if tool_profile == "guided":
        recommendations.append("Use inspect_source to compare provenance and access signals before citing this source.")
    return recommendations[:3]


def _guided_follow_up_status(status: str | None) -> str:
    normalized = str(status or "").strip()
    if normalized in {"succeeded", "partial", "needs_disambiguation", "abstained", "failed"}:
        return normalized
    if normalized == "answered":
        return "succeeded"
    if normalized == "insufficient_evidence":
        return "partial"
    return "partial"


def _guided_contract_fields(
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
) -> dict[str, Any]:
    evidence, leads = build_evidence_records(sources=sources, leads=unverified_leads)
    routing_summary = build_routing_decision(
        query=query,
        intent=intent,
        strategy_metadata=strategy_metadata,
        coverage_summary=coverage_summary,
    ).model_dump(by_alias=True)
    result_status = _guided_follow_up_status(status)
    return {
        "resultStatus": result_status,
        "answerability": classify_answerability(
            status=status,
            evidence=evidence,
            leads=leads,
            evidence_gaps=evidence_gaps,
        ),
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
        },
        "coverageSummary": coverage_summary,
        "evidence": evidence,
        "leads": leads,
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
    sources = [
        _guided_source_record_from_structured_source(source, index=index)
        for index, source in enumerate(payload.get("evidence") or [], start=1)
        if isinstance(source, dict)
    ]
    if not sources:
        sources = [source for source in payload.get("sources") or [] if isinstance(source, dict)]
    if not sources:
        sources = [
            _guided_source_record_from_structured_source(source, index=index)
            for index, source in enumerate(payload.get("structuredSources") or [], start=1)
            if isinstance(source, dict)
        ]
    if not sources:
        query = str(record.query or payload.get("query") or "")
        sources = [
            _guided_source_record_from_paper(query, paper, index=index)
            for index, paper in enumerate(record.papers, start=1)
            if isinstance(paper, dict)
        ]
    unverified_leads = [lead for lead in payload.get("unverifiedLeads") or [] if isinstance(lead, dict)]
    if not unverified_leads:
        unverified_leads = [
            _guided_source_record_from_structured_source(source, index=index)
            for index, source in enumerate(payload.get("leads") or [], start=1)
            if isinstance(source, dict)
        ]
    if not unverified_leads:
        unverified_leads = [
            _guided_source_record_from_structured_source(source, index=index)
            for index, source in enumerate(payload.get("candidateLeads") or [], start=1)
            if isinstance(source, dict)
        ] or _guided_unverified_leads_from_sources(sources)
    verified_findings = _guided_session_findings(payload, sources)
    evidence_gaps = list(payload.get("evidenceGaps") or [])
    coverage = cast(dict[str, Any] | None, payload.get("coverage") or payload.get("coverageSummary"))
    status = _guided_follow_up_status(payload.get("status") or payload.get("answerStatus"))
    failure_summary = _guided_failure_summary(
        failure_summary=cast(dict[str, Any] | None, payload.get("failureSummary")),
        status=status,
        sources=sources,
        evidence_gaps=evidence_gaps,
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
        "trustSummary": _guided_trust_summary(sources, evidence_gaps),
        "coverage": coverage,
        "failureSummary": failure_summary,
        "resultMeaning": payload.get("resultMeaning")
        or _guided_result_meaning(
            status=status,
            verified_findings=verified_findings,
            evidence_gaps=evidence_gaps,
            coverage=coverage,
            failure_summary=failure_summary,
            source_count=len(sources),
        ),
        "nextActions": payload.get("nextActions")
        or _guided_next_actions(
            search_session_id=record.search_session_id,
            status=status,
            has_sources=bool(sources),
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


def _guided_follow_up_answer_mode(question: str, session_strategy_metadata: dict[str, Any]) -> str:
    lowered = question.lower()
    if any(marker in lowered for marker in ("compare", "versus", "vs", "tradeoff", "tradeoffs")):
        return "comparison"
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
        return "qa"
    follow_up_mode = str(session_strategy_metadata.get("followUpMode") or "").strip().lower()
    if follow_up_mode in {"qa", "claim_check", "comparison"}:
        return follow_up_mode
    return "qa"


def _answer_follow_up_from_session_state(
    *,
    question: str,
    session_state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if session_state is None:
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

    return {
        "searchSessionId": session_state["searchSessionId"],
        "answerStatus": "answered",
        "answer": " ".join(answer_parts),
        "evidence": [],
        "selectedEvidenceIds": follow_up_decision.selected_evidence_ids,
        "selectedLeadIds": follow_up_decision.selected_lead_ids,
        "unsupportedAsks": [],
        "followUpQuestions": [],
        "verifiedFindings": session_state["verifiedFindings"],
        "sources": session_state["sources"],
        "unverifiedLeads": session_state["unverifiedLeads"],
        "evidenceGaps": session_state["evidenceGaps"],
        "trustSummary": session_state["trustSummary"],
        "coverage": session_state["coverage"],
        "failureSummary": session_state["failureSummary"],
        "resultMeaning": session_state["resultMeaning"],
        "nextActions": session_state["nextActions"],
        "resultState": _guided_result_state(
            status="answered",
            sources=sources,
            evidence_gaps=evidence_gaps,
            search_session_id=str(session_state.get("searchSessionId") or ""),
        ),
        **_guided_contract_fields(
            query=str(session_state.get("query") or ""),
            intent=str(session_state.get("intent") or "discovery"),
            status="answered",
            sources=sources,
            unverified_leads=cast(list[dict[str, Any]], session_state.get("unverifiedLeads") or []),
            evidence_gaps=evidence_gaps,
            coverage_summary=coverage,
            strategy_metadata=cast(dict[str, Any] | None, session_state.get("strategyMetadata")),
            timeline=cast(dict[str, Any] | None, session_state.get("timeline")),
        ),
        "executionProvenance": _guided_execution_provenance_payload(
            execution_mode="session_introspection",
            answer_source="saved_session_metadata",
            passes_run=0,
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

    async def _dispatch_internal(tool_name: str, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        return await dispatch_tool(
            tool_name,
            tool_arguments,
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
        elif best_match is not None:
            _key_conflict_fields = {"author", "year", "venue"}
            _best_conflicting = list(best_match.get("conflictingFields") or []) if isinstance(best_match, dict) else []
            _key_conflict_count = len(_key_conflict_fields & set(_best_conflicting))
            if resolution_confidence in {"high", "medium"} and _key_conflict_count >= 2:
                status = "needs_disambiguation"
            elif resolution_confidence in {"high", "medium"}:
                status = "resolved"
            else:
                status = "multiple_candidates"
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
            next_actions = [
                "The title matched but author, year, or venue conflicts make this result unsafe to cite directly.",
                "Use research with the title plus author or year to find the correct paper.",
                "Review the conflicting fields in bestMatch before treating this as a confirmed citation.",
            ]
        elif status in {"resolved", "multiple_candidates"}:
            next_actions = [
                "Use research with the resolved title or identifier to gather broader context.",
                "Inspect the resolved metadata before citing or expanding it.",
            ]
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
            "nextActions": next_actions,
            "searchSessionId": raw.get("searchSessionId"),
        }

    if name == "research":
        normalized_research_arguments, research_normalization = _guided_normalize_research_arguments(arguments)
        research_args = cast(ResearchArgs, TOOL_INPUT_MODELS[name].model_validate(normalized_research_arguments))
        intent = "discovery"
        if _guided_is_known_item_query(research_args.query):
            intent = "known_item"
        elif _guided_is_mixed_intent_query(research_args.query, research_args.focus):
            intent = "mixed"
        elif detect_regulatory_intent(research_args.query, research_args.focus):
            intent = "regulatory"

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
            evidence_gaps: list[str] = []
            status = "succeeded" if paper is not None else ("partial" if resolved.get("alternatives") else "abstained")
            trust_summary = _guided_trust_summary(sources, evidence_gaps)
            failure_summary = _guided_failure_summary(
                failure_summary=None,
                status=status,
                sources=sources,
                evidence_gaps=evidence_gaps,
            )
            response = {
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
                ),
                "nextActions": _guided_next_actions(
                    search_session_id=cast(str | None, resolved.get("searchSessionId")),
                    status=status,
                    has_sources=bool(sources),
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
            }
            response.update(
                _guided_contract_fields(
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
            return response

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

            def _summarize_guided_smart_runs() -> dict[str, Any]:
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
                status = _guided_research_status(
                    intent=status_intent,
                    sources=sources,
                    findings=verified_findings,
                    unverified_leads_count=len(unverified_leads),
                    coverage_summary=merged_coverage,
                    failure_summary=merged_failure_summary,
                    clarification=cast(dict[str, Any] | None, clarification),
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
                smart_summary = _summarize_guided_smart_runs()
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
                    smart_summary = _summarize_guided_smart_runs()
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
            failure_summary = _guided_failure_summary(
                failure_summary=merged_failure_summary,
                status=status,
                sources=sources,
                evidence_gaps=evidence_gaps,
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
            trust_summary = _guided_trust_summary(sources, evidence_gaps)
            strategy_metadata = _guided_strategy_metadata_from_runs(smart_runs)
            if review_pass_reason:
                strategy_metadata["reviewPassReason"] = review_pass_reason
            regulatory_timeline = next(
                (
                    smart.get("regulatoryTimeline")
                    for smart in smart_runs
                    if smart.get("regulatoryTimeline") is not None
                ),
                primary_smart.get("regulatoryTimeline"),
            )
            contract_fields = _guided_contract_fields(
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
                ),
                "nextActions": _guided_next_actions(
                    search_session_id=search_session_id,
                    status=status,
                    has_sources=bool(sources),
                ),
                "clarification": smart_summary.get("clarification"),
                "regulatoryTimeline": regulatory_timeline,
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
            return response

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
        status = _guided_research_status(
            intent=intent,
            sources=sources,
            findings=verified_findings,
            unverified_leads_count=len(unverified_leads),
            coverage_summary=cast(dict[str, Any] | None, raw.get("coverageSummary")),
            failure_summary=cast(dict[str, Any] | None, raw.get("failureSummary")),
            clarification=None,
        )
        failure_summary = _guided_failure_summary(
            failure_summary=cast(dict[str, Any] | None, raw.get("failureSummary")),
            status=status,
            sources=sources,
            evidence_gaps=evidence_gaps,
        )
        response = {
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
            ),
            "nextActions": _guided_next_actions(
                search_session_id=cast(str | None, raw.get("searchSessionId")),
                status=status,
                has_sources=bool(sources),
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
        }
        response.update(
            _guided_contract_fields(
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
        return response

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
        session_answer = _answer_follow_up_from_session_state(
            question=follow_up_args.question,
            session_state=session_state,
        )
        if session_answer is not None:
            session_answer["inputNormalization"] = _guided_normalization_payload(follow_up_normalization)
            session_answer["sessionResolution"] = session_resolution
            return session_answer
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
            return response
        if agentic_runtime is None:
            evidence_gaps = [follow_up_args.question]
            trust_summary = _guided_trust_summary([], evidence_gaps)
            failure_summary = _guided_failure_summary(
                failure_summary={
                    "outcome": "total_failure",
                    "whatFailed": "Grounded follow-up requires the smart runtime to be enabled.",
                    "whatStillWorked": "The saved search session can still be inspected source by source.",
                    "fallbackAttempted": False,
                    "fallbackMode": None,
                    "primaryPathFailureReason": "smart_runtime_unavailable",
                    "completenessImpact": "No grounded synthesis was attempted.",
                    "recommendedNextAction": "inspect_source",
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
                ),
                "resultState": _guided_result_state(
                    status="partial",
                    sources=[],
                    evidence_gaps=evidence_gaps,
                    search_session_id=follow_up_args.search_session_id,
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
            return response
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
            return response
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
        answer_status = str(ask.get("answerStatus") or "answered")
        guided_status = "partial" if answer_status == "insufficient_evidence" else answer_status
        verified_findings = _guided_findings_from_sources(sources)
        trust_summary = _guided_trust_summary(sources, evidence_gaps)
        failure_summary = _guided_failure_summary(
            failure_summary=cast(dict[str, Any] | None, ask.get("failureSummary")),
            status=guided_status,
            sources=sources,
            evidence_gaps=evidence_gaps,
        )
        response = {
            "searchSessionId": follow_up_args.search_session_id,
            "answerStatus": answer_status,
            "answer": ask.get("answer"),
            "evidence": ask.get("evidence") or [],
            "unsupportedAsks": ask.get("unsupportedAsks") or [],
            "followUpQuestions": ask.get("followUpQuestions") or [],
            "verifiedFindings": verified_findings,
            "sources": sources,
            "unverifiedLeads": unverified_leads,
            "evidenceGaps": evidence_gaps,
            "trustSummary": trust_summary,
            "coverage": ask.get("coverageSummary"),
            "failureSummary": failure_summary,
            "resultMeaning": _guided_result_meaning(
                status=guided_status,
                verified_findings=verified_findings,
                evidence_gaps=evidence_gaps,
                coverage=cast(dict[str, Any] | None, ask.get("coverageSummary")),
                failure_summary=failure_summary,
                source_count=len(sources),
            ),
            "nextActions": _guided_next_actions(
                search_session_id=follow_up_args.search_session_id,
                status=guided_status,
                has_sources=bool(sources),
            ),
            "resultState": _guided_result_state(
                status=guided_status,
                sources=sources,
                evidence_gaps=evidence_gaps,
                search_session_id=follow_up_args.search_session_id,
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
        }
        response.update(
            _guided_contract_fields(
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
                coverage_summary=cast(dict[str, Any] | None, ask.get("coverageSummary")),
                strategy_metadata=follow_up_strategy_metadata,
            )
        )
        response["selectedEvidenceIds"] = [
            str(item.get("evidenceId") or "").strip()
            for item in response.get("evidence") or []
            if isinstance(item, dict) and str(item.get("evidenceId") or "").strip()
        ]
        response["selectedLeadIds"] = [
            str(item.get("evidenceId") or "").strip()
            for item in response.get("leads") or []
            if isinstance(item, dict) and str(item.get("evidenceId") or "").strip()
        ]
        session_answer = _answer_follow_up_from_session_state(
            question=follow_up_args.question,
            session_state=session_state,
        )
        if answer_status == "answered" and not _guided_is_usable_answer_text(ask.get("answer")):
            if session_answer is not None:
                session_answer["inputNormalization"] = _guided_normalization_payload(follow_up_normalization)
                session_answer["sessionResolution"] = session_resolution
                return session_answer
            response["answerStatus"] = "insufficient_evidence"
            response["answer"] = None
            response["resultMeaning"] = _guided_result_meaning(
                status="partial",
                verified_findings=verified_findings,
                evidence_gaps=evidence_gaps,
                coverage=cast(dict[str, Any] | None, ask.get("coverageSummary")),
                failure_summary=failure_summary,
                source_count=len(sources),
            )
            response["resultState"] = _guided_result_state(
                status="partial",
                sources=sources,
                evidence_gaps=evidence_gaps,
                search_session_id=follow_up_args.search_session_id,
            )
            abstention_details = _guided_abstention_details_payload(
                status="insufficient_evidence",
                sources=sources,
                evidence_gaps=evidence_gaps,
                trust_summary=trust_summary,
            )
            if abstention_details is not None:
                response["abstentionDetails"] = abstention_details
            return response
        if answer_status != "answered" and session_answer is not None:
            session_answer["inputNormalization"] = _guided_normalization_payload(follow_up_normalization)
            session_answer["sessionResolution"] = session_resolution
            return session_answer
        abstention_details = _guided_abstention_details_payload(
            status=answer_status,
            sources=sources,
            evidence_gaps=evidence_gaps,
            trust_summary=trust_summary,
        )
        if abstention_details is not None:
            response["abstentionDetails"] = abstention_details
        return response

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
            if workspace_registry is not None and inspect_args.search_session_id:
                try:
                    record = workspace_registry.get(inspect_args.search_session_id)
                    available_ids = [
                        str(candidate.get("sourceId") or "").strip()
                        for candidate in _guided_record_source_candidates(record)
                        if str(candidate.get("sourceId") or "").strip()
                    ][:8]
                except Exception:
                    available_ids = []
            evidence_gaps = [
                "Could not find sourceId "
                f"{inspect_args.source_id!r} in searchSessionId "
                f"{inspect_args.search_session_id!r}."
            ]
            trust_summary = _guided_trust_summary([], evidence_gaps)
            failure_summary = _guided_failure_summary(
                failure_summary={
                    "outcome": "needs_clarification",
                    "whatFailed": "inspect_source_source_resolution",
                    "whatStillWorked": "The server returned available source IDs for explicit retry.",
                    "fallbackAttempted": False,
                    "fallbackMode": None,
                    "primaryPathFailureReason": match_type,
                    "completenessImpact": evidence_gaps[0],
                    "recommendedNextAction": "inspect_source",
                },
                status="needs_disambiguation",
                sources=[],
                evidence_gaps=evidence_gaps,
            )
            response = {
                "searchSessionId": inspect_args.search_session_id,
                "source": None,
                "directReadRecommendations": [],
                "nextActions": [
                    "Provide an exact sourceId from the saved session.",
                    "Use inspect_source again after choosing one of the available source IDs.",
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
                    search_session_id=inspect_args.search_session_id,
                ),
                "sessionResolution": session_resolution,
                "sourceResolution": _guided_source_resolution_payload(
                    requested_source_id=inspect_args.source_id,
                    resolved_source_id=None,
                    match_type=match_type,
                    available_source_ids=available_ids,
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
        session_sources = (
            [candidate for candidate in inspect_session_state.get("sources") or [] if isinstance(candidate, dict)]
            if isinstance(inspect_session_state, dict)
            else [source]
        )
        session_leads = (
            [
                candidate
                for candidate in inspect_session_state.get("unverifiedLeads") or []
                if isinstance(candidate, dict)
            ]
            if isinstance(inspect_session_state, dict)
            else []
        )
        return {
            "searchSessionId": inspect_args.search_session_id,
            "source": source,
            "evidenceId": source.get("sourceId"),
            "selectedEvidenceIds": [str(source.get("sourceId") or "").strip()],
            "directReadRecommendations": _direct_read_recommendations(source, tool_profile=tool_profile),
            "nextActions": _guided_next_actions(
                search_session_id=inspect_args.search_session_id,
                status="succeeded",
                has_sources=True,
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
            **_guided_contract_fields(
                query=str((inspect_session_state or {}).get("query") or ""),
                intent=str((inspect_session_state or {}).get("intent") or "discovery"),
                status="succeeded",
                sources=session_sources,
                unverified_leads=session_leads,
                evidence_gaps=[],
                coverage_summary=cast(dict[str, Any] | None, (inspect_session_state or {}).get("coverage")),
                strategy_metadata=cast(dict[str, Any] | None, (inspect_session_state or {}).get("strategyMetadata")),
                timeline=cast(dict[str, Any] | None, (inspect_session_state or {}).get("timeline")),
            ),
            "inputNormalization": _guided_normalization_payload(inspect_normalization),
        }

    if name == "search_papers_smart":
        smart_args = cast(
            SmartSearchPapersArgs,
            TOOL_INPUT_MODELS[name].model_validate(arguments),
        )
        if agentic_runtime is None:
            return {
                "error": "FEATURE_NOT_CONFIGURED",
                "message": ("search_papers_smart is not available because the agentic runtime was not initialized."),
                "fallbackTools": [
                    "search_papers",
                    "search_papers_bulk",
                    "search_papers_match",
                ],
            }
        return await agentic_runtime.search_papers_smart(
            query=smart_args.query,
            limit=smart_args.limit,
            search_session_id=smart_args.search_session_id,
            mode=smart_args.mode,
            year=smart_args.year,
            venue=smart_args.venue,
            focus=smart_args.focus,
            latency_profile=smart_args.latency_profile,
            provider_budget=(
                smart_args.provider_budget.model_dump(by_alias=False)
                if smart_args.provider_budget is not None
                else None
            ),
            include_enrichment=smart_args.include_enrichment,
            ctx=ctx,
        )

    if name == "ask_result_set":
        ask_args = cast(
            AskResultSetArgs,
            TOOL_INPUT_MODELS[name].model_validate(arguments),
        )
        if agentic_runtime is None:
            return {
                "error": "FEATURE_NOT_CONFIGURED",
                "message": "ask_result_set requires the agentic runtime to be enabled.",
                "fallbackTools": [
                    "search_papers",
                    "get_paper_details",
                    "get_paper_citations",
                ],
            }
        return await agentic_runtime.ask_result_set(
            search_session_id=ask_args.search_session_id,
            question=ask_args.question,
            top_k=ask_args.top_k,
            answer_mode=ask_args.answer_mode,
            latency_profile=ask_args.latency_profile,
            ctx=ctx,
        )

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
