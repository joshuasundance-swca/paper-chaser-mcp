"""Dispatch helpers for MCP tool routing."""

import re
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any, Callable, cast

from .agentic.planner import detect_regulatory_intent, query_facets, query_terms
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
from .identifiers import resolve_doi_from_paper_payload
from .models import TOOL_INPUT_MODELS, CitationFormatsResponse, RuntimeSummary, dump_jsonable
from .models.common import CitationFormat, ExportLink
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
    }
    smart_provider_order = [
        "openai",
        "azure-openai",
        "anthropic",
        "nvidia",
        "google",
        "mistral",
        "huggingface",
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
    if query_similarity >= 0.25 and title_has_anchor:
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
    for key in ("sourceId", "paperId", "canonicalId", "recommendedExpansionId", "citationText", "canonicalUrl", "url"):
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
) -> str:
    if verified_findings:
        return f"This result contains {len(verified_findings)} verified finding(s) grounded in the returned sources."
    if status == "partial":
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
            return "abstained"
        if primary_sources:
            return "partial" if failure_summary is not None else "succeeded"
        return "needs_disambiguation" if sources else "abstained"
    if len(findings) >= 2:
        return "partial" if failure_summary is not None else "succeeded"
    if sources:
        return "partial"
    return "abstained"


def _guided_summary(intent: str, status: str, findings: list[dict[str, Any]], sources: list[dict[str, Any]]) -> str:
    if findings:
        claims = "; ".join(str(finding["claim"]) for finding in findings[:3])
        return f"{intent.replace('_', ' ').title()} evidence grounded in {len(findings)} verified source(s): {claims}."
    if sources:
        return (
            "The search found some source leads, but the evidence stayed too weak, off-topic, or incomplete "
            "for a grounded summary."
        )
    if status == "needs_disambiguation":
        return "The request needs a more specific anchor before the system can build a grounded result."
    return "No sufficiently trustworthy evidence was found for a grounded result."


def _guided_next_actions(
    *,
    search_session_id: str | None,
    status: str,
) -> list[str]:
    actions: list[str] = []
    if search_session_id:
        actions.append(
            f"Use inspect_source with searchSessionId='{search_session_id}' and one sourceId to inspect evidence."
        )
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


def _find_record_source(
    *,
    workspace_registry: Any,
    search_session_id: str,
    source_id: str,
) -> dict[str, Any] | None:
    if workspace_registry is None:
        return None
    record = workspace_registry.get(search_session_id)
    payload = record.payload if isinstance(record.payload, dict) else {}
    for source in payload.get("sources") or []:
        if isinstance(source, dict) and str(source.get("sourceId") or "").strip() == source_id:
            return source
    for source in payload.get("structuredSources") or []:
        if not isinstance(source, dict):
            continue
        candidate_ids = {
            str(source.get("sourceId") or "").strip(),
            str(source.get("citationText") or "").strip(),
            str(source.get("canonicalUrl") or "").strip(),
        }
        if source_id in candidate_ids:
            return _guided_source_record_from_structured_source(source, index=1)
    for index, paper in enumerate(record.papers, start=1):
        if not isinstance(paper, dict):
            continue
        candidate_ids = {
            str(paper.get("paperId") or "").strip(),
            str(paper.get("sourceId") or "").strip(),
            str(paper.get("canonicalId") or "").strip(),
            str(paper.get("recommendedExpansionId") or "").strip(),
        }
        if source_id in candidate_ids:
            query = str(record.query or payload.get("query") or "")
            return _guided_source_record_from_paper(query, paper, index=index)
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
    search_session_id: str,
) -> dict[str, Any] | None:
    if workspace_registry is None:
        return None
    try:
        record = workspace_registry.get(search_session_id)
    except Exception:
        return None
    payload = record.payload if isinstance(record.payload, dict) else {}
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
        ),
        "nextActions": payload.get("nextActions")
        or _guided_next_actions(
            search_session_id=record.search_session_id,
            status=status,
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
    return facets


def _answer_follow_up_from_session_state(
    *,
    question: str,
    session_state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if session_state is None:
        return None
    facets = _guided_follow_up_introspection_facets(question)
    if not facets:
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
        if sources:
            source_titles = [
                str(source.get("title") or source.get("sourceId") or "").strip()
                for source in sources[:3]
                if str(source.get("title") or source.get("sourceId") or "").strip()
            ]
            if source_titles:
                answer_parts.append("Saved sources included: " + "; ".join(source_titles) + ".")
        else:
            unverified_leads = [lead for lead in session_state.get("unverifiedLeads") or [] if isinstance(lead, dict)]
            lead_titles = [
                str(lead.get("title") or lead.get("sourceId") or "").strip()
                for lead in unverified_leads[:3]
                if str(lead.get("title") or lead.get("sourceId") or "").strip()
            ]
            if lead_titles:
                answer_parts.append("Saved source leads included: " + "; ".join(lead_titles) + ".")
            else:
                answer_parts.append("No saved sources were available for this session.")

    if not answer_parts:
        return None

    return {
        "searchSessionId": session_state["searchSessionId"],
        "answerStatus": "answered",
        "answer": " ".join(answer_parts),
        "evidence": [],
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
            status = (
                "resolved"
                if str(raw.get("resolutionConfidence") or "low") == "high" or not alternatives
                else "multiple_candidates"
            )
        elif alternatives:
            status = "multiple_candidates"
        next_actions: list[str] = []
        if status == "regulatory_primary_source":
            next_actions = [
                "Use research for a full trust-graded regulatory pass.",
                "If the reference points to an exact CFR citation, keep that citation in the next research query.",
                "Inspect returned sources before treating the regulatory text as current and settled.",
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
            "nextActions": next_actions,
            "searchSessionId": raw.get("searchSessionId"),
        }

    if name == "research":
        research_args = cast(ResearchArgs, TOOL_INPUT_MODELS[name].model_validate(arguments))
        intent = "discovery"
        if detect_regulatory_intent(research_args.query, research_args.focus):
            intent = "regulatory"
        elif looks_like_paper_identifier(research_args.query) or looks_like_citation_query(research_args.query):
            intent = "known_item"

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
            failure_summary = _guided_failure_summary(
                failure_summary=None,
                status=status,
                sources=sources,
                evidence_gaps=evidence_gaps,
            )
            return {
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
                "trustSummary": _guided_trust_summary(sources, evidence_gaps),
                "coverage": None,
                "failureSummary": failure_summary,
                "resultMeaning": _guided_result_meaning(
                    status=status,
                    verified_findings=verified_findings,
                    evidence_gaps=evidence_gaps,
                    coverage=None,
                    failure_summary=failure_summary,
                ),
                "nextActions": _guided_next_actions(
                    search_session_id=cast(str | None, resolved.get("searchSessionId")),
                    status=status,
                ),
                "clarification": None,
            }

        if agentic_runtime is not None:
            smart = await _dispatch_internal(
                "search_papers_smart",
                {
                    "query": research_args.query,
                    "limit": research_args.limit,
                    "year": research_args.year,
                    "venue": research_args.venue,
                    "focus": research_args.focus,
                    "latencyProfile": research_args.latency_profile,
                    "providerBudget": {"allowPaidProviders": False},
                },
            )
            sources = [
                _guided_source_record_from_structured_source(source, index=index)
                for index, source in enumerate(smart.get("structuredSources") or [], start=1)
                if isinstance(source, dict)
            ]
            verified_findings = _guided_findings_from_sources(sources)
            evidence_gaps = list(smart.get("evidenceGaps") or [])
            unverified_leads = [
                _guided_source_record_from_structured_source(source, index=index)
                for index, source in enumerate(smart.get("candidateLeads") or [], start=1)
                if isinstance(source, dict)
            ] or _guided_unverified_leads_from_sources(sources)
            status = _guided_research_status(
                intent=str(smart.get("strategyMetadata", {}).get("intent") or intent),
                sources=sources,
                findings=verified_findings,
                coverage_summary=cast(dict[str, Any] | None, smart.get("coverageSummary")),
                failure_summary=cast(dict[str, Any] | None, smart.get("failureSummary")),
                clarification=cast(dict[str, Any] | None, smart.get("clarification")),
            )
            failure_summary = _guided_failure_summary(
                failure_summary=cast(dict[str, Any] | None, smart.get("failureSummary")),
                status=status,
                sources=sources,
                evidence_gaps=evidence_gaps,
            )
            return {
                "intent": str(smart.get("strategyMetadata", {}).get("intent") or intent),
                "status": status,
                "searchSessionId": smart.get("searchSessionId"),
                "summary": _guided_summary(intent, status, verified_findings, sources),
                "verifiedFindings": verified_findings,
                "sources": sources,
                "unverifiedLeads": unverified_leads,
                "evidenceGaps": evidence_gaps,
                "trustSummary": _guided_trust_summary(sources, evidence_gaps),
                "coverage": smart.get("coverageSummary"),
                "failureSummary": failure_summary,
                "resultMeaning": _guided_result_meaning(
                    status=status,
                    verified_findings=verified_findings,
                    evidence_gaps=evidence_gaps,
                    coverage=cast(dict[str, Any] | None, smart.get("coverageSummary")),
                    failure_summary=failure_summary,
                ),
                "nextActions": _guided_next_actions(
                    search_session_id=cast(str | None, smart.get("searchSessionId")),
                    status=status,
                ),
                "clarification": smart.get("clarification"),
                "regulatoryTimeline": smart.get("regulatoryTimeline"),
            }

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
        status = _guided_research_status(
            intent=intent,
            sources=sources,
            findings=verified_findings,
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
        return {
            "intent": intent,
            "status": status,
            "searchSessionId": raw.get("searchSessionId"),
            "summary": _guided_summary(intent, status, verified_findings, sources),
            "verifiedFindings": verified_findings,
            "sources": sources,
            "unverifiedLeads": unverified_leads,
            "evidenceGaps": evidence_gaps,
            "trustSummary": _guided_trust_summary(sources, evidence_gaps),
            "coverage": raw.get("coverageSummary"),
            "failureSummary": failure_summary,
            "resultMeaning": _guided_result_meaning(
                status=status,
                verified_findings=verified_findings,
                evidence_gaps=evidence_gaps,
                coverage=cast(dict[str, Any] | None, raw.get("coverageSummary")),
                failure_summary=failure_summary,
            ),
            "nextActions": _guided_next_actions(
                search_session_id=cast(str | None, raw.get("searchSessionId")),
                status=status,
            ),
            "clarification": None,
        }

    if name == "follow_up_research":
        follow_up_args = cast(FollowUpResearchArgs, TOOL_INPUT_MODELS[name].model_validate(arguments))
        session_state = _guided_session_state(
            workspace_registry=workspace_registry,
            search_session_id=follow_up_args.search_session_id,
        )
        session_answer = _answer_follow_up_from_session_state(
            question=follow_up_args.question,
            session_state=session_state,
        )
        if session_answer is not None:
            return session_answer
        if agentic_runtime is None:
            evidence_gaps = [follow_up_args.question]
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
            return {
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
                "trustSummary": _guided_trust_summary([], evidence_gaps),
                "coverage": None,
                "failureSummary": failure_summary,
                "resultMeaning": _guided_result_meaning(
                    status="partial",
                    verified_findings=[],
                    evidence_gaps=evidence_gaps,
                    coverage=None,
                    failure_summary=failure_summary,
                ),
                "nextActions": _guided_next_actions(
                    search_session_id=follow_up_args.search_session_id,
                    status="partial",
                ),
            }
        ask = await _dispatch_internal(
            "ask_result_set",
            {
                "searchSessionId": follow_up_args.search_session_id,
                "question": follow_up_args.question,
            },
        )
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
            "trustSummary": _guided_trust_summary(sources, evidence_gaps),
            "coverage": ask.get("coverageSummary"),
            "failureSummary": failure_summary,
            "resultMeaning": _guided_result_meaning(
                status=guided_status,
                verified_findings=verified_findings,
                evidence_gaps=evidence_gaps,
                coverage=cast(dict[str, Any] | None, ask.get("coverageSummary")),
                failure_summary=failure_summary,
            ),
            "nextActions": _guided_next_actions(
                search_session_id=follow_up_args.search_session_id,
                status=guided_status,
            ),
        }
        session_answer = _answer_follow_up_from_session_state(
            question=follow_up_args.question,
            session_state=session_state,
        )
        if answer_status != "answered" and session_answer is not None:
            return session_answer
        return response

    if name == "inspect_source":
        inspect_args = cast(InspectSourceArgs, TOOL_INPUT_MODELS[name].model_validate(arguments))
        source = _find_record_source(
            workspace_registry=workspace_registry,
            search_session_id=inspect_args.search_session_id,
            source_id=inspect_args.source_id,
        )
        if source is None:
            raise ValueError(
                f"Could not find sourceId {inspect_args.source_id!r} in searchSessionId "
                f"{inspect_args.search_session_id!r}."
            )
        return {
            "searchSessionId": inspect_args.search_session_id,
            "source": source,
            "directReadRecommendations": _direct_read_recommendations(source, tool_profile=tool_profile),
            "nextActions": _guided_next_actions(
                search_session_id=inspect_args.search_session_id,
                status="succeeded",
            ),
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
