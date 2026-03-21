"""Dispatch helpers for MCP tool routing."""

from typing import Any, Callable, cast

from .citation_repair import looks_like_paper_identifier, resolve_citation
from .clients.serpapi import SerpApiKeyMissingError
from .compat import augment_tool_result, build_clarification
from .models import TOOL_INPUT_MODELS, CitationFormatsResponse, dump_jsonable
from .models.common import CitationFormat, ExportLink
from .models.tools import (
    AskResultSetArgs,
    BasicSearchPapersArgs,
    ExpandResearchGraphArgs,
    GetCitationFormatsArgs,
    MapResearchLandscapeArgs,
    ResolveCitationArgs,
    SearchPapersArgs,
    SearchProvider,
    SmartSearchPapersArgs,
)
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
    if (
        context_hash is not None
        and state.context_hash is not None
        and state.context_hash != context_hash
    ):
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
    if (
        context_hash is not None
        and state.context_hash is not None
        and state.context_hash != context_hash
    ):
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
    "search_papers_arxiv": "arxiv",
}

SMART_TOOLS = {
    "search_papers_smart",
    "ask_result_set",
    "map_research_landscape",
    "expand_research_graph",
}


async def dispatch_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    client: Any,
    core_client: Any,
    openalex_client: Any,
    arxiv_client: Any,
    enable_core: bool,
    enable_semantic_scholar: bool,
    enable_openalex: bool,
    enable_arxiv: bool,
    serpapi_client: Any = None,
    enable_serpapi: bool = False,
    provider_order: list[SearchProvider] | None = None,
    provider_registry: Any = None,
    workspace_registry: Any = None,
    agentic_runtime: Any = None,
    ctx: Any = None,
    allow_elicitation: bool = True,
) -> dict[str, Any]:
    """Dispatch one MCP tool call to the correct backend implementation."""
    if name == "search_papers_smart":
        smart_args = cast(
            SmartSearchPapersArgs,
            TOOL_INPUT_MODELS[name].model_validate(arguments),
        )
        if agentic_runtime is None:
            return {
                "error": "FEATURE_NOT_CONFIGURED",
                "message": (
                    "search_papers_smart is not available because the agentic runtime "
                    "was not initialized."
                ),
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
                "message": (
                    "map_research_landscape requires the agentic runtime to be enabled."
                ),
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
                "message": (
                    "expand_research_graph requires the agentic runtime to be enabled."
                ),
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
        if provider_registry is None:
            return {
                "generatedAt": None,
                "providerOrder": list(provider_order or []),
                "providers": [],
            }
        snapshot = provider_registry.snapshot(
            enabled={
                "semantic_scholar": enable_semantic_scholar,
                "openalex": enable_openalex,
                "core": enable_core,
                "arxiv": enable_arxiv,
                "serpapi_google_scholar": enable_serpapi,
                "openai": agentic_runtime is not None,
            },
            provider_order=[
                *(provider_order or []),
                "openalex",
                "openai",
            ],
        )
        if not args_dict.get("include_recent_outcomes", True):
            for provider in snapshot.get("providers", []):
                if isinstance(provider, dict):
                    provider["recentOutcomes"] = []
        return snapshot

    if name == "search_papers":
        search_args = cast(
            SearchPapersArgs,
            TOOL_INPUT_MODELS[name].model_validate(arguments),
        )
        result = await search_papers_with_fallback(
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
            preferred_provider=search_args.preferred_provider,
            provider_order=search_args.provider_order or provider_order,
            core_client=core_client,
            semantic_client=client,
            arxiv_client=arxiv_client,
            serpapi_client=serpapi_client,
            provider_registry=provider_registry,
        )
        elicited = await _maybe_elicit_and_retry(
            tool_name=name,
            arguments=arguments,
            result=result,
            client=client,
            core_client=core_client,
            openalex_client=openalex_client,
            arxiv_client=arxiv_client,
            serpapi_client=serpapi_client,
            enable_core=enable_core,
            enable_semantic_scholar=enable_semantic_scholar,
            enable_openalex=enable_openalex,
            enable_arxiv=enable_arxiv,
            enable_serpapi=enable_serpapi,
            provider_order=provider_order,
            provider_registry=provider_registry,
            workspace_registry=workspace_registry,
            agentic_runtime=agentic_runtime,
            ctx=ctx,
            allow_elicitation=allow_elicitation,
        )
        if elicited is not None:
            return elicited
        return _finalize_tool_result(
            name,
            arguments,
            result,
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
                "Set SCHOLAR_SEARCH_ENABLE_OPENALEX=true to use this tool."
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
                "Set SCHOLAR_SEARCH_ENABLE_OPENALEX=true to use this tool."
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

    if name == "search_entities_openalex":
        if not enable_openalex:
            raise ValueError(
                "search_entities_openalex requires OpenAlex, which is disabled. "
                "Set SCHOLAR_SEARCH_ENABLE_OPENALEX=true to use this tool."
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
                "Set SCHOLAR_SEARCH_ENABLE_OPENALEX=true to use this tool."
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
                "Set SCHOLAR_SEARCH_ENABLE_OPENALEX=true to use this tool."
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
                "Set SCHOLAR_SEARCH_ENABLE_SERPAPI=true and provide SERPAPI_API_KEY."
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
                "Set SCHOLAR_SEARCH_ENABLE_SERPAPI=true and provide SERPAPI_API_KEY."
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
                "Set SCHOLAR_SEARCH_ENABLE_SERPAPI=true and provide SERPAPI_API_KEY."
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
                "Set SCHOLAR_SEARCH_ENABLE_SERPAPI=true and provide SERPAPI_API_KEY."
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
                "Set SCHOLAR_SEARCH_ENABLE_SERPAPI=true and provide SERPAPI_API_KEY."
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
        result = await search_papers_with_fallback(
            query=provider_arguments.query,
            limit=provider_arguments.limit,
            year=provider_arguments.year,
            fields=getattr(provider_arguments, "fields", None),
            venue=getattr(provider_arguments, "venue", None),
            publication_date_or_year=getattr(
                provider_arguments, "publication_date_or_year", None
            ),
            fields_of_study=getattr(provider_arguments, "fields_of_study", None),
            publication_types=getattr(provider_arguments, "publication_types", None),
            open_access_pdf=getattr(provider_arguments, "open_access_pdf", None),
            min_citation_count=getattr(provider_arguments, "min_citation_count", None),
            enable_core=enable_core,
            enable_semantic_scholar=enable_semantic_scholar,
            enable_arxiv=enable_arxiv,
            enable_serpapi=enable_serpapi,
            provider_order=[PROVIDER_SEARCH_TOOLS[name]],
            core_client=core_client,
            semantic_client=client,
            arxiv_client=arxiv_client,
            serpapi_client=serpapi_client,
            provider_registry=provider_registry,
        )
        return _finalize_tool_result(
            name,
            arguments,
            result,
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
                "Set SCHOLAR_SEARCH_ENABLE_SERPAPI=true and provide SERPAPI_API_KEY "
                "to use this tool. SerpApi is a paid service — see "
                "https://serpapi.com for details."
            )
        if serpapi_client is None:
            raise ValueError(
                "SerpApi client is not available. "
                "Set SCHOLAR_SEARCH_ENABLE_SERPAPI=true and SERPAPI_API_KEY."
            )
        try:
            raw = await serpapi_client.get_citation_formats(
                result_id=validated_cf.result_id,
            )
        except SerpApiKeyMissingError:
            raise
        raw_citations = raw.get("citations") or []
        raw_links = raw.get("links") or []
        response = CitationFormatsResponse(
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
            dump_jsonable(response),
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
            f"{name} requires OpenAlex, which is disabled. "
            "Set SCHOLAR_SEARCH_ENABLE_OPENALEX=true to use this tool."
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
        enable_core=enable_core,
        enable_semantic_scholar=enable_semantic_scholar,
        enable_openalex=enable_openalex,
        enable_arxiv=enable_arxiv,
        enable_serpapi=enable_serpapi,
        provider_order=provider_order,
        provider_registry=provider_registry,
        workspace_registry=workspace_registry,
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
    if tool_name in SMART_TOOLS or tool_name in {
        "get_provider_diagnostics",
        "get_serpapi_account_status",
    } or workspace_registry is None:
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
    enable_core: bool,
    enable_semantic_scholar: bool,
    enable_openalex: bool,
    enable_arxiv: bool,
    enable_serpapi: bool,
    provider_order: list[SearchProvider] | None,
    provider_registry: Any,
    workspace_registry: Any,
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
        resolved_result["matchFound"] = True
        resolved_result["matchStrategy"] = "elicited_identifier"
        resolved_result["normalizedQuery"] = refinement
        return _finalize_tool_result(
            "search_papers_match",
            {"query": refinement, "fields": arguments.get("fields")},
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
        enable_core=enable_core,
        enable_semantic_scholar=enable_semantic_scholar,
        enable_openalex=enable_openalex,
        enable_arxiv=enable_arxiv,
        serpapi_client=serpapi_client,
        enable_serpapi=enable_serpapi,
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
