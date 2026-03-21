"""Compatibility helpers for published schemas and agent-facing tool metadata."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .agentic.models import AgentHints, Clarification
from .agentic.workspace import WorkspaceRegistry
from .citation_repair import looks_like_citation_query, parse_citation

SEARCH_SESSION_TOOLS = {
    "search_papers",
    "search_papers_core",
    "search_papers_semantic_scholar",
    "search_papers_serpapi",
    "search_papers_arxiv",
    "search_papers_openalex",
    "search_papers_openalex_bulk",
    "search_papers_bulk",
    "get_paper_citations",
    "get_paper_citations_openalex",
    "get_paper_references",
    "get_paper_references_openalex",
    "get_paper_authors",
    "search_authors",
    "search_authors_openalex",
    "get_author_papers",
    "get_author_papers_openalex",
    "resolve_citation",
    "search_papers_smart",
}

LOW_SPECIFICITY_TOKENS = {
    "ai",
    "ml",
    "nlp",
    "paper",
    "papers",
    "research",
    "study",
    "studies",
    "model",
    "models",
}


def sanitize_published_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove client-hostile noise from advertised MCP tool schemas."""
    sanitized = _sanitize_schema_node(deepcopy(schema))
    if isinstance(sanitized, dict) and "$defs" in sanitized:
        if not _schema_uses_refs(sanitized):
            sanitized.pop("$defs", None)
    return sanitized


def _sanitize_schema_node(node: Any) -> Any:
    if isinstance(node, list):
        return [_sanitize_schema_node(item) for item in node]
    if not isinstance(node, dict):
        return node

    if "anyOf" in node:
        non_null = [
            option
            for option in node["anyOf"]
            if not (isinstance(option, dict) and option.get("type") == "null")
        ]
        if len(non_null) == 1:
            merged = _sanitize_schema_node(non_null[0])
            if isinstance(merged, dict):
                for key in ("description", "examples", "enum"):
                    if key not in merged and key in node:
                        merged[key] = node[key]
            return merged

    sanitized: dict[str, Any] = {}
    for key, value in node.items():
        if key in {"default", "examples"}:
            continue
        sanitized[key] = _sanitize_schema_node(value)
    return sanitized


def _schema_uses_refs(node: Any) -> bool:
    if isinstance(node, dict):
        if "$ref" in node:
            return True
        return any(_schema_uses_refs(value) for value in node.values())
    if isinstance(node, list):
        return any(_schema_uses_refs(item) for item in node)
    return False


def build_agent_hints(
    tool_name: str,
    result: dict[str, Any],
    arguments: dict[str, Any] | None = None,
) -> AgentHints:
    """Create concise next-step hints tailored to the current tool response."""
    warnings: list[str] = []
    arguments = arguments or {}
    if isinstance(result.get("brokerMetadata"), dict):
        metadata = result["brokerMetadata"]
        if metadata.get("resultQuality") == "low_relevance":
            warnings.append(
                "Results look weak for the full query; narrow or rephrase "
                "before trusting them."
            )
        if metadata.get("resultStatus") == "provider_failed":
            warnings.append(
                "Providers failed upstream; retry later before concluding "
                "the topic has no results."
            )
    if result.get("retrievalNote"):
        warnings.append(str(result["retrievalNote"]))

    if tool_name in {
        "search_papers",
        "search_papers_core",
        "search_papers_semantic_scholar",
        "search_papers_serpapi",
        "search_papers_arxiv",
        "search_papers_openalex",
        "search_papers_openalex_bulk",
        "search_papers_bulk",
    }:
        return AgentHints(
            nextToolCandidates=[
                "search_papers_smart",
                "get_paper_details",
                "get_paper_citations",
                "search_authors",
            ],
            whyThisNextStep=(
                "Start with one promising paper, then either inspect it directly, "
                "expand citations, or switch to the smart workflow for "
                "concept-level synthesis."
            ),
            safeRetry=(
                "Retry the same search with a tighter query, year/venue hint, or "
                "preferred provider if the first page is noisy."
            ),
            warnings=warnings,
        )
    if tool_name == "resolve_citation":
        best_match = result.get("bestMatch")
        confidence = str(result.get("resolutionConfidence") or "low")
        alternatives = result.get("alternatives") or []
        if isinstance(best_match, dict) and isinstance(best_match.get("paper"), dict):
            return AgentHints(
                nextToolCandidates=[
                    "get_paper_details",
                    "get_paper_citations",
                    "search_papers_smart",
                    "expand_research_graph",
                ],
                whyThisNextStep=(
                    "You now have a repaired paper anchor, so the highest-value "
                    "next move is to inspect it, expand citations, or continue "
                    "with grounded smart follow-up."
                ),
                safeRetry=(
                    "If confidence is not high, compare the alternatives and add one "
                    "missing clue such as the year, author surname, or venue."
                ),
                warnings=warnings
                + (
                    []
                    if confidence == "high"
                    else [
                        "The citation is only partially resolved; review the "
                        "alternatives before treating it as canonical."
                    ]
                ),
            )
        return AgentHints(
            nextToolCandidates=[
                "search_snippets",
                "search_papers",
                "search_authors",
                "search_papers_smart",
            ],
            whyThisNextStep=(
                "The citation is still ambiguous, so the best next move is to add "
                "one stronger clue or broaden into discovery."
            ),
            safeRetry=(
                "Retry with one concrete field such as a DOI fragment, year, author "
                "surname, or quoted phrase."
            ),
            warnings=warnings
            + (
                []
                if alternatives
                else [
                    "No convincing paper candidate was found from the current "
                    "citation text."
                ]
            ),
        )
    if tool_name in {"search_papers_match", "get_paper_details"}:
        next_tools = [
            "get_paper_citations",
            "get_paper_references",
            "get_paper_authors",
            "expand_research_graph",
        ]
        if (
            tool_name == "search_papers_match"
            and result.get("matchFound") is False
            and looks_like_citation_query(str(arguments.get("query") or ""))
        ):
            next_tools.insert(0, "resolve_citation")
        return AgentHints(
            nextToolCandidates=next_tools,
            whyThisNextStep=(
                "You already have a concrete paper anchor, so the highest-value "
                "next move is citation chasing or graph expansion."
            ),
            safeRetry=(
                "If this paper ID is not portable, retry via DOI or a "
                "Semantic Scholar-native lookup."
            ),
            warnings=warnings,
        )
    if tool_name in {
        "get_paper_citations",
        "get_paper_citations_openalex",
        "get_paper_references",
        "get_paper_references_openalex",
        "expand_research_graph",
    }:
        return AgentHints(
            nextToolCandidates=[
                "ask_result_set",
                "map_research_landscape",
                "get_paper_details",
                "search_authors",
            ],
            whyThisNextStep=(
                "You now have a reusable result set; ask grounded follow-up questions "
                "or cluster the frontier before expanding again."
            ),
            safeRetry=(
                "Reuse the same searchSessionId or pagination cursor exactly "
                "as returned."
            ),
            warnings=warnings,
        )
    if tool_name in {"search_authors", "search_authors_openalex"}:
        return AgentHints(
            nextToolCandidates=["get_author_info", "get_author_papers"],
            whyThisNextStep=(
                "Confirm the right person first, then pivot into that author's papers."
            ),
            safeRetry=(
                "Add affiliation, coauthor, venue, or topic clues if the "
                "name is ambiguous."
            ),
            warnings=warnings,
        )
    if tool_name in {"get_author_papers", "get_author_papers_openalex"}:
        return AgentHints(
            nextToolCandidates=[
                "get_paper_details",
                "search_papers_smart",
                "map_research_landscape",
            ],
            whyThisNextStep=(
                "Author paper lists work best as a launch point for "
                "paper-level inspection or theme mapping."
            ),
            safeRetry=(
                "Narrow with publicationDateOrYear or year if the author's "
                "output is too broad."
            ),
            warnings=warnings,
        )
    if tool_name == "ask_result_set":
        return AgentHints(
            nextToolCandidates=["map_research_landscape", "expand_research_graph"],
            whyThisNextStep=(
                "Follow the answer by either clustering the same corpus or "
                "expanding from the cited anchors."
            ),
            safeRetry=(
                "Ask a narrower question or increase topK if the evidence "
                "set feels thin."
            ),
            warnings=warnings,
        )
    if tool_name == "map_research_landscape":
        return AgentHints(
            nextToolCandidates=["ask_result_set", "expand_research_graph"],
            whyThisNextStep=(
                "Themes are most useful when followed by a pointed question "
                "or a citation expansion."
            ),
            safeRetry=(
                "Reduce maxThemes or refine the discovery query if "
                "everything collapses into one bucket."
            ),
            warnings=warnings,
        )
    if tool_name == "search_snippets":
        next_candidates = [
            "resolve_citation",
            "search_papers_match",
            "search_papers",
        ]
        if not looks_like_citation_query(str(arguments.get("query") or "")):
            next_candidates = [
                "search_papers_match",
                "search_papers",
                "resolve_citation",
            ]
        return AgentHints(
            nextToolCandidates=next_candidates,
            whyThisNextStep=(
                "Snippets are best used as recovery clues; once you have a likely "
                "paper, move back to citation repair, title matching, or broader "
                "paper discovery."
            ),
            safeRetry=(
                "Retry with a longer quote fragment or shift to resolve_citation "
                "when the input looks like a broken reference instead of a "
                "verbatim quote."
            ),
            warnings=warnings,
        )
    return AgentHints(
        nextToolCandidates=[],
        whyThisNextStep=(
            "Use the returned IDs, resourceUris, or searchSessionId to keep "
            "the workflow moving."
        ),
        safeRetry=(
            "Retry the exact call if the upstream provider had a transient failure."
        ),
        warnings=warnings,
    )


def build_clarification(
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
) -> Clarification | None:
    """Return a bounded clarification cue when the request is ambiguous."""
    if tool_name == "search_papers":
        query = str(arguments.get("query") or "").strip()
        tokens = [token.lower() for token in query.split() if token.strip()]
        if len(tokens) <= 2 and all(
            len(token) <= 4 or token in LOW_SPECIFICITY_TOKENS for token in tokens
        ):
            return Clarification(
                reason="low_specificity_query",
                question=(
                    "This topic is broad. Do you want method papers, a domain-specific "
                    "application, or only recent work?"
                ),
                options=["method focus", "application focus", "recent work only"],
                canProceedWithoutAnswer=True,
            )
    if tool_name == "search_papers_match" and result.get("matchFound") is False:
        parsed = parse_citation(str(arguments.get("query") or ""))
        if parsed.identifier:
            return Clarification(
                reason="identifier_available",
                question=(
                    "This looks more like an identifier-based lookup than a clean "
                    "title-only match. Use the DOI, arXiv ID, or URL directly."
                ),
                options=[
                    "use get_paper_details",
                    "use resolve_citation",
                    "broaden to search_papers",
                ],
                canProceedWithoutAnswer=True,
            )
        if parsed.looks_like_non_paper:
            return Clarification(
                reason="likely_non_paper_output",
                question=(
                    "This may refer to a report, thesis, dataset, or software package "
                    "rather than an indexed paper. Do you want broader discovery "
                    "instead?"
                ),
                options=["search_papers", "search_authors", "external verification"],
                canProceedWithoutAnswer=True,
            )
        if parsed.quoted_fragments and not parsed.title_candidates:
            return Clarification(
                reason="quote_fragment_only",
                question=(
                    "This looks like a quote fragment rather than a stable title. "
                    "Would you like quote recovery or broader citation repair?"
                ),
                options=[
                    "use search_snippets",
                    "use resolve_citation",
                    "broaden to search_papers",
                ],
                canProceedWithoutAnswer=True,
            )
        if parsed.year is None:
            return Clarification(
                reason="missing_year",
                question=(
                    "This citation is still ambiguous. Adding the approximate year "
                    "would usually disambiguate it fastest."
                ),
                options=[
                    "add year",
                    "use resolve_citation",
                    "broaden to search_papers",
                ],
                canProceedWithoutAnswer=True,
            )
        if not parsed.author_surnames:
            return Clarification(
                reason="missing_author",
                question=(
                    "This title-only lookup is close but still ambiguous. Adding one "
                    "author surname would usually disambiguate it fastest."
                ),
                options=[
                    "add author",
                    "use resolve_citation",
                    "broaden to search_papers",
                ],
                canProceedWithoutAnswer=True,
            )
        return Clarification(
            reason="near_tied_or_missing_title_match",
            question=(
                "If you know the DOI, arXiv ID, or URL, use that instead of "
                "a title-only match."
            ),
            options=["use DOI", "use arXiv ID", "broaden to search_papers"],
            canProceedWithoutAnswer=True,
        )
    if tool_name == "resolve_citation":
        parsed = parse_citation(
            str(arguments.get("citation") or ""),
            title_hint=arguments.get("titleHint"),
            author_hint=arguments.get("authorHint"),
            year_hint=arguments.get("yearHint"),
            venue_hint=arguments.get("venueHint"),
            doi_hint=arguments.get("doiHint"),
        )
        if parsed.looks_like_non_paper and not result.get("bestMatch"):
            return Clarification(
                reason="likely_non_paper_output",
                question=(
                    "This may be a report, dataset, dissertation, or software package "
                    "rather than a paper. Do you want broader discovery instead?"
                ),
                options=["search_papers", "search_authors", "external verification"],
                canProceedWithoutAnswer=True,
            )
        if str(result.get("resolutionConfidence") or "low") == "high":
            return None
        if parsed.year is None:
            return Clarification(
                reason="missing_year",
                question=(
                    "This citation is still ambiguous. Adding the approximate year "
                    "would usually disambiguate it fastest."
                ),
                options=["add year", "add author", "broaden to search_papers"],
                canProceedWithoutAnswer=True,
            )
        if not parsed.author_surnames:
            return Clarification(
                reason="missing_author",
                question=(
                    "This citation is still ambiguous. Adding one author surname "
                    "would usually disambiguate it fastest."
                ),
                options=["add author", "add venue", "broaden to search_papers"],
                canProceedWithoutAnswer=True,
            )
        if parsed.quoted_fragments and not result.get("bestMatch"):
            return Clarification(
                reason="quote_fragment_only",
                question=(
                    "This looks closer to a quote fragment than a stable citation. "
                    "Would you like quote recovery or broader discovery?"
                ),
                options=["use search_snippets", "search_papers", "search_authors"],
                canProceedWithoutAnswer=True,
            )
        return Clarification(
            reason="ambiguous_citation",
            question=(
                "Several citation cues are plausible, but the paper is still not "
                "fully confirmed. Which field can you tighten next?"
            ),
            options=["year", "author", "venue"],
            canProceedWithoutAnswer=True,
        )
    if tool_name == "search_authors":
        data = result.get("data") or []
        query = str(arguments.get("query") or "").strip()
        if len(data) > 1 and len(query.split()) <= 3:
            return Clarification(
                reason="ambiguous_author_identity",
                question=(
                    "Several authors may match this name. Can you add an affiliation, "
                    "coauthor, venue, or topic clue?"
                ),
                options=["affiliation", "coauthor", "topic clue"],
                canProceedWithoutAnswer=True,
            )
    return None


def build_resource_uris(
    tool_name: str,
    result: dict[str, Any],
    search_session_id: str | None,
) -> list[str]:
    """Surface follow-on resource URIs from tool outputs."""
    uris: list[str] = []
    if search_session_id:
        uris.append(f"search://{search_session_id}")

    if "paperId" in result and result.get("paperId"):
        paper_id = str(result["paperId"])
        uris.append(f"paper://{paper_id}")
        uris.append(f"trail://paper/{paper_id}?direction=citations")
        uris.append(f"trail://paper/{paper_id}?direction=references")
    if "authorId" in result and result.get("authorId"):
        uris.append(f"author://{result['authorId']}")

    for item in (result.get("data") or [])[:3]:
        if not isinstance(item, dict):
            continue
        if item.get("paperId"):
            uris.append(f"paper://{item['paperId']}")
        if item.get("authorId"):
            uris.append(f"author://{item['authorId']}")

    for item in (result.get("results") or [])[:3]:
        if isinstance(item, dict) and isinstance(item.get("paper"), dict):
            paper = item["paper"]
            if paper.get("paperId"):
                uris.append(f"paper://{paper['paperId']}")
    for item in [result.get("bestMatch"), *(result.get("alternatives") or [])][:3]:
        if isinstance(item, dict) and isinstance(item.get("paper"), dict):
            paper = item["paper"]
            if paper.get("paperId"):
                paper_id = str(paper["paperId"])
                uris.append(f"paper://{paper_id}")
                uris.append(f"trail://paper/{paper_id}?direction=citations")
                uris.append(f"trail://paper/{paper_id}?direction=references")

    deduped: list[str] = []
    seen: set[str] = set()
    for uri in uris:
        if uri not in seen:
            seen.add(uri)
            deduped.append(uri)
    return deduped


def augment_tool_result(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
    workspace_registry: WorkspaceRegistry,
) -> dict[str, Any]:
    """Add agent-facing metadata without changing the core result contract."""
    response = dict(result)
    search_session_id = response.get("searchSessionId")

    if tool_name in SEARCH_SESSION_TOOLS and not search_session_id:
        session_metadata = _result_set_metadata(tool_name, arguments, response)
        record = workspace_registry.save_result_set(
            source_tool=tool_name,
            payload=response,
            query=_query_hint(tool_name, arguments),
            metadata=session_metadata,
        )
        search_session_id = record.search_session_id
        response["searchSessionId"] = search_session_id

    response["agentHints"] = build_agent_hints(
        tool_name,
        response,
        arguments,
    ).model_dump(
        by_alias=True,
        exclude_none=True,
    )
    clarification = build_clarification(tool_name, arguments, response)
    if clarification is not None:
        response["clarification"] = clarification.model_dump(
            by_alias=True,
            exclude_none=True,
        )
    response["resourceUris"] = build_resource_uris(
        tool_name, response, search_session_id
    )
    return response


def _query_hint(tool_name: str, arguments: dict[str, Any]) -> str | None:
    if tool_name == "resolve_citation":
        return arguments.get("citation")
    if tool_name.startswith("search_"):
        return arguments.get("query")
    return None


def _result_set_metadata(
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"tool": tool_name}
    if tool_name in {
        "get_paper_citations",
        "get_paper_citations_openalex",
        "get_paper_references",
        "get_paper_references_openalex",
    }:
        metadata["trailParentPaperId"] = arguments.get("paper_id")
        metadata["trailDirection"] = (
            "citations" if "citations" in tool_name else "references"
        )
    if tool_name == "search_papers_smart" and "strategyMetadata" in result:
        metadata["strategyMetadata"] = result["strategyMetadata"]
    return metadata
