"""Citation resolution orchestration (public + provider-layered helpers).

Phase 9a extracted this module from ``paper_chaser_mcp/citation_repair/_core.py``.
It owns the async :func:`resolve_citation` entry point, its per-strategy helpers
(`_resolve_identifier_candidate`, `_resolve_title_candidates`,
`_resolve_snippet_candidates`, `_resolve_sparse_metadata_candidates`),
response serialization, candidate filtering / abstention, and the
`_build_famous_citation_candidate` bridge that turns a famous-paper registry
hit into a :class:`RankedCitationCandidate`.
"""

from __future__ import annotations

import logging
from typing import Any

from ..models import (
    CitationResolutionCandidate,
    CitationResolutionResponse,
    Paper,
    dump_jsonable,
)
from ..models.tools import DEFAULT_SEARCH_PROVIDER_ORDER
from .candidates import (
    ParsedCitation,
    _classify_resolution_confidence,
    _lookup_famous_citation,
    _sparse_search_queries,
    classify_known_item_resolution_state,
    parse_citation,
)
from .normalization import (
    _dedupe_strings,
    _normalize_identifier_for_openalex,
    _normalize_identifier_for_semantic_scholar,
    normalize_citation_text,
)
from .ranking import (
    RankedCitationCandidate,
    _publication_preference_score,
    _rank_candidate,
)

logger = logging.getLogger("paper-chaser-mcp.citation-repair")


def _build_famous_citation_candidate(
    parsed: ParsedCitation,
    paper: dict[str, Any],
) -> RankedCitationCandidate:
    """Wrap a famous-paper registry hit in a ``RankedCitationCandidate``."""
    matched_fields = ["authors", "year"]
    if parsed.venue_hints and any(hint.lower() in str(paper.get("venue") or "").lower() for hint in parsed.venue_hints):
        matched_fields.append("venue")
    author_overlap = sum(
        1
        for surname in parsed.author_surnames
        if any(surname.lower() in str(author.get("name") or "").lower() for author in paper.get("authors") or [])
    )
    return RankedCitationCandidate(
        paper=paper,
        score=0.97,
        resolution_strategy="identifier",
        matched_fields=matched_fields,
        conflicting_fields=[],
        title_similarity=1.0,
        year_delta=0,
        author_overlap=author_overlap,
        candidate_count=1,
        why_selected=("Matched canonical entry in famous-paper registry via authors + year."),
    )


async def resolve_citation(
    *,
    citation: str,
    max_candidates: int,
    client: Any,
    enable_core: bool,
    enable_semantic_scholar: bool,
    enable_openalex: bool,
    enable_arxiv: bool,
    enable_serpapi: bool,
    core_client: Any,
    openalex_client: Any,
    arxiv_client: Any,
    serpapi_client: Any,
    title_hint: str | None = None,
    author_hint: str | None = None,
    year_hint: str | None = None,
    venue_hint: str | None = None,
    doi_hint: str | None = None,
    include_enrichment: bool = False,
    enrichment_service: Any = None,
) -> dict[str, Any]:
    """Resolve an incomplete or almost-right citation to canonical papers."""
    parsed = parse_citation(
        citation,
        title_hint=title_hint,
        author_hint=author_hint,
        year_hint=year_hint,
        venue_hint=venue_hint,
        doi_hint=doi_hint,
    )
    candidate_map: dict[str, RankedCitationCandidate] = {}
    stage_order: list[str] = []

    if parsed.looks_like_regulatory and not parsed.identifier:
        return _serialize_citation_response(
            citation=citation,
            parsed=parsed,
            candidates=[],
        )

    famous_paper = _lookup_famous_citation(parsed)
    if famous_paper is not None:
        famous_candidate = _build_famous_citation_candidate(parsed, famous_paper)
        response = _serialize_citation_response(
            citation=citation,
            parsed=parsed,
            candidates=[famous_candidate],
        )
        response["resolutionStrategy"] = famous_candidate.resolution_strategy
        if include_enrichment and enrichment_service is not None:
            response = await _enrich_best_match(
                response,
                enrichment_service=enrichment_service,
                detail_client=client,
            )
        return response

    if parsed.identifier:
        identifier_candidate = await _resolve_identifier_candidate(
            parsed=parsed,
            client=client,
            openalex_client=openalex_client,
            enable_openalex=enable_openalex,
        )
        if identifier_candidate is not None:
            candidate_map[_paper_identity_key(identifier_candidate.paper)] = identifier_candidate
            stage_order.append("identifier")
            if identifier_candidate.score >= 0.95:
                response = _serialize_citation_response(
                    citation=citation,
                    parsed=parsed,
                    candidates=[identifier_candidate],
                )
                if include_enrichment and enrichment_service is not None:
                    response = await _enrich_best_match(
                        response,
                        enrichment_service=enrichment_service,
                        detail_client=client,
                    )
                return response

    title_candidates = parsed.title_candidates[:3]
    if title_candidates:
        matched_candidates = await _resolve_title_candidates(
            parsed=parsed,
            title_candidates=title_candidates,
            client=client,
        )
        for candidate in matched_candidates:
            _merge_ranked_candidate(candidate_map, candidate)
        if matched_candidates:
            stage_order.append("title_match")
            ranked_candidates = _ranked_candidates(
                candidate_map,
                max_candidates=max_candidates,
            )
            response = _serialize_citation_response(
                citation=citation,
                parsed=parsed,
                candidates=ranked_candidates,
            )
            if (
                _best_candidate_confidence(ranked_candidates) == "high"
                and response.get("knownItemResolutionState") != "needs_disambiguation"
            ):
                if include_enrichment and enrichment_service is not None:
                    response = await _enrich_best_match(
                        response,
                        enrichment_service=enrichment_service,
                        detail_client=client,
                    )
                response["resolutionStrategy"] = ranked_candidates[0].resolution_strategy
                return response

    snippet_candidates = await _resolve_snippet_candidates(
        parsed=parsed,
        max_candidates=max_candidates,
        client=client,
    )
    for candidate in snippet_candidates:
        _merge_ranked_candidate(candidate_map, candidate)
    if snippet_candidates:
        stage_order.append("snippet_recovery")

    sparse_candidates = await _resolve_sparse_metadata_candidates(
        parsed=parsed,
        max_candidates=max_candidates,
        client=client,
        enable_core=enable_core,
        enable_semantic_scholar=enable_semantic_scholar,
        enable_arxiv=enable_arxiv,
        enable_serpapi=enable_serpapi,
        core_client=core_client,
        arxiv_client=arxiv_client,
        serpapi_client=serpapi_client,
        enable_openalex=enable_openalex,
        openalex_client=openalex_client,
    )
    for candidate in sparse_candidates:
        _merge_ranked_candidate(candidate_map, candidate)
    if sparse_candidates:
        stage_order.append("sparse_metadata")

    ranked_candidates = _ranked_candidates(
        candidate_map,
        max_candidates=max_candidates,
    )
    response = _serialize_citation_response(
        citation=citation,
        parsed=parsed,
        candidates=ranked_candidates,
    )
    if include_enrichment and enrichment_service is not None:
        response = await _enrich_best_match(
            response,
            enrichment_service=enrichment_service,
            detail_client=client,
        )
    if stage_order:
        response["resolutionStrategy"] = (
            ranked_candidates[0].resolution_strategy if ranked_candidates else stage_order[-1]
        )
    return response


async def _enrich_best_match(
    response: dict[str, Any],
    *,
    enrichment_service: Any,
    detail_client: Any | None,
) -> dict[str, Any]:
    from ..enrichment import (
        attach_enrichments_to_paper_payload,
        hydrate_paper_for_enrichment,
    )

    best_match = response.get("bestMatch")
    if not isinstance(best_match, dict) or not isinstance(
        best_match.get("paper"),
        dict,
    ):
        return response
    hydrated_paper = await hydrate_paper_for_enrichment(
        best_match["paper"],
        detail_client=detail_client,
    )
    enriched_paper = await enrichment_service.enrich_paper_payload(
        hydrated_paper,
        query=(hydrated_paper.get("title") or best_match["paper"].get("title") or response.get("normalizedCitation")),
    )
    updated_response = dict(response)
    updated_best_match = dict(best_match)
    updated_best_match["paper"] = attach_enrichments_to_paper_payload(
        best_match["paper"],
        enriched_paper=enriched_paper,
    )
    updated_response["bestMatch"] = updated_best_match
    return updated_response


def _serialize_citation_response(
    *,
    citation: str,
    parsed: ParsedCitation,
    candidates: list[RankedCitationCandidate],
) -> dict[str, Any]:
    best = candidates[0] if candidates else None
    runner_up_score = candidates[1].score if len(candidates) > 1 else None
    confidence = _classify_resolution_confidence(
        best_score=best.score if best is not None else None,
        runner_up_score=runner_up_score,
        matched_fields=best.matched_fields if best is not None else [],
        conflicting_fields=best.conflicting_fields if best is not None else [],
        resolution_strategy=best.resolution_strategy if best is not None else "none",
        title_similarity=best.title_similarity if best is not None else None,
    )
    resolution_state = classify_known_item_resolution_state(
        resolution_confidence=confidence,
        resolution_strategy=best.resolution_strategy if best is not None else "none",
        matched_fields=best.matched_fields if best is not None else [],
        conflicting_fields=best.conflicting_fields if best is not None else [],
        title_similarity=best.title_similarity if best is not None else None,
        year_delta=best.year_delta if best is not None else None,
        author_overlap=best.author_overlap if best is not None else None,
        best_score=best.score if best is not None else None,
        runner_up_score=runner_up_score,
        candidate_count=len(candidates),
        has_best_match=best is not None,
    )
    if resolution_state == "needs_disambiguation" and confidence == "high":
        confidence = "medium"
    abstain = bool(
        best is not None
        and (
            confidence == "low"
            or (
                parsed.looks_like_non_paper
                and best.resolution_strategy
                not in {
                    "identifier",
                    "identifier_openalex",
                    "exact_title",
                    "openalex_exact_title",
                    "crossref_exact_title",
                }
                and "identifier" not in best.matched_fields
            )
        )
    )
    if abstain:
        alternatives = _abstention_candidates(candidates)
    else:
        alternative_confidence = "medium" if resolution_state == "needs_disambiguation" else confidence
        alternatives = _filtered_alternative_candidates(
            candidates=candidates,
            confidence=alternative_confidence,
        )
    best_for_response = None if abstain else best
    serializable_candidates = ([best] if best is not None else []) if not abstain else alternatives
    if not abstain and best is not None:
        serializable_candidates = [best, *alternatives]
    response = CitationResolutionResponse(
        bestMatch=_to_candidate_model(best_for_response) if best_for_response is not None else None,
        alternatives=[_to_candidate_model(candidate) for candidate in alternatives],
        resolutionConfidence=confidence,
        resolutionStrategy=best.resolution_strategy if best is not None else "none",
        matchedFields=best.matched_fields if best is not None else [],
        conflictingFields=best.conflicting_fields if best is not None else [],
        normalizedCitation=parsed.normalized_text or normalize_citation_text(citation),
        extractedFields=_parsed_fields_payload(parsed),
        inferredFields=_inferred_fields_payload(parsed, serializable_candidates),
        candidateCount=len(serializable_candidates),
        knownItemResolutionState=resolution_state,
        message=_resolution_message(
            parsed=parsed,
            candidates=serializable_candidates,
            confidence=confidence,
        ),
    )
    return dump_jsonable(response)


def _abstention_candidates(
    candidates: list[RankedCitationCandidate],
) -> list[RankedCitationCandidate]:
    abstained: list[RankedCitationCandidate] = []
    for candidate in candidates:
        if candidate.score < 0.4:
            continue
        if (
            candidate.title_similarity < 0.5
            and candidate.author_overlap == 0
            and "identifier" not in candidate.matched_fields
        ):
            continue
        abstained.append(candidate)
    if abstained:
        return abstained[:3]
    for candidate in candidates:
        if candidate.score < 0.25:
            continue
        if (
            candidate.title_similarity < 0.35
            and candidate.author_overlap == 0
            and "identifier" not in candidate.matched_fields
            and "year" not in candidate.matched_fields
        ):
            continue
        abstained.append(candidate)
    return abstained[:3]


def _filtered_alternative_candidates(
    *,
    candidates: list[RankedCitationCandidate],
    confidence: str,
) -> list[RankedCitationCandidate]:
    if len(candidates) <= 1:
        return []
    best = candidates[0]
    filtered: list[RankedCitationCandidate] = []
    for candidate in candidates[1:]:
        if candidate.score < 0.4:
            continue
        if confidence == "high":
            if candidate.score < max(0.5, best.score - 0.18):
                continue
            if (
                "identifier" not in candidate.matched_fields
                and "title" not in candidate.matched_fields
                and candidate.author_overlap == 0
            ):
                continue
            if candidate.title_similarity < 0.72 and candidate.author_overlap == 0:
                continue
            if (
                "title" in candidate.conflicting_fields
                and candidate.title_similarity < 0.78
                and "identifier" not in candidate.matched_fields
            ):
                continue
        else:
            if (
                candidate.title_similarity < 0.55
                and candidate.author_overlap == 0
                and "identifier" not in candidate.matched_fields
            ):
                continue
            if (
                candidate.author_overlap == 0
                and "author" in candidate.conflicting_fields
                and candidate.year_delta not in {0, 1}
                and "identifier" not in candidate.matched_fields
            ):
                continue
        filtered.append(candidate)
    return filtered


def _to_candidate_model(
    candidate: RankedCitationCandidate,
) -> CitationResolutionCandidate:
    return CitationResolutionCandidate(
        paper=Paper.model_validate(candidate.paper),
        score=round(candidate.score, 6),
        resolutionStrategy=candidate.resolution_strategy,
        matchedFields=candidate.matched_fields,
        conflictingFields=candidate.conflicting_fields,
        titleSimilarity=round(candidate.title_similarity, 6),
        yearDelta=candidate.year_delta,
        authorOverlap=candidate.author_overlap,
        candidateCount=candidate.candidate_count,
        whySelected=candidate.why_selected,
    )


def _parsed_fields_payload(parsed: ParsedCitation) -> dict[str, Any]:
    return {
        "identifier": parsed.identifier,
        "identifierType": parsed.identifier_type,
        "candidateYear": parsed.year,
        "quotedFragments": parsed.quoted_fragments,
        "titleCandidates": parsed.title_candidates,
        "authorSurnames": parsed.author_surnames,
        "venueHints": parsed.venue_hints,
        "volume": parsed.volume,
        "issue": parsed.issue,
        "pages": parsed.pages,
        "looksLikeNonPaper": parsed.looks_like_non_paper,
        "looksLikeRegulatory": parsed.looks_like_regulatory,
    }


def _inferred_fields_payload(
    parsed: ParsedCitation,
    candidates: list[RankedCitationCandidate],
) -> dict[str, Any]:
    if parsed.looks_like_regulatory:
        likely_output_type = "regulatory_primary_source"
    elif parsed.looks_like_non_paper:
        likely_output_type = "non_paper_candidate"
    else:
        likely_output_type = "paper"
    disambiguation_fields: list[str] = []
    if candidates:
        best = candidates[0]
        if "year" in best.conflicting_fields:
            disambiguation_fields.append("year")
        if "author" in best.conflicting_fields:
            disambiguation_fields.append("author")
        if "venue" in best.conflicting_fields:
            disambiguation_fields.append("venue")
    return {
        "likelyOutputType": likely_output_type,
        "needsDisambiguationBy": disambiguation_fields,
        "primaryTitleCandidate": (parsed.title_candidates[0] if parsed.title_candidates else None),
    }


def _resolution_message(
    *,
    parsed: ParsedCitation,
    candidates: list[RankedCitationCandidate],
    confidence: str,
) -> str:
    if not candidates:
        if parsed.looks_like_regulatory:
            return (
                "This input looks like a Federal Register or CFR citation rather than a paper. "
                "Use search_federal_register for discovery, get_federal_register_document for one notice or rule, "
                "or get_cfr_text for codified text."
            )
        if parsed.looks_like_non_paper:
            return (
                "No confident paper match was found. This input may refer to a "
                "dataset, report, dissertation, software package, or another "
                "non-paper output."
            )
        return (
            "No confident paper match was found. Try adding an author surname, "
            "year, DOI fragment, quote fragment, or venue hint."
        )
    best = candidates[0]
    if parsed.looks_like_regulatory and confidence != "high":
        return (
            "This input looks regulatory rather than paper-like. Prefer the Federal Register or CFR tools over "
            "forcing a scholarly citation match."
        )
    if parsed.looks_like_non_paper and confidence != "high":
        return (
            "This citation looks report-like or otherwise outside the indexed paper surface. "
            "The server is returning low-confidence alternatives instead of forcing a paper match."
        )
    if confidence == "high":
        return (
            "Resolved the citation with strong field agreement. Use the returned "
            "paper as an anchor for details, citations, or grounded follow-up."
        )
    if confidence == "medium":
        return (
            "One candidate leads, but some fields disagree or the citation is "
            "still somewhat incomplete. Review the alternatives before citing it."
        )
    if best.conflicting_fields:
        fields = ", ".join(best.conflicting_fields)
        return (
            "The leading candidate is plausible but not fully confirmed. The "
            f"fastest way to disambiguate is to confirm the {fields}."
        )
    return (
        "The server found plausible candidates but cannot confidently force a "
        "single canonical citation from the available clues."
    )


async def _resolve_identifier_candidate(
    *,
    parsed: ParsedCitation,
    client: Any,
    openalex_client: Any,
    enable_openalex: bool,
) -> RankedCitationCandidate | None:
    if not parsed.identifier:
        return None
    semantic_identifier = _normalize_identifier_for_semantic_scholar(parsed.identifier, parsed.identifier_type)
    openalex_identifier = _normalize_identifier_for_openalex(parsed.identifier, parsed.identifier_type)
    last_error: Exception | None = None
    for strategy, resolver in (
        (
            "identifier",
            lambda: client.get_paper_details(
                paper_id=semantic_identifier,
                fields=None,
            ),
        ),
        (
            "identifier_openalex",
            lambda: openalex_client.get_paper_details(paper_id=openalex_identifier),
        ),
    ):
        if strategy == "identifier_openalex" and not enable_openalex:
            continue
        if strategy == "identifier_openalex" and not openalex_identifier:
            continue
        try:
            paper = dump_jsonable(await resolver())
        except Exception as exc:
            last_error = exc
            logger.debug(
                "Citation identifier resolution via %s failed for %r: %s",
                strategy,
                parsed.identifier,
                exc,
            )
        else:
            return _rank_candidate(
                paper=paper,
                parsed=parsed,
                resolution_strategy=strategy,
                candidate_count=1,
                snippet_text=None,
            )
    if last_error is not None:
        logger.debug(
            "No citation identifier candidate resolved for %r after fallback attempts.",
            parsed.identifier,
        )
    return None


async def _resolve_title_candidates(
    *,
    parsed: ParsedCitation,
    title_candidates: list[str],
    client: Any,
) -> list[RankedCitationCandidate]:
    ranked: list[RankedCitationCandidate] = []
    for title_candidate in title_candidates:
        try:
            payload = dump_jsonable(await client.search_papers_match(query=title_candidate, fields=None))
        except Exception as exc:
            logger.debug(
                "Citation title recovery failed for %r: %s",
                title_candidate,
                exc,
            )
        else:
            if not payload.get("paperId"):
                continue
            strategy = str(payload.get("matchStrategy") or "title_match")
            candidate = _rank_candidate(
                paper=payload,
                parsed=parsed,
                resolution_strategy=strategy,
                candidate_count=int(payload.get("candidateCount") or 1),
                snippet_text=None,
            )
            ranked.append(candidate)
            if (
                _classify_resolution_confidence(
                    best_score=candidate.score,
                    runner_up_score=None,
                    matched_fields=candidate.matched_fields,
                    conflicting_fields=candidate.conflicting_fields,
                    resolution_strategy=candidate.resolution_strategy,
                    title_similarity=candidate.title_similarity,
                )
                == "high"
            ):
                break
    return ranked


def _ranked_candidates(
    candidate_map: dict[str, RankedCitationCandidate],
    *,
    max_candidates: int,
) -> list[RankedCitationCandidate]:
    return sorted(
        candidate_map.values(),
        key=lambda item: (
            -item.score,
            item.year_delta if item.year_delta is not None else 999,
            -item.author_overlap,
            -_publication_preference_score(item.paper),
            -item.title_similarity,
        ),
    )[:max_candidates]


def _best_candidate_confidence(
    candidates: list[RankedCitationCandidate],
) -> Any:
    best = candidates[0] if candidates else None
    runner_up_score = candidates[1].score if len(candidates) > 1 else None
    return _classify_resolution_confidence(
        best_score=best.score if best is not None else None,
        runner_up_score=runner_up_score,
        matched_fields=best.matched_fields if best is not None else [],
        conflicting_fields=best.conflicting_fields if best is not None else [],
        resolution_strategy=best.resolution_strategy if best is not None else "none",
        title_similarity=best.title_similarity if best is not None else None,
    )


async def _resolve_snippet_candidates(
    *,
    parsed: ParsedCitation,
    max_candidates: int,
    client: Any,
) -> list[RankedCitationCandidate]:
    if parsed.quoted_fragments:
        snippet_query = max(parsed.quoted_fragments, key=len)
    elif parsed.title_candidates:
        snippet_query = parsed.title_candidates[0]
    else:
        return []
    try:
        snippet_payload = dump_jsonable(await client.search_snippets(query=snippet_query, limit=max_candidates))
    except Exception:
        return []

    ranked: list[RankedCitationCandidate] = []
    for item in (snippet_payload.get("data") or [])[:max_candidates]:
        if not isinstance(item, dict):
            continue
        paper = item.get("paper")
        if not isinstance(paper, dict) or not paper.get("paperId"):
            continue
        enriched = paper
        try:
            enriched = dump_jsonable(
                await client.get_paper_details(
                    paper_id=str(paper["paperId"]),
                    fields=None,
                )
            )
        except Exception as exc:
            logger.debug(
                "Citation snippet enrichment failed for paper %r: %s",
                paper.get("paperId"),
                exc,
            )
        ranked.append(
            _rank_candidate(
                paper=enriched,
                parsed=parsed,
                resolution_strategy="snippet_recovery",
                candidate_count=len(snippet_payload.get("data") or []),
                snippet_text=_snippet_text(item),
            )
        )
    return ranked


async def _resolve_sparse_metadata_candidates(
    *,
    parsed: ParsedCitation,
    max_candidates: int,
    client: Any,
    enable_core: bool,
    enable_semantic_scholar: bool,
    enable_arxiv: bool,
    enable_serpapi: bool,
    core_client: Any,
    arxiv_client: Any,
    serpapi_client: Any,
    enable_openalex: bool,
    openalex_client: Any,
) -> list[RankedCitationCandidate]:
    from ..search_executor import SearchClientBundle, SearchExecutor

    search_queries = _sparse_search_queries(parsed)
    ranked: list[RankedCitationCandidate] = []
    search_executor = SearchExecutor()
    for query in search_queries[:3]:
        try:
            search_trace = await search_executor.search_with_fallback(
                query=query,
                limit=max(max_candidates * 4, 12),
                year=str(parsed.year) if parsed.year is not None else None,
                fields=None,
                venue=parsed.venue_hints[:1] or None,
                preferred_provider=("semantic_scholar" if enable_semantic_scholar else None),
                provider_order=None,
                default_provider_order=DEFAULT_SEARCH_PROVIDER_ORDER,
                ss_only_filters=[],
                enabled={
                    "core": enable_core,
                    "semantic_scholar": enable_semantic_scholar,
                    "arxiv": enable_arxiv,
                    "serpapi_google_scholar": enable_serpapi,
                    "openalex": False,
                },
                clients=SearchClientBundle(
                    core_client=core_client,
                    semantic_client=client,
                    arxiv_client=arxiv_client,
                    serpapi_client=serpapi_client,
                ),
                provider_registry=None,
                allow_default_hedging=False,
                publication_date_or_year=None,
                fields_of_study=None,
                publication_types=None,
                open_access_pdf=None,
                min_citation_count=None,
            )
        except Exception as exc:
            logger.debug(
                "Citation sparse metadata recovery failed for %r: %s",
                query,
                exc,
            )
        else:
            data = search_trace.result.model_dump(by_alias=True)["data"] if search_trace.result is not None else []
            for paper in data[: max_candidates * 2]:
                if not isinstance(paper, dict):
                    continue
                ranked.append(
                    _rank_candidate(
                        paper=paper,
                        parsed=parsed,
                        resolution_strategy="sparse_metadata",
                        candidate_count=len(data),
                        snippet_text=None,
                    )
                )
            if ranked:
                break

    if enable_openalex and openalex_client is not None:
        openalex_query = search_queries[0] if search_queries else parsed.normalized_text
        try:
            openalex_payload = dump_jsonable(
                await openalex_client.search(
                    query=openalex_query,
                    limit=max_candidates,
                    year=str(parsed.year) if parsed.year is not None else None,
                )
            )
        except Exception:
            openalex_payload = {}
        for paper in (openalex_payload.get("data") or [])[:max_candidates]:
            if not isinstance(paper, dict):
                continue
            ranked.append(
                _rank_candidate(
                    paper=paper,
                    parsed=parsed,
                    resolution_strategy="openalex_metadata",
                    candidate_count=len(openalex_payload.get("data") or []),
                    snippet_text=None,
                )
            )
    return ranked


def _merge_ranked_candidate(
    candidate_map: dict[str, RankedCitationCandidate],
    candidate: RankedCitationCandidate,
) -> None:
    key = _paper_identity_key(candidate.paper)
    existing = candidate_map.get(key)
    if existing is None:
        candidate_map[key] = candidate
        return
    existing.score = max(existing.score, candidate.score)
    existing.title_similarity = max(
        existing.title_similarity,
        candidate.title_similarity,
    )
    existing.author_overlap = max(existing.author_overlap, candidate.author_overlap)
    if existing.year_delta is None:
        existing.year_delta = candidate.year_delta
    elif candidate.year_delta is not None:
        existing.year_delta = min(existing.year_delta, candidate.year_delta)
    existing.matched_fields = _dedupe_strings([*existing.matched_fields, *candidate.matched_fields])
    existing.conflicting_fields = _dedupe_strings([*existing.conflicting_fields, *candidate.conflicting_fields])
    if candidate.score >= existing.score:
        existing.resolution_strategy = candidate.resolution_strategy
        existing.why_selected = candidate.why_selected
        existing.candidate_count = candidate.candidate_count


def _paper_identity_key(paper: dict[str, Any]) -> str:
    for candidate in (
        paper.get("paperId"),
        paper.get("canonicalId"),
        paper.get("recommendedExpansionId"),
        paper.get("sourceId"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    title = normalize_citation_text(str(paper.get("title") or "")).lower()
    year = str(paper.get("year") or "")
    return f"title:{title}|year:{year}"


def _snippet_text(item: dict[str, Any]) -> str | None:
    snippet = item.get("snippet")
    if isinstance(snippet, dict) and snippet.get("text"):
        return str(snippet["text"])
    if item.get("text"):
        return str(item["text"])
    return None
