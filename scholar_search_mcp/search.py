"""Search fallback and result-shaping helpers."""

import logging
import re
from collections.abc import Sequence
from typing import Any, Literal, Optional

from .clients.serpapi import SerpApiKeyMissingError
from .models import (
    ArxivSearchResponse,
    BrokerAttempt,
    BrokerMetadata,
    CoreSearchResponse,
    Paper,
    SearchResponse,
    SemanticSearchResponse,
)
from .models.tools import (
    DEFAULT_SEARCH_PROVIDER_ORDER,
    SearchPapersArgs,
    SearchProvider,
)

logger = logging.getLogger("scholar-search-mcp")

SEMANTIC_SCHOLAR_ONLY_FIELDS = (
    "publication_date_or_year",
    "fields_of_study",
    "publication_types",
    "open_access_pdf",
    "min_citation_count",
)
SEMANTIC_SCHOLAR_ONLY_FILTER_ALIASES = {
    field_name: SearchPapersArgs.model_fields[field_name].alias or field_name
    for field_name in SEMANTIC_SCHOLAR_ONLY_FIELDS
}

# Common academic/research terms that should not count as distinctive query
# tokens when checking result relevance.  Short tokens (<6 chars) are already
# excluded by the length threshold, so only longer common words are listed here.
_RELEVANCE_STOPWORDS: frozenset[str] = frozenset(
    {
        "across",
        "analysis",
        "another",
        "approach",
        "article",
        "authors",
        "before",
        "between",
        "beyond",
        "dataset",
        "during",
        "enable",
        "every",
        "findings",
        "further",
        "general",
        "however",
        "improve",
        "including",
        "information",
        "learning",
        "methods",
        "models",
        "neural",
        "novel",
        "number",
        "obtain",
        "papers",
        "perform",
        "present",
        "problem",
        "proposed",
        "provide",
        "recent",
        "report",
        "research",
        "results",
        "review",
        "should",
        "significant",
        "specific",
        "studies",
        "system",
        "technique",
        "through",
        "towards",
        "training",
        "various",
        "within",
        "without",
        "would",
    }
)


def _distinctive_query_tokens(query: str) -> list[str]:
    """Return lowercase tokens from *query* that are long and non-trivial.

    A token is considered distinctive when it is at least 6 characters long
    and not in the academic stopword list.  Short common words and well-known
    academic filler terms are excluded so that only tokens capable of
    meaningfully constraining the result set are returned.
    """
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    return [t for t in tokens if len(t) >= 6 and t not in _RELEVANCE_STOPWORDS]


def _has_unmatched_distinctive_tokens(query: str, papers: list[Paper]) -> bool:
    """Return True if any distinctive query token appears in no result text.

    A "distinctive" token is at least 6 characters long and not in the
    academic stopword list (see ``_distinctive_query_tokens``).  When such a
    token is absent from every returned paper's title *and* abstract the
    results are likely a weak match for that part of the query — e.g. when a
    gibberish token appears in the query but Semantic Scholar returns papers
    that only match the generic words around it.

    Returns False when there are no distinctive tokens (short or all-stopword
    query) or when all distinctive tokens appear in at least one result.
    """
    distinctive = _distinctive_query_tokens(query)
    if not distinctive:
        return False
    result_text = " ".join(
        " ".join(filter(None, [p.title, p.abstract])) for p in papers
    ).lower()
    return any(token not in result_text for token in distinctive)


def _enrich_ss_paper(paper: Paper) -> Paper:
    """Return a copy of a Semantic Scholar paper enriched with provenance fields.

    ``sourceId`` is the Semantic Scholar-native ``paperId`` hash.
    ``canonicalId`` follows the documented priority order:
    DOI > paperId > arXiv ID > sourceId.
    """
    external_ids: dict[str, Any] = (paper.model_extra or {}).get("externalIds") or {}
    doi: str | None = external_ids.get("DOI") or None
    arxiv_id: str | None = external_ids.get("ArXiv") or None
    paper_id: str | None = paper.paper_id
    source_id = paper_id
    canonical_id: str | None = doi or paper_id or arxiv_id or source_id
    return paper.model_copy(
        update={
            "source": paper.source or "semantic_scholar",
            "source_id": source_id,
            "canonical_id": canonical_id,
            "recommended_expansion_id": canonical_id,
            "expansion_id_status": "portable",
        }
    )


def _dump_search_response(response: SearchResponse) -> dict[str, Any]:
    result: dict[str, Any] = {
        "total": response.total,
        "offset": response.offset,
        "data": [
            paper.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
            for paper in response.data
        ],
    }
    if response.broker_metadata is not None:
        result["brokerMetadata"] = response.broker_metadata.model_dump(by_alias=True)
    return result


def _result_quality(
    provider_used: str,
) -> Literal["strong", "low_relevance", "lexical", "unknown"]:
    """Return a result-quality signal for the given provider.

    - ``"strong"``: Semantic Scholar used semantic/relevance ranking.
    - ``"low_relevance"``: Semantic Scholar returned results but distinctive
      query tokens were absent from all result titles/abstracts; the caller
      should apply the low-relevance override separately via ``_metadata``.
    - ``"lexical"``: CORE or arXiv used keyword-only matching; results may be
      false positives for unusual or nonsense queries.
    - ``"unknown"``: SerpApi path or no results — match quality is undetermined.
    """
    if provider_used == "semantic_scholar":
        return "strong"
    if provider_used in ("core", "arxiv"):
        return "lexical"
    return "unknown"


def _compute_result_status(
    provider_used: str,
    attempts: list[BrokerAttempt],
) -> Literal["returned_results", "no_results", "provider_failed"]:
    """Determine the top-level outcome of a search broker run.

    - ``"returned_results"``: at least one provider returned data.
    - ``"provider_failed"``: every active (non-skipped) attempt failed with an
      upstream error and no provider returned even an empty result set.  This
      signals a transient outage rather than a genuinely empty query.
    - ``"no_results"``: all providers responded but none returned results (true
      empty result set for this query).
    """
    if provider_used != "none":
        return "returned_results"
    active_attempts = [a for a in attempts if a.status != "skipped"]
    if active_attempts and all(a.status == "failed" for a in active_attempts):
        return "provider_failed"
    return "no_results"


def _metadata(
    *,
    provider_used: str,
    attempts: list[BrokerAttempt],
    ss_only_filters: list[str],
    venue: Sequence[str] | None = None,
    preferred_provider: SearchProvider | None = None,
    provider_order: Sequence[SearchProvider] | None = None,
    low_relevance: bool = False,
) -> BrokerMetadata:
    """Build consistent broker metadata for ``search_papers`` responses.

    When *low_relevance* is True and the provider is Semantic Scholar, the
    result quality is downgraded from ``"strong"`` to ``"low_relevance"`` and
    the ``nextStepHint`` includes an explicit warning so agents do not treat
    the results as a healthy discovery set.

    When ``provider_used`` is ``"none"`` and every active provider raised an
    upstream error, ``resultStatus`` is set to ``"provider_failed"`` and the
    ``nextStepHint`` explicitly describes the failure so agents do not confuse
    a transient outage with a genuinely empty query.
    """
    routing_steered = preferred_provider is not None or provider_order is not None
    provider_labels = {
        "core": "CORE",
        "semantic_scholar": "Semantic Scholar",
        "serpapi_google_scholar": "SerpApi Google Scholar",
        "arxiv": "arXiv",
    }
    portability_guidance = (
        "For Semantic Scholar expansion tools such as get_paper_citations, "
        "get_paper_references, get_paper_authors, or author pivots, prefer "
        "paper.recommendedExpansionId when it is present. If a result reports "
        "paper.expansionIdStatus='not_portable', do not retry with brokered "
        "paperId/sourceId/canonicalId values; resolve the paper through DOI or "
        "a Semantic Scholar-native lookup first. "
    )

    bulk_is_pivot = provider_used != "semantic_scholar"

    if provider_used == "semantic_scholar":
        if venue:
            bulk_guidance = (
                "search_papers_bulk can broaden this into a larger Semantic Scholar "
                "retrieval flow, but that is a semantic pivot because bulk search "
                "does not preserve venue filtering. Its default ordering is also "
                "NOT relevance-ranked, so it is not 'page 2' of these results. "
                "For citation-ranked bulk retrieval, pass sort='citationCount:desc'. "
            )
        elif routing_steered:
            bulk_guidance = (
                "If you need many more Semantic Scholar results for the same topic, "
                "use search_papers_bulk — but note it uses exhaustive corpus traversal "
                "with an internal ordering that is NOT relevance-ranked, and it "
                "leaves the brokered routing preferences behind. "
                "It is not 'page 2' of these results; "
                "expect different result ordering. "
                "For citation-ranked bulk retrieval, pass sort='citationCount:desc'. "
            )
        else:
            bulk_guidance = (
                "If you need many more Semantic Scholar results for the same topic, "
                "use search_papers_bulk — but note it uses exhaustive corpus traversal "
                "with an internal ordering that is NOT relevance-ranked. "
                "It is not 'page 2' of these results; "
                "expect different result ordering. "
                "For citation-ranked bulk retrieval, pass sort='citationCount:desc'. "
            )
    elif provider_used in provider_labels:
        bulk_guidance = (
            "search_papers_bulk uses Semantic Scholar regardless of which provider "
            f"search_papers used — calling it now is a provider pivot away from "
            f"{provider_labels[provider_used]}, not a continuation. "
        )
    else:
        bulk_guidance = ""

    quality = _result_quality(provider_used)
    if low_relevance and quality == "strong":
        quality = "low_relevance"
        quality_guidance = (
            "brokerMetadata.resultQuality='low_relevance': one or more distinctive "
            "query tokens were not found in any returned result title or abstract — "
            "the results are likely a weak or irrelevant match for the full query. "
            "Do not treat these as a trustworthy discovery set. "
            "Consider rephrasing the query, broadening it, or trying a different "
            "provider via providerOrder. "
        )
    elif quality == "lexical":
        quality_guidance = (
            "brokerMetadata.resultQuality='lexical': results are keyword matches "
            f"from {provider_labels.get(provider_used, provider_used)} and may "
            "include false positives — verify topical relevance before proceeding. "
        )
    elif quality == "unknown" and provider_used != "none":
        quality_guidance = (
            "brokerMetadata.resultQuality='unknown': match quality is not "
            "determined for this provider — verify topical relevance. "
        )
    else:
        quality_guidance = ""

    result_status = _compute_result_status(provider_used, attempts)

    if provider_used == "serpapi_google_scholar":
        next_step_hint = (
            "Results are from SerpApi Google Scholar. Papers that include "
            "scholarResultId can be passed to get_paper_citation_formats for "
            "MLA, APA, BibTeX, and other export formats. "
            + portability_guidance
            + quality_guidance
            + bulk_guidance
            + "To expand from a paper use get_paper_citations or get_paper_references."
        )
    elif provider_used == "none":
        if result_status == "provider_failed":
            failed_labels = [
                provider_labels.get(a.provider, a.provider)
                for a in attempts
                if a.status == "failed"
            ]
            if len(failed_labels) == 1:
                next_step_hint = (
                    f"{failed_labels[0]} returned an upstream error "
                    f"(brokerMetadata.resultStatus='provider_failed'). "
                    "This is likely a transient outage, not an empty result set. "
                    "Retry the call later, or use search_papers to try other "
                    "providers. "
                    "Error details are in brokerMetadata.attemptedProviders."
                )
            else:
                failed_str = ", ".join(failed_labels)
                next_step_hint = (
                    f"All attempted providers ({failed_str}) returned upstream "
                    f"errors (brokerMetadata.resultStatus='provider_failed'). "
                    "This is likely a transient outage, not an empty result set. "
                    "Retry the call later, or use search_papers with a different "
                    "providerOrder. "
                    "Error details are in brokerMetadata.attemptedProviders."
                )
        else:
            next_step_hint = (
                "No provider returned results. Try broadening the query, changing "
                "providerOrder, or using search_papers_bulk with a different query."
            )
    else:
        next_step_hint = (
            "Inspect the results. "
            + (portability_guidance if provider_used in {"core", "arxiv"} else "")
            + quality_guidance
            + bulk_guidance
            + "To expand from a paper use get_paper_citations or get_paper_references."
        )
    return BrokerMetadata(
        provider_used=provider_used,
        result_status=result_status,
        attempted_providers=attempts,
        semantic_scholar_only_filters=ss_only_filters,
        result_quality=quality,
        bulk_search_is_provider_pivot=bulk_is_pivot,
        next_step_hint=next_step_hint,
    )


def _effective_provider_order(
    *,
    preferred_provider: SearchProvider | None,
    provider_order: Sequence[SearchProvider] | None,
) -> list[SearchProvider]:
    order = list(provider_order or DEFAULT_SEARCH_PROVIDER_ORDER)
    if preferred_provider is None:
        return order
    return [preferred_provider] + [
        provider for provider in order if provider != preferred_provider
    ]


def _skip_for_ss_only_filters(
    provider: SearchProvider, ss_only_filters: list[str]
) -> BrokerAttempt | None:
    if provider not in {"core", "serpapi_google_scholar"} or not ss_only_filters:
        return None
    return BrokerAttempt(
        provider=provider,
        status="skipped",
        reason=(
            "Skipped because Semantic Scholar-only filters were requested: "
            + ", ".join(ss_only_filters)
        ),
    )


def _disabled_provider_attempt(provider: SearchProvider) -> BrokerAttempt:
    disabled_reasons = {
        "core": "Disabled by SCHOLAR_SEARCH_ENABLE_CORE=false.",
        "semantic_scholar": "Disabled by SCHOLAR_SEARCH_ENABLE_SEMANTIC_SCHOLAR=false.",
        "serpapi_google_scholar": "Disabled by SCHOLAR_SEARCH_ENABLE_SERPAPI=false.",
        "arxiv": "Disabled by SCHOLAR_SEARCH_ENABLE_ARXIV=false.",
    }
    return BrokerAttempt(
        provider=provider,
        status="skipped",
        reason=disabled_reasons[provider],
    )


def _earlier_result_attempt(provider: SearchProvider) -> BrokerAttempt:
    return BrokerAttempt(
        provider=provider,
        status="skipped",
        reason="Skipped because an earlier provider already returned results.",
    )


def _core_result(core_response: dict[str, Any], limit: int) -> SearchResponse:
    core_search = CoreSearchResponse.model_validate(core_response)
    return SearchResponse(
        total=core_search.total or len(core_search.entries),
        offset=0,
        data=core_search.entries[:limit],
    )


def _semantic_result(s2_response: dict[str, Any], limit: int) -> SearchResponse:
    semantic_search = SemanticSearchResponse.model_validate(s2_response)
    return SearchResponse(
        total=semantic_search.total or len(semantic_search.data),
        offset=semantic_search.offset,
        data=[_enrich_ss_paper(paper) for paper in semantic_search.data[:limit]],
    )


def _serpapi_result(serpapi_papers: list[dict[str, Any]], limit: int) -> SearchResponse:
    validated: list[Paper] = [Paper.model_validate(p) for p in serpapi_papers]
    return SearchResponse(
        total=len(validated),
        offset=0,
        data=validated[:limit],
    )


def _arxiv_result(arxiv_response: dict[str, Any], limit: int) -> SearchResponse:
    arxiv_search = ArxivSearchResponse.model_validate(arxiv_response)
    return SearchResponse(
        total=arxiv_search.total_results,
        offset=0,
        data=arxiv_search.entries[:limit],
    )


def _merge_search_results(
    s2_response: dict[str, Any],
    arxiv_response: dict[str, Any],
    limit: int,
) -> dict[str, Any]:
    """Merge Semantic Scholar and arXiv results into the shared response shape."""
    semantic_search = SemanticSearchResponse.model_validate(s2_response)
    arxiv_search = ArxivSearchResponse.model_validate(arxiv_response)

    s2_data = [_enrich_ss_paper(paper) for paper in semantic_search.data]
    arxiv_entries = list(arxiv_search.entries)

    seen_arxiv_ids = set()
    for paper in s2_data:
        external_ids = (paper.model_extra or {}).get("externalIds") or {}
        arxiv_id = external_ids.get("ArXiv")
        if arxiv_id:
            seen_arxiv_ids.add(str(arxiv_id))

    merged = list(s2_data)
    for paper in arxiv_entries:
        arxiv_id = paper.paper_id or ""
        if arxiv_id and arxiv_id not in seen_arxiv_ids:
            seen_arxiv_ids.add(arxiv_id)
            merged.append(paper)

    return _dump_search_response(
        SearchResponse(
            total=len(merged[:limit]),
            offset=semantic_search.offset,
            data=merged[:limit],
        )
    )


def _core_response_to_merged(
    core_response: dict[str, Any], limit: int
) -> dict[str, Any]:
    """Convert CORE search response to unified shape (total, offset, data)."""
    core_search = CoreSearchResponse.model_validate(core_response)
    return _dump_search_response(
        SearchResponse(
            total=core_search.total or len(core_search.entries),
            offset=0,
            data=core_search.entries[:limit],
        )
    )


async def search_papers_with_fallback(
    *,
    query: str,
    limit: int,
    year: Optional[str],
    fields: Optional[list[str]],
    venue: Optional[list[str]],
    enable_core: bool,
    enable_semantic_scholar: bool,
    enable_arxiv: bool,
    enable_serpapi: bool = False,
    core_client: Any,
    semantic_client: Any,
    arxiv_client: Any,
    serpapi_client: Any = None,
    publication_date_or_year: Optional[str] = None,
    fields_of_study: Optional[str] = None,
    publication_types: Optional[str] = None,
    open_access_pdf: Optional[bool] = None,
    min_citation_count: Optional[int] = None,
    preferred_provider: SearchProvider | None = None,
    provider_order: Sequence[SearchProvider] | None = None,
) -> dict[str, Any]:
    """Execute the search broker chain using the configured provider order.

    CORE is skipped when any Semantic Scholar-only filter is requested
    (``publicationDateOrYear``, ``fieldsOfStudy``, ``publicationTypes``,
    ``openAccessPdf``, ``minCitationCount``) because CORE does not support those
    parameters, and silently returning un-filtered CORE results would violate the
    caller's intent.

    SerpApi Google Scholar is included in the fallback chain when
    ``enable_serpapi=True`` (opt-in, paid API).  It is also skipped when any
    Semantic Scholar-only filter is requested, because those filters have no
    equivalent in Google Scholar.

    Pagination is intentionally not supported here: each provider uses a different
    continuation mechanism and mixing pages from different backends would produce
    incorrect results.  For paginated retrieval use ``search_papers_bulk``
    (Semantic Scholar) or other provider-specific tools.
    """
    ss_only_filters = [
        SEMANTIC_SCHOLAR_ONLY_FILTER_ALIASES[field_name]
        for field_name, value in (
            ("publication_date_or_year", publication_date_or_year),
            ("fields_of_study", fields_of_study),
            ("publication_types", publication_types),
            ("open_access_pdf", open_access_pdf),
            ("min_citation_count", min_citation_count),
        )
        if value is not None
    ]
    effective_order = _effective_provider_order(
        preferred_provider=preferred_provider,
        provider_order=provider_order,
    )

    result: SearchResponse | None = None
    provider_used = "none"
    attempts: list[BrokerAttempt] = []
    processed_providers = 0
    for index, provider in enumerate(effective_order):
        processed_providers = index + 1
        skip_for_filters = _skip_for_ss_only_filters(provider, ss_only_filters)
        if skip_for_filters is not None:
            attempts.append(skip_for_filters)
            continue

        if provider == "core":
            if not enable_core:
                attempts.append(_disabled_provider_attempt(provider))
                continue
            try:
                core_response = await core_client.search(
                    query=query,
                    limit=limit,
                    start=0,
                    year=year,
                )
                result = _core_result(core_response, limit)
                if result.data:
                    provider_used = provider
                    attempts.append(
                        BrokerAttempt(provider=provider, status="returned_results")
                    )
                    logger.info("search_papers: using CORE API results")
                    break
                attempts.append(
                    BrokerAttempt(provider=provider, status="returned_no_results")
                )
                result = None
            except Exception as exc:
                attempts.append(
                    BrokerAttempt(provider=provider, status="failed", reason=str(exc))
                )
                logger.info(
                    "search_papers: CORE failed (%s), falling back to next channel",
                    exc,
                )
                result = None
            continue

        if provider == "semantic_scholar":
            if not enable_semantic_scholar:
                attempts.append(_disabled_provider_attempt(provider))
                continue
            try:
                s2_response = await semantic_client.search_papers(
                    query=query,
                    limit=limit,
                    fields=fields,
                    year=year,
                    venue=venue,
                    publication_date_or_year=publication_date_or_year,
                    fields_of_study=fields_of_study,
                    publication_types=publication_types,
                    open_access_pdf=open_access_pdf,
                    min_citation_count=min_citation_count,
                )
                result = _semantic_result(s2_response, limit)
                if result.data:
                    provider_used = provider
                    attempts.append(
                        BrokerAttempt(provider=provider, status="returned_results")
                    )
                    logger.info("search_papers: using Semantic Scholar results")
                    break
                attempts.append(
                    BrokerAttempt(provider=provider, status="returned_no_results")
                )
                result = None
            except Exception as exc:
                attempts.append(
                    BrokerAttempt(provider=provider, status="failed", reason=str(exc))
                )
                logger.info(
                    "search_papers: Semantic Scholar failed (%s), "
                    "falling back to next channel",
                    exc,
                )
                result = None
            continue

        if provider == "serpapi_google_scholar":
            if not enable_serpapi:
                attempts.append(_disabled_provider_attempt(provider))
                continue
            if serpapi_client is None:
                attempts.append(
                    BrokerAttempt(
                        provider=provider,
                        status="skipped",
                        reason="Enabled but no SerpApi client is configured.",
                    )
                )
                continue
            try:
                serpapi_papers = await serpapi_client.search(
                    query=query,
                    limit=limit,
                    year=year,
                )
                result = _serpapi_result(serpapi_papers, limit)
                if result.data:
                    provider_used = provider
                    attempts.append(
                        BrokerAttempt(provider=provider, status="returned_results")
                    )
                    logger.info("search_papers: using SerpApi Google Scholar results")
                    break
                attempts.append(
                    BrokerAttempt(provider=provider, status="returned_no_results")
                )
                result = None
            except SerpApiKeyMissingError:
                raise
            except Exception as exc:
                attempts.append(
                    BrokerAttempt(provider=provider, status="failed", reason=str(exc))
                )
                logger.info(
                    "search_papers: SerpApi failed (%s), falling back to next channel",
                    exc,
                )
                result = None
            continue

        if not enable_arxiv:
            attempts.append(_disabled_provider_attempt(provider))
            continue

        arxiv_response = await arxiv_client.search(
            query=query,
            limit=limit,
            year=year,
        )
        result = _arxiv_result(arxiv_response, limit)
        if result.data:
            provider_used = provider
            attempts.append(BrokerAttempt(provider=provider, status="returned_results"))
            logger.info("search_papers: using arXiv results")
            break
        attempts.append(BrokerAttempt(provider=provider, status="returned_no_results"))
        result = None

    if provider_used != "none":
        attempts.extend(
            _earlier_result_attempt(provider)
            for provider in effective_order[processed_providers:]
        )
        if result is not None:
            low_relevance = (
                provider_used == "semantic_scholar"
                and _has_unmatched_distinctive_tokens(query, result.data)
            )
            result.broker_metadata = _metadata(
                provider_used=provider_used,
                attempts=attempts,
                ss_only_filters=ss_only_filters,
                venue=venue,
                preferred_provider=preferred_provider,
                provider_order=provider_order,
                low_relevance=low_relevance,
            )

    if result is None:
        return _dump_search_response(
            SearchResponse(
                broker_metadata=_metadata(
                    provider_used="none",
                    attempts=attempts,
                    ss_only_filters=ss_only_filters,
                    venue=venue,
                    preferred_provider=preferred_provider,
                    provider_order=provider_order,
                ),
            )
        )
    return _dump_search_response(result)
