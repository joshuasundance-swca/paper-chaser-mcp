"""Search fallback and result-shaping helpers."""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Optional

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
from .provider_runtime import (
    ProviderDiagnosticsRegistry,
    ProviderStatusBucket,
    provider_is_paywalled,
)
from .search_executor import (
    ProviderSearchRequest,
    SearchClientBundle,
    SearchExecutor,
    arxiv_result,
    core_result,
    earlier_result_attempt,
    enrich_semantic_scholar_paper,
    provider_attempt_from_outcome,
    semantic_result,
    serpapi_result,
)

logger = logging.getLogger("paper-chaser-mcp")

_SEARCH_EXECUTOR = SearchExecutor()

BROKER_HEDGE_DELAY_SECONDS = 0.35
HEDGE_ELIGIBLE_PROVIDERS: frozenset[SearchProvider] = frozenset({"semantic_scholar", "arxiv", "core"})

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
_TITLEISH_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'/-]*")


@dataclass
class _BrokerProviderResult:
    provider: SearchProvider
    attempt: BrokerAttempt
    response: SearchResponse | None = None

    @property
    def has_results(self) -> bool:
        return self.response is not None and bool(self.response.data)


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
    result_text = " ".join(" ".join(filter(None, [p.title, p.abstract])) for p in papers).lower()
    return any(token not in result_text for token in distinctive)


def _is_title_like_query(query: str) -> bool:
    """Return True when *query* looks more like a known-item title than a topic.

    This heuristic is intentionally conservative and is only used together with
    the low-relevance check. We only suppress brokered Semantic Scholar results
    when the query looks title-like *and* the returned results failed the
    distinctive-token relevance screen.
    """

    normalized = " ".join(str(query or "").split())
    if not normalized:
        return False
    lowered = normalized.lower()
    if any(marker in lowered for marker in ("doi:", "arxiv:", "site:", "author:")):
        return False
    if re.search(r"\b(?:19|20)\d{2}\b", normalized):
        return False
    words = _TITLEISH_WORD_RE.findall(normalized)
    if not 4 <= len(words) <= 18:
        return False
    capitalized = sum(1 for word in words if word[:1].isupper())
    capitalized_ratio = capitalized / max(len(words), 1)
    has_title_punctuation = any(mark in normalized for mark in (":", '"', "“", "”", "-", "–", "—"))
    return has_title_punctuation or capitalized_ratio >= 0.6


def _enrich_ss_paper(paper: Paper) -> Paper:
    """Return a copy of a Semantic Scholar paper enriched with provenance fields."""

    return enrich_semantic_scholar_paper(paper)


def _dump_search_response(response: SearchResponse) -> dict[str, Any]:
    result: dict[str, Any] = {
        "total": response.total,
        "offset": response.offset,
        "data": [paper.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True) for paper in response.data],
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
        - ``"unknown"``: SerpApi, ScholarAPI, or no results — match quality is undetermined.
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
    suppressed_low_relevance_title: bool = False,
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
        "scholarapi": "ScholarAPI",
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
    incompatible_filter_skips = [
        attempt
        for attempt in attempts
        if attempt.status == "skipped"
        and attempt.reason
        and (
            "Semantic Scholar-only filters were requested" in attempt.reason
            or "cannot honor requested advanced filters" in attempt.reason
        )
    ]

    if provider_used == "serpapi_google_scholar":
        next_step_hint = (
            "Results are from SerpApi Google Scholar, a paid provider path. Papers that include "
            "scholarResultId can be passed to get_paper_citation_formats for "
            "MLA, APA, BibTeX, and other export formats. "
            + portability_guidance
            + quality_guidance
            + bulk_guidance
            + "To expand from a paper use get_paper_citations or get_paper_references."
        )
    elif provider_used == "scholarapi":
        next_step_hint = (
            "Results are from ScholarAPI, a paid provider path. "
            "If the task needs more results ordered by indexed_at rather than relevance, "
            "continue with list_papers_scholarapi. "
            "If the task needs accessible full text or binary PDFs, use "
            "get_paper_text_scholarapi, get_paper_texts_scholarapi, or "
            "get_paper_pdf_scholarapi with the ScholarAPI paper id. "
            + quality_guidance
            + bulk_guidance
            + (
                "Use get_paper_citations or get_paper_references only after "
                "you re-anchor the paper into a Semantic Scholar-compatible id."
            )
        )
    elif provider_used == "none":
        if suppressed_low_relevance_title:
            next_step_hint = (
                "Semantic Scholar returned low-relevance results for a title-like query, "
                "so the broker suppressed them instead of surfacing likely false positives. "
                "Try search_papers_match for exact-title recovery, refine the title, or "
                "change providerOrder."
            )
        elif incompatible_filter_skips and all(attempt.status == "skipped" for attempt in attempts):
            next_step_hint = (
                "No routed provider was allowed to run because one or more requested advanced filters "
                "could not be honored by that provider chain. Inspect brokerMetadata.attemptedProviders, "
                "remove the incompatible filters, route to a provider that supports them, or use "
                "search_papers_semantic_scholar for the widest advanced-filter support."
            )
        elif result_status == "provider_failed":
            failed_labels = [provider_labels.get(a.provider, a.provider) for a in attempts if a.status == "failed"]
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
        paid_provider_used=(provider_is_paywalled(provider_used) if provider_used != "none" else False),
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
    return _SEARCH_EXECUTOR.effective_provider_order(
        preferred_provider=preferred_provider,
        provider_order=provider_order,
        default_order=DEFAULT_SEARCH_PROVIDER_ORDER,
    )


def _provider_enabled(
    provider: SearchProvider,
    *,
    enable_core: bool,
    enable_semantic_scholar: bool,
    enable_arxiv: bool,
    enable_serpapi: bool,
    enable_scholarapi: bool,
    serpapi_client: Any,
    scholarapi_client: Any,
) -> bool:
    return _SEARCH_EXECUTOR.provider_enabled(
        provider,
        enabled={
            "core": enable_core,
            "semantic_scholar": enable_semantic_scholar,
            "arxiv": enable_arxiv,
            "serpapi_google_scholar": enable_serpapi,
            "scholarapi": enable_scholarapi,
            "openalex": False,
        },
        clients=SearchClientBundle(
            core_client=True,
            semantic_client=True,
            arxiv_client=True,
            serpapi_client=serpapi_client,
            scholarapi_client=scholarapi_client,
        ),
    )


def _hedge_target_index(
    *,
    effective_order: Sequence[SearchProvider],
    current_index: int,
    ss_only_filters: list[str],
    enable_core: bool,
    enable_semantic_scholar: bool,
    enable_arxiv: bool,
    enable_serpapi: bool,
    enable_scholarapi: bool,
    serpapi_client: Any,
    scholarapi_client: Any,
) -> int | None:
    return _SEARCH_EXECUTOR.hedge_target_index(
        effective_order=effective_order,
        current_index=current_index,
        ss_only_filters=ss_only_filters,
        enabled={
            "core": enable_core,
            "semantic_scholar": enable_semantic_scholar,
            "arxiv": enable_arxiv,
            "serpapi_google_scholar": enable_serpapi,
            "scholarapi": enable_scholarapi,
            "openalex": False,
        },
        clients=SearchClientBundle(
            core_client=True,
            semantic_client=True,
            arxiv_client=True,
            serpapi_client=serpapi_client,
            scholarapi_client=scholarapi_client,
        ),
    )


def _skip_for_ss_only_filters(provider: SearchProvider, ss_only_filters: list[str]) -> BrokerAttempt | None:
    return _SEARCH_EXECUTOR.skip_for_semantic_scholar_only_filters(provider, ss_only_filters)


def _disabled_provider_attempt(provider: SearchProvider) -> BrokerAttempt:
    disabled_reasons = {
        "core": "Disabled by PAPER_CHASER_ENABLE_CORE=false.",
        "semantic_scholar": "Disabled by PAPER_CHASER_ENABLE_SEMANTIC_SCHOLAR=false.",
        "serpapi_google_scholar": "Disabled by PAPER_CHASER_ENABLE_SERPAPI=false.",
        "scholarapi": "Disabled by PAPER_CHASER_ENABLE_SCHOLARAPI=false.",
        "arxiv": "Disabled by PAPER_CHASER_ENABLE_ARXIV=false.",
    }
    return BrokerAttempt(
        provider=provider,
        status="skipped",
        reason=disabled_reasons[provider],
    )


def _earlier_result_attempt(provider: SearchProvider) -> BrokerAttempt:
    return earlier_result_attempt(provider)


def _attempt_from_outcome(
    provider: SearchProvider,
    *,
    status_bucket: ProviderStatusBucket,
    reason: str | None,
) -> BrokerAttempt:
    return provider_attempt_from_outcome(
        provider,
        status_bucket=status_bucket,
        reason=reason,
    )


def _core_result(core_response: dict[str, Any], limit: int) -> SearchResponse:
    return core_result(core_response, limit)


def _semantic_result(s2_response: dict[str, Any], limit: int) -> SearchResponse:
    return semantic_result(s2_response, limit)


def _serpapi_result(serpapi_papers: list[dict[str, Any]], limit: int) -> SearchResponse:
    return serpapi_result(serpapi_papers, limit)


def _arxiv_result(arxiv_response: dict[str, Any], limit: int) -> SearchResponse:
    return arxiv_result(arxiv_response, limit)


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


def _core_response_to_merged(core_response: dict[str, Any], limit: int) -> dict[str, Any]:
    """Convert CORE search response to unified shape (total, offset, data)."""
    core_search = CoreSearchResponse.model_validate(core_response)
    return _dump_search_response(
        SearchResponse(
            total=core_search.total or len(core_search.entries),
            offset=0,
            data=core_search.entries[:limit],
        )
    )


def _build_provider_search_request(
    *,
    query: str,
    limit: int,
    year: str | None,
    fields: list[str] | None,
    venue: list[str] | None,
    publication_date_or_year: str | None,
    fields_of_study: str | None,
    publication_types: str | None,
    open_access_pdf: bool | None,
    min_citation_count: int | None,
) -> ProviderSearchRequest:
    return ProviderSearchRequest(
        query=query,
        limit=limit,
        year=year,
        fields=fields,
        venue=venue,
        publication_date_or_year=publication_date_or_year,
        fields_of_study=fields_of_study,
        publication_types=publication_types,
        open_access_pdf=open_access_pdf,
        min_citation_count=min_citation_count,
    )


async def _run_provider_search(
    *,
    provider: SearchProvider,
    query: str,
    limit: int,
    year: Optional[str],
    fields: Optional[list[str]],
    venue: Optional[list[str]],
    core_client: Any,
    semantic_client: Any,
    arxiv_client: Any,
    serpapi_client: Any = None,
    scholarapi_client: Any = None,
    publication_date_or_year: Optional[str] = None,
    fields_of_study: Optional[str] = None,
    publication_types: Optional[str] = None,
    open_access_pdf: Optional[bool] = None,
    min_citation_count: Optional[int] = None,
    provider_registry: ProviderDiagnosticsRegistry | None = None,
) -> _BrokerProviderResult:
    result = await _SEARCH_EXECUTOR.execute_provider(
        provider=provider,
        request=_build_provider_search_request(
            query=query,
            limit=limit,
            year=year,
            fields=fields,
            venue=venue,
            publication_date_or_year=publication_date_or_year,
            fields_of_study=fields_of_study,
            publication_types=publication_types,
            open_access_pdf=open_access_pdf,
            min_citation_count=min_citation_count,
        ),
        clients=SearchClientBundle(
            core_client=core_client,
            semantic_client=semantic_client,
            arxiv_client=arxiv_client,
            serpapi_client=serpapi_client,
        ),
        provider_registry=provider_registry,
    )
    return _BrokerProviderResult(
        provider=provider,
        attempt=result.attempt,
        response=result.response,
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
    enable_scholarapi: bool = False,
    core_client: Any,
    semantic_client: Any,
    arxiv_client: Any,
    serpapi_client: Any = None,
    scholarapi_client: Any = None,
    publication_date_or_year: Optional[str] = None,
    fields_of_study: Optional[str] = None,
    publication_types: Optional[str] = None,
    open_access_pdf: Optional[bool] = None,
    min_citation_count: Optional[int] = None,
    preferred_provider: SearchProvider | None = None,
    provider_order: Sequence[SearchProvider] | None = None,
    provider_registry: ProviderDiagnosticsRegistry | None = None,
    allow_default_hedging: bool = False,
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

    ScholarAPI is also available as an explicit opt-in broker target when
    ``enable_scholarapi=True`` and ``providerOrder`` or ``preferredProvider``
    routes to it. It is intentionally kept out of the default broker order so
    the free default path remains stable. When explicitly routed, ScholarAPI
    can honor the simple ``year`` filter plus ``openAccessPdf`` via its own
    PDF-availability filter, while unsupported advanced filters still cause the
    broker to skip ScholarAPI rather than silently widening the query.

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
        if value is not None and (field_name != "open_access_pdf" or value is True)
    ]
    trace = await _SEARCH_EXECUTOR.search_with_fallback(
        query=query,
        limit=limit,
        year=year,
        fields=fields,
        venue=venue,
        preferred_provider=preferred_provider,
        provider_order=provider_order,
        default_provider_order=DEFAULT_SEARCH_PROVIDER_ORDER,
        ss_only_filters=ss_only_filters,
        enabled={
            "core": enable_core,
            "semantic_scholar": enable_semantic_scholar,
            "arxiv": enable_arxiv,
            "serpapi_google_scholar": enable_serpapi,
            "scholarapi": enable_scholarapi,
            "openalex": False,
        },
        clients=SearchClientBundle(
            core_client=core_client,
            semantic_client=semantic_client,
            arxiv_client=arxiv_client,
            serpapi_client=serpapi_client,
            scholarapi_client=scholarapi_client,
        ),
        provider_registry=provider_registry,
        allow_default_hedging=allow_default_hedging,
        hedge_delay_seconds=BROKER_HEDGE_DELAY_SECONDS,
        publication_date_or_year=publication_date_or_year,
        fields_of_study=fields_of_study,
        publication_types=publication_types,
        open_access_pdf=open_access_pdf,
        min_citation_count=min_citation_count,
    )
    result = trace.result
    provider_used = trace.provider_used
    attempts = list(trace.attempts)
    effective_order = trace.effective_order
    processed_providers = trace.processed_providers
    low_relevance = False

    if provider_used != "none":
        if result is not None:
            low_relevance = provider_used == "semantic_scholar" and _has_unmatched_distinctive_tokens(
                query, result.data
            )
            suppress_low_relevance_title = low_relevance and _is_title_like_query(query)
            if suppress_low_relevance_title:
                for attempt in attempts:
                    if attempt.provider == provider_used and attempt.status == "returned_results":
                        attempt.status = "returned_no_results"
                        attempt.reason = (
                            "returned low-relevance results for a title-like query; broker suppressed "
                            "them to avoid obvious false positives"
                        )
                        break
                return _dump_search_response(
                    SearchResponse(
                        broker_metadata=_metadata(
                            provider_used="none",
                            attempts=attempts,
                            ss_only_filters=ss_only_filters,
                            venue=venue,
                            preferred_provider=preferred_provider,
                            provider_order=provider_order,
                            suppressed_low_relevance_title=True,
                        ),
                    )
                )
        attempts.extend(_earlier_result_attempt(provider) for provider in effective_order[processed_providers:])
        if result is not None:
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
                    ss_only_filters=trace.ss_only_filters,
                    venue=venue,
                    preferred_provider=preferred_provider,
                    provider_order=provider_order,
                ),
            )
        )
    return _dump_search_response(result)
