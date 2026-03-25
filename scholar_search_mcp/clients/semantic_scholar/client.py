"""Semantic Scholar API client."""

import logging
import re
import time
from difflib import SequenceMatcher
from typing import Any, Optional

from ...citation_repair import build_match_metadata
from ...constants import (
    API_BASE_URL,
    DEFAULT_AUTHOR_FIELDS,
    DEFAULT_PAPER_FIELDS,
    MAX_429_RETRIES,
    RECOMMENDATIONS_BASE_URL,
    SEMANTIC_SCHOLAR_MIN_INTERVAL,
)
from ...models import (
    AuthorListResponse,
    AuthorProfile,
    BatchAuthorResponse,
    BatchPaperResponse,
    BulkSearchResponse,
    Paper,
    PaperAuthorListResponse,
    PaperListResponse,
    RecommendationResponse,
    SemanticSearchResponse,
    SnippetSearchResponse,
    dump_jsonable,
)
from ...transport import asyncio, httpx, maybe_close_async_resource

logger = logging.getLogger("scholar-search-mcp")
_AUTHOR_QUERY_QUOTES_PATTERN = re.compile(r'["“”‘’`]+')
_AUTHOR_QUERY_PUNCTUATION_PATTERN = re.compile(r"[,;()]+")
_AUTHOR_QUERY_INITIAL_PERIOD_PATTERN = re.compile(r"(?<=\w)\.(?=\s|$)")
_AUTHOR_QUERY_WHITESPACE_PATTERN = re.compile(r"\s+")
_TITLE_LOOKUP_QUOTES_PATTERN = re.compile(r'["“”‘’`]+')
_TITLE_LOOKUP_PUNCTUATION_PATTERN = re.compile(r"[-–—:;,/?!()]+")
_TITLE_LOOKUP_KEY_PATTERN = re.compile(r"[^0-9a-z]+")
# Keep fuzzy-title recovery conservative: exact normalized matches win, and
# otherwise require very high similarity to avoid promoting unrelated papers.
_TITLE_MATCH_SIMILARITY_THRESHOLD = 0.92
# Fallback search window: use a larger window so that famous papers (e.g.
# "Attention Is All You Need") are more likely to appear in the candidate list
# even when relevance ranking places them below position 10.  100 is the
# maximum the /paper/search endpoint accepts; prefer accuracy over speed in
# this rare fallback path.
_TITLE_MATCH_FALLBACK_LIMIT = 100
# Regex for a bare arXiv ID (new format YYMM.NNNNN or YYMM.NNNN, optional vN,
# or old category/NNNNNNN format) without any prefix.
_BARE_ARXIV_ID_PATTERN = re.compile(
    r"^(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z][\w.-]+/\d{7}(?:v\d+)?)$",
    re.IGNORECASE,
)
# arxiv.org URL patterns: https://arxiv.org/abs/NNNNN or https://arxiv.org/pdf/NNNNN
_ARXIV_URL_PATTERN = re.compile(
    r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/([^\s/?#]+)",
    re.IGNORECASE,
)


class SemanticScholarClient:
    """Semantic Scholar API client.

    One shared rate limiter (``_rate_lock`` + ``_last_request_time``) enforces the
    documented ``1 request per second`` ceiling across *all* endpoints.  The limiter
    lives on the instance so that a single application-level client instance shares
    the budget, matching the guide's advice to keep one limiter per service client.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.headers: dict[str, str] = {}
        if api_key:
            self.headers["x-api-key"] = api_key
        # Lazily initialized in async context (avoids event-loop binding issues).
        self._rate_lock: Optional[asyncio.Lock] = None
        self._last_request_time: float = 0.0
        self._http_client: Any | None = None

    def _get_rate_lock(self) -> asyncio.Lock:
        if self._rate_lock is None:
            self._rate_lock = asyncio.Lock()
        return self._rate_lock

    def _get_http_client(self) -> Any:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def _pace(self) -> None:
        """Enforce the 1 request/second ceiling before every outbound request."""
        lock = self._get_rate_lock()
        async with lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < SEMANTIC_SCHOLAR_MIN_INTERVAL:
                await asyncio.sleep(SEMANTIC_SCHOLAR_MIN_INTERVAL - elapsed)
            self._last_request_time = time.monotonic()

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        json_data: Optional[dict[str, Any]] = None,
        base_url: Optional[str] = None,
        max_retries: int = 4,
        max_429_retries: Optional[int] = None,
        base_delay: float = 1.0,
    ) -> Any:
        """Send HTTP request with rate pacing and exponential backoff on 429.

        ``_pace()`` is called before *every* attempt (including retries after
        a 429) so that the shared cross-endpoint rate limit is honoured even
        when a previous attempt was rejected.
        """
        resolved_base = base_url if base_url is not None else API_BASE_URL
        url = f"{resolved_base}/{endpoint}"
        retry_429_limit = MAX_429_RETRIES if max_429_retries is None else max(0, max_429_retries)
        total_attempts = max(max_retries, retry_429_limit) + 1

        client = self._get_http_client()
        for attempt in range(total_attempts):
            await self._pace()
            response = await client.request(
                method=method,
                url=url,
                headers=self.headers,
                params=params,
                json=json_data,
            )

            if response.status_code == 429:
                if attempt < retry_429_limit:
                    delay = base_delay * (2**attempt)
                    retry_after = response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        delay = max(delay, float(retry_after))
                    logger.warning(
                        "Rate limited (429), retrying in %.1fs (%s/%s)",
                        delay,
                        attempt + 1,
                        retry_429_limit,
                    )
                    await asyncio.sleep(delay)
                    continue

                response.raise_for_status()
            response.raise_for_status()
            return response.json()

        raise RuntimeError("Semantic Scholar request retry loop exited unexpectedly")

    async def aclose(self) -> None:
        """Close the shared HTTP client, if one has been created."""
        client, self._http_client = self._http_client, None
        await maybe_close_async_resource(client)

    # ------------------------------------------------------------------
    # Paper search
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_match_response(response: Any) -> Paper:
        """Normalize best-match responses to one paper payload.

        The upstream endpoint normally returns a single paper object, but some
        observed responses have wrapped the match inside ``data[0]`` while also
        including top-level paper keys with null values. Prefer the nested paper
        payload when present so callers always receive one unambiguous
        paper-shaped response.
        """
        if isinstance(response, dict):
            nested_matches = response.get("data")
            if isinstance(nested_matches, list):
                for candidate in nested_matches:
                    if isinstance(candidate, dict):
                        return Paper.model_validate(candidate)
        return Paper.model_validate(response)

    @staticmethod
    def _normalize_nested_paper_list_response(
        response: Any,
        nested_key: str,
    ) -> PaperListResponse:
        """Normalize list endpoints that wrap each paper under a nested key."""
        if isinstance(response, dict):
            normalized = dict(response)
            raw_items = normalized.get("data")
            if raw_items is None:
                normalized["data"] = []
                response = normalized
                return PaperListResponse.model_validate(response)
            if isinstance(raw_items, list):
                normalized_items: list[dict[str, Any]] = []
                for item in raw_items:
                    if not isinstance(item, dict):
                        continue
                    nested_paper = item.get(nested_key)
                    if isinstance(nested_paper, dict):
                        paper_payload = dict(nested_paper)
                    elif "paperId" in item or "title" in item:
                        paper_payload = dict(item)
                    else:
                        continue
                    if paper_payload.get("authors") is None:
                        paper_payload["authors"] = []
                    normalized_items.append(paper_payload)
                normalized["data"] = normalized_items
            response = normalized
        return PaperListResponse.model_validate(response)

    @staticmethod
    def _status_code_from_error(exc: Exception) -> int | None:
        response = getattr(exc, "response", None)
        return getattr(response, "status_code", None)

    @staticmethod
    def _normalize_author_search_query(query: str) -> str:
        normalized = _AUTHOR_QUERY_QUOTES_PATTERN.sub(" ", query.strip())
        normalized = _AUTHOR_QUERY_PUNCTUATION_PATTERN.sub(" ", normalized)
        normalized = _AUTHOR_QUERY_INITIAL_PERIOD_PATTERN.sub("", normalized)
        normalized = _AUTHOR_QUERY_WHITESPACE_PATTERN.sub(" ", normalized).strip()
        return normalized or query.strip()

    @staticmethod
    def _normalize_paper_id(paper_id: str) -> str:
        """Normalize a paper identifier to a form accepted by Semantic Scholar.

        Semantic Scholar's ``/paper/{id}`` endpoint requires specific prefixes
        (``ARXIV:``, ``DOI:``, ``URL:``, etc.) in uppercase.  This helper
        normalizes the most common variants so callers can pass natural-language
        identifiers such as ``arXiv:1706.03762``, ``1706.03762``, or an
        ``arxiv.org`` URL and still reach the correct paper.

        Normalization rules (applied in order):
        1. ``arXiv:<id>`` / ``arxiv:<id>`` → ``ARXIV:<id>``
        2. Bare arXiv IDs (new-style ``YYMM.NNNNN`` or old-style
           ``category/NNNNNNN``) → ``ARXIV:<id>``
        3. ``https://arxiv.org/abs/<id>`` (or ``/pdf/``) → ``ARXIV:<id>``
        4. Everything else is returned unchanged.
        """
        stripped = paper_id.strip()
        # 1. Normalize case of arXiv: prefix
        if re.match(r"^arxiv:", stripped, re.IGNORECASE):
            return "ARXIV:" + stripped[len("arxiv:") :]
        # 2. Bare arXiv ID (no prefix)
        if _BARE_ARXIV_ID_PATTERN.match(stripped):
            return "ARXIV:" + stripped
        # 3. arxiv.org URL → ARXIV:<id>
        url_match = _ARXIV_URL_PATTERN.match(stripped)
        if url_match:
            return "ARXIV:" + url_match.group(1)
        return stripped

    @staticmethod
    def _paper_id_portability_hint() -> str:
        return (
            "If this paper came from brokered CORE, arXiv, or SerpApi results, retry "
            "with paper.recommendedExpansionId when it is present. If "
            "paper.expansionIdStatus is not_portable, resolve the paper through "
            "DOI or a Semantic Scholar-native lookup instead of reusing "
            "provider-specific paperId, sourceId, or canonicalId values."
        )

    @staticmethod
    def _enrich_bulk_paper(paper: Paper) -> Paper:
        """Enrich a bulk search result paper with SS expansion ID portability fields.

        Mirrors ``_enrich_ss_paper`` in ``search.py`` so that ``search_papers_bulk``
        results expose the same ``recommendedExpansionId`` / ``expansionIdStatus``
        signals as brokered ``search_papers_semantic_scholar`` results.
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

    @staticmethod
    def _normalize_title_lookup_query(query: str) -> str:
        normalized = _TITLE_LOOKUP_QUOTES_PATTERN.sub(" ", query.strip())
        normalized = _TITLE_LOOKUP_PUNCTUATION_PATTERN.sub(" ", normalized)
        normalized = _AUTHOR_QUERY_WHITESPACE_PATTERN.sub(" ", normalized).strip()
        return normalized or query.strip()

    @classmethod
    def _title_lookup_queries(cls, query: str) -> list[str]:
        queries: list[str] = []
        normalized = cls._normalize_title_lookup_query(query)
        for candidate in (
            query.strip(),
            normalized,
            query.strip().lower(),
            normalized.lower(),
        ):
            if candidate and candidate not in queries:
                queries.append(candidate)
        return queries

    @staticmethod
    def _normalize_title_match_key(value: str) -> str:
        normalized = _TITLE_LOOKUP_KEY_PATTERN.sub(" ", value.lower())
        normalized = _AUTHOR_QUERY_WHITESPACE_PATTERN.sub(" ", normalized).strip()
        return normalized

    @classmethod
    def _pick_title_match_candidate(
        cls,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        target_key = cls._normalize_title_match_key(query)
        if not target_key:
            return None

        best_candidate: dict[str, Any] | None = None
        best_score: tuple[int, float] = (-1, -1.0)
        for candidate in candidates:
            title = candidate.get("title")
            if not isinstance(title, str) or not title.strip():
                continue
            candidate_key = cls._normalize_title_match_key(title)
            if not candidate_key:
                continue
            exact = candidate_key == target_key
            containment = target_key in candidate_key or candidate_key in target_key
            ratio = SequenceMatcher(None, target_key, candidate_key).ratio()
            score = (2 if exact else 1 if containment else 0, ratio)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            return None

        exactish, ratio = best_score
        if exactish > 0 or ratio >= _TITLE_MATCH_SIMILARITY_THRESHOLD:
            return best_candidate
        return None

    @staticmethod
    def _build_fallback_match_payload(
        matched: dict[str, Any],
        query: str,
        candidate_query: str,
        strategy: str,
        candidate_count: int | None = None,
        match_provider: str | None = None,
    ) -> dict[str, Any]:
        """Build a match payload for a fallback-found paper.

        Sets ``matchFound``, ``matchStrategy``, and (when the effective query
        differs from the original) ``normalizedQuery`` so callers can tell which
        query variant actually produced the match.
        """
        payload = dump_jsonable(Paper.model_validate(matched))
        payload["matchFound"] = True
        payload["matchStrategy"] = strategy
        if match_provider is not None:
            payload["matchProvider"] = match_provider
        if candidate_query != query.strip():
            payload["normalizedQuery"] = candidate_query
        payload.update(
            build_match_metadata(
                query=query,
                paper=payload,
                candidate_count=candidate_count,
                resolution_strategy=strategy,
            )
        )
        return payload

    @classmethod
    def _pick_exact_title_match_candidate(
        cls,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        target_key = cls._normalize_title_match_key(query)
        if not target_key:
            return None
        for candidate in candidates:
            title = candidate.get("title")
            if not isinstance(title, str) or not title.strip():
                continue
            if cls._normalize_title_match_key(title) == target_key:
                return candidate
        return None

    @staticmethod
    def _normalize_crossref_work_to_paper(work: dict[str, Any]) -> dict[str, Any]:
        doi = str(work.get("doi") or "").strip() or None
        url = str(work.get("url") or "").strip() or None
        recommended_expansion_id = f"DOI:{doi}" if doi else None
        return {
            "paperId": doi or url or str(work.get("title") or "crossref-match"),
            "title": work.get("title"),
            "year": work.get("year"),
            "authors": work.get("authors") or [],
            "citationCount": work.get("citationCount"),
            "venue": work.get("venue"),
            "publicationDate": work.get("publicationDate"),
            "url": url,
            "source": "crossref",
            "sourceId": doi or url,
            "canonicalId": doi or url,
            "recommendedExpansionId": recommended_expansion_id,
            "expansionIdStatus": "portable" if recommended_expansion_id is not None else "not_portable",
        }

    async def _recover_exact_title_cross_provider(
        self,
        query: str,
        *,
        openalex_client: Any | None,
        enable_openalex: bool,
        crossref_client: Any | None,
        enable_crossref: bool,
    ) -> dict[str, Any] | None:
        if enable_openalex and openalex_client is not None:
            try:
                openalex_response = await openalex_client.search(query=query, limit=10)
            except Exception:
                openalex_response = None
            openalex_data = (openalex_response or {}).get("data") or []
            if isinstance(openalex_data, list):
                matched = self._pick_exact_title_match_candidate(query, openalex_data)
                if matched is not None:
                    return self._build_fallback_match_payload(
                        matched,
                        query,
                        query,
                        "openalex_exact_title",
                        candidate_count=len(openalex_data),
                        match_provider="openalex",
                    )

        if enable_crossref and crossref_client is not None:
            try:
                crossref_match = await crossref_client.search_work(query)
            except Exception:
                crossref_match = None
            if isinstance(crossref_match, dict):
                normalized = self._normalize_crossref_work_to_paper(crossref_match)
                matched = self._pick_exact_title_match_candidate(query, [normalized])
                if matched is not None:
                    return self._build_fallback_match_payload(
                        matched,
                        query,
                        query,
                        "crossref_exact_title",
                        candidate_count=1,
                        match_provider="crossref",
                    )

        return None

    async def _search_papers_match_fallback(
        self,
        query: str,
        fields: Optional[list[str]] = None,
        *,
        openalex_client: Any | None = None,
        enable_openalex: bool = False,
        crossref_client: Any | None = None,
        enable_crossref: bool = False,
    ) -> dict[str, Any]:
        candidate_queries = self._title_lookup_queries(query)
        # Also try quoted-phrase variants so that Semantic Scholar treats the
        # title as an exact phrase rather than separate keywords.  Unquoted
        # keyword search can rank generic topic papers above the specific paper
        # the agent wants when the title contains common words (e.g. "Attention
        # Is All You Need" competes with hundreds of papers about attention
        # mechanisms).  Quoted variants are appended after the plain variants so
        # the cheap unquoted paths are tried first.
        quoted_candidates = [f'"{q}"' for q in candidate_queries]
        all_search_queries = candidate_queries + quoted_candidates
        for candidate_query in all_search_queries:
            try:
                fallback_response = await self.search_papers(
                    candidate_query,
                    limit=_TITLE_MATCH_FALLBACK_LIMIT,
                    fields=fields,
                )
            except httpx.HTTPStatusError:
                continue
            matched = self._pick_title_match_candidate(
                query,
                fallback_response.get("data", []),
            )
            if matched is None:
                continue
            return self._build_fallback_match_payload(
                matched,
                query,
                candidate_query,
                "fuzzy_search",
                candidate_count=len(fallback_response.get("data", [])),
                match_provider="semantic_scholar",
            )

        external_match = await self._recover_exact_title_cross_provider(
            query,
            openalex_client=openalex_client,
            enable_openalex=enable_openalex,
            crossref_client=crossref_client,
            enable_crossref=enable_crossref,
        )
        if external_match is not None:
            return external_match

        # Final fallback: citation-sorted bulk search.  Relevance-ranked search
        # may bury a very famous paper behind topic papers with higher textual
        # overlap (e.g. "Attention Is All You Need" drowned by attention surveys).
        # Sorting by citation count descending ensures canonical, highly-cited
        # papers appear at the top regardless of relevance ranking.  This is a
        # deliberate last resort — it only runs after every other strategy has
        # failed, so the extra API call is acceptable.
        for candidate_query in candidate_queries:
            try:
                bulk_response = await self.search_papers_bulk(
                    candidate_query,
                    fields=fields,
                    sort="citationCount:desc",
                    limit=_TITLE_MATCH_FALLBACK_LIMIT,
                )
            except httpx.HTTPStatusError:
                continue
            matched = self._pick_title_match_candidate(
                query,
                bulk_response.get("data", []),
            )
            if matched is None:
                continue
            return self._build_fallback_match_payload(
                matched,
                query,
                candidate_query,
                "citation_ranked",
                candidate_count=len(bulk_response.get("data", [])),
                match_provider="semantic_scholar",
            )

        return {
            "paperId": None,
            "title": None,
            "query": query,
            "matchFound": False,
            "matchStrategy": "none",
            "matchConfidence": "low",
            "matchedFields": [],
            "titleSimilarity": 0.0,
            "yearDelta": None,
            "authorOverlap": 0,
            "candidateCount": 0,
            "normalizedQueriesTried": all_search_queries,
            "message": (
                "No Semantic Scholar title match was found. If you have a DOI, "
                "arXiv ID, or URL for this item, use get_paper_details instead "
                "(e.g. get_paper_details(paper_id='arXiv:1706.03762')). "
                "Otherwise this query may refer to a dissertation, software "
                "release, report, or other output outside the indexed paper "
                "surface. Try search_papers or search_authors for broader "
                "discovery."
            ),
        }

    async def search_papers(
        self,
        query: str,
        limit: int = 10,
        fields: Optional[list[str]] = None,
        year: Optional[str] = None,
        venue: Optional[list[str]] = None,
        offset: Optional[int] = None,
        publication_date_or_year: Optional[str] = None,
        fields_of_study: Optional[str] = None,
        publication_types: Optional[str] = None,
        open_access_pdf: Optional[bool] = None,
        min_citation_count: Optional[int] = None,
    ) -> dict[str, Any]:
        """Relevance-ranked paper search (``/paper/search``)."""
        params: dict[str, Any] = {
            "query": query,
            "limit": min(limit, 100),
            "fields": ",".join(fields or DEFAULT_PAPER_FIELDS),
        }
        if offset is not None:
            params["offset"] = offset
        if year:
            params["year"] = year
        if venue:
            params["venue"] = ",".join(venue)
        if publication_date_or_year:
            params["publicationDateOrYear"] = publication_date_or_year
        if fields_of_study:
            params["fieldsOfStudy"] = fields_of_study
        if publication_types:
            params["publicationTypes"] = publication_types
        if open_access_pdf:
            params["openAccessPdf"] = ""
        if min_citation_count is not None:
            params["minCitationCount"] = min_citation_count
        response = await self._request("GET", "paper/search", params=params)
        return dump_jsonable(SemanticSearchResponse.model_validate(response))

    async def search_papers_bulk(
        self,
        query: str,
        fields: Optional[list[str]] = None,
        token: Optional[str] = None,
        sort: Optional[str] = None,
        limit: int = 100,
        year: Optional[str] = None,
        publication_date_or_year: Optional[str] = None,
        fields_of_study: Optional[str] = None,
        publication_types: Optional[str] = None,
        open_access_pdf: Optional[bool] = None,
        min_citation_count: Optional[int] = None,
    ) -> dict[str, Any]:
        """Bulk paper search with token-based pagination (``/paper/search/bulk``).

        Supports advanced boolean query syntax and provider-sized bulk pages.
        The upstream endpoint may ignore small ``limit`` values and still return
        a full bulk batch, so the client truncates the returned ``data`` list to
        the requested limit after normalization while preserving the returned
        ``token`` for continuation. Repeat the call with the returned ``token``
        until the response contains no token to paginate through up to
        10,000,000 results.

        When custom *fields* are supplied and the upstream endpoint responds with
        400 (e.g. because the bulk endpoint does not support every field that
        ``/paper/search`` accepts), the request is automatically retried with the
        default field set so agents always receive results. A ``fieldsDropped``
        flag and explanatory message are added to the response in that case.
        """
        params: dict[str, Any] = {
            "query": query,
            "limit": min(limit, 1000),
            "fields": ",".join(fields or DEFAULT_PAPER_FIELDS),
        }
        if token:
            params["token"] = token
        if sort:
            params["sort"] = sort
        if year:
            params["year"] = year
        if publication_date_or_year:
            params["publicationDateOrYear"] = publication_date_or_year
        if fields_of_study:
            params["fieldsOfStudy"] = fields_of_study
        if publication_types:
            params["publicationTypes"] = publication_types
        if open_access_pdf:
            params["openAccessPdf"] = ""
        if min_citation_count is not None:
            params["minCitationCount"] = min_citation_count

        fields_dropped: bool = False
        try:
            response = await self._request("GET", "paper/search/bulk", params=params)
        except httpx.HTTPStatusError as exc:
            status_code = self._status_code_from_error(exc)
            if status_code == 400 and fields:
                # The bulk endpoint supports fewer fields than /paper/search.
                # Retry with default fields so agents always get results.
                params["fields"] = ",".join(DEFAULT_PAPER_FIELDS)
                response = await self._request("GET", "paper/search/bulk", params=params)
                fields_dropped = True
            else:
                raise

        parsed = BulkSearchResponse.model_validate(response)
        if len(parsed.data) > limit:
            parsed.data = parsed.data[:limit]
        parsed.data = [self._enrich_bulk_paper(paper) for paper in parsed.data]
        result = dump_jsonable(parsed)
        if fields_dropped:
            result["fieldsDropped"] = True
            result["message"] = (
                "One or more requested fields are not supported by the "
                "Semantic Scholar bulk search endpoint and were dropped. "
                "Results use the default field set. "
                "To retrieve specific fields, use search_papers or "
                "search_papers_semantic_scholar instead."
            )
        return result

    async def search_papers_match(
        self,
        query: str,
        fields: Optional[list[str]] = None,
        *,
        openalex_client: Any | None = None,
        enable_openalex: bool = False,
        crossref_client: Any | None = None,
        enable_crossref: bool = False,
    ) -> dict[str, Any]:
        """Best title-match paper search (``/paper/search/match``).

        Returns the single paper whose title most closely matches *query*.
        Adds ``matchFound`` and ``matchStrategy`` fields to the response so
        agents can distinguish a confirmed match from the structured no-match
        payload returned by the fallback path.

        Tries each variant produced by ``_title_lookup_queries``
        (original, punctuation-normalized, lowercase, lowercase-punct-normalized)
        against the primary ``/paper/search/match`` endpoint so that common
        title-case and punctuation differences do not cause spurious no-match
        results.
        """
        fields_str = ",".join(fields or DEFAULT_PAPER_FIELDS)
        candidate_queries = self._title_lookup_queries(query)
        for candidate_query in candidate_queries:
            params: dict[str, Any] = {
                "query": candidate_query,
                "fields": fields_str,
            }
            try:
                response = await self._request("GET", "paper/search/match", params=params)
            except httpx.HTTPStatusError as exc:
                status_code = self._status_code_from_error(exc)
                if status_code in {400, 404}:
                    continue
                raise
            paper = self._normalize_match_response(response)
            if paper.paper_id is None:
                continue
            result = dump_jsonable(paper)
            result["matchFound"] = True
            result["matchStrategy"] = "exact_title"
            result["matchProvider"] = "semantic_scholar"
            if candidate_query != query:
                result["normalizedQuery"] = candidate_query
            result.update(
                build_match_metadata(
                    query=query,
                    paper=result,
                    candidate_count=1,
                    resolution_strategy="exact_title",
                )
            )
            return result
        return await self._search_papers_match_fallback(
            query,
            fields=fields,
            openalex_client=openalex_client,
            enable_openalex=enable_openalex,
            crossref_client=crossref_client,
            enable_crossref=enable_crossref,
        )

    async def paper_autocomplete(self, query: str) -> dict[str, Any]:
        """Query completion for paper titles (``/paper/autocomplete``).

        Designed for typeahead or interactive search UI.  Returns a list of
        matching title completions; raw API response is returned as-is.
        """
        params: dict[str, Any] = {"query": query}
        return await self._request("GET", "paper/autocomplete", params=params)

    # ------------------------------------------------------------------
    # Paper detail / sub-resources
    # ------------------------------------------------------------------

    async def get_paper_details(
        self,
        paper_id: str,
        fields: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Get paper details (``/paper/{paper_id}``)."""
        normalized_id = self._normalize_paper_id(paper_id)
        params = {"fields": ",".join(fields or DEFAULT_PAPER_FIELDS)}
        try:
            response = await self._request("GET", f"paper/{normalized_id}", params=params)
        except httpx.HTTPStatusError as exc:
            status_code = self._status_code_from_error(exc)
            if status_code == 404:
                raise ValueError(
                    f"Semantic Scholar could not find paper {paper_id!r}"
                    + (f" (normalized to {normalized_id!r})" if normalized_id != paper_id else "")
                    + ". "
                    + self._paper_id_portability_hint()
                ) from exc
            if status_code == 400:
                raise ValueError(
                    f"Semantic Scholar rejected paper identifier {paper_id!r}"
                    + (f" (normalized to {normalized_id!r})" if normalized_id != paper_id else "")
                    + " for get_paper_details. Use a Semantic Scholar-compatible "
                    "paper identifier. " + self._paper_id_portability_hint()
                ) from exc
            raise
        return dump_jsonable(Paper.model_validate(response))

    async def get_paper_citations(
        self,
        paper_id: str,
        limit: int = 100,
        fields: Optional[list[str]] = None,
        offset: Optional[int] = None,
    ) -> dict[str, Any]:
        """Get papers that cite this paper (``/paper/{paper_id}/citations``)."""
        params: dict[str, Any] = {
            "limit": min(limit, 1000),
            "fields": ",".join(fields or DEFAULT_PAPER_FIELDS),
        }
        if offset is not None:
            params["offset"] = offset
        response = await self._request(
            "GET",
            f"paper/{paper_id}/citations",
            params=params,
        )
        return dump_jsonable(self._normalize_nested_paper_list_response(response, "citingPaper"))

    async def get_paper_references(
        self,
        paper_id: str,
        limit: int = 100,
        fields: Optional[list[str]] = None,
        offset: Optional[int] = None,
    ) -> dict[str, Any]:
        """Get paper references (``/paper/{paper_id}/references``)."""
        params: dict[str, Any] = {
            "limit": min(limit, 1000),
            "fields": ",".join(fields or DEFAULT_PAPER_FIELDS),
        }
        if offset is not None:
            params["offset"] = offset
        response = await self._request(
            "GET",
            f"paper/{paper_id}/references",
            params=params,
        )
        return dump_jsonable(self._normalize_nested_paper_list_response(response, "citedPaper"))

    async def get_paper_authors(
        self,
        paper_id: str,
        limit: int = 100,
        fields: Optional[list[str]] = None,
        offset: Optional[int] = None,
    ) -> dict[str, Any]:
        """Get authors of a paper (``/paper/{paper_id}/authors``)."""
        params: dict[str, Any] = {
            "limit": min(limit, 1000),
            "fields": ",".join(fields or DEFAULT_AUTHOR_FIELDS),
        }
        if offset is not None:
            params["offset"] = offset
        try:
            response = await self._request(
                "GET",
                f"paper/{paper_id}/authors",
                params=params,
            )
        except httpx.HTTPStatusError as exc:
            status_code = self._status_code_from_error(exc)
            if status_code == 404:
                raise ValueError(
                    f"Semantic Scholar could not find paper {paper_id!r} for "
                    "get_paper_authors. This tool only accepts Semantic "
                    "Scholar-compatible paper identifiers. "
                    f"{self._paper_id_portability_hint()}"
                ) from exc
            if status_code == 400:
                raise ValueError(
                    f"Semantic Scholar rejected paper identifier {paper_id!r} for "
                    "get_paper_authors. Use a Semantic Scholar-compatible paper "
                    f"identifier. {self._paper_id_portability_hint()}"
                ) from exc
            raise
        return dump_jsonable(PaperAuthorListResponse.model_validate(response))

    # ------------------------------------------------------------------
    # Author endpoints
    # ------------------------------------------------------------------

    async def get_author_info(
        self,
        author_id: str,
        fields: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Get author info (``/author/{author_id}``)."""
        params = {"fields": ",".join(fields or DEFAULT_AUTHOR_FIELDS)}
        try:
            response = await self._request("GET", f"author/{author_id}", params=params)
        except httpx.HTTPStatusError as exc:
            status_code = self._status_code_from_error(exc)
            if status_code == 400:
                raise ValueError(
                    f"Semantic Scholar rejected get_author_info for author "
                    f"{author_id!r}. Requested author fields: {params['fields']}. "
                    "Use only supported author fields and pass a Semantic Scholar "
                    "authorId returned by search_authors or get_paper_authors."
                ) from exc
            if status_code == 404:
                raise ValueError(
                    f"Semantic Scholar could not find author {author_id!r}. "
                    "Use a Semantic Scholar authorId returned by search_authors "
                    "or get_paper_authors."
                ) from exc
            raise
        return dump_jsonable(AuthorProfile.model_validate(response))

    async def get_author_papers(
        self,
        author_id: str,
        limit: int = 100,
        fields: Optional[list[str]] = None,
        offset: Optional[int] = None,
        publication_date_or_year: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get papers by an author (``/author/{author_id}/papers``)."""
        params: dict[str, Any] = {
            "limit": min(limit, 1000),
            "fields": ",".join(fields or DEFAULT_PAPER_FIELDS),
        }
        if offset is not None:
            params["offset"] = offset
        if publication_date_or_year:
            # publicationDateOrYear uses ':' as the range separator (e.g. "2022:"),
            # not '-' which is the year-parameter style (e.g. "2022-").
            # Normalize a single trailing '-' to ':' so the common shorthand
            # "2022-" or "2022-03-05-" works the same as the API-native
            # "2022:" or "2022-03-05:".
            normalized_date = (
                publication_date_or_year[:-1] + ":"
                if publication_date_or_year.endswith("-")
                else publication_date_or_year
            )
            params["publicationDateOrYear"] = normalized_date
        try:
            response = await self._request(
                "GET",
                f"author/{author_id}/papers",
                params=params,
            )
        except httpx.HTTPStatusError as exc:
            status_code = self._status_code_from_error(exc)
            if status_code == 400:
                filter_hint = (
                    f" Check that publicationDateOrYear={publication_date_or_year!r} "
                    "uses a valid format (e.g. '2022', '2022:', '2022-03-05:', "
                    "'2020-01-01:2023-12-31'). Open-ended ranges require a trailing "
                    "colon ('2022:'), not a trailing hyphen."
                    if publication_date_or_year
                    else " Use a Semantic Scholar authorId returned by search_authors or get_paper_authors."
                )
                raise ValueError(
                    f"Semantic Scholar rejected get_author_papers for author {author_id!r}.{filter_hint}"
                ) from exc
            if status_code == 404:
                raise ValueError(
                    f"Semantic Scholar could not find author {author_id!r} for "
                    "get_author_papers. Use a Semantic Scholar authorId returned by "
                    "search_authors or get_paper_authors."
                ) from exc
            raise
        return dump_jsonable(PaperListResponse.model_validate(response))

    async def search_authors(
        self,
        query: str,
        limit: int = 10,
        fields: Optional[list[str]] = None,
        offset: Optional[int] = None,
    ) -> dict[str, Any]:
        """Search for authors by name (``/author/search``)."""
        normalized_query = self._normalize_author_search_query(query)
        params: dict[str, Any] = {
            "query": normalized_query,
            "limit": min(limit, 1000),
            "fields": ",".join(fields or DEFAULT_AUTHOR_FIELDS),
        }
        if offset is not None:
            params["offset"] = offset
        try:
            response = await self._request("GET", "author/search", params=params)
        except httpx.HTTPStatusError as exc:
            if self._status_code_from_error(exc) == 400:
                raise ValueError(
                    f"Semantic Scholar rejected author search query {query!r}. "
                    "search_authors only supports plain-text name queries. "
                    f"The MCP server normalized the query to {normalized_query!r}; "
                    "if the request still fails, retry without quotes, boolean "
                    "operators, or other special syntax. For common names, add "
                    "affiliation, coauthor, venue, or topic clues and then confirm "
                    "the best candidate with get_author_info/get_author_papers."
                ) from exc
            raise
        return dump_jsonable(AuthorListResponse.model_validate(response))

    async def batch_get_authors(
        self,
        author_ids: list[str],
        fields: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Batch fetch up to 1,000 authors (``POST /author/batch``)."""
        json_data = {"ids": author_ids[:1000]}
        params = {"fields": ",".join(fields or DEFAULT_AUTHOR_FIELDS)}
        response = await self._request(
            "POST",
            "author/batch",
            params=params,
            json_data=json_data,
        )
        return dump_jsonable(BatchAuthorResponse.model_validate(response))

    # ------------------------------------------------------------------
    # Snippet search
    # ------------------------------------------------------------------

    async def search_snippets(
        self,
        query: str,
        limit: int = 10,
        fields: Optional[list[str]] = None,
        year: Optional[str] = None,
        publication_date_or_year: Optional[str] = None,
        fields_of_study: Optional[str] = None,
        min_citation_count: Optional[int] = None,
        venue: Optional[str] = None,
    ) -> dict[str, Any]:
        """Search for matching text snippets (``/snippet/search``)."""
        params: dict[str, Any] = {
            "query": query,
            "limit": min(limit, 100),
        }
        if fields:
            params["fields"] = ",".join(fields)
        if year:
            params["year"] = year
        if publication_date_or_year:
            params["publicationDateOrYear"] = publication_date_or_year
        if fields_of_study:
            params["fieldsOfStudy"] = fields_of_study
        if min_citation_count is not None:
            params["minCitationCount"] = min_citation_count
        if venue:
            params["venue"] = venue
        try:
            response = await self._request(
                "GET",
                "snippet/search",
                params=params,
                max_429_retries=0,
            )
        except httpx.HTTPStatusError as exc:
            status_code = self._status_code_from_error(exc)
            if status_code in {400, 404, 429, 500, 502, 503, 504}:
                payload = dump_jsonable(SnippetSearchResponse(data=[]))
                payload["query"] = query
                payload["degraded"] = True
                payload["message"] = (
                    "Semantic Scholar snippet search could not serve this query, so "
                    "the server returned an empty result instead of surfacing the raw "
                    "provider error. Retry with a shorter plain-text phrase or fall "
                    "back to search_papers_match/search_papers."
                )
                payload["providerStatusCode"] = status_code
                return payload
            raise
        return dump_jsonable(SnippetSearchResponse.model_validate(response))

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    async def get_recommendations(
        self,
        paper_id: str,
        limit: int = 10,
        fields: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Get paper recommendations from one seed paper (GET).

        Uses ``/recommendations/v1/papers/forpaper/{paper_id}``.
        """
        params: dict[str, Any] = {
            "limit": min(limit, 100),
            "fields": ",".join(fields or DEFAULT_PAPER_FIELDS),
        }
        response = await self._request(
            "GET",
            f"papers/forpaper/{paper_id}",
            params=params,
            base_url=RECOMMENDATIONS_BASE_URL,
        )
        return dump_jsonable(RecommendationResponse.model_validate(response))

    async def get_recommendations_post(
        self,
        positive_paper_ids: list[str],
        negative_paper_ids: Optional[list[str]] = None,
        limit: int = 10,
        fields: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Get recommendations from positive and negative seed sets (POST).

        Uses ``POST /recommendations/v1/papers``.  More flexible than the single-seed
        GET route and typically produces better results for guided retrieval.
        """
        json_data: dict[str, Any] = {
            "positivePaperIds": positive_paper_ids,
        }
        if negative_paper_ids:
            json_data["negativePaperIds"] = negative_paper_ids
        params: dict[str, Any] = {
            "limit": min(limit, 100),
            "fields": ",".join(fields or DEFAULT_PAPER_FIELDS),
        }
        response = await self._request(
            "POST",
            "papers",
            params=params,
            json_data=json_data,
            base_url=RECOMMENDATIONS_BASE_URL,
        )
        return dump_jsonable(RecommendationResponse.model_validate(response))

    # ------------------------------------------------------------------
    # Batch paper lookup
    # ------------------------------------------------------------------

    async def batch_get_papers(
        self,
        paper_ids: list[str],
        fields: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Batch fetch up to 500 papers (``POST /paper/batch``)."""
        json_data = {"ids": paper_ids[:500]}
        params = {"fields": ",".join(fields or DEFAULT_PAPER_FIELDS)}
        response = await self._request(
            "POST",
            "paper/batch",
            params=params,
            json_data=json_data,
        )
        return dump_jsonable(BatchPaperResponse.model_validate(response))
