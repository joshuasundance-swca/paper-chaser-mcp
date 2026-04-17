"""Typed tool argument models and schema registry."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..constants import SUPPORTED_AUTHOR_FIELDS, SUPPORTED_PAPER_FIELDS
from .ecos import EcosDocumentKind


class ToolArgsModel(BaseModel):
    """Base MCP tool input model."""

    model_config = ConfigDict(extra="forbid")


OPAQUE_CURSOR_FIELD_DESCRIPTION = (
    "Continuation cursor from a previous response's pagination.nextCursor "
    "(an opaque server-issued cursor). Pass it back exactly as returned; do "
    "not derive, edit, or fabricate it, and do not reuse it across a "
    "different tool or query flow. Omit to start from the beginning."
)

KnownItemResolutionState = Literal[
    "resolved_exact",
    "resolved_probable",
    "needs_disambiguation",
]
"""Execution-provenance label for known-item / resolve_reference outcomes.

- ``resolved_exact``: a single high-confidence match (identifier round-trip, or
  exact-title match with strongly corroborating author/year fields).
- ``resolved_probable``: one best match but with weaker agreement (fuzzy title,
  mid-range confidence, or a single conflicting metadata field).
- ``needs_disambiguation``: multiple near-tie candidates, no best match, or a
  lone candidate that fails the probable threshold."""


SUPPORTED_AUTHOR_FIELDS_TEXT = ", ".join(SUPPORTED_AUTHOR_FIELDS)
SUPPORTED_PAPER_FIELDS_TEXT = ", ".join(SUPPORTED_PAPER_FIELDS)
AUTHOR_FIELDS_DESCRIPTION = "Fields to return. Supported values: " + SUPPORTED_AUTHOR_FIELDS_TEXT
PAPER_FIELDS_DESCRIPTION = "Paper fields to return. Supported values: " + SUPPORTED_PAPER_FIELDS_TEXT
AUTHOR_ID_DESCRIPTION = "Semantic Scholar author ID from search_authors or get_paper_authors"
AUTHOR_SEARCH_QUERY_DESCRIPTION = (
    "Author name to search for. Plain-text only; initials and exact-name "
    "punctuation may be normalized before the upstream request."
)
OPENALEX_WORK_ID_DESCRIPTION = (
    "OpenAlex work identifier for OpenAlex-specific tools. Accepts an OpenAlex W-id, OpenAlex work URL, or DOI."
)
OPENALEX_AUTHOR_ID_DESCRIPTION = (
    "OpenAlex author identifier for OpenAlex-specific tools. Accepts an OpenAlex A-id or OpenAlex author URL."
)
SEMANTIC_SCHOLAR_EXPANSION_PAPER_ID_DESCRIPTION = (
    "Paper identifier for Semantic Scholar expansion tools. Prefer "
    "paper.recommendedExpansionId when brokered search results provide it. If "
    "paper.expansionIdStatus is not_portable, do not reuse brokered "
    "paperId/sourceId/canonicalId values directly; resolve the paper through a "
    "DOI or Semantic Scholar-native lookup first."
)
ENRICHMENT_PAPER_ID_DESCRIPTION = (
    "Existing paper identifier for enrichment tools. Accepts a bare DOI, DOI URL, "
    "canonicalId, recommendedExpansionId, or any paperId that already embeds a DOI."
)
DOI_INPUT_DESCRIPTION = "Bare DOI or DOI URL. When both doi and paper_id are supplied, the explicit doi wins."
INCLUDE_ENRICHMENT_DESCRIPTION = (
    "When true, add post-resolution Crossref, Unpaywall, and OpenAlex enrichment "
    "to the final resolved paper output. Matching, ranking, and retrieval behavior "
    "stay unchanged."
)


def _clamp_limit(value: int | None, default: int, maximum: int) -> int:
    if value is None:
        return default
    return min(max(int(value), 1), maximum)


def _validate_author_fields(fields: list[str] | None) -> list[str] | None:
    if fields is None:
        return None
    unsupported = [field for field in fields if field not in SUPPORTED_AUTHOR_FIELDS]
    if unsupported:
        raise ValueError(
            "Unsupported author fields: "
            + ", ".join(unsupported)
            + ". Supported values: "
            + SUPPORTED_AUTHOR_FIELDS_TEXT
            + "."
        )
    return fields


def _validate_paper_fields(fields: list[str] | None) -> list[str] | None:
    if fields is None:
        return None
    unsupported = [field for field in fields if field not in SUPPORTED_PAPER_FIELDS]
    if unsupported:
        raise ValueError(
            "Unsupported paper fields: "
            + ", ".join(unsupported)
            + ". Supported values: "
            + SUPPORTED_PAPER_FIELDS_TEXT
            + "."
        )
    return fields


SearchProvider = Literal[
    "core",
    "semantic_scholar",
    "serpapi_google_scholar",
    "scholarapi",
    "arxiv",
]
OpenAlexEntityType = Literal["source", "institution", "topic"]
LatencyProfile = Literal["fast", "balanced", "deep"]

SEARCH_PROVIDER_ALIASES: dict[str, SearchProvider] = {
    "core": "core",
    "semantic_scholar": "semantic_scholar",
    "serpapi": "serpapi_google_scholar",
    "serpapi_google_scholar": "serpapi_google_scholar",
    "scholarapi": "scholarapi",
    "arxiv": "arxiv",
}

DEFAULT_SEARCH_PROVIDER_ORDER: tuple[SearchProvider, ...] = (
    "semantic_scholar",
    "arxiv",
    "core",
    "serpapi_google_scholar",
)


def _supported_provider_names() -> str:
    return ", ".join(SEARCH_PROVIDER_ALIASES)


def _normalize_provider_name(value: object) -> SearchProvider:
    if not isinstance(value, str):
        raise ValueError("Provider names must be strings. Supported values: " + _supported_provider_names())
    normalized = value.strip().lower()
    try:
        return SEARCH_PROVIDER_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported provider {value!r}. Supported values: {_supported_provider_names()}.") from exc


def _validate_provider_order(
    value: list[object] | None,
) -> list[SearchProvider] | None:
    if value is None:
        return None
    if not value:
        raise ValueError("Provider order must contain at least one provider")
    seen: set[SearchProvider] = set()
    duplicates: list[SearchProvider] = []
    normalized_providers = [_normalize_provider_name(provider) for provider in value]
    for provider in normalized_providers:
        if provider in seen:
            duplicates.append(provider)
        seen.add(provider)
    if duplicates:
        duplicate_text = ", ".join(duplicates)
        raise ValueError(f"Provider order cannot repeat providers: {duplicate_text}")
    return normalized_providers


class ResearchArgs(ToolArgsModel):
    query: str = Field(description="Natural-language research request or topic to investigate.")
    limit: int = Field(
        default=5,
        description="Max evidence-bearing sources to keep in the guided result set (default 5, max 10).",
    )
    year: str | None = Field(default=None, description="Optional year or year range hint, e.g. 2022:2025.")
    venue: str | None = Field(default=None, description="Optional venue hint to narrow the request.")
    focus: str | None = Field(default=None, description="Optional focus hint to steer the research goal.")

    @model_validator(mode="before")
    @classmethod
    def drop_deprecated_latency_profile(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        normalized.pop("latencyProfile", None)
        normalized.pop("latency_profile", None)
        return normalized

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 5, 10)


class FollowUpResearchArgs(ToolArgsModel):
    search_session_id: str | None = Field(
        default=None,
        alias="searchSessionId",
        description=(
            "Optional searchSessionId returned by the guided research tool. "
            "When omitted, the server will reuse the most recent compatible saved session only if that choice is "
            "unambiguous."
        ),
    )
    question: str = Field(description="Grounded follow-up question about the saved research result set.")


class ResolveReferenceArgs(ToolArgsModel):
    reference: str = Field(description="Citation, DOI, URL, title fragment, or regulatory reference to resolve.")


class InspectSourceArgs(ToolArgsModel):
    search_session_id: str | None = Field(
        default=None,
        alias="searchSessionId",
        description=(
            "Optional searchSessionId returned by the guided research tool. "
            "When omitted, the server will reuse the most recent compatible saved session only if that choice is "
            "unambiguous."
        ),
    )
    source_id: str = Field(
        alias="sourceId",
        description="Canonical sourceId or session-local source alias returned in the guided research sources list.",
    )


class GetRuntimeStatusArgs(ToolArgsModel):
    pass


class BasicSearchPapersArgs(ToolArgsModel):
    query: str = Field(description="Search query")
    limit: int = Field(default=10, description="Max results (default 10, max 100)")
    year: str | None = Field(
        default=None,
        description="Year filter, e.g. '2020-2023' or '2023'",
    )

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 10, 100)


class SearchPapersBaseArgs(BasicSearchPapersArgs):
    fields: list[str] | None = Field(default=None, description="Fields to return")
    venue: list[str] | None = Field(
        default=None,
        description="Venue names to filter",
    )
    publication_date_or_year: str | None = Field(
        default=None,
        alias="publicationDateOrYear",
        description=("Date or date-range filter, e.g. '2019-03-05', '2016:2020', '2010-'"),
    )
    fields_of_study: str | None = Field(
        default=None,
        alias="fieldsOfStudy",
        description="Comma-separated fields of study filter",
    )
    publication_types: str | None = Field(
        default=None,
        alias="publicationTypes",
        description="Comma-separated publication types filter",
    )
    open_access_pdf: bool | None = Field(
        default=None,
        alias="openAccessPdf",
        description="Only return papers with a public PDF",
    )
    min_citation_count: int | None = Field(
        default=None,
        alias="minCitationCount",
        description="Minimum citation count filter",
    )


class SearchPapersArgs(SearchPapersBaseArgs):
    preferred_provider: SearchProvider | None = Field(
        default=None,
        alias="preferredProvider",
        description=(
            "Optional provider to try first before continuing the broker fallback "
            "chain. One of: core, semantic_scholar, serpapi, "
            "serpapi_google_scholar, scholarapi, arxiv."
        ),
    )
    provider_order: list[SearchProvider] | None = Field(
        default=None,
        alias="providerOrder",
        description=(
            "Optional ordered provider chain override for this call. Defaults to "
            "semantic_scholar, arxiv, core, serpapi_google_scholar. Omit providers "
            "to skip them for this request. `serpapi` is accepted as a shorthand "
            "for `serpapi_google_scholar`. ScholarAPI is supported as an explicit "
            "opt-in broker target via `scholarapi`, but it is not part of the default order."
        ),
    )

    @field_validator("preferred_provider", mode="before")
    @classmethod
    def normalize_preferred_provider(cls, value: object) -> SearchProvider | None:
        if value is None:
            return None
        return _normalize_provider_name(value)

    @field_validator("provider_order", mode="before")
    @classmethod
    def validate_provider_order(cls, value: list[object] | None) -> list[SearchProvider] | None:
        return _validate_provider_order(value)


class ProviderSearchPapersArgs(SearchPapersBaseArgs):
    """Shared provider-specific single-source paper search arguments."""


class MinimalProviderSearchPapersArgs(BasicSearchPapersArgs):
    """Provider-specific search args for backends that only honor query/year."""


class SemanticProviderSearchPapersArgs(SearchPapersBaseArgs):
    """Semantic Scholar single-source search arguments."""


class ScholarApiSearchArgs(ToolArgsModel):
    query: str = Field(description="ScholarAPI search query")
    limit: int = Field(default=10, description="Max results (default 10, max 1000)")
    cursor: str | None = Field(default=None, description=OPAQUE_CURSOR_FIELD_DESCRIPTION)
    indexed_after: str | None = Field(
        default=None,
        description="Optional RFC3339 UTC lower bound for indexed_at.",
    )
    indexed_before: str | None = Field(
        default=None,
        description="Optional RFC3339 UTC upper bound for indexed_at.",
    )
    published_after: str | None = Field(
        default=None,
        description="Optional ISO date lower bound for published_date.",
    )
    published_before: str | None = Field(
        default=None,
        description="Optional ISO date upper bound for published_date.",
    )
    has_text: bool | None = Field(
        default=None,
        description="When true, only return records with full text available.",
    )
    has_pdf: bool | None = Field(
        default=None,
        description="When true, only return records with PDF available.",
    )

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 10, 1000)


class ScholarApiListArgs(ToolArgsModel):
    query: str | None = Field(default=None, description="Optional ScholarAPI keyword or phrase filter.")
    limit: int = Field(default=100, description="Max results (default 100, max 1000)")
    cursor: str | None = Field(default=None, description=OPAQUE_CURSOR_FIELD_DESCRIPTION)
    indexed_after: str | None = Field(
        default=None,
        description="Optional RFC3339 UTC lower bound for indexed_at.",
    )
    indexed_before: str | None = Field(
        default=None,
        description="Optional RFC3339 UTC upper bound for indexed_at.",
    )
    published_after: str | None = Field(
        default=None,
        description="Optional ISO date lower bound for published_date.",
    )
    published_before: str | None = Field(
        default=None,
        description="Optional ISO date upper bound for published_date.",
    )
    has_text: bool | None = Field(
        default=None,
        description="When true, only return records with full text available.",
    )
    has_pdf: bool | None = Field(
        default=None,
        description="When true, only return records with PDF available.",
    )

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 100, 1000)


class OpenAlexBulkSearchPapersArgs(BasicSearchPapersArgs):
    limit: int = Field(default=100, description="Max results (default 100, max 200)")
    cursor: str | None = Field(
        default=None,
        description=OPAQUE_CURSOR_FIELD_DESCRIPTION,
    )

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 100, 200)


class BulkSearchPapersArgs(ToolArgsModel):
    query: str = Field(
        description=(
            "Search query; supports boolean/fuzzy syntax: +AND, |OR, -negate, "
            '"phrases", prefix*, ~N edit-distance, (precedence)'
        )
    )
    fields: list[str] | None = Field(default=None, description="Fields to return")
    cursor: str | None = Field(
        default=None,
        description=(
            "Continuation cursor from a previous response's pagination.nextCursor "
            "(an opaque server-issued cursor for this bulk-search query). Pass it "
            "back exactly as returned; do not derive, edit, or fabricate it, and "
            "do not reuse it across a different query flow. Omit to start a new "
            "search."
        ),
    )
    sort: str | None = Field(
        default=None,
        description=("Sort order, e.g. 'paperId:asc', 'citationCount:desc', 'publicationDate:desc'"),
    )
    year: str | None = Field(
        default=None,
        description="Year filter, e.g. '2020-2023' or '2023'",
    )
    publication_date_or_year: str | None = Field(
        default=None,
        alias="publicationDateOrYear",
        description="Date or date-range filter",
    )
    fields_of_study: str | None = Field(
        default=None,
        alias="fieldsOfStudy",
        description="Comma-separated fields of study filter",
    )
    publication_types: str | None = Field(
        default=None,
        alias="publicationTypes",
        description="Comma-separated publication types filter",
    )
    open_access_pdf: bool | None = Field(
        default=None,
        alias="openAccessPdf",
        description="Only return papers with a public PDF",
    )
    min_citation_count: int | None = Field(
        default=None,
        alias="minCitationCount",
        description="Minimum citation count filter",
    )
    limit: int = Field(
        default=100,
        description=(
            "Max papers returned per call (default 100, max 1000). The upstream "
            "Semantic Scholar bulk endpoint may still fetch its larger provider "
            "batch internally, so prefer search_papers or "
            "search_papers_semantic_scholar for small targeted pages."
        ),
    )

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 100, 1000)


class PaperMatchArgs(ToolArgsModel):
    query: str = Field(description="Paper title to find the best match for")
    fields: list[str] | None = Field(default=None, description="Fields to return")
    include_enrichment: bool = Field(
        default=False,
        alias="includeEnrichment",
        description=INCLUDE_ENRICHMENT_DESCRIPTION,
    )


class ResolveCitationArgs(ToolArgsModel):
    citation: str = Field(
        description=(
            "Partial, malformed, or almost-right citation text to repair. "
            "Can include DOI/arXiv/URL fragments, author names, venue hints, "
            "quote fragments, or approximate years."
        )
    )
    max_candidates: int = Field(
        default=5,
        alias="maxCandidates",
        description="Maximum ranked candidates to return (default 5, max 5).",
    )
    title_hint: str | None = Field(
        default=None,
        alias="titleHint",
        description="Optional title fragment hint when the citation text is sparse.",
    )
    author_hint: str | None = Field(
        default=None,
        alias="authorHint",
        description="Optional author surname or author-name hint.",
    )
    year_hint: str | None = Field(
        default=None,
        alias="yearHint",
        description="Optional approximate year hint, e.g. '2009' or 'around 2001'.",
    )
    venue_hint: str | None = Field(
        default=None,
        alias="venueHint",
        description="Optional venue or journal hint.",
    )
    doi_hint: str | None = Field(
        default=None,
        alias="doiHint",
        description="Optional DOI or DOI fragment hint.",
    )
    include_enrichment: bool = Field(
        default=False,
        alias="includeEnrichment",
        description=INCLUDE_ENRICHMENT_DESCRIPTION,
    )

    @field_validator("max_candidates", mode="before")
    @classmethod
    def clamp_max_candidates(cls, value: int | None) -> int:
        return _clamp_limit(value, 5, 5)


class PaperAutocompleteArgs(ToolArgsModel):
    query: str = Field(description="Partial paper title for typeahead completion")


class OpenAlexPaperAutocompleteArgs(ToolArgsModel):
    query: str = Field(description="Partial paper title for OpenAlex typeahead.")
    limit: int = Field(
        default=10,
        description="Max matches (default 10, max 20).",
    )

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 10, 20)


class PaperLookupArgs(ToolArgsModel):
    paper_id: str = Field(description="Paper ID (DOI, ArXiv ID, S2 ID, etc.)")
    fields: list[str] | None = Field(default=None, description="Fields to return")
    include_enrichment: bool = Field(
        default=False,
        alias="includeEnrichment",
        description=INCLUDE_ENRICHMENT_DESCRIPTION,
    )


class PaperEnrichmentLookupArgs(ToolArgsModel):
    paper_id: str | None = Field(
        default=None,
        description=ENRICHMENT_PAPER_ID_DESCRIPTION,
    )
    doi: str | None = Field(
        default=None,
        description=DOI_INPUT_DESCRIPTION,
    )


class ScholarApiPaperTextArgs(ToolArgsModel):
    paper_id: str = Field(
        description=(
            "ScholarAPI paper id for full-text retrieval. Accepts either a raw id "
            "or a ScholarAPI:<id> value returned by search."
        )
    )


class ScholarApiPaperTextsArgs(ToolArgsModel):
    paper_ids: list[str] = Field(
        description=(
            "ScholarAPI paper ids to retrieve as full text (max 100). Each item may be "
            "a raw id or a ScholarAPI:<id> value returned by search."
        )
    )

    @field_validator("paper_ids")
    @classmethod
    def validate_paper_ids(cls, value: list[str]) -> list[str]:
        if len(value) > 100:
            raise ValueError(f"Maximum 100 ScholarAPI paper IDs per batch request, got {len(value)}")
        return value


class ScholarApiPaperPdfArgs(ToolArgsModel):
    paper_id: str = Field(
        description=(
            "ScholarAPI paper id for PDF retrieval. Accepts either a raw id or a "
            "ScholarAPI:<id> value returned by search."
        )
    )


class CrossrefEnrichmentArgs(PaperEnrichmentLookupArgs):
    query: str | None = Field(
        default=None,
        description=(
            "Optional title or bibliographic query fallback used only when no DOI can be resolved from the inputs."
        ),
    )

    @model_validator(mode="after")
    def require_lookup_material(self) -> "CrossrefEnrichmentArgs":
        if self.paper_id or self.doi or self.query:
            return self
        raise ValueError("Provide at least one of paper_id, doi, or query.")


class UnpaywallEnrichmentArgs(PaperEnrichmentLookupArgs):
    @model_validator(mode="after")
    def require_identifier(self) -> "UnpaywallEnrichmentArgs":
        if self.paper_id or self.doi:
            return self
        raise ValueError("Provide paper_id or doi for Unpaywall lookups.")


class EnrichPaperArgs(CrossrefEnrichmentArgs):
    """Run Crossref, Unpaywall, and OpenAlex enrichment for one known paper or DOI."""


class OpenAlexPaperLookupArgs(ToolArgsModel):
    paper_id: str = Field(description=OPENALEX_WORK_ID_DESCRIPTION)


class OpenAlexPaperListArgs(OpenAlexPaperLookupArgs):
    limit: int = Field(
        default=100,
        description="Max results (default 100, max 200)",
    )
    cursor: str | None = Field(
        default=None,
        description=OPAQUE_CURSOR_FIELD_DESCRIPTION,
    )

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 100, 200)


class PaperListArgs(PaperLookupArgs):
    paper_id: str = Field(description=SEMANTIC_SCHOLAR_EXPANSION_PAPER_ID_DESCRIPTION)
    limit: int = Field(
        default=100,
        description="Max results (default 100, max 1000)",
    )
    cursor: str | None = Field(
        default=None,
        description=OPAQUE_CURSOR_FIELD_DESCRIPTION,
    )

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 100, 1000)


class PaperAuthorsArgs(ToolArgsModel):
    paper_id: str = Field(description=SEMANTIC_SCHOLAR_EXPANSION_PAPER_ID_DESCRIPTION)
    fields: list[str] | None = Field(
        default=None,
        description=AUTHOR_FIELDS_DESCRIPTION,
    )
    limit: int = Field(
        default=100,
        description="Max results (default 100, max 1000)",
    )
    cursor: str | None = Field(
        default=None,
        description=OPAQUE_CURSOR_FIELD_DESCRIPTION,
    )

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 100, 1000)

    @field_validator("fields")
    @classmethod
    def validate_fields(cls, value: list[str] | None) -> list[str] | None:
        return _validate_author_fields(value)


class AuthorInfoArgs(ToolArgsModel):
    author_id: str = Field(description=AUTHOR_ID_DESCRIPTION)
    fields: list[str] | None = Field(
        default=None,
        description=AUTHOR_FIELDS_DESCRIPTION,
    )

    @field_validator("fields")
    @classmethod
    def validate_fields(cls, value: list[str] | None) -> list[str] | None:
        return _validate_author_fields(value)


class AuthorPapersArgs(AuthorInfoArgs):
    fields: list[str] | None = Field(
        default=None,
        description=PAPER_FIELDS_DESCRIPTION,
    )
    limit: int = Field(
        default=100,
        description="Max results (default 100, max 1000)",
    )
    cursor: str | None = Field(
        default=None,
        description=OPAQUE_CURSOR_FIELD_DESCRIPTION,
    )
    publication_date_or_year: str | None = Field(
        default=None,
        alias="publicationDateOrYear",
        description=(
            "Date or date-range filter. Accepted formats: '2019' (year), "
            "'2019-03-05' (date), '2022:' (from 2022 onwards), "
            "':2021' (up to 2021), '2020:2023' (range), "
            "'2020-06-01:2023-12-31' (date range). "
            "Note: use a trailing colon for open-ended ranges ('2022:'), "
            "not a trailing hyphen. A trailing hyphen (e.g. '2022-' or "
            "'2022-03-05-') is automatically normalized to the correct "
            "colon form ('2022:' or '2022-03-05:')."
        ),
    )

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 100, 1000)

    @field_validator("fields")
    @classmethod
    def validate_fields(cls, value: list[str] | None) -> list[str] | None:
        return _validate_paper_fields(value)


class OpenAlexAuthorInfoArgs(ToolArgsModel):
    author_id: str = Field(description=OPENALEX_AUTHOR_ID_DESCRIPTION)


class OpenAlexAuthorPapersArgs(OpenAlexAuthorInfoArgs):
    limit: int = Field(
        default=100,
        description="Max results (default 100, max 200)",
    )
    cursor: str | None = Field(
        default=None,
        description=OPAQUE_CURSOR_FIELD_DESCRIPTION,
    )
    year: str | None = Field(
        default=None,
        description="Year filter, e.g. '2020-2023' or '2023'",
    )

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 100, 200)


class AuthorSearchArgs(ToolArgsModel):
    query: str = Field(description=AUTHOR_SEARCH_QUERY_DESCRIPTION)
    fields: list[str] | None = Field(
        default=None,
        description=AUTHOR_FIELDS_DESCRIPTION,
    )
    limit: int = Field(
        default=10,
        description="Max results (default 10, max 1000)",
    )
    cursor: str | None = Field(
        default=None,
        description=OPAQUE_CURSOR_FIELD_DESCRIPTION,
    )

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 10, 1000)

    @field_validator("fields")
    @classmethod
    def validate_fields(cls, value: list[str] | None) -> list[str] | None:
        return _validate_author_fields(value)


class OpenAlexAuthorSearchArgs(ToolArgsModel):
    query: str = Field(description="Author name to search in OpenAlex.")
    limit: int = Field(
        default=10,
        description="Max results (default 10, max 200)",
    )
    cursor: str | None = Field(
        default=None,
        description=OPAQUE_CURSOR_FIELD_DESCRIPTION,
    )

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 10, 200)


class OpenAlexEntitySearchArgs(ToolArgsModel):
    query: str = Field(description="Entity search query for OpenAlex.")
    entity_type: OpenAlexEntityType = Field(
        alias="entityType",
        description="Entity family to search: source, institution, or topic.",
    )
    limit: int = Field(
        default=10,
        description="Max results (default 10, max 50).",
    )
    cursor: str | None = Field(
        default=None,
        description=OPAQUE_CURSOR_FIELD_DESCRIPTION,
    )

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 10, 50)


class OpenAlexEntityPaperSearchArgs(ToolArgsModel):
    entity_type: OpenAlexEntityType = Field(
        alias="entityType",
        description="Entity family to pivot from: source, institution, or topic.",
    )
    entity_id: str = Field(
        alias="entityId",
        description="OpenAlex entity identifier to pivot from.",
    )
    limit: int = Field(
        default=100,
        description="Max papers (default 100, max 200).",
    )
    cursor: str | None = Field(
        default=None,
        description=OPAQUE_CURSOR_FIELD_DESCRIPTION,
    )
    year: str | None = Field(
        default=None,
        description="Optional year filter, e.g. '2023' or '2020-2024'.",
    )

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 100, 200)


class BatchGetAuthorsArgs(ToolArgsModel):
    author_ids: list[str] = Field(description="List of author IDs (up to 1000)")
    fields: list[str] | None = Field(
        default=None,
        description=AUTHOR_FIELDS_DESCRIPTION,
    )

    @field_validator("author_ids")
    @classmethod
    def validate_author_ids(cls, value: list[str]) -> list[str]:
        if len(value) > 1000:
            raise ValueError(f"Maximum 1000 author IDs per batch request, got {len(value)}")
        return value

    @field_validator("fields")
    @classmethod
    def validate_fields(cls, value: list[str] | None) -> list[str] | None:
        return _validate_author_fields(value)


class SnippetSearchArgs(ToolArgsModel):
    query: str = Field(description="Text snippet to search for")
    fields: list[str] | None = Field(default=None, description="Fields to return")
    limit: int = Field(
        default=10,
        description="Max results (default 10, max 100)",
    )
    year: str | None = Field(
        default=None,
        description="Year filter, e.g. '2020-2023' or '2023'",
    )
    publication_date_or_year: str | None = Field(
        default=None,
        alias="publicationDateOrYear",
        description="Date or date-range filter",
    )
    fields_of_study: str | None = Field(
        default=None,
        alias="fieldsOfStudy",
        description="Comma-separated fields of study filter",
    )
    min_citation_count: int | None = Field(
        default=None,
        alias="minCitationCount",
        description="Minimum citation count filter",
    )
    venue: str | None = Field(
        default=None,
        description="Comma-separated venue filter",
    )

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 10, 100)


class RecommendationArgs(PaperLookupArgs):
    limit: int = Field(default=10, description="Max results (default 10, max 100)")

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 10, 100)


class PostRecommendationsArgs(ToolArgsModel):
    positive_paper_ids: list[str] = Field(
        alias="positivePaperIds",
        description="Paper IDs to use as positive seeds",
    )
    negative_paper_ids: list[str] | None = Field(
        default=None,
        alias="negativePaperIds",
        description="Paper IDs to use as negative seeds (optional)",
    )
    fields: list[str] | None = Field(default=None, description="Fields to return")
    limit: int = Field(default=10, description="Max results (default 10, max 100)")

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 10, 100)


class BatchGetPapersArgs(ToolArgsModel):
    paper_ids: list[str] = Field(description="List of paper IDs (up to 500)")
    fields: list[str] | None = Field(default=None, description="Fields to return")

    @field_validator("paper_ids")
    @classmethod
    def validate_paper_ids(cls, value: list[str]) -> list[str]:
        if len(value) > 500:
            raise ValueError(f"Maximum 500 paper IDs per batch request, got {len(value)}")
        return value


class GetCitationFormatsArgs(ToolArgsModel):
    result_id: str = Field(
        description=(
            "Scholar result_id for the paper. "
            "Use paper.scholarResultId (not paper.sourceId) from a "
            "serpapi_google_scholar search_papers result — "
            "paper.scholarResultId is the raw Scholar result_id and is the "
            "correct identifier for this tool. "
            "paper.sourceId may be a cluster_id or cites_id instead of a "
            "result_id when result_id was absent, so it cannot be used here. "
            "If paper.scholarResultId is absent the paper cannot be used with "
            "this tool. "
            "This is a paid SerpApi request (cached for 1 hour by SerpApi). "
            "Only works when PAPER_CHASER_ENABLE_SERPAPI=true and "
            "SERPAPI_API_KEY is set."
        ),
    )


class SerpApiCitedByArgs(ToolArgsModel):
    cites_id: str = Field(
        alias="citesId",
        description="Google Scholar cites_id from a SerpApi cited_by link.",
    )
    query: str | None = Field(
        default=None,
        description="Optional extra Scholar query refinement within the cited-by set.",
    )
    limit: int = Field(
        default=10,
        description="Max results (default 10, max 20).",
    )
    cursor: str | None = Field(
        default=None,
        description=OPAQUE_CURSOR_FIELD_DESCRIPTION,
    )
    year: str | None = Field(
        default=None,
        description="Optional year or year-range filter.",
    )

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 10, 20)


class SerpApiVersionsArgs(ToolArgsModel):
    cluster_id: str = Field(
        alias="clusterId",
        description="Google Scholar cluster_id from a SerpApi versions link.",
    )
    limit: int = Field(
        default=10,
        description="Max results (default 10, max 20).",
    )
    cursor: str | None = Field(
        default=None,
        description=OPAQUE_CURSOR_FIELD_DESCRIPTION,
    )

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 10, 20)


class SerpApiAuthorProfileArgs(ToolArgsModel):
    author_id: str = Field(
        alias="authorId",
        description="Google Scholar author_id from a SerpApi author profile or result.",
    )


class SerpApiAuthorArticlesArgs(ToolArgsModel):
    author_id: str = Field(
        alias="authorId",
        description="Google Scholar author_id from a SerpApi author profile or result.",
    )
    limit: int = Field(
        default=10,
        description="Max results (default 10, max 20).",
    )
    cursor: str | None = Field(
        default=None,
        description=OPAQUE_CURSOR_FIELD_DESCRIPTION,
    )
    sort: str | None = Field(
        default=None,
        description="Optional SerpApi author sort hint, e.g. 'title' or 'pubdate'.",
    )

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 10, 20)


class ProviderDiagnosticsArgs(ToolArgsModel):
    include_recent_outcomes: bool = Field(
        default=True,
        alias="includeRecentOutcomes",
        description="Whether to include recent per-provider outcome envelopes.",
    )


class SerpApiAccountStatusArgs(ToolArgsModel):
    pass


class ProviderBudgetArgs(ToolArgsModel):
    max_total_calls: int | None = Field(
        default=None,
        alias="maxTotalCalls",
        description="Optional cap across all provider calls during one smart search.",
    )
    max_semantic_scholar_calls: int | None = Field(
        default=None,
        alias="maxSemanticScholarCalls",
        description="Optional Semantic Scholar call cap for one smart search.",
    )
    max_openalex_calls: int | None = Field(
        default=None,
        alias="maxOpenAlexCalls",
        description="Optional OpenAlex call cap for one smart search.",
    )
    max_core_calls: int | None = Field(
        default=None,
        alias="maxCoreCalls",
        description="Optional CORE call cap for one smart search.",
    )
    max_arxiv_calls: int | None = Field(
        default=None,
        alias="maxArxivCalls",
        description="Optional arXiv call cap for one smart search.",
    )
    max_serpapi_calls: int | None = Field(
        default=None,
        alias="maxSerpApiCalls",
        description="Optional SerpApi call cap for one smart search.",
    )
    max_scholarapi_calls: int | None = Field(
        default=None,
        alias="maxScholarApiCalls",
        description="Optional ScholarAPI call cap for one smart search.",
    )
    allow_paid_providers: bool = Field(
        default=True,
        alias="allowPaidProviders",
        description="Set false to disallow paid providers such as SerpApi or ScholarAPI.",
    )


class SmartSearchPapersArgs(ToolArgsModel):
    query: str = Field(
        description=(
            "Concept-level or known-item research query. Natural language is "
            "allowed; exact identifiers like DOI, arXiv ID, or URL are also "
            "accepted."
        )
    )
    limit: int = Field(
        default=10,
        description="Max smart-ranked results to return (default 10, max 25)",
    )
    search_session_id: str | None = Field(
        default=None,
        alias="searchSessionId",
        description=("Optional prior searchSessionId to continue refining an existing research workspace."),
    )
    mode: Literal[
        "auto",
        "discovery",
        "review",
        "known_item",
        "author",
        "citation",
        "regulatory",
    ] = Field(
        default="auto",
        description=(
            "Task shape hint. Use auto unless the agent already knows whether "
            "this is discovery, literature review, known-item lookup, author "
            "pivot, citation chasing, or a regulatory primary-source workflow."
        ),
    )
    year: str | None = Field(
        default=None,
        description="Optional year or year-range hint, e.g. '2023' or '2020-2024'.",
    )
    venue: str | None = Field(
        default=None,
        description="Optional venue hint to keep the search focused.",
    )
    focus: str | None = Field(
        default=None,
        description="Optional subtopic, method, or application focus hint.",
    )
    latency_profile: LatencyProfile = Field(
        default="deep",
        alias="latencyProfile",
        description=(
            "Quality-first search depth control: deep is the default for the best "
            "smart-search coverage, balanced trades a bit of quality for lower latency, "
            "and fast is intended for smoke tests or interactive debugging only."
        ),
    )
    provider_budget: ProviderBudgetArgs | None = Field(
        default=None,
        alias="providerBudget",
        description=(
            "Optional per-request provider budget for advanced clients. Use this "
            "to cap total provider calls, provider-specific calls, or paid usage."
        ),
    )
    include_enrichment: bool = Field(
        default=False,
        alias="includeEnrichment",
        description=(
            "When true, enrich only the final returned smart hits after retrieval, fusion, and ranking complete."
        ),
    )

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 10, 25)


class AskResultSetArgs(ToolArgsModel):
    search_session_id: str = Field(
        alias="searchSessionId",
        description="searchSessionId from search_papers_smart or a reusable list tool.",
    )
    question: str = Field(description=("Follow-up question to answer using only papers from the saved result set."))
    top_k: int = Field(
        default=8,
        alias="topK",
        description=("Number of evidence papers to retrieve from the result set (max 12)."),
    )
    answer_mode: Literal["qa", "claim_check", "comparison"] = Field(
        default="qa",
        alias="answerMode",
        description="Answer style: grounded QA, claim checking, or comparison.",
    )
    latency_profile: LatencyProfile = Field(
        default="deep",
        alias="latencyProfile",
        description=(
            "Quality-first control for grounded answer synthesis. Deep is the default, "
            "balanced lowers latency, and fast is for smoke tests only."
        ),
    )

    @field_validator("top_k", mode="before")
    @classmethod
    def clamp_top_k(cls, value: int | None) -> int:
        return _clamp_limit(value, 8, 12)


class MapResearchLandscapeArgs(ToolArgsModel):
    search_session_id: str = Field(
        alias="searchSessionId",
        description="searchSessionId for the saved paper result set to cluster.",
    )
    max_themes: int = Field(
        default=5,
        alias="maxThemes",
        description="Maximum number of themes to return (default 5, max 5).",
    )
    latency_profile: LatencyProfile = Field(
        default="deep",
        alias="latencyProfile",
        description=(
            "Quality-first control for theme labeling and summarization. Deep is the default, "
            "balanced lowers latency, and fast is for smoke tests only."
        ),
    )

    @field_validator("max_themes", mode="before")
    @classmethod
    def clamp_max_themes(cls, value: int | None) -> int:
        return _clamp_limit(value, 5, 5)


class ExpandResearchGraphArgs(ToolArgsModel):
    seed_paper_ids: list[str] | None = Field(
        default=None,
        alias="seedPaperIds",
        description=(
            "Optional paper IDs to expand from. Use recommendedExpansionId when "
            "available for Semantic Scholar-based expansion."
        ),
    )
    seed_search_session_id: str | None = Field(
        default=None,
        alias="seedSearchSessionId",
        description=(
            "Optional searchSessionId whose paper results should be used as graph "
            "seeds when explicit seedPaperIds are not provided."
        ),
    )
    direction: Literal["citations", "references", "authors"] = Field(
        default="citations",
        description="Graph expansion direction.",
    )
    hops: int = Field(
        default=1,
        description="Expansion depth in hops (default 1, max 2).",
    )
    per_seed_limit: int = Field(
        default=25,
        alias="perSeedLimit",
        description="Max frontier items to fetch per seed (default 25, max 50).",
    )
    latency_profile: LatencyProfile = Field(
        default="deep",
        alias="latencyProfile",
        description=(
            "Quality-first control for graph ranking and scoring. Deep is the default, "
            "balanced lowers latency, and fast is for smoke tests only."
        ),
    )

    @field_validator("hops", mode="before")
    @classmethod
    def clamp_hops(cls, value: int | None) -> int:
        return _clamp_limit(value, 1, 2)

    @field_validator("per_seed_limit", mode="before")
    @classmethod
    def clamp_per_seed_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 25, 50)


class SearchSpeciesEcosArgs(ToolArgsModel):
    query: str = Field(description="Common-name or scientific-name species query for ECOS.")
    limit: int = Field(
        default=10,
        description="Max species hits to return (default 10, max 25).",
    )
    match_mode: Literal["auto", "exact", "prefix"] = Field(
        default="auto",
        alias="matchMode",
        description=(
            "auto tries exact common/scientific-name matching first, then "
            "prefix matching. exact and prefix restrict the search strategy."
        ),
    )

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 10, 25)


class EcosSpeciesLookupArgs(ToolArgsModel):
    species_id: str = Field(description="ECOS species id or ECOS species URL.")


class ListSpeciesDocumentsEcosArgs(EcosSpeciesLookupArgs):
    document_kinds: list[EcosDocumentKind] | None = Field(
        default=None,
        alias="documentKinds",
        description=(
            "Optional document-kind filter. Supported values: recovery_plan, "
            "five_year_review, biological_opinion, federal_register, "
            "other_recovery_doc, conservation_plan_link."
        ),
    )


class GetDocumentTextEcosArgs(ToolArgsModel):
    url: str = Field(description="Absolute or ECOS-relative document URL to fetch and convert.")


class SearchFederalRegisterArgs(ToolArgsModel):
    query: str = Field(description="Federal Register search query.")
    limit: int = Field(default=10, description="Max documents to return (default 10, max 25).")
    agencies: list[str] | None = Field(
        default=None,
        description="Optional agency slug filter list, e.g. fish-and-wildlife-service.",
    )
    document_types: list[Literal["RULE", "PRORULE", "NOTICE", "PRESDOCU"]] | None = Field(
        default=None,
        alias="documentTypes",
        description="Optional Federal Register document-type filter.",
    )
    publication_date_from: str | None = Field(
        default=None,
        alias="publicationDateFrom",
        description="Optional publication-date lower bound in YYYY-MM-DD format.",
    )
    publication_date_to: str | None = Field(
        default=None,
        alias="publicationDateTo",
        description="Optional publication-date upper bound in YYYY-MM-DD format.",
    )
    cfr_citation: str | None = Field(
        default=None,
        alias="cfrCitation",
        description="Optional CFR citation filter, e.g. 50 CFR 17 or 40 CFR 122.26.",
    )
    cfr_title: int | None = Field(default=None, alias="cfrTitle", description="Optional CFR title filter.")
    cfr_part: int | None = Field(default=None, alias="cfrPart", description="Optional CFR part filter.")
    document_number: str | None = Field(
        default=None,
        alias="documentNumber",
        description="Optional explicit Federal Register document number filter.",
    )

    @field_validator("limit", mode="before")
    @classmethod
    def clamp_limit(cls, value: int | None) -> int:
        return _clamp_limit(value, 10, 25)


class GetFederalRegisterDocumentArgs(ToolArgsModel):
    identifier: str = Field(
        description="Federal Register document number, FR citation, or GovInfo FR link.",
    )


class GetCfrTextArgs(ToolArgsModel):
    title_number: int = Field(alias="titleNumber", description="CFR title number, e.g. 40.")
    part_number: int = Field(alias="partNumber", description="CFR part number, e.g. 122.")
    section_number: str | None = Field(
        default=None,
        alias="sectionNumber",
        description="Optional CFR section suffix, e.g. 26 for 40 CFR 122.26.",
    )
    revision_year: int | None = Field(
        default=None,
        alias="revisionYear",
        description="Optional CFR revision year used during volume resolution.",
    )
    effective_date: str | None = Field(
        default=None,
        alias="effectiveDate",
        description="Optional effective date hint in YYYY-MM-DD format.",
    )


TOOL_INPUT_MODELS: dict[str, type[ToolArgsModel]] = {
    "research": ResearchArgs,
    "follow_up_research": FollowUpResearchArgs,
    "resolve_reference": ResolveReferenceArgs,
    "inspect_source": InspectSourceArgs,
    "get_runtime_status": GetRuntimeStatusArgs,
    "search_papers": SearchPapersArgs,
    "search_papers_core": MinimalProviderSearchPapersArgs,
    "search_papers_semantic_scholar": SemanticProviderSearchPapersArgs,
    "search_papers_serpapi": MinimalProviderSearchPapersArgs,
    "search_papers_scholarapi": ScholarApiSearchArgs,
    "search_papers_arxiv": MinimalProviderSearchPapersArgs,
    "search_papers_openalex": MinimalProviderSearchPapersArgs,
    "search_papers_openalex_bulk": OpenAlexBulkSearchPapersArgs,
    "list_papers_scholarapi": ScholarApiListArgs,
    "search_papers_bulk": BulkSearchPapersArgs,
    "search_papers_match": PaperMatchArgs,
    "resolve_citation": ResolveCitationArgs,
    "paper_autocomplete": PaperAutocompleteArgs,
    "paper_autocomplete_openalex": OpenAlexPaperAutocompleteArgs,
    "get_paper_details": PaperLookupArgs,
    "get_paper_text_scholarapi": ScholarApiPaperTextArgs,
    "get_paper_texts_scholarapi": ScholarApiPaperTextsArgs,
    "get_paper_pdf_scholarapi": ScholarApiPaperPdfArgs,
    "get_paper_metadata_crossref": CrossrefEnrichmentArgs,
    "get_paper_open_access_unpaywall": UnpaywallEnrichmentArgs,
    "enrich_paper": EnrichPaperArgs,
    "get_paper_details_openalex": OpenAlexPaperLookupArgs,
    "get_paper_citations": PaperListArgs,
    "get_paper_citations_openalex": OpenAlexPaperListArgs,
    "get_paper_references": PaperListArgs,
    "get_paper_references_openalex": OpenAlexPaperListArgs,
    "get_paper_authors": PaperAuthorsArgs,
    "get_author_info": AuthorInfoArgs,
    "get_author_info_openalex": OpenAlexAuthorInfoArgs,
    "get_author_papers": AuthorPapersArgs,
    "get_author_papers_openalex": OpenAlexAuthorPapersArgs,
    "search_authors": AuthorSearchArgs,
    "search_authors_openalex": OpenAlexAuthorSearchArgs,
    "search_entities_openalex": OpenAlexEntitySearchArgs,
    "search_papers_openalex_by_entity": OpenAlexEntityPaperSearchArgs,
    "batch_get_authors": BatchGetAuthorsArgs,
    "search_snippets": SnippetSearchArgs,
    "get_paper_recommendations": RecommendationArgs,
    "get_paper_recommendations_post": PostRecommendationsArgs,
    "batch_get_papers": BatchGetPapersArgs,
    "get_paper_citation_formats": GetCitationFormatsArgs,
    "search_papers_serpapi_cited_by": SerpApiCitedByArgs,
    "search_papers_serpapi_versions": SerpApiVersionsArgs,
    "get_author_profile_serpapi": SerpApiAuthorProfileArgs,
    "get_author_articles_serpapi": SerpApiAuthorArticlesArgs,
    "get_serpapi_account_status": SerpApiAccountStatusArgs,
    "get_provider_diagnostics": ProviderDiagnosticsArgs,
    "search_species_ecos": SearchSpeciesEcosArgs,
    "get_species_profile_ecos": EcosSpeciesLookupArgs,
    "list_species_documents_ecos": ListSpeciesDocumentsEcosArgs,
    "get_document_text_ecos": GetDocumentTextEcosArgs,
    "search_federal_register": SearchFederalRegisterArgs,
    "get_federal_register_document": GetFederalRegisterDocumentArgs,
    "get_cfr_text": GetCfrTextArgs,
    "search_papers_smart": SmartSearchPapersArgs,
    "ask_result_set": AskResultSetArgs,
    "map_research_landscape": MapResearchLandscapeArgs,
    "expand_research_graph": ExpandResearchGraphArgs,
}
