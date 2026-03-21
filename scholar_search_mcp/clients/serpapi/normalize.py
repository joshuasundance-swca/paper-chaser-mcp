"""Normalize SerpApi Google Scholar organic results into the shared Paper model."""

import re
from typing import Any, Literal, Optional

from ...models import Author, Paper, dump_jsonable


def _extract_year_from_summary(summary: str) -> Optional[int]:
    """Extract the first 4-digit year (1900-2099) from a publication summary."""
    match = re.search(r"\b(19|20)\d{2}\b", summary)
    if match:
        try:
            return int(match.group())
        except ValueError:
            pass
    return None


def _extract_doi_from_link(link: str) -> Optional[str]:
    """Try to extract a DOI from a URL string."""
    match = re.search(r"10\.\d{4,9}/\S+", link)
    return match.group(0).rstrip(".,;)") if match else None


def _parse_year_range(year: str) -> tuple[Optional[int], Optional[int]]:
    """Parse a year or year-range string into (low, high) integer pair.

    Accepts ``"2023"`` (single year), ``"2020-2023"`` (inclusive range), and
    open-ended variants like ``"2020-"`` or ``"-2023"``.
    """
    year = year.strip()
    if "-" in year:
        parts = year.split("-", 1)
        low_str, high_str = parts[0].strip(), parts[1].strip()
        year_low: Optional[int] = None
        year_high: Optional[int] = None
        if len(low_str) >= 4 and low_str[:4].isdigit():
            try:
                year_low = int(low_str[:4])
            except ValueError:
                pass
        if len(high_str) >= 4 and high_str[:4].isdigit():
            try:
                year_high = int(high_str[:4])
            except ValueError:
                pass
        return year_low, year_high
    else:
        if len(year) >= 4 and year[:4].isdigit():
            try:
                y = int(year[:4])
                return y, y
            except ValueError:
                pass
    return None, None


def _authors_from_text(text: str | None) -> list[Author]:
    authors: list[Author] = []
    if not isinstance(text, str):
        return authors
    for raw_name in text.split(","):
        name = raw_name.strip()
        if name:
            authors.append(Author(name=name))
    return authors


def _build_paper(
    *,
    title: str,
    paper_id: str | None,
    source_id: str | None,
    canonical_id: str | None,
    recommended_expansion_id: str | None,
    authors: list[Author],
    abstract: str | None,
    year: int | None,
    venue: str | None,
    url: str | None,
    pdf_url: str | None = None,
    citation_count: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expansion_id_status: Literal["portable", "not_portable"] = (
        "portable" if recommended_expansion_id else "not_portable"
    )
    paper = Paper(
        paperId=paper_id,
        title=title,
        abstract=abstract,
        year=year,
        authors=authors,
        citationCount=citation_count,
        referenceCount=None,
        influentialCitationCount=None,
        venue=venue,
        publicationTypes=None,
        publicationDate=None,
        url=url,
        pdfUrl=pdf_url,
        source="serpapi_google_scholar",
        sourceId=source_id,
        canonicalId=canonical_id,
        recommendedExpansionId=recommended_expansion_id,
        expansionIdStatus=expansion_id_status,
        scholarResultId=paper_id,
    )
    if extra:
        paper = paper.model_copy(update=extra)
    return dump_jsonable(paper)


def normalize_organic_result(result: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Convert one SerpApi organic Scholar result to a normalized paper dict.

    Returns a ``dump_jsonable``-serialized dict compatible with ``Paper``, or
    ``None`` if the result lacks a title (minimum required field).
    """
    title = (result.get("title") or "").strip()
    if not title:
        return None

    # --- Provider identifiers ---
    result_id: Optional[str] = result.get("result_id") or None
    inline_links: dict[str, Any] = result.get("inline_links") or {}

    versions: dict[str, Any] = inline_links.get("versions") or {}
    cluster_id: Optional[str] = (
        str(versions["cluster_id"]) if versions.get("cluster_id") else None
    )

    cited_by: dict[str, Any] = inline_links.get("cited_by") or {}
    cites_id: Optional[str] = (
        str(cited_by["cites_id"]) if cited_by.get("cites_id") else None
    )

    # sourceId priority: result_id > cluster_id > cites_id
    source_id: Optional[str] = result_id or cluster_id or cites_id

    # --- URL and DOI ---
    link: Optional[str] = result.get("link") or None
    doi: Optional[str] = None
    if link:
        doi = _extract_doi_from_link(link)

    # canonicalId priority: DOI > cluster_id > result_id > sourceId
    canonical_id: Optional[str] = doi or cluster_id or result_id or source_id
    recommended_expansion_id: Optional[str] = doi or None

    # --- Authors ---
    authors: list[Author] = []
    pub_info: dict[str, Any] = result.get("publication_info") or {}
    for author in pub_info.get("authors") or []:
        if isinstance(author, dict):
            name = (author.get("name") or "").strip()
            if name:
                authors.append(Author(name=name))

    # --- Year and venue from publication_info.summary ---
    year: Optional[int] = None
    venue: Optional[str] = None
    summary: str = (pub_info.get("summary") or "").strip()
    if summary:
        year = _extract_year_from_summary(summary)
        # Summaries look like "Author Names - Journal Name, 2023" or
        # "Journal Name, 2023". Extract the venue-ish component.
        summary_parts = summary.split(" - ", 1)
        venue_part = summary_parts[-1].strip() if len(summary_parts) > 1 else ""
        if not venue_part:
            venue_part = summary_parts[0].strip()
        # Strip trailing year / comma from venue
        venue_candidate = re.sub(r",?\s*(19|20)\d{2}.*$", "", venue_part).strip()
        if venue_candidate and venue_candidate != summary.strip():
            venue = venue_candidate or None

    # --- Abstract/snippet: Scholar snippet is excerpt, not a full abstract ---
    abstract: Optional[str] = (result.get("snippet") or "").strip() or None

    # --- PDF URL from resources ---
    pdf_url: Optional[str] = None
    for resource in result.get("resources") or []:
        if isinstance(resource, dict):
            fmt = (resource.get("file_format") or "").lower()
            res_title = (resource.get("title") or "").lower()
            res_link = resource.get("link") or ""
            if res_link and ("pdf" in fmt or "pdf" in res_title):
                pdf_url = res_link
                break

    # --- Citation count ---
    citation_count: Optional[int] = None
    cited_by_total = cited_by.get("total")
    if cited_by_total is not None:
        try:
            citation_count = int(cited_by_total)
        except (ValueError, TypeError):
            pass

    extra: dict[str, Any] = {}
    if cluster_id:
        extra["scholarClusterId"] = cluster_id
    if cites_id:
        extra["scholarCitesId"] = cites_id
    return _build_paper(
        title=title,
        paper_id=result_id,
        source_id=source_id,
        canonical_id=canonical_id,
        recommended_expansion_id=recommended_expansion_id,
        authors=authors,
        abstract=abstract,
        year=year,
        venue=venue,
        url=link,
        pdf_url=pdf_url,
        citation_count=citation_count,
        extra=extra,
    )


def normalize_author_article_result(result: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Convert one SerpApi author article entry to the shared Paper shape."""

    title = (result.get("title") or "").strip()
    if not title:
        return None
    link = result.get("link")
    citation_id = result.get("citation_id")
    doi = _extract_doi_from_link(link) if isinstance(link, str) else None
    cited_by = result.get("cited_by") or {}
    cites_id = str(cited_by.get("cites_id")) if cited_by.get("cites_id") else None
    citation_count: Optional[int] = None
    cited_by_value = cited_by.get("value")
    if cited_by_value is not None:
        try:
            citation_count = int(cited_by_value)
        except (TypeError, ValueError):
            pass
    year = None
    raw_year = result.get("year")
    if raw_year is not None:
        try:
            year = int(raw_year)
        except (TypeError, ValueError):
            year = None
    publication = result.get("publication")
    if isinstance(publication, str) and publication.strip():
        venue = publication.strip()
    else:
        venue = None
    extra: dict[str, Any] = {}
    if cites_id:
        extra["scholarCitesId"] = cites_id
    if citation_id:
        extra["scholarCitationId"] = citation_id
    return _build_paper(
        title=title,
        paper_id=str(citation_id) if citation_id else None,
        source_id=str(citation_id) if citation_id else cites_id,
        canonical_id=doi or (str(citation_id) if citation_id else cites_id),
        recommended_expansion_id=doi,
        authors=_authors_from_text(result.get("authors")),
        abstract=None,
        year=year,
        venue=venue,
        url=link if isinstance(link, str) else None,
        citation_count=citation_count,
        extra=extra,
    )
