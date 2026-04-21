"""Guided citation/verification helpers (Phase 3 extraction).

Extracted from :mod:`paper_chaser_mcp.dispatch._core`. The 8 functions here
build citation payloads and normalize verification/access-status axes for
guided source records. ``_core.py`` re-imports every symbol defined here at
the bottom of its module to preserve the pre-Phase-3 dispatch seam.

Module-level constants (``_REGULATORY_SOURCE_TYPES``, the access-status
implication frozensets) stay in ``_core.py`` for now; this module imports
them so the functions keep their original semantics without duplicating
the source of truth.
"""

from __future__ import annotations

import re
from typing import Any, cast

from ...identifiers import resolve_doi_from_paper_payload
from .._core import (
    _ACCESS_STATUS_IMPLIES_BODY,
    _ACCESS_STATUS_IMPLIES_QA,
    _REGULATORY_SOURCE_TYPES,
    _deduplicate_authors,
)



def _assign_verification_status(
    *,
    source_type: str,
    has_doi: bool = False,
    has_doi_resolution: bool = False,
    full_text_url_found: bool = False,
    body_text_embedded: bool = False,
) -> str:
    """Assign a verification-status label for a source.

    Regulatory sources earn ``verified_primary_source`` ONLY when the caller
    confirms body-text is embedded/available (``body_text_embedded=True``).
    URL-only regulatory hits — i.e. the provider found a canonical URL but no
    inline markdown/body — fall back to ``verified_metadata``; the access
    axis expresses URL discovery via ``accessStatus="url_verified"``. This is
    the P0-2 split: ``verified_primary_source`` attests a primary source that
    was actually readable, not merely linked.
    """
    if source_type in _REGULATORY_SOURCE_TYPES:
        if body_text_embedded:
            return "verified_primary_source"
        return "verified_metadata"
    if has_doi and has_doi_resolution:
        return "verified_metadata"
    if source_type == "scholarly_article" and has_doi:
        return "verified_metadata"
    if has_doi:
        return "verified_metadata"
    return "unverified"




def _guided_normalize_access_axes(candidate: dict[str, Any]) -> tuple[str, bool, bool, bool, bool]:
    """Return normalized access status + split access flags.

    ``fullTextUrlFound`` tracks URL discovery only. ``fullTextObserved`` tracks
    actual full-text/body availability for the saved/current source record.
    """

    raw_access_status = str(candidate.get("accessStatus") or "").strip()
    explicit_url_found = bool(candidate.get("fullTextUrlFound"))
    explicit_body_text_embedded = bool(candidate.get("bodyTextEmbedded"))
    explicit_qa_readable_text = bool(candidate.get("qaReadableText"))
    explicit_full_text_retrieved = bool(candidate.get("fullTextRetrieved"))
    legacy_full_text_observed = bool(candidate.get("fullTextObserved"))
    has_full_text_locator = bool(
        candidate.get("canonicalUrl")
        or candidate.get("retrievedUrl")
        or candidate.get("url")
        or candidate.get("pdfUrl")
    )
    has_split_signals = any(
        key in candidate
        for key in (
            "fullTextUrlFound",
            "bodyTextEmbedded",
            "qaReadableText",
            "fullTextRetrieved",
            "accessStatus",
        )
    )

    full_text_url_found = explicit_url_found or (legacy_full_text_observed and not has_split_signals)
    body_text_embedded = explicit_body_text_embedded or raw_access_status in _ACCESS_STATUS_IMPLIES_BODY
    qa_readable_text = (
        explicit_qa_readable_text or explicit_full_text_retrieved or raw_access_status in _ACCESS_STATUS_IMPLIES_QA
    )
    if qa_readable_text:
        body_text_embedded = True
    if (
        not full_text_url_found
        and has_full_text_locator
        and raw_access_status
        in {
            "url_verified",
            "oa_verified",
            "oa_uncertain",
            "pdf_available",
            "full_text_verified",
            "full_text_retrieved",
        }
    ):
        full_text_url_found = True

    if raw_access_status in {"full_text_verified", "full_text_retrieved"}:
        if qa_readable_text:
            access_status = "qa_readable_text"
        elif body_text_embedded:
            access_status = "body_text_embedded"
        elif full_text_url_found:
            access_status = "url_verified"
        elif bool(candidate.get("abstractObserved")):
            access_status = "abstract_only"
        else:
            access_status = "access_unverified"
    elif raw_access_status == "qa_readable_text":
        access_status = "qa_readable_text"
    elif raw_access_status == "body_text_embedded":
        access_status = "body_text_embedded"
    elif body_text_embedded:
        access_status = "qa_readable_text" if qa_readable_text else "body_text_embedded"
    elif raw_access_status:
        access_status = raw_access_status
    elif full_text_url_found:
        access_status = "url_verified"
    elif bool(candidate.get("abstractObserved")):
        access_status = "abstract_only"
    else:
        access_status = "access_unverified"

    full_text_observed = (
        body_text_embedded
        or qa_readable_text
        or explicit_full_text_retrieved
        or (legacy_full_text_observed and not has_split_signals)
    )
    return access_status, full_text_url_found, full_text_observed, body_text_embedded, qa_readable_text




def _guided_normalize_verification_status(
    candidate: dict[str, Any],
    *,
    source_type: str,
    full_text_url_found: bool,
    body_text_embedded: bool,
) -> str:
    raw_citation = candidate.get("citation")
    citation = cast(dict[str, Any], raw_citation) if isinstance(raw_citation, dict) else {}
    explicit_verification_status = str(candidate.get("verificationStatus") or "").strip()
    normalized_verification_status = explicit_verification_status or _assign_verification_status(
        source_type=source_type,
        has_doi=bool(candidate.get("doi") or citation.get("doi")),
        has_doi_resolution=bool(candidate.get("doi") or citation.get("doi")),
        full_text_url_found=full_text_url_found,
        body_text_embedded=body_text_embedded,
    )
    if (
        normalized_verification_status == "verified_primary_source"
        and source_type in _REGULATORY_SOURCE_TYPES
        and not body_text_embedded
    ):
        return "verified_metadata"
    return normalized_verification_status




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
    raw_authors = list(
        dict.fromkeys(
            str(author.get("name") or "").strip()
            for author in (paper.get("authors") or [])
            if isinstance(author, dict) and str(author.get("name") or "").strip()
        )
    )
    authors = _deduplicate_authors(raw_authors)
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


