"""Guided source-record helpers (Phase 3 extraction).

Extracted from :mod:`paper_chaser_mcp.dispatch._core`. The 12 functions here
build and merge guided source-record payloads (including the
FederalRegisterDocument projection) for downstream trust/coverage logic.
"""

from __future__ import annotations

from typing import Any

from ...guided_semantic import strip_null_fields
from ...identifiers import resolve_doi_from_paper_payload
from ..normalization import _guided_normalize_source_locator, _guided_normalize_whitespace
from ..relevance import _paper_topical_relevance, compute_topical_relevance
from .citations import (
    _guided_citation_from_paper,
    _guided_citation_from_structured_source,
    _guided_normalize_access_axes,
    _guided_normalize_verification_status,
    _guided_open_access_route,
)


def _guided_source_id(candidate: dict[str, Any], *, fallback_prefix: str, index: int) -> str:
    for key in (
        "sourceId",
        "evidenceId",
        "paperId",
        "canonicalId",
        "recommendedExpansionId",
        "citationText",
        "canonicalUrl",
        "url",
    ):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    title = str(candidate.get("title") or "").strip()
    if title:
        return title
    return f"{fallback_prefix}-{index}"


def _guided_source_record_from_structured_source(source: dict[str, Any], *, index: int) -> dict[str, Any]:
    _source_type = source.get("sourceType") or "unknown"
    _access_status, _full_text_url_found, _full_text_observed, _body_text_embedded, _qa_readable_text = (
        _guided_normalize_access_axes(source)
    )
    _default_verification = _guided_normalize_verification_status(
        source,
        source_type=str(_source_type),
        full_text_url_found=_full_text_url_found,
        body_text_embedded=_body_text_embedded,
    )
    topical_relevance = source.get("topicalRelevance") or "weak_match"
    weak_match_reason = str(source.get("whyClassifiedAsWeakMatch") or "").strip() or None
    if weak_match_reason is None and topical_relevance in {"weak_match", "off_topic"}:
        weak_match_reason = str(source.get("note") or source.get("whyNotVerified") or "").strip() or None
    normalized_open_access_source = {
        **source,
        "accessStatus": _access_status,
        "fullTextUrlFound": _full_text_url_found,
        "fullTextObserved": _full_text_observed,
    }
    return {
        "sourceId": _guided_source_id(source, fallback_prefix="source", index=index),
        "title": source.get("title"),
        "provider": source.get("provider"),
        "sourceType": _source_type,
        "verificationStatus": _default_verification,
        "accessStatus": _access_status,
        "topicalRelevance": topical_relevance,
        "confidence": source.get("confidence") or "medium",
        "isPrimarySource": bool(source.get("isPrimarySource")),
        "canonicalUrl": source.get("canonicalUrl"),
        "retrievedUrl": source.get("retrievedUrl"),
        "fullTextUrlFound": _full_text_url_found,
        "fullTextObserved": _full_text_observed,
        "bodyTextEmbedded": _body_text_embedded,
        "qaReadableText": _qa_readable_text,
        "abstractObserved": bool(source.get("abstractObserved")),
        "openAccessRoute": _guided_open_access_route(normalized_open_access_source),
        "citationText": source.get("citationText"),
        "citation": _guided_citation_from_structured_source(source),
        "date": source.get("date"),
        "note": source.get("note"),
        "whyClassifiedAsWeakMatch": weak_match_reason,
    }


def _guided_source_record_from_paper(query: str, paper: dict[str, Any], *, index: int) -> dict[str, Any]:
    canonical_url = paper.get("canonicalUrl") or paper.get("url") or paper.get("pdfUrl")
    source_type = paper.get("sourceType") or "scholarly_article"
    doi, _ = resolve_doi_from_paper_payload(paper)
    _access_status, _full_text_url_found, _full_text_observed, _body_text_embedded, _qa_readable_text = (
        _guided_normalize_access_axes(paper)
    )
    verification_status = _guided_normalize_verification_status(
        {**paper, "doi": doi},
        source_type=str(source_type),
        full_text_url_found=_full_text_url_found,
        body_text_embedded=_body_text_embedded,
    )
    # Backward-compatible default: a scholarly article with basic descriptive
    # metadata (title + at least one author OR a venue) should remain
    # ``verified_metadata`` rather than silently drop to ``unverified`` just
    # because it lacks a DOI. ``unverified`` is reserved for records truly
    # missing descriptive metadata. (ws-dispatch-contract-trust / finding #3.)
    if (
        verification_status == "unverified"
        and str(source_type) == "scholarly_article"
        and str(paper.get("title") or "").strip()
    ):
        _authors = paper.get("authors") or []
        _has_author = False
        if isinstance(_authors, list):
            for _author in _authors:
                if isinstance(_author, str) and _author.strip():
                    _has_author = True
                    break
                if isinstance(_author, dict) and str(_author.get("name") or "").strip():
                    _has_author = True
                    break
        _has_venue = bool(str(paper.get("venue") or "").strip())
        if _has_author or _has_venue:
            verification_status = "verified_metadata"
    # Access-status split (P0-2): an explicit accessStatus on the input wins;
    # otherwise derive from the strongest signal available. ``body_text_embedded``
    # implies inline body content; a URL-only hit becomes ``url_verified``
    # (distinct from the deprecated ``full_text_verified`` which used to be
    # emitted for any URL-found regulatory or scholarly record).
    topical_relevance = _paper_topical_relevance(query, paper)
    confidence = paper.get("confidence") or ("high" if topical_relevance == "on_topic" else "medium")
    weak_match_reason = str(paper.get("whyClassifiedAsWeakMatch") or "").strip() or None
    if weak_match_reason is None and topical_relevance in {"weak_match", "off_topic"}:
        weak_match_reason = str(paper.get("note") or paper.get("venue") or "").strip() or None
    normalized_open_access_paper = {
        **paper,
        "accessStatus": _access_status,
        "fullTextUrlFound": _full_text_url_found,
        "fullTextObserved": _full_text_observed,
        "canonicalUrl": canonical_url,
        "retrievedUrl": paper.get("retrievedUrl") or canonical_url,
    }
    return strip_null_fields(
        {
            "sourceId": _guided_source_id(paper, fallback_prefix="paper", index=index),
            "title": paper.get("title"),
            "provider": paper.get("source"),
            "sourceType": source_type,
            "verificationStatus": verification_status,
            "accessStatus": _access_status,
            "topicalRelevance": topical_relevance,
            "confidence": confidence,
            "isPrimarySource": bool(paper.get("isPrimarySource")),
            "canonicalUrl": canonical_url,
            "retrievedUrl": paper.get("retrievedUrl") or canonical_url,
            "fullTextUrlFound": _full_text_url_found,
            "fullTextObserved": _full_text_observed,
            "bodyTextEmbedded": _body_text_embedded,
            "qaReadableText": _qa_readable_text,
            "abstractObserved": bool(paper.get("abstractObserved")),
            "openAccessRoute": _guided_open_access_route(normalized_open_access_paper),
            "citationText": str(paper.get("canonicalId") or paper.get("paperId") or "") or None,
            "citation": _guided_citation_from_paper(paper, canonical_url),
            "date": paper.get("publicationDate") or paper.get("year"),
            "note": paper.get("note") or paper.get("venue"),
            "whyClassifiedAsWeakMatch": weak_match_reason,
        }
    )


def _guided_sources_from_fr_documents(query: str, documents: list[Any]) -> list[dict[str, Any]]:
    """Convert FederalRegisterDocument objects into guided source records."""
    sources: list[dict[str, Any]] = []
    for index, doc in enumerate(documents or [], start=1):
        # Support both attribute-style (Pydantic model) and dict-style access
        def _get(attr: str, default: Any = None) -> Any:
            if isinstance(doc, dict):
                return doc.get(attr, default)
            return getattr(doc, attr, default)

        title = _get("title")
        if not title:
            continue
        html_url = _get("htmlUrl") or _get("bodyHtmlUrl")
        pdf_url = _get("pdfUrl")
        canonical_url = html_url or _get("govInfoLink") or pdf_url
        doc_number = _get("documentNumber")
        doc_type = str(_get("documentType") or "").strip()
        pub_date = _get("publicationDate")
        citation = _get("citation")
        agencies_raw = _get("agencies") or []
        agency_names = [
            str(getattr(a, "name", None) or (a.get("name") if isinstance(a, dict) else "") or "").strip()
            for a in agencies_raw
        ]
        agency_str = "; ".join(n for n in agency_names if n) or None
        cfr_refs = _get("cfrReferences") or []
        abstract = _get("abstract")

        # Build citation text from FR citation or document number
        citation_text = citation or (f"Fed. Reg. No. {doc_number}" if doc_number else None)

        # FR documents are authoritative primary sources, but topical
        # relevance must still be *computed* from the query — the FR search
        # API sometimes returns adjacent/off-topic rules that should be
        # flagged for escalation rather than folded into evidence as on-topic.
        source_type = "federal_register_rule" if "rule" in doc_type.lower() else "regulatory_document"
        note_parts = [agency_str] if agency_str else []
        if cfr_refs:
            note_parts.append("CFR: " + ", ".join(str(r) for r in cfr_refs[:3]))

        fr_source_candidate = {
            "title": title,
            "abstract": abstract,
            "venue": "Federal Register",
            "note": "; ".join(note_parts) if note_parts else None,
        }
        topical_relevance = compute_topical_relevance(query, fr_source_candidate)

        sources.append(
            {
                "sourceId": f"fr-{doc_number}" if doc_number else f"fr-source-{index}",
                "title": title,
                "provider": "federal_register",
                "sourceType": source_type,
                # P0-2: URL-found alone does not imply body-text-embedded.
                # FR docs here are URL-discovered metadata; body must be
                # fetched separately to earn ``verified_primary_source``.
                "verificationStatus": "verified_metadata",
                "accessStatus": ("url_verified" if html_url else ("pdf_available" if pdf_url else "access_unverified")),
                "topicalRelevance": topical_relevance,
                "confidence": "high" if topical_relevance == "on_topic" else "medium",
                "isPrimarySource": True,
                "canonicalUrl": canonical_url,
                "retrievedUrl": canonical_url,
                "fullTextUrlFound": bool(html_url),
                "bodyTextEmbedded": False,
                "qaReadableText": False,
                "abstractObserved": bool(abstract),
                "openAccessRoute": "open_access" if canonical_url else None,
                "citationText": citation_text,
                "citation": {
                    "title": title,
                    "authors": agency_names or [],
                    "year": pub_date[:4] if isinstance(pub_date, str) and len(pub_date) >= 4 else None,
                    "venue": "Federal Register",
                    "canonicalId": citation_text,
                    "citationText": citation_text,
                },
                "date": pub_date,
                "note": "; ".join(note_parts) if note_parts else None,
            }
        )
    return sources


def _guided_extract_source_id(arguments: dict[str, Any]) -> Any:
    return next(
        (
            arguments.get(key)
            for key in ("sourceId", "source_id", "evidenceId", "evidence_id", "source", "sourceRef", "leadId", "id")
            if arguments.get(key) is not None
        ),
        None,
    )


def _guided_dedupe_source_records(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for source in sources:
        key = (
            str(source.get("sourceId") or "").strip(),
            str(source.get("canonicalUrl") or "").strip(),
            str(source.get("title") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def _guided_source_matches_reference(candidate: dict[str, Any], reference: Any) -> bool:
    normalized_reference = _guided_normalize_whitespace(reference)
    if not normalized_reference:
        return False
    lowered_reference = normalized_reference.lower()
    normalized_locator = _guided_normalize_source_locator(normalized_reference)
    for value in (
        candidate.get("sourceId"),
        candidate.get("sourceAlias"),
        candidate.get("citationText"),
        candidate.get("canonicalUrl"),
        candidate.get("retrievedUrl"),
        candidate.get("title"),
    ):
        normalized_candidate = _guided_normalize_whitespace(value)
        if not normalized_candidate:
            continue
        if normalized_candidate.lower() == lowered_reference:
            return True
        if normalized_locator and _guided_normalize_source_locator(normalized_candidate) == normalized_locator:
            return True
    return False


def _guided_source_records_share_surface(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_titles = {
        _guided_normalize_whitespace(left.get("title")).lower(),
        _guided_normalize_whitespace(left.get("citationText")).lower(),
    } - {""}
    right_titles = {
        _guided_normalize_whitespace(right.get("title")).lower(),
        _guided_normalize_whitespace(right.get("citationText")).lower(),
    } - {""}
    if left_titles and right_titles and left_titles & right_titles:
        return True

    left_locators = {
        _guided_normalize_source_locator(left.get("canonicalUrl")),
        _guided_normalize_source_locator(left.get("retrievedUrl")),
    } - {""}
    right_locators = {
        _guided_normalize_source_locator(right.get("canonicalUrl")),
        _guided_normalize_source_locator(right.get("retrievedUrl")),
    } - {""}
    return bool(left_locators and right_locators and left_locators & right_locators)


def _guided_source_identity(source: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(source.get("sourceId") or "").strip(),
        str(source.get("canonicalUrl") or "").strip(),
        str(source.get("title") or "").strip().lower(),
    )


def _guided_merge_source_records(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary)
    for key, value in secondary.items():
        if key not in merged or merged[key] in (None, "", [], {}):
            merged[key] = value
            continue
        if isinstance(value, bool) and value and not bool(merged[key]):
            merged[key] = True
    access_status, full_text_url_found, full_text_observed, body_text_embedded, qa_readable_text = (
        _guided_normalize_access_axes(merged)
    )
    source_type = str(merged.get("sourceType") or "unknown")
    merged["accessStatus"] = access_status
    merged["fullTextUrlFound"] = full_text_url_found
    merged["fullTextObserved"] = full_text_observed
    merged["bodyTextEmbedded"] = body_text_embedded
    merged["qaReadableText"] = qa_readable_text
    merged["verificationStatus"] = _guided_normalize_verification_status(
        merged,
        source_type=source_type,
        full_text_url_found=full_text_url_found,
        body_text_embedded=body_text_embedded,
    )
    return merged


def _guided_merge_source_record_sets(*record_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    ordered_keys: list[tuple[str, str, str]] = []
    for record_set in record_sets:
        for record in record_set:
            canonical_record = _guided_merge_source_records({}, record)
            key = _guided_source_identity(canonical_record)
            if key not in merged_by_key:
                merged_by_key[key] = canonical_record
                ordered_keys.append(key)
            else:
                merged_by_key[key] = _guided_merge_source_records(merged_by_key[key], canonical_record)
    return [merged_by_key[key] for key in ordered_keys]


def _guided_source_coverage_summary(
    *,
    sources: list[dict[str, Any]],
    leads: list[dict[str, Any]],
    base_coverage: dict[str, Any] | None,
) -> dict[str, Any] | None:
    summary = dict(base_coverage or {})
    visible_records = [record for record in [*sources, *leads] if isinstance(record, dict)]
    if not summary and not visible_records:
        return None
    summary["totalSources"] = len(visible_records)
    by_access_status: dict[str, int] = {}
    for record in visible_records:
        access_status = str(record.get("accessStatus") or "access_unverified").strip() or "access_unverified"
        by_access_status[access_status] = by_access_status.get(access_status, 0) + 1
    summary["byAccessStatus"] = by_access_status
    return summary
