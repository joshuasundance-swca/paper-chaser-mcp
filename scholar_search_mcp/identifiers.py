"""Identifier normalization helpers shared across enrichment and lookup flows."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import unquote

DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)


def _trim_doi(value: str) -> str:
    normalized = value.strip().rstrip(".,;")
    while normalized.endswith(")") and normalized.count("(") < normalized.count(")"):
        normalized = normalized[:-1]
    while normalized.endswith("]") and normalized.count("[") < normalized.count("]"):
        normalized = normalized[:-1]
    while normalized.endswith("}") and normalized.count("{") < normalized.count("}"):
        normalized = normalized[:-1]
    return normalized.strip()


def normalize_doi(value: Any) -> str | None:
    """Normalize one DOI-like value to the bare DOI string."""

    if not isinstance(value, str):
        return None
    normalized = unquote(value).strip()
    if not normalized:
        return None
    lowered = normalized.lower()
    if lowered.startswith("doi:"):
        normalized = normalized[4:].strip()
    match = DOI_PATTERN.search(normalized)
    if match is None:
        return None
    doi = _trim_doi(match.group(0))
    return doi.lower() or None


def resolve_doi_from_paper_payload(paper: Any) -> tuple[str | None, str | None]:
    """Resolve a DOI from a normalized paper payload or Paper model."""

    if paper is None:
        return None, None

    if hasattr(paper, "model_dump"):
        payload = paper.model_dump(by_alias=True)
    elif isinstance(paper, Mapping):
        payload = dict(paper)
    else:
        return None, None

    external_ids = payload.get("externalIds")
    if not isinstance(external_ids, Mapping):
        model_extra = payload.get("model_extra")
        if isinstance(model_extra, Mapping):
            external_ids = model_extra.get("externalIds")

    candidates: tuple[tuple[str, Any], ...] = (
        ("doi", payload.get("doi")),
        ("canonical_id", payload.get("canonicalId")),
        ("recommended_expansion_id", payload.get("recommendedExpansionId")),
        ("paper_id", payload.get("paperId")),
        ("source_id", payload.get("sourceId")),
        ("url", payload.get("url")),
        ("pdf_url", payload.get("pdfUrl")),
        (
            "external_ids",
            external_ids.get("DOI") if isinstance(external_ids, Mapping) else None,
        ),
    )
    for source, candidate in candidates:
        doi = normalize_doi(candidate)
        if doi:
            return doi, source
    return None, None


def resolve_doi_inputs(
    *,
    doi: str | None = None,
    paper_id: str | None = None,
    paper: Any = None,
) -> tuple[str | None, str | None]:
    """Resolve a DOI from explicit DOI input, identifier input, or a paper payload."""

    explicit = normalize_doi(doi)
    if explicit:
        return explicit, "doi"

    from_identifier = normalize_doi(paper_id)
    if from_identifier:
        return from_identifier, "paper_id"

    return resolve_doi_from_paper_payload(paper)
