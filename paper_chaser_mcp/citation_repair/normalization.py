"""Stateless text and identifier normalization helpers for :mod:`citation_repair`.

Phase 9a extracted this module verbatim from
``paper_chaser_mcp/citation_repair/_core.py``. It owns the regex patterns,
vocabulary constants, and pure-function normalizers used by the parser, the
candidate builders, and the resolver orchestration. Everything here is
stdlib-only so downstream modules (``candidates``, ``api``, ``ranking``) can
depend on it without triggering circular imports.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
ARXIV_RE = re.compile(
    r"(?:arxiv:)?(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z][\w.-]+/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
PAGES_RE = re.compile(r"\b\d{1,4}\s*[-:]\s*\d{1,4}\b")
QUOTED_RE = re.compile(r'["“”](.+?)["“”]')
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'/-]*")
REGULATORY_CITATION_RE = re.compile(
    r"\b\d+\s*(?:F\.?\s*R\.?|FED(?:ERAL)?\.?\s+REG(?:ISTER)?\.?)\s*\d+\b|\b\d+\s+CFR\b",
    re.IGNORECASE,
)
VENUE_HINTS = (
    "annual review",
    "annual review of ecology",
    "ecological applications",
    "ecology letters",
    "nature sustainability",
    "nature",
    "science",
    "pnas",
    "proceedings of the national academy of sciences",
    "journal of applied ecology",
    "global change biology",
    "environmental science",
    "environmental research letters",
    "acl",
    "cvpr",
    "emnlp",
    "iclr",
    "icml",
    "naacl",
    "neurips",
    "nips",
)
NON_PAPER_TERMS = {
    "dataset",
    "datasheet",
    "dissertation",
    "guidance",
    "guidelines",
    "handbook",
    "manual",
    "package",
    "policy",
    "report",
    "software",
    "standard",
    "thesis",
    "whitepaper",
}
REGULATORY_TERMS = {
    "federal register",
    "fed. reg",
    "cfr",
    "code of federal regulations",
    "final rule",
    "proposed rule",
    "rulemaking",
    "notice of",
}
GENERIC_TITLE_WORDS = {
    "and",
    "the",
    "for",
    "with",
    "from",
    "that",
    "this",
    "these",
    "those",
    "using",
    "study",
    "studies",
    "paper",
    "papers",
    "framework",
}


def normalize_citation_text(value: str) -> str:
    """Collapse whitespace while preserving the user's wording."""
    return " ".join(str(value or "").strip().split())


def looks_like_url(value: str) -> bool:
    """Return True when *value* looks like a URL."""
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def looks_like_paper_identifier(value: str) -> bool:
    """Return True when *value* resembles a DOI, arXiv ID, or URL."""
    normalized = normalize_citation_text(value)
    lowered = normalized.lower()
    return bool(
        normalized
        and (
            DOI_RE.search(normalized)
            or ARXIV_RE.search(normalized)
            or looks_like_url(normalized)
            or lowered.startswith("doi:")
            or lowered.startswith("arxiv:")
        )
    )


def _venue_hint_in_text(text: str, venue: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9-]){re.escape(venue)}(?![A-Za-z0-9-])", text))


def _normalize_identifier_for_semantic_scholar(identifier: str, identifier_type: str | None) -> str:
    normalized = normalize_citation_text(identifier)
    if not normalized:
        return normalized
    lowered = normalized.lower()
    if identifier_type == "doi" or DOI_RE.fullmatch(normalized):
        if lowered.startswith("doi:"):
            return f"DOI:{normalized[4:].strip()}"
        doi_match = DOI_RE.search(normalized)
        if doi_match:
            return f"DOI:{doi_match.group(0)}"
    if identifier_type == "arxiv" or ARXIV_RE.fullmatch(normalized):
        if lowered.startswith("arxiv:"):
            return f"ARXIV:{normalized[6:].strip()}"
        return f"ARXIV:{normalized}"
    if identifier_type == "url" and looks_like_url(normalized):
        parsed = urlparse(normalized)
        if parsed.netloc.lower().endswith("semanticscholar.org"):
            path_parts = [part for part in parsed.path.split("/") if part]
            if path_parts:
                candidate = path_parts[-1]
                if re.fullmatch(r"[A-Fa-f0-9]{40}", candidate):
                    return candidate
        return f"URL:{normalized}"
    return normalized


def _normalize_identifier_for_openalex(identifier: str, identifier_type: str | None) -> str | None:
    normalized = normalize_citation_text(identifier)
    if not normalized:
        return None
    lowered = normalized.lower()
    if identifier_type == "doi" or DOI_RE.fullmatch(normalized):
        if lowered.startswith("doi:"):
            normalized = normalized[4:].strip()
        elif lowered.startswith("https://doi.org/"):
            normalized = normalized[16:]
        elif lowered.startswith("http://doi.org/"):
            normalized = normalized[15:]
        doi_match = DOI_RE.search(normalized)
        return doi_match.group(0) if doi_match else None
    if lowered.startswith("https://openalex.org/w") or lowered.startswith("http://openalex.org/w"):
        return normalized.rstrip("/").rsplit("/", 1)[-1]
    if re.fullmatch(r"W\d+", normalized, re.IGNORECASE):
        return normalized
    return None


def _dedupe_strings(values: Any) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_citation_text(str(value))
        lowered = normalized.lower()
        if not normalized or lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(normalized)
    return deduped
