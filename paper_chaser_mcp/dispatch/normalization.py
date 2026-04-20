"""Pure whitespace/citation/year normalization helpers for guided dispatch.

These helpers canonicalize free-form query text, source locators, CFR/FR
citations, and year hints so that the guided research/follow-up surface can
detect duplicates, produce stable cursor context hashes, and render
repair notes without every call site reinventing the same regexes.

Moved from ``paper_chaser_mcp.dispatch._core`` as part of the Phase 2
refactor. Behavior is preserved verbatim; only the module boundary moves.
"""

from __future__ import annotations

import re
from typing import Any

_GUIDED_QUERY_PREFIX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^\s*(?:please\s+|kindly\s+)?(?:help\s+me\s+)?(?:find|search(?:\s+for)?|look\s+up|research|summarize|show)\s+"
        r"(?:papers?|literature|studies|evidence|sources?|information)\s+(?:about|on|for)\s+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:please\s+|kindly\s+)?(?:help\s+me\s+)?(?:find|search(?:\s+for)?|look\s+up|research|summarize|show)\s+",
        re.IGNORECASE,
    ),
)

_GUIDED_CFR_SECTION_RE = re.compile(
    r"\b(?P<title>\d{1,2})\s*c\.?\s*f\.?\s*r\.?\s*(?P<part>\d{1,4})\s*(?:[.\-:/]\s*|\s+)(?P<section>\d{1,4})\b",
    re.IGNORECASE,
)
_GUIDED_CFR_PART_RE = re.compile(
    r"\b(?P<title>\d{1,2})\s*c\.?\s*f\.?\s*r\.?\s*part\s*(?P<part>\d{1,4})\b",
    re.IGNORECASE,
)
_GUIDED_FR_CITATION_RE = re.compile(
    r"\b(?P<volume>\d+)\s*f\.?\s*r\.?\s*(?P<page>\d+)\b",
    re.IGNORECASE,
)
_GUIDED_YEAR_RANGE_RE = re.compile(r"\b(?P<start>(?:19|20)\d{2})\s*[-:/]\s*(?P<end>(?:19|20)\d{2})\b")
_GUIDED_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _guided_normalize_whitespace(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _guided_normalize_source_locator(value: Any) -> str:
    normalized = _guided_normalize_whitespace(value).lower().rstrip("/")
    if not normalized:
        return ""
    normalized = re.sub(r"^https?://", "", normalized)
    normalized = re.sub(r"^www\.", "", normalized)
    return normalized


def _guided_strip_research_prefix(query: str) -> str:
    stripped = query
    for pattern in _GUIDED_QUERY_PREFIX_PATTERNS:
        candidate = pattern.sub("", stripped, count=1).strip()
        if candidate and candidate != stripped:
            return candidate
    return stripped


def _guided_normalize_citation_surface(text: str) -> str:
    normalized = text
    normalized = _GUIDED_CFR_SECTION_RE.sub(r"\g<title> CFR \g<part>.\g<section>", normalized)
    normalized = _GUIDED_CFR_PART_RE.sub(r"\g<title> CFR Part \g<part>", normalized)
    normalized = _GUIDED_FR_CITATION_RE.sub(r"\g<volume> FR \g<page>", normalized)
    return normalized


def _guided_normalize_year_hint(value: Any) -> str | None:
    text = _guided_normalize_whitespace(value)
    if not text:
        return None
    range_match = _GUIDED_YEAR_RANGE_RE.search(text)
    if range_match:
        start = range_match.group("start")
        end = range_match.group("end")
        return f"{start}:{end}" if start <= end else f"{end}:{start}"
    years = _GUIDED_YEAR_RE.findall(text)
    if len(years) >= 2:
        start, end = years[0], years[1]
        return f"{start}:{end}" if start <= end else f"{end}:{start}"
    if years:
        return years[0]
    return text


__all__ = (
    "_GUIDED_QUERY_PREFIX_PATTERNS",
    "_GUIDED_CFR_SECTION_RE",
    "_GUIDED_CFR_PART_RE",
    "_GUIDED_FR_CITATION_RE",
    "_GUIDED_YEAR_RANGE_RE",
    "_GUIDED_YEAR_RE",
    "_guided_normalize_whitespace",
    "_guided_normalize_source_locator",
    "_guided_strip_research_prefix",
    "_guided_normalize_citation_surface",
    "_guided_normalize_year_hint",
)
