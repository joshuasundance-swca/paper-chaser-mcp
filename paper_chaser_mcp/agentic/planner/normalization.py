"""Query normalization, facet extraction, and title-shape heuristics.

These helpers are the lowest layer of planner text processing and have no
dependency on planner decisions or intent labels. The two title-shape helpers
(``looks_like_exact_title`` and ``looks_like_near_known_item_query``) consult
``detect_regulatory_intent`` and ``_query_starts_broad`` through lazy imports
to avoid the ``normalization ↔ regulatory ↔ specificity`` import cycle.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from .constants import (
    FACET_SPLIT_RE,
    GENERIC_EVIDENCE_WORDS,
    QUERY_FACET_TOKEN_ALLOWLIST,
    QUERYISH_TITLE_BLOCKERS,
    STRONG_REGULATORY_TITLE_BLOCKERS,
    TITLE_STOPWORDS,
)


def normalize_query(query: str) -> str:
    """Collapse whitespace and keep the original wording intact."""
    return " ".join(query.strip().split())


def query_facets(query: str) -> list[str]:
    """Extract compact multi-token facets that should stay visible in results."""
    normalized = re.sub(r"[-_/]+", " ", normalize_query(query).lower())
    facets: list[str] = []
    seen: set[str] = set()
    for segment in FACET_SPLIT_RE.split(normalized):
        tokens = [
            token
            for token in re.findall(r"[A-Za-z0-9]{3,}", segment)
            if (token not in GENERIC_EVIDENCE_WORDS or token in QUERY_FACET_TOKEN_ALLOWLIST)
        ]
        if len(tokens) >= 2:
            facet = " ".join(tokens[:3])
            if facet not in seen:
                seen.add(facet)
                facets.append(facet)
    if facets:
        return facets[:3]

    fallback_tokens = [
        token for token in re.findall(r"[A-Za-z0-9]{3,}", normalized) if token not in GENERIC_EVIDENCE_WORDS
    ]
    for token in fallback_tokens[:3]:
        if token not in seen:
            seen.add(token)
            facets.append(token)
    return facets


def query_terms(query: str) -> list[str]:
    """Return distinctive normalized query terms for lightweight coverage checks."""
    normalized = re.sub(r"[-_/]+", " ", normalize_query(query).lower())
    terms: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[A-Za-z0-9]{3,}", normalized):
        if token in seen:
            continue
        if token in GENERIC_EVIDENCE_WORDS and token not in QUERY_FACET_TOKEN_ALLOWLIST:
            continue
        seen.add(token)
        terms.append(token)
    return terms[:8]


def looks_like_url(query: str) -> bool:
    """Return True when *query* is a plausible URL."""
    if not query:
        return False
    parsed = urlparse(query)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def looks_like_exact_title(query: str) -> bool:
    """Heuristically identify likely exact-title lookups."""
    # Lazy imports break the normalization↔regulatory↔specificity cycle.
    from .regulatory import detect_regulatory_intent
    from .specificity import _query_starts_broad

    normalized = normalize_query(query)
    if not normalized or normalized.endswith("?"):
        return False
    if detect_regulatory_intent(normalized):
        return False
    lowered = normalized.lower()
    if any(marker in lowered for marker in STRONG_REGULATORY_TITLE_BLOCKERS):
        return False
    if re.search(r"\b\d+\s*(?:cfr|f\.?\s*r\.?)\b", lowered):
        return False
    if any(phrase in lowered for phrase in ("what does", "what is", "what are", "include representative")):
        return False
    stripped = normalized.strip("\"'")
    words = re.findall(r"[A-Za-z][A-Za-z0-9'/-]*", stripped)
    if not 2 <= len(words) <= 24:
        return False
    significant_words = [word for word in words if len(word) > 2 and word.lower() not in TITLE_STOPWORDS]
    if len(significant_words) < 2:
        return False
    queryish_count = sum(word.lower() in QUERYISH_TITLE_BLOCKERS for word in significant_words)
    if queryish_count >= 3:
        return False
    title_like_words = [word for word in significant_words if word[:1].isupper() or word.isupper() or "-" in word]
    if len(title_like_words) >= max(2, int(len(significant_words) * 0.45)):
        return True
    if (
        queryish_count == 0
        and not _query_starts_broad(normalized)
        and 4 <= len(significant_words) <= 12
        and any(word.lower() in TITLE_STOPWORDS for word in words)
    ):
        return True
    return bool(re.search(r"\([A-Z][A-Za-z]+(?:\s+[a-z][A-Za-z-]+)+\)", stripped)) and len(significant_words) >= 6


def looks_like_near_known_item_query(query: str) -> bool:
    """Heuristically identify short near-known-item prompts.

    This catches prompts like model/paper/system names (for example CLIP,
    Toolformer, DSPy, or Med-PaLM M) that are not exact-title lookups but still
    behave like anchored known-item requests for topical-relevance purposes.
    """
    # Lazy import avoids the normalization↔regulatory cycle.
    from .regulatory import detect_regulatory_intent

    normalized = normalize_query(query)
    if not normalized or normalized.endswith("?"):
        return False
    if detect_regulatory_intent(normalized):
        return False
    lowered = normalized.lower()
    if any(marker in lowered for marker in STRONG_REGULATORY_TITLE_BLOCKERS):
        return False
    if any(
        phrase in lowered
        for phrase in (
            "what is",
            "what are",
            "how does",
            "how do",
            "compare",
            "survey",
            "review",
            "literature",
            "state of the art",
            "overview",
            "introduction",
            "tutorial",
        )
    ):
        return False
    raw_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9+./:-]*", query.strip("\"' "))
    terms = query_terms(normalized)
    if not raw_tokens or not terms or len(terms) > 4 or len(raw_tokens) > 5:
        return False
    queryish_count = sum(term.lower() in QUERYISH_TITLE_BLOCKERS for term in terms)
    if queryish_count >= 2:
        return False

    def _identifierish(token: str) -> bool:
        stripped = token.strip()
        if len(stripped) <= 1:
            return stripped.isupper()
        upper_count = sum(char.isupper() for char in stripped)
        lower_count = sum(char.islower() for char in stripped)
        return bool(
            stripped.isupper()
            or "-" in stripped
            or any(char.isdigit() for char in stripped)
            or re.search(r"[a-z][A-Z]|[A-Z][a-z]+[A-Z]", stripped)
            or (upper_count >= 2 and lower_count >= 1)
        )

    if any(_identifierish(token) for token in raw_tokens):
        return True
    return len(raw_tokens) == 1 and raw_tokens[0][:1].isupper() and len(raw_tokens[0]) >= 5


__all__ = [
    "looks_like_exact_title",
    "looks_like_near_known_item_query",
    "looks_like_url",
    "normalize_query",
    "query_facets",
    "query_terms",
]
