"""Pure topical-relevance helpers extracted from ``dispatch._core``.

These are the canonical signals used by the guided research, follow-up, and
``inspect_source`` paths to decide whether a retrieved paper/document is
``on_topic``, ``weak_match``, or ``off_topic`` for a given query. They are
deliberately side-effect free so they can be unit-tested directly and reused
by future extractions without pulling in dispatch's heavy dependencies.

Keep this module small. New relevance heuristics should land here; anything
that touches I/O, provider clients, or session state belongs elsewhere.
"""

from __future__ import annotations

import re
from typing import Any

from ..agentic.planner import query_facets, query_terms


def _tokenize_relevance_text(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", value.lower()))


def _facet_match(tokens: set[str], facet: str) -> bool:
    facet_tokens = _tokenize_relevance_text(facet)
    return bool(facet_tokens) and facet_tokens.issubset(tokens)


def _topical_relevance_from_signals(
    *,
    query_similarity: float,
    title_facet_coverage: float,
    title_anchor_coverage: float,
    query_facet_coverage: float,
    query_anchor_coverage: float,
) -> str:
    title_has_anchor = title_facet_coverage > 0 or title_anchor_coverage > 0
    body_has_anchor = query_facet_coverage > 0 or query_anchor_coverage > 0
    has_facet_signal = title_facet_coverage > 0 or query_facet_coverage > 0
    # Require a multi-token phrase match (facet) for the standard threshold, or a
    # strict majority of query terms when no phrase match exists.  A single-token
    # title hit with low similarity is a weak signal, not grounded evidence.
    if title_has_anchor and ((has_facet_signal and query_similarity >= 0.25) or query_similarity > 0.5):
        return "on_topic"
    if query_similarity < 0.12 or not (title_has_anchor or body_has_anchor):
        return "off_topic"
    return "weak_match"


def compute_topical_relevance(query: str, source: dict[str, Any]) -> str:
    """Canonical topical-relevance classifier shared by guided paths.

    Used by both the paper/smart retrieval path and the Federal Register
    document converter so that research and inspect_source produce the same
    relevance label for the same ``(query, source)`` pair. Returns one of
    ``on_topic``, ``weak_match``, or ``off_topic``.
    """
    facets = query_facets(query)
    terms = query_terms(query)
    title_tokens = _tokenize_relevance_text(str(source.get("title") or ""))
    body_text_parts = [
        str(source.get("title") or ""),
        str(source.get("abstract") or ""),
        str(source.get("venue") or ""),
        str(source.get("note") or ""),
    ]
    paper_tokens = _tokenize_relevance_text(" ".join(part for part in body_text_parts if part))
    matched_terms = [term for term in terms if term in paper_tokens]
    matched_title_terms = [term for term in terms if term in title_tokens]
    matched_facets = [facet for facet in facets if _facet_match(paper_tokens, facet)]
    matched_title_facets = [facet for facet in facets if _facet_match(title_tokens, facet)]
    term_coverage = len(matched_terms) / len(terms) if terms else 0.0
    title_term_coverage = len(matched_title_terms) / len(terms) if terms else 0.0
    query_similarity = max(term_coverage, title_term_coverage)
    return _topical_relevance_from_signals(
        query_similarity=query_similarity,
        title_facet_coverage=(len(matched_title_facets) / len(facets) if facets else 0.0),
        title_anchor_coverage=title_term_coverage,
        query_facet_coverage=(len(matched_facets) / len(facets) if facets else 0.0),
        query_anchor_coverage=term_coverage,
    )


def _paper_topical_relevance(query: str, paper: dict[str, Any]) -> str:
    """Backward-compatible wrapper around :func:`compute_topical_relevance`."""
    return compute_topical_relevance(query, paper)


__all__ = (
    "_tokenize_relevance_text",
    "_facet_match",
    "_topical_relevance_from_signals",
    "compute_topical_relevance",
    "_paper_topical_relevance",
)
