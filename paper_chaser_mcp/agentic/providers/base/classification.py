"""Deterministic lexical relevance classification helpers."""

from __future__ import annotations

import re
from typing import Any

from ...provider_helpers import COMMON_QUERY_WORDS, _lexical_similarity, _tokenize

_FALLBACK_RELEVANCE_STOPWORDS = COMMON_QUERY_WORDS | {
    "current",
    "evidence",
    "federal",
    "history",
    "latest",
    "literature",
    "peer",
    "recent",
    "regulatory",
    "reports",
    "review",
    "scientific",
    "scholarship",
    "studies",
    "study",
    "systematic",
}


def _fallback_query_terms(query: str) -> list[str]:
    return [token for token in _tokenize(query) if token not in _FALLBACK_RELEVANCE_STOPWORDS and len(token) >= 4]


def _fallback_query_facets(query: str) -> list[list[str]]:
    lowered = str(query or "").lower()
    segments = re.split(r"\b(?:for|in|on|about|into|within|across|via|through|regarding|around|under)\b", lowered)
    facets: list[list[str]] = []
    for segment in segments:
        tokens = [
            token for token in _tokenize(segment) if token not in _FALLBACK_RELEVANCE_STOPWORDS and len(token) >= 4
        ]
        if len(tokens) >= 2:
            facets.append(tokens[:3])
    return facets[:3]


def classify_relevance_without_llm(
    *,
    query: str,
    paper: dict[str, Any],
) -> dict[str, Any]:
    title = str(paper.get("title") or "")
    abstract = str(paper.get("abstract") or "")
    paper_text = " ".join(part for part in [title, abstract] if part)
    similarity = _lexical_similarity(query, paper_text)
    query_terms = _fallback_query_terms(query)
    title_tokens = set(_tokenize(title))
    body_tokens = set(_tokenize(paper_text))
    title_hits = [term for term in query_terms if term in title_tokens]
    body_hits = [term for term in query_terms if term in body_tokens]
    facets = _fallback_query_facets(query)

    facet_title_hits = 0
    facet_body_hits = 0
    for facet in facets:
        required = len(facet) if len(facet) <= 2 else 2
        if sum(token in title_tokens for token in facet) >= required:
            facet_title_hits += 1
        if sum(token in body_tokens for token in facet) >= required:
            facet_body_hits += 1

    title_anchor_coverage = (len(title_hits) / len(query_terms)) if query_terms else 0.0
    title_facet_coverage = (facet_title_hits / len(facets)) if facets else 0.0
    body_facet_coverage = (facet_body_hits / len(facets)) if facets else 0.0

    if (title_facet_coverage > 0.0 or title_anchor_coverage >= 0.34) and similarity >= 0.2:
        classification = "on_topic"
        rationale = "Deterministic fallback found strong title or facet overlap with the query."
    elif similarity < 0.08 or (not body_hits and body_facet_coverage == 0.0):
        classification = "off_topic"
        rationale = "Deterministic fallback found little direct overlap beyond generic research language."
    else:
        classification = "weak_match"
        rationale = (
            "Deterministic fallback found partial semantic overlap without enough direct title or facet support."
        )

    return {
        "classification": classification,
        "rationale": rationale,
        "fallback": True,
        "provenance": "deterministic_fallback",
    }
