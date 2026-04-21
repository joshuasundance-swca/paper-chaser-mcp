"""Scoring / ranking helpers for :mod:`paper_chaser_mcp.citation_repair`.

Phase 2 Track B Step 2 extracted this module verbatim from
``paper_chaser_mcp/citation_repair/_core.py``. It owns the candidate scoring
engine (:func:`_rank_candidate`), its supporting score helpers
(:func:`_author_overlap`, :func:`_year_delta`, :func:`_venue_overlap`,
:func:`_identifier_hit`, :func:`_snippet_alignment`,
:func:`_source_confidence`, :func:`_publication_preference_score`), the
title-similarity helpers (:func:`_title_similarity`,
:func:`_apply_length_penalty`, :func:`_token_overlap_ratio`,
:func:`_weighted_token_overlap_ratio`), and the
:class:`RankedCitationCandidate` dataclass.

The scoring weights encode today's behavior (including the known bias where
a large ``year_delta`` can fully zero the score on a perfect title hit) and
are the surface Phase 9b will rebalance. Any weight change must land
together with its sibling update in
``tests/test_citation_repair_ranking.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from ._core import (
    GENERIC_TITLE_WORDS,
    WORD_RE,
    ParsedCitation,
    _why_selected,
    normalize_citation_text,
)

__all__ = (
    "RankedCitationCandidate",
    "_rank_candidate",
    "_author_overlap",
    "_year_delta",
    "_venue_overlap",
    "_identifier_hit",
    "_snippet_alignment",
    "_token_overlap_ratio",
    "_publication_preference_score",
    "_title_similarity",
    "_apply_length_penalty",
    "_weighted_token_overlap_ratio",
    "_source_confidence",
    "_surname",
)


@dataclass(slots=True)
class RankedCitationCandidate:
    """One ranked citation-repair candidate before serialization."""

    paper: dict[str, Any]
    score: float
    resolution_strategy: str
    matched_fields: list[str]
    conflicting_fields: list[str]
    title_similarity: float
    year_delta: int | None
    author_overlap: int
    candidate_count: int | None
    why_selected: str


def _rank_candidate(
    *,
    paper: dict[str, Any],
    parsed: ParsedCitation,
    resolution_strategy: str,
    candidate_count: int | None,
    snippet_text: str | None,
) -> RankedCitationCandidate:
    # Deferred import avoids a circular dependency: ``_core`` re-exports
    # symbols from this module for backward compatibility, and
    # ``_dedupe_strings`` lives in ``_core``.
    from ._core import _dedupe_strings

    upstream_title_similarity = paper.get("titleSimilarity")
    title_similarity = max(
        _title_similarity(parsed, paper),
        float(upstream_title_similarity or 0.0),
    )
    author_overlap = max(
        _author_overlap(parsed, paper),
        int(paper.get("authorOverlap") or 0),
    )
    year_delta = _year_delta(parsed, paper)
    if year_delta is None and paper.get("yearDelta") is not None:
        try:
            year_delta = int(paper["yearDelta"])
        except (TypeError, ValueError):
            year_delta = None
    venue_overlap = _venue_overlap(parsed, paper)
    identifier_hit = resolution_strategy.startswith("identifier") or _identifier_hit(parsed, paper)
    snippet_alignment = _snippet_alignment(parsed, paper, snippet_text=snippet_text)
    source_confidence = _source_confidence(resolution_strategy)
    publication_preference = _publication_preference_score(paper)
    upstream_confidence = str(paper.get("matchConfidence") or "").lower()
    year_conflict = parsed.year is not None and year_delta is not None and year_delta > 1
    author_conflict = bool(parsed.author_surnames) and author_overlap == 0
    venue_conflict = bool(parsed.venue_hints) and not venue_overlap
    key_conflict_count = sum(1 for conflict in (author_conflict, year_conflict, venue_conflict) if conflict)

    score = 0.0
    if identifier_hit:
        score += 0.55
    score += title_similarity * 0.35
    if author_overlap >= 1:
        score += 0.08
    if author_overlap >= 2:
        score += 0.17
    if year_delta == 0:
        score += 0.08
    elif year_delta == 1:
        score += 0.04
    elif year_delta is not None and year_delta > 1:
        score -= min(year_delta, 10) * 0.04
        if year_delta > 5:
            score -= 0.06
    if venue_overlap:
        score += 0.05
    if snippet_alignment > 0:
        score += min(snippet_alignment, 1.0) * 0.05
    score += source_confidence * 0.05
    score += publication_preference * 0.03
    upstream_bonus = 0.0
    if upstream_confidence == "high":
        upstream_bonus = 0.15
    elif upstream_confidence == "medium":
        upstream_bonus = 0.08
    if key_conflict_count >= 2:
        upstream_bonus = min(upstream_bonus, 0.04)
    elif year_conflict:
        upstream_bonus = min(upstream_bonus, 0.06)
    score += upstream_bonus
    score = max(0.0, min(score, 1.0))

    matched_fields: list[str] = []
    conflicting_fields: list[str] = []
    matched_fields.extend(str(field) for field in paper.get("matchedFields") or [])
    conflicting_fields.extend(str(field) for field in paper.get("conflictingFields") or [])
    if identifier_hit:
        matched_fields.append("identifier")
    if title_similarity >= 0.72:
        matched_fields.append("title")
    elif parsed.title_candidates and not identifier_hit:
        conflicting_fields.append("title")
    if author_overlap > 0:
        matched_fields.append("author")
    elif parsed.author_surnames:
        conflicting_fields.append("author")
    if year_delta == 0:
        matched_fields.append("year")
    elif year_conflict:
        conflicting_fields.append("year")
    if venue_overlap:
        matched_fields.append("venue")
    elif parsed.venue_hints:
        conflicting_fields.append("venue")
    if snippet_alignment >= 0.35:
        matched_fields.append("snippet")

    why_selected = _why_selected(
        matched_fields=matched_fields,
        conflicting_fields=conflicting_fields,
        paper=paper,
        parsed=parsed,
        resolution_strategy=resolution_strategy,
    )
    return RankedCitationCandidate(
        paper=paper,
        score=score,
        resolution_strategy=resolution_strategy,
        matched_fields=_dedupe_strings(matched_fields),
        conflicting_fields=_dedupe_strings(conflicting_fields),
        title_similarity=title_similarity,
        year_delta=year_delta,
        author_overlap=author_overlap,
        candidate_count=candidate_count,
        why_selected=why_selected,
    )


def _title_similarity(parsed: ParsedCitation, paper: dict[str, Any]) -> float:
    title = normalize_citation_text(str(paper.get("title") or "")).lower()
    if not title:
        return 0.0
    candidates = parsed.title_candidates or [parsed.normalized_text]
    best = 0.0
    for candidate in candidates:
        normalized_candidate = normalize_citation_text(candidate).lower()
        if not normalized_candidate:
            continue
        raw = max(
            SequenceMatcher(None, normalized_candidate, title).ratio(),
            _token_overlap_ratio(normalized_candidate, title),
            _weighted_token_overlap_ratio(normalized_candidate, title),
        )
        raw = _apply_length_penalty(raw, normalized_candidate, title)
        best = max(best, raw)
    return best


def _apply_length_penalty(similarity: float, candidate: str, title: str) -> float:
    """Reduce similarity when candidate and title lengths differ significantly."""
    cand_len = len(candidate)
    title_len = len(title)
    if cand_len == 0 or title_len == 0:
        return similarity
    ratio = min(cand_len, title_len) / max(cand_len, title_len)
    if ratio >= 0.7:
        return similarity
    # Scale penalty: at ratio=0.5 penalty ~0.15, at ratio=0.2 penalty ~0.35
    penalty = (0.7 - ratio) * 0.7
    return max(0.0, similarity - penalty)


def _author_overlap(parsed: ParsedCitation, paper: dict[str, Any]) -> int:
    if not parsed.author_surnames:
        return 0
    author_names = {
        _surname(str(author.get("name") or ""))
        for author in (paper.get("authors") or [])
        if isinstance(author, dict) and author.get("name")
    }
    return sum(1 for surname in parsed.author_surnames if surname in author_names)


def _year_delta(parsed: ParsedCitation, paper: dict[str, Any]) -> int | None:
    if parsed.year is None or paper.get("year") is None:
        return None
    try:
        return abs(int(paper["year"]) - int(parsed.year))
    except (TypeError, ValueError):
        return None


def _venue_overlap(parsed: ParsedCitation, paper: dict[str, Any]) -> bool:
    if not parsed.venue_hints:
        return False
    venue = normalize_citation_text(str(paper.get("venue") or "")).lower()
    if not venue:
        return False
    return any(hint.lower() in venue or venue in hint.lower() for hint in parsed.venue_hints)


def _identifier_hit(parsed: ParsedCitation, paper: dict[str, Any]) -> bool:
    if not parsed.identifier:
        return False
    lowered_identifier = parsed.identifier.lower()
    external_ids = paper.get("externalIds") or {}
    candidates = [
        str(paper.get("paperId") or ""),
        str(paper.get("canonicalId") or ""),
        str(paper.get("recommendedExpansionId") or ""),
        str(external_ids.get("DOI") or ""),
        str(external_ids.get("ArXiv") or ""),
    ]
    if parsed.identifier_type == "doi":
        normalized_identifier = lowered_identifier.removeprefix("doi:")
        return any(
            normalized_identifier == candidate.lower().removeprefix("doi:") for candidate in candidates if candidate
        )
    if parsed.identifier_type == "arxiv":
        normalized_identifier = lowered_identifier.removeprefix("arxiv:")
        return any(
            normalized_identifier == candidate.lower().removeprefix("arxiv:") for candidate in candidates if candidate
        )
    return any(lowered_identifier == candidate.lower() for candidate in candidates if candidate)


def _snippet_alignment(
    parsed: ParsedCitation,
    paper: dict[str, Any],
    *,
    snippet_text: str | None,
) -> float:
    if not snippet_text:
        return 0.0
    paper_text = " ".join(
        part for part in [str(paper.get("title") or ""), str(paper.get("abstract") or "")] if part
    ).lower()
    if not paper_text:
        return 0.0
    return _token_overlap_ratio(snippet_text.lower(), paper_text)


def _source_confidence(strategy: str) -> float:
    mapping = {
        "identifier": 1.0,
        "identifier_openalex": 0.92,
        "exact_title": 0.9,
        "openalex_exact_title": 0.9,
        "crossref_exact_title": 0.84,
        "fuzzy_search": 0.82,
        "citation_ranked": 0.74,
        "snippet_recovery": 0.7,
        "sparse_metadata": 0.65,
        "openalex_metadata": 0.6,
    }
    return mapping.get(strategy, 0.55)


def _publication_preference_score(paper: dict[str, Any]) -> float:
    publication_types_raw = paper.get("publicationTypes")
    if isinstance(publication_types_raw, str):
        publication_types = {publication_types_raw.lower()}
    elif isinstance(publication_types_raw, list):
        publication_types = {str(value).lower() for value in publication_types_raw}
    else:
        publication_types = set()
    source = str(paper.get("source") or "").lower()
    venue = normalize_citation_text(str(paper.get("venue") or "")).lower()
    external_ids = paper.get("externalIds") or {}
    doi = str(external_ids.get("DOI") or paper.get("doi") or "").strip()

    score = 0.0
    if doi or str(paper.get("canonicalId") or "").lower().startswith("10."):
        score += 1.0
    if venue and "arxiv" not in venue:
        score += 0.6
    if publication_types & {"journal-article", "proceedings-article", "conference", "conference-paper"}:
        score += 0.8
    if source == "arxiv" or "preprint" in publication_types or venue == "arxiv":
        score -= 1.2
    return score


def _surname(name: str) -> str:
    words = [word.lower() for word in WORD_RE.findall(name)]
    return words[-1] if words else ""


def _token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = {token for token in re.findall(r"[a-z0-9]{3,}", left.lower()) if token not in GENERIC_TITLE_WORDS}
    right_tokens = {token for token in re.findall(r"[a-z0-9]{3,}", right.lower()) if token not in GENERIC_TITLE_WORDS}
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens & right_tokens
    return len(intersection) / len(left_tokens)


def _weighted_token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = [token for token in re.findall(r"[a-z0-9]{3,}", left.lower()) if token not in GENERIC_TITLE_WORDS]
    right_token_set = {
        token for token in re.findall(r"[a-z0-9]{3,}", right.lower()) if token not in GENERIC_TITLE_WORDS
    }
    if not left_tokens or not right_token_set:
        return 0.0
    matched_weight = sum(max(len(token) - 2, 1) for token in left_tokens if token in right_token_set)
    total_weight = sum(max(len(token) - 2, 1) for token in left_tokens)
    return matched_weight / total_weight if total_weight else 0.0
