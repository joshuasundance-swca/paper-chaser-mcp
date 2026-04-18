"""Deterministic scoring helpers for comparative/selection follow-ups.

Implements the P1-2 ``topRecommendation`` contract described in
``docs/ux-remediation-checklist.md``. All scoring is stdlib-only and
deterministic so that comparison/selection follow-ups produce a stable
recommendation without an additional LLM call.

Public surface:

* :func:`infer_comparative_axis` -- map a follow-up question into one of the
  supported axes (``recency``, ``authority``, ``beginner``, ``coverage``,
  ``relevance_fallback``).
* :func:`score_papers_for_comparative_axis` -- produce a ``paper_id -> score``
  dict (scores in ``[0.0, 1.0]``) for the supplied axis.
"""

from __future__ import annotations

import math
import re
from typing import Any, Literal

from ..models.common import Paper

ComparativeAxis = Literal["recency", "authority", "beginner", "coverage", "relevance_fallback"]

# Ordered list of (axis, markers). The first axis whose marker appears in the
# question wins; order matters because some markers (e.g. "most recent") are
# more specific than the generic beginner/coverage cues.
_AXIS_MARKERS: tuple[tuple[ComparativeAxis, tuple[str, ...]], ...] = (
    (
        "recency",
        (
            "most recent",
            "most up to date",
            "most up-to-date",
            "latest paper",
            "latest study",
            "latest research",
            "newest",
            "up-to-date",
        ),
    ),
    (
        "authority",
        (
            "most authoritative",
            "most cited",
            "most influential",
            "seminal",
            "foundational",
            "highest citation",
            "highly cited",
        ),
    ),
    (
        "beginner",
        (
            "beginner-friendly",
            "beginner friendly",
            "most accessible",
            "easiest to",
            "best starting point",
            "starting point",
            "should i start with",
            "where should i start",
            "introduction to",
            "primer",
        ),
    ),
    (
        "coverage",
        (
            "most comprehensive",
            "broadest",
            "widest coverage",
            "covers the most",
            "most thorough",
            "covers everything",
        ),
    ),
)


def infer_comparative_axis(question: str) -> ComparativeAxis:
    """Infer the comparative axis from a follow-up question."""
    lowered = (question or "").lower()
    for axis, markers in _AXIS_MARKERS:
        if any(marker in lowered for marker in markers):
            return axis
    return "relevance_fallback"


_BEGINNER_TITLE_MARKERS: tuple[str, ...] = (
    "introduction",
    "primer",
    "tutorial",
    "overview",
    "survey",
    "review",
    "beginner",
    "guide",
    "getting started",
)


_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "between",
        "by",
        "compared",
        "comparison",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "most",
        "of",
        "on",
        "or",
        "over",
        "paper",
        "papers",
        "should",
        "study",
        "studies",
        "than",
        "that",
        "the",
        "their",
        "them",
        "these",
        "this",
        "to",
        "under",
        "versus",
        "vs",
        "was",
        "were",
        "what",
        "which",
        "while",
        "who",
        "why",
        "with",
    }
)


def _year_score(paper: Paper, min_year: int, max_year: int) -> float:
    year = paper.year
    if year is None:
        return 0.0
    if max_year <= min_year:
        return 1.0
    return max(0.0, min(1.0, (year - min_year) / (max_year - min_year)))


def _authority_score(paper: Paper, max_log: float) -> float:
    citations = paper.citation_count
    if citations is None or citations <= 0:
        return 0.0
    if max_log <= 0.0:
        return 0.0
    return max(0.0, min(1.0, math.log(1 + citations) / max_log))


def _beginner_score(paper: Paper) -> float:
    title = (paper.title or "").lower()
    abstract = (paper.abstract or "").lower()
    if any(marker in title for marker in _BEGINNER_TITLE_MARKERS):
        return 1.0
    if any(marker in abstract for marker in ("review", "survey", "introductory", "tutorial")):
        return 0.5
    return 0.1


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {tok for tok in tokens if len(tok) > 2 and tok not in _STOPWORDS}


def _coverage_score(paper: Paper, question_tokens: set[str]) -> float:
    if not question_tokens:
        return 0.0
    haystack = f"{paper.title or ''} {paper.abstract or ''}"
    paper_tokens = _tokenize(haystack)
    if not paper_tokens:
        return 0.0
    overlap = len(question_tokens & paper_tokens)
    return max(0.0, min(1.0, overlap / len(question_tokens)))


def score_papers_for_comparative_axis(
    papers: list[Paper],
    question: str,
    axis: ComparativeAxis,
    *,
    relevance_scores: dict[str, float] | None = None,
    provider_bundle: Any = None,
) -> dict[str, float]:
    """Return ``{paper_id: score}`` for ``papers`` under ``axis``.

    Scores are deterministic, stdlib-only, and clipped to ``[0.0, 1.0]``.
    ``provider_bundle`` is accepted for forward compatibility but unused.
    """
    del provider_bundle  # reserved for future LLM-backed axis extensions
    result: dict[str, float] = {}
    if not papers:
        return result

    if axis == "recency":
        years = [p.year for p in papers if p.year is not None]
        min_year = min(years) if years else 0
        max_year = max(years) if years else 0
        for paper in papers:
            pid = _paper_key(paper)
            if pid:
                result[pid] = _year_score(paper, min_year, max_year)
        return result

    if axis == "authority":
        log_values = [math.log(1 + c) for c in (p.citation_count for p in papers) if c and c > 0]
        max_log = max(log_values) if log_values else 0.0
        for paper in papers:
            pid = _paper_key(paper)
            if pid:
                result[pid] = _authority_score(paper, max_log)
        return result

    if axis == "beginner":
        for paper in papers:
            pid = _paper_key(paper)
            if pid:
                result[pid] = _beginner_score(paper)
        return result

    if axis == "coverage":
        question_tokens = _tokenize(question or "")
        for paper in papers:
            pid = _paper_key(paper)
            if pid:
                result[pid] = _coverage_score(paper, question_tokens)
        return result

    # relevance_fallback: use externally supplied relevance scores when
    # available, else 0.0.
    for paper in papers:
        pid = _paper_key(paper)
        if not pid:
            continue
        if relevance_scores and pid in relevance_scores:
            result[pid] = max(0.0, min(1.0, float(relevance_scores[pid])))
        else:
            result[pid] = 0.0
    return result


def _paper_key(paper: Paper) -> str:
    return str(paper.paper_id or paper.canonical_id or "").strip()


__all__ = [
    "ComparativeAxis",
    "infer_comparative_axis",
    "score_papers_for_comparative_axis",
]
