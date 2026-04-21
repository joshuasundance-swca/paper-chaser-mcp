"""Landscape/inspect subgraph helpers extracted from ``_core`` (Phase 7b).

Pure module-level helpers used by ``AgenticRuntime.map_research_landscape``.
The async class method remains on ``_core.AgenticRuntime``; only the
side-effect-free theme/cluster/gap helpers live here.
"""

from __future__ import annotations

import re
import statistics
from typing import Any

from ..models import LandscapeTheme
from ..provider_base import ModelProviderBundle
from .shared_state import _THEME_LABEL_STOPWORDS
from .source_records import _paper_text


def _label_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{3,}", text.lower())


def _theme_terms_from_papers(seed_terms: list[str], papers: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for paper in papers[:8]:
        for token in _label_tokens(str(paper.get("title") or "")):
            if token in _THEME_LABEL_STOPWORDS or token.isdigit():
                continue
            counts[token] = counts.get(token, 0) + 3
        for token in _label_tokens(str(paper.get("abstract") or "")):
            if token in _THEME_LABEL_STOPWORDS or token.isdigit():
                continue
            counts[token] = counts.get(token, 0) + 1
    for term in seed_terms:
        for token in _label_tokens(term):
            if token in _THEME_LABEL_STOPWORDS or token.isdigit():
                continue
            counts[token] = counts.get(token, 0) + 2
    return [term for term, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def _normalized_theme_label(raw_label: str) -> str:
    parts = [segment.strip() for segment in re.split(r"[/|,:;\-]+", raw_label) if segment.strip()]
    if not parts:
        return ""
    return " / ".join(part.title() for part in parts[:2])


def _finalize_theme_label(
    *,
    raw_label: str,
    seed_terms: list[str],
    papers: list[dict[str, Any]],
) -> str:
    normalized = " ".join(raw_label.split())
    parts = [segment.strip() for segment in re.split(r"[/|,:;\-]+", normalized) if segment.strip()]
    if normalized and parts:
        part_tokens = [_label_tokens(part) for part in parts]
        part_meaningful = [[token for token in tokens if token not in _THEME_LABEL_STOPWORDS] for tokens in part_tokens]
        if all(tokens for tokens in part_meaningful):
            return _normalized_theme_label(normalized)
    derived_terms = _theme_terms_from_papers(seed_terms, papers)
    if len(derived_terms) >= 2:
        return " / ".join(term.title() for term in derived_terms[:2])
    if derived_terms:
        return derived_terms[0].title()
    if normalized:
        tokens = [token for token in _label_tokens(normalized) if token not in _THEME_LABEL_STOPWORDS]
        if tokens:
            return " / ".join(token.title() for token in tokens[:2])
    return "General theme"


def _top_terms_for_cluster(papers: list[dict[str, Any]]) -> list[str]:
    tokens: dict[str, int] = {}
    for paper in papers[:8]:
        for token in _paper_text(paper).lower().split():
            cleaned = "".join(character for character in token if character.isalnum())
            if len(cleaned) < 4:
                continue
            tokens[cleaned] = tokens.get(cleaned, 0) + 1
    return [
        term
        for term, _ in sorted(
            tokens.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:4]
    ]


async def _cluster_papers(
    *,
    papers: list[dict[str, Any]],
    provider_bundle: ModelProviderBundle,
    max_themes: int,
) -> list[list[dict[str, Any]]]:
    if not papers:
        return []
    remaining = list(papers)
    clusters: list[list[dict[str, Any]]] = []
    threshold = 0.22
    while remaining and len(clusters) < max(max_themes, 1):
        seed = remaining.pop(0)
        cluster = [seed]
        seed_text = _paper_text(seed)
        candidate_texts = [_paper_text(candidate) for candidate in remaining]
        similarities = await provider_bundle.abatched_similarity(
            seed_text,
            candidate_texts,
        )
        rest: list[dict[str, Any]] = []
        for candidate, similarity in zip(remaining, similarities, strict=False):
            if similarity >= threshold:
                cluster.append(candidate)
            else:
                rest.append(candidate)
        clusters.append(cluster)
        remaining = rest
    if remaining:
        clusters[-1].extend(remaining)
    return clusters[:max_themes]


def _compute_gaps(papers: list[dict[str, Any]]) -> list[str]:
    if not papers:
        return ["No papers were available to analyze for gaps."]
    years: list[int] = [paper["year"] for paper in papers if isinstance(paper.get("year"), int)]
    venues: set[str] = {
        str(paper["venue"]) for paper in papers if isinstance(paper.get("venue"), str) and paper.get("venue")
    }
    gaps: list[str] = []
    if years and max(years) - min(years) <= 1:
        gaps.append(
            "The current result set is concentrated in a narrow time window; earlier foundational work may be missing."
        )
    if len(venues) <= 1:
        gaps.append("Most papers cluster around one venue or source, so cross-community coverage may still be thin.")
    if not gaps:
        gaps.append(
            "Methodological diversity looks reasonable, but targeted "
            "negative-result or benchmark papers may still be underrepresented."
        )
    return gaps


def _compute_disagreements(papers: list[dict[str, Any]]) -> list[str]:
    if len(papers) < 3:
        return ["The result set is still small, so disagreements are not yet obvious."]
    years: list[int] = [paper["year"] for paper in papers if isinstance(paper.get("year"), int)]
    if years and statistics.pstdev(years) >= 2.5:
        return [
            "The papers span different periods, so assumptions and evaluation "
            "norms may disagree across older and newer work."
        ]
    return [
        "Evaluation setups and coverage differ across the returned papers, so "
        "direct comparisons should be made carefully."
    ]


def _suggest_next_searches(
    papers: list[dict[str, Any]],
    themes: list[LandscapeTheme],
) -> list[str]:
    suggestions: list[str] = []
    if themes:
        suggestions.append(f"{themes[0].title} benchmark papers")
        suggestions.append(f"{themes[0].title} survey")
    recent_years: list[int] = sorted(
        {paper["year"] for paper in papers if isinstance(paper.get("year"), int)},
        reverse=True,
    )
    if recent_years:
        suggestions.append(f"{recent_years[0]} follow-up work")
    deduped: list[str] = []
    seen: set[str] = set()
    for suggestion in suggestions:
        if suggestion not in seen:
            seen.add(suggestion)
            deduped.append(suggestion)
    return deduped[:3]


__all__ = [
    "_cluster_papers",
    "_compute_disagreements",
    "_compute_gaps",
    "_finalize_theme_label",
    "_label_tokens",
    "_normalized_theme_label",
    "_suggest_next_searches",
    "_theme_terms_from_papers",
    "_top_terms_for_cluster",
]
