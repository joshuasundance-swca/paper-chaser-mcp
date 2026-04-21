"""Graph-expansion helpers extracted in Phase 7b from ``graphs/_core``.

These pure helpers score and filter a frontier of related papers during
``expand_research_graph`` / graph-research workflows. The orchestrating
``AgenticRuntime._expand_seed`` and ``_semantic_recommendation_candidates``
methods stay on the class because they coordinate provider fan-out and
context progress events; the math and text-shaping helpers move here so
they become directly testable and the ``_core`` class body shrinks.
"""

from __future__ import annotations

import re
import time
from typing import Any

from ..planner import query_facets, query_terms
from ..provider_base import ModelProviderBundle
from .shared_state import _GRAPH_GENERIC_TERMS
from .source_records import _graph_topic_tokens, _paper_text


async def _graph_frontier_scores(
    *,
    seed: dict[str, Any],
    related_papers: list[dict[str, Any]],
    provider_bundle: ModelProviderBundle,
    intent_text: str | None = None,
) -> list[float]:
    if not related_papers:
        return []
    seed_title = str(seed.get("title") or "")
    seed_terms = [term for term in query_terms(seed_title or _paper_text(seed)) if term not in _GRAPH_GENERIC_TERMS]
    seed_facets = query_facets(seed_title or _paper_text(seed))
    seed_term_set = _graph_topic_tokens(seed_title or _paper_text(seed))
    normalized_intent_text = (intent_text or "").strip()
    intent_terms = [term for term in query_terms(normalized_intent_text) if term not in _GRAPH_GENERIC_TERMS]
    intent_facets = query_facets(normalized_intent_text)
    intent_term_set = _graph_topic_tokens(normalized_intent_text)
    query_similarities = await provider_bundle.abatched_similarity(
        _paper_text(seed),
        [_paper_text(related) for related in related_papers],
    )
    if normalized_intent_text:
        intent_similarities = await provider_bundle.abatched_similarity(
            normalized_intent_text,
            [_paper_text(related) for related in related_papers],
        )
    else:
        intent_similarities = query_similarities
    scores: list[float] = []
    for related, query_similarity, intent_similarity in zip(
        related_papers,
        query_similarities,
        intent_similarities,
        strict=False,
    ):
        related_title = str(related.get("title") or "")
        related_text = _paper_text(related).lower()
        related_tokens = _graph_topic_tokens(related_text)
        related_title_tokens = _graph_topic_tokens(related_title.lower())
        anchor_overlap = sum(term in related_tokens for term in seed_terms) / len(seed_terms) if seed_terms else 0.0
        intent_anchor_overlap = (
            sum(term in related_tokens for term in intent_terms) / len(intent_terms) if intent_terms else 0.0
        )
        facet_overlap = 0.0
        if seed_facets:
            matched_facets = 0
            for facet in seed_facets:
                facet_tokens = re.findall(r"[a-z0-9]{3,}", facet.lower())
                if not facet_tokens:
                    continue
                required = len(facet_tokens) if len(facet_tokens) <= 2 else 2
                if sum(token in related_tokens for token in facet_tokens) >= required:
                    matched_facets += 1
            facet_overlap = matched_facets / len(seed_facets)
        intent_facet_overlap = 0.0
        if intent_facets:
            matched_intent_facets = 0
            for facet in intent_facets:
                facet_tokens = re.findall(r"[a-z0-9]{3,}", facet.lower())
                if not facet_tokens:
                    continue
                required = len(facet_tokens) if len(facet_tokens) <= 2 else 2
                if sum(token in related_tokens for token in facet_tokens) >= required:
                    matched_intent_facets += 1
            intent_facet_overlap = matched_intent_facets / len(intent_facets)
        title_overlap = 0.0
        if seed_term_set and related_title_tokens:
            title_overlap = len(seed_term_set & related_title_tokens) / len(seed_term_set)
        intent_title_overlap = 0.0
        if intent_term_set and related_title_tokens:
            intent_title_overlap = len(intent_term_set & related_title_tokens) / len(intent_term_set)

        citation_count = related.get("citationCount")
        citation_bonus = 0.0
        if isinstance(citation_count, int) and citation_count > 0:
            citation_bonus = min(citation_count / 5000.0, 0.08)
        year = related.get("year")
        recency_bonus = 0.0
        if isinstance(year, int):
            current_year = time.gmtime().tm_year
            recency_bonus = max(0.0, 0.03 - max(0, current_year - year) * 0.005)
        topic_penalty = 0.0
        if seed_terms and anchor_overlap == 0.0:
            topic_penalty += 0.26
        elif seed_terms and anchor_overlap < 0.25:
            topic_penalty += 0.1
        if intent_terms and intent_anchor_overlap == 0.0:
            topic_penalty += 0.24
        elif intent_terms and intent_anchor_overlap < 0.25:
            topic_penalty += 0.1
        if seed_facets and facet_overlap == 0.0:
            topic_penalty += 0.2
        elif seed_facets and facet_overlap < 0.5:
            topic_penalty += 0.08
        if intent_facets and intent_facet_overlap == 0.0:
            topic_penalty += 0.2
        elif intent_facets and intent_facet_overlap < 0.5:
            topic_penalty += 0.08
        if title_overlap == 0.0:
            topic_penalty += 0.12
        elif title_overlap < 0.2:
            topic_penalty += 0.05
        if intent_term_set and intent_title_overlap == 0.0:
            topic_penalty += 0.14
        elif intent_term_set and intent_title_overlap < 0.2:
            topic_penalty += 0.06
        if seed_terms and intent_terms and anchor_overlap == 0.0 and intent_anchor_overlap == 0.0:
            topic_penalty += 0.12
        score = (
            (query_similarity * 0.28)
            + (intent_similarity * 0.24)
            + (anchor_overlap * 0.12)
            + (intent_anchor_overlap * 0.16)
            + (facet_overlap * 0.08)
            + (intent_facet_overlap * 0.12)
            + (title_overlap * 0.05)
            + (intent_title_overlap * 0.09)
            + citation_bonus
            + recency_bonus
            - topic_penalty
        )
        scores.append(round(max(score, 0.0), 6))
    return scores


def _graph_intent_text(
    record: Any | None,
    resolved_seeds: list[dict[str, Any]],
) -> str:
    if record is not None:
        metadata = record.metadata if isinstance(record.metadata, dict) else {}
        strategy_metadata = metadata.get("strategyMetadata")
        if isinstance(strategy_metadata, dict):
            normalized_query = strategy_metadata.get("normalizedQuery")
            if isinstance(normalized_query, str) and normalized_query.strip():
                return normalized_query.strip()
        original_query = metadata.get("originalQuery")
        if isinstance(original_query, str) and original_query.strip():
            return original_query.strip()
        if isinstance(record.query, str) and record.query.strip():
            return record.query.strip()
    for seed in resolved_seeds:
        for candidate in (seed.get("title"), seed.get("paperId")):
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return ""


def _filter_graph_frontier(
    ranked_related: list[tuple[dict[str, Any], float]],
) -> list[tuple[dict[str, Any], float]]:
    if not ranked_related:
        return []
    best_score = max(score for _, score in ranked_related)
    threshold = max(0.18, best_score * 0.45)
    retained = [(paper, score) for paper, score in ranked_related if score >= threshold]
    if retained:
        return retained
    return ranked_related[: min(3, len(ranked_related))]
