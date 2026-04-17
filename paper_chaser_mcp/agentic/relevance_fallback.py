"""Deterministic three-way relevance fallback with rich provenance.

This module is used when the LLM batch relevance classifier fails (even after
retry) so we do not regress to a blanket ``weak_match`` label. It reuses the
existing lexical signals exposed by :mod:`provider_base` and adds explicit
``relevanceSource``/``relevanceConfidence``/``relevanceReason`` metadata so the
rest of the pipeline can surface a degraded-mode signal to callers.
"""

from __future__ import annotations

from typing import Any, Literal

from .provider_base import (
    _fallback_query_facets,
    _fallback_query_terms,
    classify_relevance_without_llm,
    relevance_paper_identifier,
)
from .provider_helpers import _lexical_similarity, _tokenize

RelevanceLabel = Literal["on_topic", "weak_match", "off_topic"]
RelevanceSource = Literal["llm", "llm_retry", "deterministic_tier", "hybrid"]


def _signal_profile(query: str, paper: dict[str, Any]) -> dict[str, float]:
    title = str(paper.get("title") or "")
    abstract = str(paper.get("abstract") or "")
    paper_text = " ".join(part for part in [title, abstract] if part)
    similarity = _lexical_similarity(query, paper_text)
    query_terms = _fallback_query_terms(query)
    title_tokens = set(_tokenize(title))
    body_tokens = set(_tokenize(paper_text))
    title_hits = sum(1 for term in query_terms if term in title_tokens) if query_terms else 0
    body_hits = sum(1 for term in query_terms if term in body_tokens) if query_terms else 0
    facets = _fallback_query_facets(query)
    facet_title_hits = 0
    facet_body_hits = 0
    for facet in facets:
        required = len(facet) if len(facet) <= 2 else 2
        if sum(token in title_tokens for token in facet) >= required:
            facet_title_hits += 1
        if sum(token in body_tokens for token in facet) >= required:
            facet_body_hits += 1
    return {
        "similarity": similarity,
        "title_anchor_coverage": (title_hits / len(query_terms)) if query_terms else 0.0,
        "body_anchor_coverage": (body_hits / len(query_terms)) if query_terms else 0.0,
        "title_facet_coverage": (facet_title_hits / len(facets)) if facets else 0.0,
        "body_facet_coverage": (facet_body_hits / len(facets)) if facets else 0.0,
    }


def _confidence_for(label: RelevanceLabel, profile: dict[str, float]) -> float:
    similarity = profile["similarity"]
    if label == "on_topic":
        base = 0.55 + 0.25 * profile["title_facet_coverage"] + 0.15 * profile["title_anchor_coverage"]
    elif label == "off_topic":
        # Less overlap → more confident it's off-topic.
        base = 0.55 + 0.30 * (1.0 - min(similarity * 4.0, 1.0))
    else:
        base = 0.35
    return round(max(0.2, min(0.9, base)), 3)


_RATIONALE_MAX_CHARS = 150


def _compose_classification_rationale(label: RelevanceLabel, profile: dict[str, float]) -> str:
    """Compose a compact (<150 char) user-facing rationale from signal profile.

    Parameterized over already-computed signals (similarity, anchor/facet
    coverage) so this stays domain-agnostic. Distinct from the verbose
    ``relevanceReason`` debug string kept for diagnostics.
    """
    similarity = profile["similarity"]
    title_anchor = profile["title_anchor_coverage"]
    body_anchor = profile["body_anchor_coverage"]
    title_facet = profile["title_facet_coverage"]
    body_facet = profile["body_facet_coverage"]
    if label == "on_topic":
        text = (
            f"Strong topical overlap: title anchors {title_anchor:.0%}, facet coverage {title_facet:.0%}."
        )
    elif label == "off_topic":
        if title_anchor == 0 and body_anchor == 0:
            text = "No query-term overlap in title or abstract; low lexical similarity."
        else:
            text = (
                f"Low topical overlap: lexical similarity {similarity:.2f}, "
                f"title anchors {title_anchor:.0%}."
            )
    else:  # weak_match
        if title_anchor == 0 and body_anchor > 0:
            text = (
                f"Query terms appear only in abstract (body anchors {body_anchor:.0%}); "
                "title lacks direct overlap."
            )
        elif 0.0 < title_facet < 1.0:
            text = (
                f"Partial facet coverage ({title_facet:.0%}); some query aspects missing in title."
            )
        elif body_facet > 0 and title_facet == 0:
            text = f"Facets mentioned in abstract ({body_facet:.0%}) but absent from title."
        else:
            text = (
                f"Borderline match: similarity {similarity:.2f}, title anchors {title_anchor:.0%}."
            )
    return text[:_RATIONALE_MAX_CHARS]


def classify_paper_deterministic(
    *,
    query: str,
    paper: dict[str, Any],
    reason: str = "batch_classifier_failed",
) -> dict[str, Any]:
    """Return a single classification entry with deterministic-tier provenance.

    The return shape is a superset of :func:`classify_relevance_without_llm` so
    existing consumers that only look at ``classification``/``rationale``/
    ``fallback``/``provenance`` keep working. Callers that want richer
    diagnostics can read ``relevanceSource``, ``relevanceConfidence`` and
    ``relevanceReason``.
    """
    base = classify_relevance_without_llm(query=query, paper=paper)
    profile = _signal_profile(query, paper)
    label: RelevanceLabel = base["classification"]
    confidence = _confidence_for(label, profile)
    signals = (
        f"titleAnchorCoverage={profile['title_anchor_coverage']:.2f}, "
        f"titleFacetCoverage={profile['title_facet_coverage']:.2f}, "
        f"bodyFacetCoverage={profile['body_facet_coverage']:.2f}, "
        f"lexicalSimilarity={profile['similarity']:.2f}"
    )
    relevance_reason = (
        f"Deterministic three-way tier (trigger={reason}). {base['rationale']} Signals: {signals}."
    )
    base.update(
        {
            "relevanceSource": "deterministic_tier",
            "relevanceConfidence": confidence,
            "relevanceReason": relevance_reason,
            "classificationRationale": _compose_classification_rationale(label, profile),
            "degradedTrigger": reason,
        }
    )
    return base


def classify_batch_deterministic(
    *,
    query: str,
    papers: list[dict[str, Any]],
    reason: str = "batch_classifier_failed",
) -> dict[str, dict[str, Any]]:
    """Map paperId -> deterministic three-way classification entry.

    Used as the last-resort fallback when the LLM batch call and a retry both
    fail. Guarantees that callers receive a mix of labels grounded in concrete
    signals rather than a blanket ``weak_match``.
    """
    return {
        relevance_paper_identifier(paper, index): classify_paper_deterministic(
            query=query,
            paper=paper,
            reason=reason,
        )
        for index, paper in enumerate(papers)
    }


def annotate_llm_entry(
    entry: dict[str, Any],
    *,
    source: RelevanceSource = "llm",
    confidence: float = 0.88,
) -> dict[str, Any]:
    """Annotate an LLM-produced classification entry with provenance metadata."""
    entry.setdefault("relevanceSource", source)
    entry.setdefault("relevanceConfidence", confidence)
    rationale = str(entry.get("rationale") or "").strip()
    if rationale:
        entry.setdefault("relevanceReason", rationale)
        entry.setdefault("classificationRationale", rationale[:_RATIONALE_MAX_CHARS].rstrip())
    return entry


def classification_provenance_counts(
    entries: dict[str, dict[str, Any]],
    *,
    degraded_ratio_threshold: float = 0.4,
) -> dict[str, Any]:
    """Summarize classification provenance across a batch.

    Returns a dict shaped for ``trustSummary.classificationProvenance`` with
    per-source counters, total count, degraded-ratio, and a boolean
    ``degradedClassification`` flag that trips when more than
    ``degraded_ratio_threshold`` of entries came from a non-primary path
    (retry, deterministic tier, or hybrid).
    """
    total = len(entries)
    counts: dict[str, int] = {"llm": 0, "llm_retry": 0, "deterministic_tier": 0, "hybrid": 0}
    for entry in entries.values():
        source = str(entry.get("relevanceSource") or "").strip()
        if source not in counts:
            # Legacy entries (no relevanceSource). Infer from existing flags.
            if bool(entry.get("fallback")):
                source = "deterministic_tier"
            else:
                source = "llm"
        counts[source] = counts.get(source, 0) + 1
    degraded_count = counts["llm_retry"] + counts["deterministic_tier"] + counts["hybrid"]
    degraded_ratio = (degraded_count / total) if total else 0.0
    return {
        "counts": counts,
        "total": total,
        "degradedCount": degraded_count,
        "degradedRatio": round(degraded_ratio, 3),
        "degradedClassification": bool(total and degraded_ratio > degraded_ratio_threshold),
    }


__all__ = [
    "RelevanceLabel",
    "RelevanceSource",
    "annotate_llm_entry",
    "classification_provenance_counts",
    "classify_batch_deterministic",
    "classify_paper_deterministic",
    "_compose_classification_rationale",
]
