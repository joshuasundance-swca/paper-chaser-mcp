"""Tests for LLM classification provenance surfaces.

When the deterministic topical-relevance gate disagrees with the LLM
classification, we must keep the deterministic verdict as the effective label
but still surface the raw LLM signal + a ``classificationSource`` tag so
agents can observe the override. A counter is also emitted in
``strategyMetadata.llmClassificationOverrides`` so smart-search callers can
see how often the override fired.

These tests cover:

1. ``_classify_topical_relevance_with_provenance`` records both verdicts and
   marks ``llm_override_ignored`` in the clear-fast-path + disagreement case,
   and tags the source correctly for the weak-match tiebreaker and middle-zone
   LLM paths.
2. ``search_papers_smart`` propagates the override into
   ``strategyMetadata.llmClassificationOverrides`` and the per-result
   ``llmClassification`` / ``classificationSource`` fields when at least one
   disagreement occurs.
"""

from __future__ import annotations

from typing import Any

import pytest

from paper_chaser_mcp.agentic.graphs import (
    TopicalRelevanceClassification,
    _classify_topical_relevance_with_provenance,
)
from tests.helpers import RecordingOpenAlexClient, RecordingSemanticClient
from tests.test_smart_tools import _deterministic_runtime


def _paper(title: str, *, abstract: str = "") -> dict[str, Any]:
    return {
        "paperId": title.lower().replace(" ", "-"),
        "title": title,
        "abstract": abstract or f"Abstract discussing {title}.",
        "year": 2024,
        "source": "semantic_scholar",
    }


def test_provenance_fast_path_on_topic_with_llm_disagreement_flags_override() -> None:
    query = "microplastic effects in freshwater benthic macroinvertebrates"
    paper = _paper(
        "Microplastic Effects on Freshwater Benthic Macroinvertebrates",
        abstract=(
            "Study of microplastic exposure thresholds for freshwater benthic "
            "macroinvertebrate communities in river systems."
        ),
    )

    result = _classify_topical_relevance_with_provenance(
        query=query,
        paper=paper,
        query_similarity=0.92,  # fast-path on_topic
        llm_classification="off_topic",
    )

    assert isinstance(result, TopicalRelevanceClassification)
    assert result.effective == "on_topic"  # deterministic wins
    assert result.deterministic == "on_topic"
    assert result.llm == "off_topic"
    assert result.source == "deterministic"
    assert result.llm_override_ignored is True


def test_provenance_fast_path_off_topic_with_llm_disagreement_flags_override() -> None:
    query = "microplastic effects in freshwater benthic macroinvertebrates"
    paper = _paper(
        "Medieval Trade Routes Across the Silk Road",
        abstract="A survey of 14th-century overland trade.",
    )

    result = _classify_topical_relevance_with_provenance(
        query=query,
        paper=paper,
        query_similarity=0.05,  # fast-path off_topic: sim<0.12 and no signal
        llm_classification="on_topic",
    )

    assert result.effective == "off_topic"
    assert result.deterministic == "off_topic"
    assert result.llm == "on_topic"
    assert result.source == "deterministic"
    assert result.llm_override_ignored is True


def test_provenance_weak_match_with_llm_classification_is_llm_tiebreaker() -> None:
    query = "microplastic effects in freshwater benthic macroinvertebrates"
    # Title has no query anchors (→ no title signal); abstract has one term
    # (→ body signal only), so deterministic lands in the middle weak_match
    # zone for similarity in [0.12, 0.50].
    paper = _paper(
        "General survey methodology overview",
        abstract="A methods paper mentioning microplastic sampling in passing.",
    )

    result = _classify_topical_relevance_with_provenance(
        query=query,
        paper=paper,
        query_similarity=0.30,
        llm_classification="on_topic",
    )

    assert result.deterministic == "weak_match"
    assert result.effective == "on_topic"
    assert result.llm == "on_topic"
    assert result.source == "llm_tiebreaker"
    assert result.llm_override_ignored is False


def test_provenance_no_llm_signal_returns_deterministic_source() -> None:
    query = "microplastic effects in freshwater benthic macroinvertebrates"
    paper = _paper(
        "General survey methodology overview",
        abstract="A methods paper mentioning microplastic sampling in passing.",
    )

    result = _classify_topical_relevance_with_provenance(
        query=query,
        paper=paper,
        query_similarity=0.30,
        llm_classification=None,
    )

    assert result.llm is None
    assert result.source == "deterministic"
    assert result.llm_override_ignored is False


@pytest.mark.asyncio
async def test_search_papers_smart_records_llm_override_in_strategy_metadata() -> None:
    """Smart search should expose ``llmClassificationOverrides`` and the raw
    LLM verdict on each hit when the deterministic fast-path overrode the LLM
    signal.
    """

    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()

    strong_paper = {
        "paperId": "strong-match",
        "title": "Microplastic Effects on Freshwater Benthic Macroinvertebrates",
        "abstract": (
            "Comprehensive synthesis of microplastic exposure in freshwater benthic "
            "macroinvertebrates across river and lake systems."
        ),
        "year": 2024,
        "source": "semantic_scholar",
    }
    weak_paper = {
        "paperId": "weak-middle",
        "title": "Microplastic survey methodology overview",
        "abstract": "General methods paper focused on laboratory sampling protocols.",
        "year": 2024,
        "source": "semantic_scholar",
    }

    async def semantic_search(**kwargs: object) -> dict[str, Any]:
        semantic.calls.append(("search_papers", dict(kwargs)))
        return {"total": 2, "offset": 0, "data": [strong_paper, weak_paper]}

    async def empty_openalex_search(**kwargs: object) -> dict[str, Any]:
        openalex.calls.append(("search", dict(kwargs)))
        return {"total": 0, "offset": 0, "data": []}

    semantic.search_papers = semantic_search  # type: ignore[method-assign]
    openalex.search = empty_openalex_search  # type: ignore[method-assign]

    _, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)

    # Force similarity so the strong paper hits the deterministic fast-path
    # (>0.5) while the weak paper stays in the middle zone.
    async def _forced_similarity(query: str, texts: list[str], **_: object) -> list[float]:
        del query
        scores: list[float] = []
        for text in texts:
            lowered = text.lower()
            if "benthic" in lowered:
                scores.append(0.92)
            else:
                scores.append(0.30)
        return scores

    runtime._provider_bundle.abatched_similarity = _forced_similarity  # type: ignore[method-assign]
    runtime._deterministic_bundle.abatched_similarity = _forced_similarity  # type: ignore[method-assign]

    # Mock LLM batch to return classifications for BOTH papers. The strong
    # paper's LLM verdict disagrees with the deterministic fast-path, which
    # should trigger the override counter. The weak-match paper gets a
    # matching classification so it exercises the llm_tiebreaker path.
    async def _relevance_batch(**_: object) -> dict[str, dict[str, str]]:
        return {
            "strong-match": {
                "classification": "off_topic",
                "rationale": "LLM disagrees with the deterministic fast-path.",
            },
            "weak-middle": {
                "classification": "on_topic",
                "rationale": "LLM says on_topic for the middle-zone candidate.",
            },
        }

    runtime._provider_bundle.aclassify_relevance_batch = _relevance_batch  # type: ignore[method-assign]
    runtime._deterministic_bundle.aclassify_relevance_batch = _relevance_batch  # type: ignore[method-assign]

    payload = await runtime.search_papers_smart(
        query="microplastic effects in freshwater benthic macroinvertebrates",
        limit=5,
    )

    strategy = payload["strategyMetadata"]
    assert strategy.get("llmClassificationOverrides", 0) >= 1

    results_by_id = {hit["paper"]["paperId"]: hit for hit in payload["results"]}
    strong_hit = results_by_id["strong-match"]
    assert strong_hit["topicalRelevance"] == "on_topic"  # deterministic wins
    assert strong_hit["llmClassification"] == "off_topic"
    assert strong_hit["classificationSource"] == "deterministic"

    weak_hit = results_by_id["weak-middle"]
    # The weak-match deterministic + on_topic LLM → LLM breaks the tie.
    assert weak_hit["llmClassification"] == "on_topic"
    assert weak_hit["classificationSource"] == "llm_tiebreaker"
    assert weak_hit["topicalRelevance"] == "on_topic"

    structured_by_id = {src["sourceId"]: src for src in payload["structuredSources"] if src.get("sourceId")}
    assert structured_by_id["strong-match"]["llmClassification"] == "off_topic"
    assert structured_by_id["strong-match"]["classificationSource"] == "deterministic"
