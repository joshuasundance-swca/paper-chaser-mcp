"""Tests for Workstream D: heritage_cultural_resources intent family.

Covers:
1. LLM-provided ``intent_family`` is preserved by ``classify_query``.
2. Deterministic fallback sets ``intent_family`` when the LLM does not,
   using the existing ``_detect_cultural_resource_intent`` helper.
3. ``_rank_regulatory_documents`` prefers cultural-resource documents
   when ``cultural_resource_boost=True``.
4. Queries with no cultural-resource signals leave ``intent_family`` at ``None``
   (no regression on existing intents).
"""

from __future__ import annotations

import pytest

from paper_chaser_mcp.agentic.graphs import _rank_regulatory_documents
from paper_chaser_mcp.agentic.models import PlannerDecision
from paper_chaser_mcp.agentic.planner import classify_query


class _Bundle:
    def __init__(self, planner: PlannerDecision) -> None:
        self._planner = planner

    async def aplan_search(self, **kwargs: object) -> PlannerDecision:
        return self._planner.model_copy(deep=True)


@pytest.mark.asyncio
async def test_classify_query_preserves_llm_supplied_intent_family() -> None:
    """If the LLM sets ``intent_family``, classify_query must preserve it."""
    seed = PlannerDecision(
        intent="regulatory",
        followUpMode="qa",
        intentFamily="heritage_cultural_resources",
    )
    _, planner = await classify_query(
        query="Section 106 consultation for tribal cultural resources",
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_Bundle(seed),  # type: ignore[arg-type]
    )
    assert planner.intent_family == "heritage_cultural_resources"


@pytest.mark.asyncio
async def test_classify_query_fallback_sets_heritage_family_for_nhpa_query() -> None:
    """When the LLM omits ``intent_family`` but cultural markers are present,
    the deterministic fallback should fill it in."""
    seed = PlannerDecision(intent="regulatory", followUpMode="qa")
    _, planner = await classify_query(
        query="NHPA Section 106 THPO consultation sacred site",
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_Bundle(seed),  # type: ignore[arg-type]
    )
    assert planner.intent_family == "heritage_cultural_resources"


@pytest.mark.asyncio
async def test_classify_query_fallback_fires_across_intents() -> None:
    """The fallback is cross-cutting; it should also apply when the LLM routes
    the query to a non-regulatory intent (e.g., discovery)."""
    seed = PlannerDecision(intent="discovery", followUpMode="qa")
    _, planner = await classify_query(
        query="archaeological resources act compliance literature review",
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_Bundle(seed),  # type: ignore[arg-type]
    )
    assert planner.intent_family == "heritage_cultural_resources"


@pytest.mark.asyncio
async def test_classify_query_leaves_intent_family_none_for_non_cultural_query() -> None:
    """Non-cultural queries must leave ``intent_family`` at ``None``."""
    seed = PlannerDecision(intent="discovery", followUpMode="qa")
    _, planner = await classify_query(
        query="graph neural networks for molecular property prediction",
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_Bundle(seed),  # type: ignore[arg-type]
    )
    assert planner.intent_family is None


def test_rank_regulatory_documents_prefers_cultural_resource_doc_when_boost_on() -> None:
    """When ``cultural_resource_boost=True`` a cultural-resource FR notice
    should outrank a generic environmental notice."""
    cultural_notice = {
        "title": "Notice of Section 106 NHPA tribal consultation for cultural heritage sites",
        "abstract": "Tribal historic preservation office coordination for archaeological resources.",
        "documentType": "notice",
        "publicationDate": "2023-04-15",
    }
    generic_notice = {
        "title": "Notice of air quality permit modification",
        "abstract": "General environmental compliance notice for industrial emissions.",
        "documentType": "notice",
        "publicationDate": "2023-04-15",
    }

    ranked_on = _rank_regulatory_documents(
        [generic_notice, cultural_notice],
        subject_terms=set(),
        priority_terms=set(),
        facet_terms=[],
        prefer_guidance=False,
        prefer_recent=False,
        cultural_resource_boost=True,
    )
    assert ranked_on[0] is cultural_notice

    ranked_off = _rank_regulatory_documents(
        [cultural_notice, generic_notice],
        subject_terms=set(),
        priority_terms=set(),
        facet_terms=[],
        prefer_guidance=False,
        prefer_recent=False,
        cultural_resource_boost=False,
    )
    # Without the boost the two notices score identically on these inputs,
    # so ordering falls back to the stable input order.
    assert cultural_notice in ranked_off
    assert generic_notice in ranked_off
