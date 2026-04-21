"""Phase 7c-1: tests for the extracted ``planner.orchestrator`` submodule."""

from __future__ import annotations

from typing import Any

import pytest

from paper_chaser_mcp.agentic.config import AgenticConfig
from paper_chaser_mcp.agentic.models import ExpansionCandidate, PlannerDecision
from paper_chaser_mcp.agentic.planner import orchestrator as orchestrator_module
from paper_chaser_mcp.agentic.planner._core import (
    classify_query,
    grounded_expansion_candidates,
    speculative_expansion_candidates,
)


def _make_config() -> AgenticConfig:
    return AgenticConfig(
        enabled=True,
        provider="deterministic",
        planner_model="deterministic",
        synthesis_model="deterministic",
        embedding_model="deterministic",
        index_backend="memory",
        session_ttl_seconds=1800,
        enable_trace_log=False,
        max_grounded_variants=3,
        max_speculative_variants=3,
    )


class _PlannerBundle:
    def __init__(self, planner: PlannerDecision) -> None:
        self._planner = planner

    async def aplan_search(self, **kwargs: Any) -> PlannerDecision:
        return self._planner.model_copy(deep=True)


class _ExpansionBundle:
    def __init__(
        self,
        *,
        grounded: list[ExpansionCandidate] | None = None,
        speculative: list[ExpansionCandidate] | None = None,
    ) -> None:
        self._grounded = grounded or []
        self._speculative = speculative or []

    async def asuggest_grounded_expansions(
        self, *, query: str, papers: list[dict[str, Any]], max_variants: int
    ) -> list[ExpansionCandidate]:
        return list(self._grounded)[:max_variants]

    async def asuggest_speculative_expansions(
        self,
        *,
        query: str,
        evidence_texts: list[str],
        max_variants: int,
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> list[ExpansionCandidate]:
        return list(self._speculative)[:max_variants]


@pytest.mark.asyncio
async def test_classify_query_normalizes_and_returns_planner() -> None:
    seed = PlannerDecision(intent="discovery")
    normalized, planner = await classify_query(
        query="  coral reef bleaching  ",
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_PlannerBundle(seed),  # type: ignore[arg-type]
    )
    assert normalized == "coral reef bleaching"
    assert planner.intent == "discovery"


@pytest.mark.asyncio
async def test_classify_query_explicit_mode_overrides_intent() -> None:
    seed = PlannerDecision(intent="discovery")
    _, planner = await classify_query(
        query="systematic review of microplastic toxicity",
        mode="review",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_PlannerBundle(seed),  # type: ignore[arg-type]
    )
    assert planner.intent == "review"
    assert planner.intent_source == "explicit"


@pytest.mark.asyncio
async def test_grounded_expansion_candidates_adds_constraints_and_dedupes() -> None:
    bundle = _ExpansionBundle(
        grounded=[
            ExpansionCandidate(
                variant="thermal stress adaptation mechanisms",
                source="from_retrieved_evidence",
                rationale="r1",
            ),
        ]
    )
    result = await grounded_expansion_candidates(
        original_query="coral bleaching",
        papers=[{"title": "bleaching", "abstract": "coral"}],
        config=_make_config(),
        provider_bundle=bundle,  # type: ignore[arg-type]
        focus="reef",
    )
    assert result[0].source == "from_input"
    assert "reef" in result[0].variant
    assert any(c.source == "from_retrieved_evidence" for c in result)
    variants_lower = [c.variant.lower() for c in result]
    assert len(set(variants_lower)) == len(variants_lower)


@pytest.mark.asyncio
async def test_speculative_expansion_candidates_respects_bound() -> None:
    bundle = _ExpansionBundle(
        speculative=[ExpansionCandidate(variant=f"variant {i}", source="speculative", rationale="r") for i in range(10)]
    )
    result = await speculative_expansion_candidates(
        original_query="climate",
        papers=[{"title": "x", "abstract": "y"}],
        config=_make_config(),
        provider_bundle=bundle,  # type: ignore[arg-type]
    )
    assert len(result) == 3


def test_core_symbols_identity_matches_orchestrator() -> None:
    assert classify_query is orchestrator_module.classify_query
    assert grounded_expansion_candidates is orchestrator_module.grounded_expansion_candidates
    assert speculative_expansion_candidates is orchestrator_module.speculative_expansion_candidates
