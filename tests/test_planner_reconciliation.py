"""Phase 7c-1: tests for the extracted ``planner.reconciliation`` submodule."""

from __future__ import annotations

import pytest

from paper_chaser_mcp.agentic.models import PlannerDecision
from paper_chaser_mcp.agentic.planner import reconciliation as reconciliation_module
from paper_chaser_mcp.agentic.planner._core import (
    _derive_regulatory_intent,
    _has_literature_corroboration,
    _VALID_REGULATORY_INTENTS,
)


def _make_planner(
    *,
    intent: str = "discovery",
    regulatory_subintent: str | None = None,
    secondary_intents: list[str] | None = None,
    retrieval_hypotheses: list[str] | None = None,
    search_angles: list[str] | None = None,
    candidate_concepts: list[str] | None = None,
) -> PlannerDecision:
    return PlannerDecision(
        intent=intent,  # type: ignore[arg-type]
        regulatorySubintent=regulatory_subintent,  # type: ignore[arg-type]
        secondaryIntents=secondary_intents or [],  # type: ignore[arg-type]
        retrievalHypotheses=retrieval_hypotheses or [],
        searchAngles=search_angles or [],
        candidateConcepts=candidate_concepts or [],
    )


def test_valid_regulatory_intents_contains_expected_labels() -> None:
    assert "current_cfr_text" in _VALID_REGULATORY_INTENTS
    assert "hybrid_regulatory_plus_literature" in _VALID_REGULATORY_INTENTS
    assert "unspecified" not in _VALID_REGULATORY_INTENTS


def test_has_literature_corroboration_detects_review_secondary() -> None:
    planner = _make_planner(secondary_intents=["review"])
    assert _has_literature_corroboration(planner=planner, query="EPA rule", focus=None) is True


def test_has_literature_corroboration_detects_hypothesis_marker() -> None:
    planner = _make_planner(retrieval_hypotheses=["Systematic review of compliance outcomes"])
    assert _has_literature_corroboration(planner=planner, query="stormwater", focus=None) is True


def test_has_literature_corroboration_rejects_regulation_only() -> None:
    planner = _make_planner(retrieval_hypotheses=["CFR text lookup"])
    assert _has_literature_corroboration(planner=planner, query="what does EPA require", focus=None) is False


def test_derive_regulatory_intent_preserves_valid_llm_label() -> None:
    planner = _make_planner(intent="regulatory", regulatory_subintent="current_cfr_text")
    assert _derive_regulatory_intent(planner=planner, query="40 CFR 122", focus=None) == "current_cfr_text"


def test_derive_regulatory_intent_returns_none_for_non_regulatory() -> None:
    planner = _make_planner(intent="discovery")
    assert _derive_regulatory_intent(planner=planner, query="coral reef bleaching", focus=None) is None


def test_derive_regulatory_intent_demotes_hybrid_without_corroboration() -> None:
    planner = _make_planner(
        intent="regulatory",
        regulatory_subintent="hybrid_regulatory_plus_literature",
    )
    result = _derive_regulatory_intent(planner=planner, query="EPA stormwater rule", focus=None)
    assert result != "hybrid_regulatory_plus_literature"


def test_reconciliation_submodule_exposes_expected_symbols() -> None:
    expected = {"_VALID_REGULATORY_INTENTS", "_has_literature_corroboration", "_derive_regulatory_intent"}
    missing = expected - set(dir(reconciliation_module))
    assert not missing, f"reconciliation submodule missing: {missing}"


def test_core_symbols_identity_matches_reconciliation_submodule() -> None:
    assert _VALID_REGULATORY_INTENTS is reconciliation_module._VALID_REGULATORY_INTENTS
    assert _has_literature_corroboration is reconciliation_module._has_literature_corroboration
    assert _derive_regulatory_intent is reconciliation_module._derive_regulatory_intent


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
