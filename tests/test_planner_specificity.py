"""Phase 6 red-bar: pin specificity / ambiguity estimation helpers."""

from __future__ import annotations

from paper_chaser_mcp.agentic import planner as legacy_planner
from paper_chaser_mcp.agentic.models import IntentCandidate
from paper_chaser_mcp.agentic.planner.specificity import (
    _estimate_ambiguity_level,
    _estimate_query_specificity,
    _is_definitional_query,
    _looks_broad_concept_query,
    _query_starts_broad,
)


def test_query_starts_broad_detects_question_starters() -> None:
    assert _query_starts_broad("what is the transformer architecture")
    assert _query_starts_broad("how does attention work")
    assert _query_starts_broad("compare BERT and GPT")
    assert not _query_starts_broad("Attention Is All You Need")


def test_is_definitional_query_matches_patterns() -> None:
    assert _is_definitional_query("what is retrieval augmented generation")
    assert _is_definitional_query("define reinforcement learning")
    assert _is_definitional_query("overview of climate feedback loops")
    assert not _is_definitional_query("Attention Is All You Need")
    assert not _is_definitional_query("")


def test_looks_broad_concept_query_flags_open_ended_questions() -> None:
    assert _looks_broad_concept_query(
        normalized_query="what are the latest methods for evidence different studies review",
        focus=None,
        year=None,
        venue=None,
    )
    assert not _looks_broad_concept_query(
        normalized_query="Attention Is All You Need",
        focus=None,
        year=None,
        venue=None,
    )


def test_estimate_query_specificity_narrow_vs_broad() -> None:
    # DOI / URL-bearing queries are strong known-item → high.
    assert (
        _estimate_query_specificity(
            normalized_query="10.1234/abcd.efgh",
            focus=None,
            year=None,
            venue=None,
        )
        == "high"
    )
    # Broad conceptual question with queryish stopwords → low.
    assert (
        _estimate_query_specificity(
            normalized_query="what are the latest methods for evidence different studies review",
            focus=None,
            year=None,
            venue=None,
        )
        == "low"
    )


def test_estimate_ambiguity_level_handles_confidence_mix() -> None:
    # Single candidate + high specificity → low ambiguity.
    candidates_single = [
        IntentCandidate(intent="discovery", confidence="high", source="planner", rationale=""),
    ]
    assert (
        _estimate_ambiguity_level(
            candidates=candidates_single,
            routing_confidence="high",
            query_specificity="high",
        )
        == "low"
    )
    # Tied confidence between two candidates → high ambiguity.
    candidates_tied = [
        IntentCandidate(intent="discovery", confidence="medium", source="planner", rationale=""),
        IntentCandidate(intent="review", confidence="medium", source="heuristic", rationale=""),
    ]
    assert (
        _estimate_ambiguity_level(
            candidates=candidates_tied,
            routing_confidence="medium",
            query_specificity="medium",
        )
        == "high"
    )
    # Low routing confidence → always high ambiguity.
    assert (
        _estimate_ambiguity_level(
            candidates=candidates_single,
            routing_confidence="low",
            query_specificity="high",
        )
        == "high"
    )


def test_legacy_planner_reexports_specificity_helpers() -> None:
    for name in (
        "_estimate_query_specificity",
        "_estimate_ambiguity_level",
        "_query_starts_broad",
        "_is_definitional_query",
        "_looks_broad_concept_query",
    ):
        assert hasattr(legacy_planner, name), f"legacy planner missing {name}"
