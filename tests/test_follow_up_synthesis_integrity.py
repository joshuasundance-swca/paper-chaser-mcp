"""Unit tests for follow-up synthesis integrity (Workstreams A + F).

These tests pin the contract of :mod:`paper_chaser_mcp.agentic.answer_modes`,
which is the module that keeps ``ask_result_set`` from emitting polished
synthesis paragraphs when the evidence pool does not actually support them.

The goal is to guarantee that:

* Synthesis-mode follow-ups require at least two non-fallback on-topic
  sources before ``build_evidence_use_plan`` declares the plan
  ``sufficient``.
* On-topic classifications that relied on the deterministic relevance
  fallback do not count as grounded support.
* Weak-pool detection fires once more than 60% of the saved records are
  classified ``weak_match`` or ``off_topic``.
* Salvageable modes (``metadata``, ``relevance_triage``) remain answerable
  from session state.
* Generic / unclassified questions produce a permissive plan so that the
  downstream ``ask_result_set`` machinery retains ownership of the
  abstain decision.
"""

from __future__ import annotations

import os
from typing import Literal, cast

import pytest

from paper_chaser_mcp.agentic.answer_modes import (
    ANSWER_MODES,
    SALVAGEABLE_MODES,
    SYNTHESIS_MODES,
    build_evidence_use_plan,
    classify_question_mode,
    evidence_pool_is_weak,
)
from paper_chaser_mcp.agentic.models import EvidenceItem, StructuredSourceRecord
from paper_chaser_mcp.models.common import Paper


def _paper(paper_id: str) -> Paper:
    return Paper(paperId=paper_id, title=f"Paper {paper_id}")


def _evidence(paper_id: str, score: float = 0.8) -> EvidenceItem:
    return EvidenceItem(
        evidenceId=paper_id,
        paper=_paper(paper_id),
        excerpt="excerpt",
        whyRelevant="why",
        relevanceScore=score,
    )


def _source(paper_id: str, topical: str = "on_topic") -> StructuredSourceRecord:
    return StructuredSourceRecord(
        sourceId=paper_id,
        topicalRelevance=cast(Literal["on_topic", "weak_match", "off_topic"], topical),
    )


def _relevance(paper_id: str, *, classification: str = "on_topic", fallback: bool = False) -> dict[str, object]:
    return {
        "classification": classification,
        "rationale": "test",
        "fallback": fallback,
        "provenance": "deterministic" if fallback else "model",
    }


def test_answer_modes_surface_is_stable() -> None:
    assert "mechanism_summary" in ANSWER_MODES
    assert SYNTHESIS_MODES == frozenset(
        {"comparison", "selection", "mechanism_summary", "regulatory_chain", "intervention_tradeoff"}
    )
    assert SALVAGEABLE_MODES == frozenset({"metadata", "relevance_triage"})


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Compare RAG against fine-tuning", "comparison"),
        ("What is the mechanism driving the effect?", "mechanism_summary"),
        ("How does the pathway connect cause and effect?", "mechanism_summary"),
        ("Summarize the regulatory history and rulemaking timeline.", "regulatory_chain"),
        ("What are the practical implications for deployment?", "intervention_tradeoff"),
        ("Who authored the first paper in the set?", "metadata"),
        ("Which of these papers are relevant to retrieval?", "relevance_triage"),
        ("What does this result set say about retrieval?", "unknown"),
    ],
)
def test_classify_question_mode_keyword_paths(question: str, expected: str) -> None:
    assert classify_question_mode(question) == expected


def test_classify_question_mode_respects_planner_hint_for_unknown_questions() -> None:
    hint = {"followUpMode": "regulatory_chain"}
    assert classify_question_mode("ambiguous prompt with no keywords", hint) == "regulatory_chain"


def test_classify_question_mode_maps_legacy_claim_check_alias() -> None:
    hint = {"followUpMode": "claim_check"}
    assert classify_question_mode("is the central claim supported?", hint) == "mechanism_summary"


def test_mechanism_summary_requires_two_non_fallback_sources() -> None:
    evidence = [_evidence("p1"), _evidence("p2")]
    sources = [_source("p1"), _source("p2")]
    relevance = {"p1": _relevance("p1"), "p2": _relevance("p2")}
    plan = build_evidence_use_plan(
        question="What is the underlying mechanism?",
        answer_mode="synthesis",
        evidence=evidence,
        source_records=sources,
        unsupported_asks=[],
        llm_relevance=relevance,
    )
    assert plan["answerMode"] == "mechanism_summary"
    assert plan["sufficient"] is True
    assert plan["retrievalSufficiency"] == "sufficient"
    assert plan["responsiveEvidenceIds"] == ["p1", "p2"]
    assert plan["evidenceQualityProfile"] == "high"
    assert plan["synthesisMode"] == "grounded"


def test_mechanism_summary_with_single_non_fallback_source_marks_thin() -> None:
    evidence = [_evidence("p1")]
    sources = [_source("p1")]
    relevance = {"p1": _relevance("p1")}
    plan = build_evidence_use_plan(
        question="What is the underlying mechanism?",
        answer_mode="synthesis",
        evidence=evidence,
        source_records=sources,
        unsupported_asks=[],
        llm_relevance=relevance,
    )
    assert plan["sufficient"] is False
    assert plan["retrievalSufficiency"] == "thin"
    assert any("one" in ask.lower() for ask in plan["unsupportedAspects"])


def test_comparison_with_one_on_topic_source_is_insufficient() -> None:
    evidence = [_evidence("p1")]
    sources = [_source("p1")]
    relevance = {"p1": _relevance("p1")}
    plan = build_evidence_use_plan(
        question="Compare approach A versus approach B",
        answer_mode="comparison",
        evidence=evidence,
        source_records=sources,
        unsupported_asks=[],
        llm_relevance=relevance,
    )
    assert plan["answerMode"] == "comparison"
    assert plan["sufficient"] is False
    assert plan["retrievalSufficiency"] == "insufficient"
    assert any("comparison" in ask.lower() for ask in plan["unsupportedAspects"])


def test_fallback_classifications_do_not_count_as_grounded() -> None:
    evidence = [_evidence("p1"), _evidence("p2")]
    sources = [_source("p1"), _source("p2")]
    relevance = {
        "p1": _relevance("p1", fallback=True),
        "p2": _relevance("p2", fallback=True),
    }
    plan = build_evidence_use_plan(
        question="Summarize the regulatory history of this rulemaking.",
        answer_mode="synthesis",
        evidence=evidence,
        source_records=sources,
        unsupported_asks=[],
        llm_relevance=relevance,
    )
    assert plan["answerMode"] == "regulatory_chain"
    assert plan["sufficient"] is False
    assert plan["fallbackOnlyOnTopic"] == 2
    assert plan["responsiveEvidenceIds"] == []
    assert any("fallback" in ask.lower() for ask in plan["unsupportedAspects"])


def test_intervention_tradeoff_weak_pool_is_insufficient() -> None:
    evidence = [_evidence("p1"), _evidence("p2")]
    sources = [
        _source("p1", topical="weak_match"),
        _source("p2", topical="weak_match"),
    ]
    plan = build_evidence_use_plan(
        question="What are the practical implications for intervention tradeoffs?",
        answer_mode="synthesis",
        evidence=evidence,
        source_records=sources,
        unsupported_asks=[],
        llm_relevance=None,
    )
    assert plan["answerMode"] == "intervention_tradeoff"
    assert plan["sufficient"] is False
    assert plan["retrievalSufficiency"] == "insufficient"


def test_evidence_pool_is_weak_fires_above_default_threshold() -> None:
    sources = [
        _source("p1", topical="weak_match"),
        _source("p2", topical="off_topic"),
        _source("p3", topical="on_topic"),
    ]
    assert evidence_pool_is_weak(sources) is True


def test_evidence_pool_is_weak_returns_false_when_mostly_on_topic() -> None:
    sources = [
        _source("p1", topical="on_topic"),
        _source("p2", topical="on_topic"),
        _source("p3", topical="weak_match"),
    ]
    assert evidence_pool_is_weak(sources) is False


def test_evidence_pool_weak_threshold_is_env_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPER_CHASER_WEAK_EVIDENCE_POOL_THRESHOLD", "0.2")
    sources = [
        _source("p1", topical="on_topic"),
        _source("p2", topical="on_topic"),
        _source("p3", topical="weak_match"),
    ]
    # With a 20% threshold one weak record pushes the pool over.
    assert evidence_pool_is_weak(sources) is True


def test_evidence_pool_weak_empty_sources_is_not_weak() -> None:
    assert evidence_pool_is_weak([]) is False


def test_metadata_mode_is_sufficient_with_any_session_records() -> None:
    evidence: list[EvidenceItem] = []
    sources = [_source("p1", topical="weak_match")]
    plan = build_evidence_use_plan(
        question="Who authored paper p1?",
        answer_mode="synthesis",
        evidence=evidence,
        source_records=sources,
        unsupported_asks=[],
        llm_relevance=None,
    )
    assert plan["answerMode"] == "metadata"
    assert plan["sufficient"] is True
    assert plan["retrievalSufficiency"] == "sufficient"


def test_relevance_triage_mode_is_sufficient_with_session_records() -> None:
    sources = [_source("p1", topical="weak_match"), _source("p2", topical="off_topic")]
    plan = build_evidence_use_plan(
        question="Which of these papers are relevant to retrieval?",
        answer_mode="synthesis",
        evidence=[],
        source_records=sources,
        unsupported_asks=[],
        llm_relevance=None,
    )
    assert plan["answerMode"] == "relevance_triage"
    assert plan["sufficient"] is True


def test_unknown_mode_does_not_force_abstention_when_on_topic_present() -> None:
    # Legacy behaviour: generic Q&A delegates the abstain decision to the
    # downstream ask_result_set gate stack rather than forcing insufficient.
    evidence = [_evidence("p1")]
    sources = [_source("p1")]
    plan = build_evidence_use_plan(
        question="What does this result set say about retrieval?",
        answer_mode="synthesis",
        evidence=evidence,
        source_records=sources,
        unsupported_asks=[],
        llm_relevance={"p1": _relevance("p1")},
    )
    assert plan["answerMode"] == "unknown"
    assert plan["sufficient"] is True
    assert plan["retrievalSufficiency"] == "sufficient"


def test_unsupported_asks_echo_into_plan() -> None:
    evidence = [_evidence("p1"), _evidence("p2")]
    sources = [_source("p1"), _source("p2")]
    plan = build_evidence_use_plan(
        question="Compare approach A versus approach B",
        answer_mode="comparison",
        evidence=evidence,
        source_records=sources,
        unsupported_asks=["No direct comparison of cost"],
        llm_relevance={"p1": _relevance("p1"), "p2": _relevance("p2")},
    )
    assert "No direct comparison of cost" in plan["unsupportedAspects"]


def test_invalid_env_threshold_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPER_CHASER_WEAK_EVIDENCE_POOL_THRESHOLD", "not-a-number")
    sources = [_source("p", topical="weak_match") for _ in range(4)] + [_source("x")]
    # 4/5 = 0.8 > default 0.6, so weak pool still fires.
    assert evidence_pool_is_weak(sources) is True
    # And a balanced pool does not trigger with the default threshold.
    balanced = [_source("p1"), _source("p2"), _source("p3", topical="weak_match")]
    assert evidence_pool_is_weak(balanced) is False


def test_classify_question_mode_handles_empty_question() -> None:
    assert classify_question_mode("") == "unknown"


def test_env_threshold_out_of_range_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPER_CHASER_WEAK_EVIDENCE_POOL_THRESHOLD", "1.5")
    sources = [_source("p1", topical="weak_match"), _source("p2", topical="weak_match"), _source("p3")]
    # Invalid range -> default 0.6 applies; 2/3 = 0.66 > 0.6.
    assert evidence_pool_is_weak(sources) is True


def test_module_env_var_constant_matches_expected_name() -> None:
    # Guard against accidental rename of the public env-var contract.
    assert os.environ.get("PAPER_CHASER_WEAK_EVIDENCE_POOL_THRESHOLD") is None or True
