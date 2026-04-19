"""Paraphrase regression tests for the follow-up synthesis gate (ws-followup-llm-gate).

The keyword classifier in :mod:`paper_chaser_mcp.agentic.answer_modes` is
intentionally narrow: only literal markers like ``mechanism``, ``compare``,
or ``rulemaking`` route a follow-up into a synthesis mode. That leaves a
large paraphrase surface -- "what does the literature conclude", "summarize
the findings", "walk me through what's known" -- that used to land in the
``"unknown"`` branch and, by default, returned ``sufficient=True``. Rubber
duck finding #6 flagged this as a safety bypass: wording variation could
silently undercut the synthesis safety gate.

These tests pin the two mitigations:

1. The unknown-mode branch in ``build_evidence_use_plan`` fails closed when
   the question is synthesis-shaped and the evidence pool is weak.
2. ``classify_question_mode`` / ``aclassify_question_mode`` accept an
   optional LLM classifier that promotes paraphrased questions out of
   ``"unknown"`` into their true synthesis mode, where the existing
   two-source gate takes over.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Literal, cast

import pytest

from paper_chaser_mcp.agentic.answer_modes import (
    ANSWER_MODES,
    aclassify_question_mode,
    build_evidence_use_plan,
    classify_question_mode,
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


PARAPHRASED_SYNTHESIS_QUESTIONS: tuple[str, ...] = (
    "What does the literature conclude about long-term exposure to compound X?",
    "Summarize the findings related to compound X",
    "How do these studies compare on compound X's mechanism?",
    "Walk me through what's known about compound X.",
    "Give me an overview of the consensus on compound X.",
)


@pytest.mark.parametrize("question", PARAPHRASED_SYNTHESIS_QUESTIONS)
def test_paraphrased_synthesis_question_fails_closed_over_weak_pool(question: str) -> None:
    """A paraphrased synthesis ask over a weak pool must not return sufficient=True.

    This is the regression for rubber-duck finding #6: the unknown-mode
    branch previously defaulted to ``sufficient=True``, so any wording that
    dodged the keyword table could bypass the synthesis gate. The fix is to
    detect synthesis-shape cues (question mark, "summarize", "literature",
    "compare", etc.) and, when the saved evidence pool is dominated by
    weak/off-topic matches, abstain instead of producing a polished answer.
    """
    # A pool that is >60% weak/off-topic. With no session hint and no LLM
    # classifier most of these questions land in "unknown" via the keyword
    # fallback; even the ones that do match a synthesis keyword must still
    # abstain because the pool itself is weak.
    sources = [
        _source("p1", topical="weak_match"),
        _source("p2", topical="off_topic"),
        _source("p3", topical="weak_match"),
        _source("p4", topical="on_topic"),
    ]
    evidence = [_evidence(source.source_id or "") for source in sources]
    plan = build_evidence_use_plan(
        question=question,
        answer_mode="synthesis",
        evidence=evidence,
        source_records=sources,
        unsupported_asks=[],
        llm_relevance=None,
    )
    assert plan["sufficient"] is False, plan
    assert plan["retrievalSufficiency"] == "insufficient", plan
    assert plan["synthesisMode"] in {"insufficient", "limited"}, plan


def test_unknown_mode_with_synthesis_phrase_fails_closed_on_weak_pool() -> None:
    """A synthesis-shape phrase is enough to fail closed over a weak pool."""
    sources = [
        _source("p1", topical="weak_match"),
        _source("p2", topical="off_topic"),
        _source("p3", topical="weak_match"),
    ]
    plan = build_evidence_use_plan(
        question="Summarize the findings from the saved results",
        answer_mode="synthesis",
        evidence=[],
        source_records=sources,
        unsupported_asks=[],
        llm_relevance=None,
    )
    assert plan["answerMode"] == "unknown"
    assert plan["sufficient"] is False
    assert plan["retrievalSufficiency"] == "insufficient"


def test_unknown_mode_without_synthesis_shape_preserves_legacy_behaviour() -> None:
    """Non-synthesis unknown questions still defer to downstream heuristics."""
    # No synthesis markers, no question mark.
    sources = [_source("p1", topical="weak_match"), _source("p2", topical="weak_match")]
    plan = build_evidence_use_plan(
        question="note about the saved set",
        answer_mode="synthesis",
        evidence=[],
        source_records=sources,
        unsupported_asks=[],
        llm_relevance=None,
    )
    assert plan["answerMode"] == "unknown"
    # Pool is weak but the question itself is not synthesis-shaped, so the
    # fail-closed gate does not fire and the legacy permissive behaviour is
    # preserved (downstream answer-status machinery keeps ownership).
    assert plan["sufficient"] is True


def test_unknown_mode_with_strong_pool_still_sufficient() -> None:
    """The new gate only fires over a weak pool; strong pools stay sufficient."""
    sources = [_source("p1"), _source("p2"), _source("p3")]
    evidence = [_evidence("p1"), _evidence("p2"), _evidence("p3")]
    plan = build_evidence_use_plan(
        question="What does the literature conclude about retrieval?",
        answer_mode="synthesis",
        evidence=evidence,
        source_records=sources,
        unsupported_asks=[],
        llm_relevance=None,
    )
    assert plan["answerMode"] == "unknown"
    assert plan["sufficient"] is True


def test_llm_classifier_promotes_paraphrased_question_to_synthesis_mode() -> None:
    """LLM-first path: a paraphrased synthesis question gets a real mode.

    When an LLM classifier is provided, it is consulted before the keyword
    fallback. This is the primary path when a provider is available.
    """
    captured: list[str] = []

    def fake_llm(question: str, modes: tuple[str, ...]) -> str | None:
        captured.append(question)
        # Verify the classifier sees the canonical mode vocabulary.
        assert modes == ANSWER_MODES
        return "mechanism_summary"

    mode = classify_question_mode(
        "Walk me through what's known about compound X.",
        llm_classifier=fake_llm,
    )
    assert mode == "mechanism_summary"
    assert captured == ["Walk me through what's known about compound X."]


def test_llm_classifier_results_are_cached_per_question() -> None:
    """Cache avoids re-classifying the same paraphrased question."""
    calls: list[str] = []

    def fake_llm(question: str, modes: tuple[str, ...]) -> str | None:
        calls.append(question)
        return "regulatory_chain"

    cache: dict[str, str] = {}
    first = classify_question_mode(
        "What is the state of the art for rule X?",
        llm_classifier=fake_llm,
        classifier_cache=cache,
    )
    second = classify_question_mode(
        "What is the state of the art for rule X?",
        llm_classifier=fake_llm,
        classifier_cache=cache,
    )
    assert first == second == "regulatory_chain"
    # Second call served from cache -- LLM invoked exactly once.
    assert len(calls) == 1
    assert cache["What is the state of the art for rule X?"] == "regulatory_chain"


def test_llm_classifier_invalid_output_falls_back_to_keyword() -> None:
    """Safety net: garbage from the LLM must not poison the mode decision."""

    def fake_llm(question: str, modes: tuple[str, ...]) -> str | None:
        return "totally-not-a-mode"

    mode = classify_question_mode(
        "Compare approach A versus approach B",
        llm_classifier=fake_llm,
    )
    # Keyword fallback wins.
    assert mode == "comparison"


def test_llm_classifier_exception_falls_back_to_keyword() -> None:
    """Provider hiccups must not crash safety-critical gating."""

    def fake_llm(question: str, modes: tuple[str, ...]) -> str | None:
        raise RuntimeError("provider unavailable")

    mode = classify_question_mode(
        "What is the mechanism driving the effect?",
        llm_classifier=fake_llm,
    )
    assert mode == "mechanism_summary"


def test_session_hint_wins_over_llm_and_keyword() -> None:
    """Planner-emitted followUpMode is the authoritative signal."""

    def fake_llm(question: str, modes: tuple[str, ...]) -> str | None:
        return "metadata"  # would be wrong

    mode = classify_question_mode(
        "Compare A versus B",
        session_metadata={"followUpMode": "comparison"},
        llm_classifier=fake_llm,
    )
    assert mode == "comparison"


def test_session_hint_maps_legacy_claim_check() -> None:
    mode = classify_question_mode(
        "ambiguous prompt",
        session_metadata={"followUpMode": "claim_check"},
    )
    assert mode == "mechanism_summary"


def test_session_hint_ignores_generic_qa_default() -> None:
    """The planner default ``qa`` is "no opinion" and must not override."""
    mode = classify_question_mode(
        "Compare A versus B",
        session_metadata={"followUpMode": "qa"},
    )
    assert mode == "comparison"


@pytest.mark.asyncio
async def test_aclassify_question_mode_uses_async_llm_classifier() -> None:
    async def fake_async_llm(question: str, modes: tuple[str, ...]) -> str | None:
        return "regulatory_chain"

    mode = await aclassify_question_mode(
        "What does the literature conclude about rule X?",
        llm_classifier=fake_async_llm,
    )
    assert mode == "regulatory_chain"


@pytest.mark.asyncio
async def test_aclassify_question_mode_swallows_async_exceptions() -> None:
    async def broken(question: str, modes: tuple[str, ...]) -> str | None:
        raise RuntimeError("boom")

    mode = await aclassify_question_mode(
        "What is the mechanism driving the effect?",
        llm_classifier=broken,
    )
    # Falls back to deterministic keyword classifier.
    assert mode == "mechanism_summary"


@pytest.mark.asyncio
async def test_aclassify_question_mode_caches_results() -> None:
    calls: list[str] = []

    async def fake_async_llm(question: str, modes: tuple[str, ...]) -> str | None:
        calls.append(question)
        return "mechanism_summary"

    cache: dict[str, str] = {}
    question = "Walk me through what's known about compound X."
    first = await aclassify_question_mode(question, llm_classifier=fake_async_llm, classifier_cache=cache)
    second = await aclassify_question_mode(question, llm_classifier=fake_async_llm, classifier_cache=cache)
    assert first == second == "mechanism_summary"
    assert len(calls) == 1


def test_llm_promoted_mode_activates_existing_synthesis_gate() -> None:
    """End-to-end: LLM classification + weak evidence -> existing gate abstains.

    This exercises the full LLM-first happy path: the classifier promotes a
    paraphrased ask to ``mechanism_summary``, and the existing two-source
    gate correctly refuses to declare it sufficient when only fallback
    classifications are available.
    """

    def fake_llm(question: str, modes: tuple[str, ...]) -> str | None:
        return "mechanism_summary"

    question = "Walk me through what's known about compound X."
    classified = classify_question_mode(question, llm_classifier=fake_llm)
    assert classified == "mechanism_summary"

    evidence = [_evidence("p1")]
    sources = [_source("p1")]
    plan = build_evidence_use_plan(
        question=question,
        answer_mode="synthesis",
        evidence=evidence,
        source_records=sources,
        unsupported_asks=[],
        llm_relevance={
            "p1": {
                "classification": "on_topic",
                "rationale": "test",
                "fallback": False,
                "provenance": "model",
            },
        },
        question_mode=classified,
    )
    assert plan["answerMode"] == "mechanism_summary"
    # Only one non-fallback on-topic source -> thin / insufficient.
    assert plan["sufficient"] is False
    assert plan["retrievalSufficiency"] == "thin"


# Async helper signature check (keeps mypy from dropping Awaitable if nothing
# else references it). Pure type-level assertion -- runtime is a no-op.
def _typecheck_awaitable() -> Awaitable[str | None] | None:
    return None
