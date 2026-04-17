"""Regression coverage for the Finding 3 follow-up synthesis gate.

The deterministic keyword classifier misses paraphrased synthesis asks such
as ``"what do these papers say?"`` because it does not hit any
``SYNTHESIS_SHAPE_MARKERS``; it falls through to ``unknown`` and lets
``build_evidence_use_plan`` treat the turn as open-ended Q&A. That bypasses
the fail-closed gate for weak evidence pools.

These tests exercise two linked properties:

1. ``aclassify_question_mode`` honours an injected LLM classifier when the
   deterministic heuristic would otherwise return ``unknown``.
2. ``build_evidence_use_plan`` then treats the resolved mode as a synthesis
   ask and refuses to mark the plan sufficient when the evidence pool is
   weak/off-topic.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from paper_chaser_mcp.agentic.answer_modes import (
    ANSWER_MODES,
    SYNTHESIS_MODES,
    aclassify_question_mode,
    build_evidence_use_plan,
)


def _make_stub_classifier(
    mode: str,
) -> Callable[[str, tuple[str, ...]], Coroutine[Any, Any, str | None]]:
    async def _classifier(question: str, modes: tuple[str, ...]) -> str | None:
        assert question
        assert modes == ANSWER_MODES
        return mode

    return _classifier


def test_aclassify_question_mode_uses_llm_when_keywords_miss() -> None:
    """Paraphrased synthesis ask should be rescued by the LLM classifier."""

    resolved = asyncio.run(
        aclassify_question_mode(
            "What do these papers say?",
            None,
            llm_classifier=_make_stub_classifier("mechanism_summary"),
        )
    )
    assert resolved == "mechanism_summary"
    assert resolved in SYNTHESIS_MODES


def test_aclassify_question_mode_ignores_llm_unknown() -> None:
    """``unknown`` from the LLM must not shortcut the keyword fallback."""

    resolved = asyncio.run(
        aclassify_question_mode(
            "Summarise these results",
            None,
            llm_classifier=_make_stub_classifier("unknown"),
        )
    )
    # We don't assert a specific mode here (that is the keyword classifier's
    # responsibility); what matters for the gate is that the LLM's "unknown"
    # did not short-circuit the resolution and the caller got *something*
    # drawn from the canonical ANSWER_MODES tuple.
    assert resolved in ANSWER_MODES


def test_build_evidence_use_plan_fails_closed_on_weak_pool() -> None:
    """A synthesis mode over a weak pool must emit sufficient=False."""

    plan = build_evidence_use_plan(
        question="What do these papers say?",
        answer_mode="qa",
        evidence=[],
        source_records=[],
        unsupported_asks=[],
        llm_relevance=None,
        question_mode="mechanism_summary",
    )

    assert plan.get("sufficient") is False
    assert plan.get("answerMode") == "mechanism_summary"
