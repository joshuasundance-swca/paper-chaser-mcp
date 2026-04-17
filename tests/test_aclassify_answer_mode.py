"""Regression tests for Finding 3: ``aclassify_answer_mode`` on LLM bundles.

The base ``ProviderBundle.aclassify_answer_mode`` is a safe stub that returns
``None`` so deterministic bundles fall through to the keyword heuristic in
:mod:`paper_chaser_mcp.agentic.answer_modes`. Concrete OpenAI / LangChain
bundles override it to perform a lightweight structured LLM call. These tests
verify that the async entry point :func:`aclassify_question_mode` honors a
valid model emission and still defaults to keyword heuristics when the
classifier returns nothing.
"""

from __future__ import annotations

import pytest

from paper_chaser_mcp.agentic.answer_modes import (
    ANSWER_MODES,
    aclassify_question_mode,
)
from paper_chaser_mcp.agentic.config import AgenticConfig
from paper_chaser_mcp.agentic.provider_base import DeterministicProviderBundle


@pytest.mark.asyncio
async def test_aclassify_question_mode_honors_valid_llm_emission() -> None:
    """When a stub bundle returns ``"mechanism_summary"`` the caller must
    honor it -- this is the contract consumed by ``ask_result_set`` via
    ``graphs.py``."""

    async def llm_classifier(question: str, modes: tuple[str, ...]) -> str | None:
        assert "mechanism_summary" in modes
        return "mechanism_summary"

    mode = await aclassify_question_mode(
        "How does metformin reduce hepatic gluconeogenesis?",
        llm_classifier=llm_classifier,
    )
    assert mode == "mechanism_summary"


@pytest.mark.asyncio
async def test_aclassify_question_mode_rejects_invalid_llm_emission() -> None:
    """Invalid emissions must be ignored and the keyword heuristic must run."""

    async def llm_classifier(question: str, modes: tuple[str, ...]) -> str | None:
        return "not-a-real-mode"

    mode = await aclassify_question_mode(
        "Summarise the methodology of this study.",
        llm_classifier=llm_classifier,
    )
    assert mode in ANSWER_MODES
    # "methodology" is a keyword-heuristic signal for methodology_summary.
    assert mode != "not-a-real-mode"


@pytest.mark.asyncio
async def test_deterministic_bundle_returns_none_preserving_keyword_fallback() -> None:
    """The deterministic base bundle must keep returning ``None`` so
    ``aclassify_question_mode`` falls through to the keyword heuristic --
    this is the behaviour the production pipeline depends on when no LLM
    provider is configured."""

    bundle = DeterministicProviderBundle(
        AgenticConfig(
            enabled=True,
            provider="deterministic",
            planner_model="deterministic-planner",
            synthesis_model="deterministic-synthesizer",
            embedding_model="deterministic-lexical",
            index_backend="memory",
            session_ttl_seconds=1800,
            enable_trace_log=False,
        )
    )
    result = await bundle.aclassify_answer_mode(
        question="What is the mechanism of action?",
        modes=ANSWER_MODES,
    )
    assert result is None
