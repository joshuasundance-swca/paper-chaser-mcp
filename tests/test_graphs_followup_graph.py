"""Phase 7b: ``agentic.graphs.followup_graph`` submodule (ask_result_set helpers)."""

from __future__ import annotations

import pytest

from paper_chaser_mcp.agentic.graphs import _core as core_module
from paper_chaser_mcp.agentic.graphs import followup_graph


@pytest.mark.parametrize(
    "name",
    [
        "_build_grounded_comparison_answer",
        "_comparison_requested",
        "_comparison_takeaway",
        "_contextualize_follow_up_question",
        "_looks_like_title_venue_list",
        "_paper_focus_phrase",
        "_shared_focus_terms",
        "_should_use_structured_comparison_answer",
    ],
)
def test_followup_graph_exports_helper(name: str) -> None:
    assert callable(getattr(followup_graph, name))


def test_followup_graph_helpers_match_core_legacy_bindings() -> None:
    assert followup_graph._build_grounded_comparison_answer is core_module._build_grounded_comparison_answer
    assert followup_graph._shared_focus_terms is core_module._shared_focus_terms
    assert followup_graph._comparison_requested is core_module._comparison_requested
    assert (
        followup_graph._should_use_structured_comparison_answer
        is core_module._should_use_structured_comparison_answer
    )
    assert followup_graph._contextualize_follow_up_question is core_module._contextualize_follow_up_question


def test_build_grounded_comparison_answer_no_papers() -> None:
    result = followup_graph._build_grounded_comparison_answer(question="q", evidence_papers=[])
    assert "not contain enough evidence" in result


def test_comparison_requested_marker() -> None:
    assert followup_graph._comparison_requested("Compare A vs B", "answer")
    assert followup_graph._comparison_requested("what", "comparison")
    assert not followup_graph._comparison_requested("tell me about X", "answer")


def test_should_use_structured_comparison_answer_delegates() -> None:
    assert followup_graph._should_use_structured_comparison_answer(
        question="compare X and Y",
        answer_mode="answer",
        answer_text="",
        evidence_papers=[],
    )


def test_contextualize_follow_up_question_non_comparison_passthrough() -> None:
    assert (
        followup_graph._contextualize_follow_up_question(
            question="hello",
            record=None,
            question_mode="answer",
        )
        == "hello"
    )


def test_facade_reexports_build_grounded_comparison_answer() -> None:
    from paper_chaser_mcp.agentic.graphs import _build_grounded_comparison_answer as facade_fn

    assert facade_fn is followup_graph._build_grounded_comparison_answer
