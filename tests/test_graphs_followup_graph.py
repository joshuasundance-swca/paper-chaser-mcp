"""Phase 7b: ``agentic.graphs.followup_graph`` submodule (ask_result_set helpers).

Every private helper is reached via an explicit ``from`` import so the
``tests/test_test_seam_inventory.py`` firewall can see the seam — attribute
access on a module-handle import would silently bypass that guard.
"""

from __future__ import annotations

from paper_chaser_mcp.agentic.graphs import (
    _build_grounded_comparison_answer as facade_build_grounded_comparison_answer,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _build_grounded_comparison_answer as core_build_grounded_comparison_answer,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _comparison_requested as core_comparison_requested,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _comparison_takeaway as core_comparison_takeaway,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _contextualize_follow_up_question as core_contextualize_follow_up_question,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _looks_like_title_venue_list as core_looks_like_title_venue_list,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _paper_focus_phrase as core_paper_focus_phrase,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _shared_focus_terms as core_shared_focus_terms,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _should_use_structured_comparison_answer as core_should_use_structured_comparison_answer,
)
from paper_chaser_mcp.agentic.graphs.followup_graph import (
    _build_grounded_comparison_answer,
    _comparison_requested,
    _comparison_takeaway,
    _contextualize_follow_up_question,
    _looks_like_title_venue_list,
    _paper_focus_phrase,
    _shared_focus_terms,
    _should_use_structured_comparison_answer,
)


def test_followup_graph_exports_build_grounded_comparison_answer() -> None:
    assert callable(_build_grounded_comparison_answer)


def test_followup_graph_exports_comparison_requested() -> None:
    assert callable(_comparison_requested)


def test_followup_graph_exports_comparison_takeaway() -> None:
    assert callable(_comparison_takeaway)


def test_followup_graph_exports_contextualize_follow_up_question() -> None:
    assert callable(_contextualize_follow_up_question)


def test_followup_graph_exports_looks_like_title_venue_list() -> None:
    assert callable(_looks_like_title_venue_list)


def test_followup_graph_exports_paper_focus_phrase() -> None:
    assert callable(_paper_focus_phrase)


def test_followup_graph_exports_shared_focus_terms() -> None:
    assert callable(_shared_focus_terms)


def test_followup_graph_exports_should_use_structured_comparison_answer() -> None:
    assert callable(_should_use_structured_comparison_answer)


def test_followup_graph_build_grounded_comparison_answer_matches_core() -> None:
    assert _build_grounded_comparison_answer is core_build_grounded_comparison_answer


def test_followup_graph_comparison_requested_matches_core() -> None:
    assert _comparison_requested is core_comparison_requested


def test_followup_graph_comparison_takeaway_matches_core() -> None:
    assert _comparison_takeaway is core_comparison_takeaway


def test_followup_graph_contextualize_follow_up_question_matches_core() -> None:
    assert _contextualize_follow_up_question is core_contextualize_follow_up_question


def test_followup_graph_looks_like_title_venue_list_matches_core() -> None:
    assert _looks_like_title_venue_list is core_looks_like_title_venue_list


def test_followup_graph_paper_focus_phrase_matches_core() -> None:
    assert _paper_focus_phrase is core_paper_focus_phrase


def test_followup_graph_shared_focus_terms_matches_core() -> None:
    assert _shared_focus_terms is core_shared_focus_terms


def test_followup_graph_should_use_structured_comparison_answer_matches_core() -> None:
    assert _should_use_structured_comparison_answer is core_should_use_structured_comparison_answer


def test_build_grounded_comparison_answer_no_papers() -> None:
    result = _build_grounded_comparison_answer(question="q", evidence_papers=[])
    assert "not contain enough evidence" in result


def test_comparison_requested_marker() -> None:
    assert _comparison_requested("Compare A vs B", "answer")
    assert _comparison_requested("what", "comparison")
    assert not _comparison_requested("tell me about X", "answer")


def test_should_use_structured_comparison_answer_delegates() -> None:
    assert _should_use_structured_comparison_answer(
        question="compare X and Y",
        answer_mode="answer",
        answer_text="",
        evidence_papers=[],
    )


def test_contextualize_follow_up_question_non_comparison_passthrough() -> None:
    assert (
        _contextualize_follow_up_question(
            question="hello",
            record=None,
            question_mode="answer",
        )
        == "hello"
    )


def test_facade_reexports_build_grounded_comparison_answer() -> None:
    assert facade_build_grounded_comparison_answer is _build_grounded_comparison_answer
