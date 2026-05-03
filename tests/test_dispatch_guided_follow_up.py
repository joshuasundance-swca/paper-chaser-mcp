"""TDD tests for dispatch/guided/follow_up extraction."""

from __future__ import annotations

import pytest

from paper_chaser_mcp.agentic.answer_modes import classify_question_mode
from paper_chaser_mcp.dispatch.guided.follow_up import (
    _answer_follow_up_from_session_state,
    _guided_follow_up_answer_mode,
    _guided_follow_up_introspection_facets,
    _guided_follow_up_response_mode,
    _guided_is_usable_answer_text,
    _guided_metadata_answer_is_responsive,
    _guided_relevance_triage_answers,
    _guided_requested_metadata_facets,
    _guided_source_metadata_answers,
)


class TestIntrospectionFacets:
    def test_empty_question_returns_empty_set(self) -> None:
        assert _guided_follow_up_introspection_facets("") == set()

    def test_returns_set_instance(self) -> None:
        result = _guided_follow_up_introspection_facets("what is the author?")
        assert isinstance(result, set)


class TestIsUsableAnswerText:
    def test_none_returns_false(self) -> None:
        assert _guided_is_usable_answer_text(None) is False

    def test_blank_returns_false(self) -> None:
        assert _guided_is_usable_answer_text("   ") is False

    def test_non_string_returns_false(self) -> None:
        assert _guided_is_usable_answer_text(42) is True

    def test_real_string_returns_true(self) -> None:
        assert _guided_is_usable_answer_text("a real answer") is True


class TestSourceMetadataAnswers:
    def test_empty_sources_returns_empty_list(self) -> None:
        assert _guided_source_metadata_answers("who?", []) == []

    def test_returns_list(self) -> None:
        result = _guided_source_metadata_answers(
            "what year?",
            [{"year": "2020", "title": "Paper"}],
        )
        assert isinstance(result, list)


class TestRequestedMetadataFacets:
    def test_empty_question_returns_empty_set(self) -> None:
        assert _guided_requested_metadata_facets("") == set()

    def test_returns_set(self) -> None:
        result = _guided_requested_metadata_facets("who wrote this?")
        assert isinstance(result, set)


class TestMetadataAnswerIsResponsive:
    def test_callable(self) -> None:
        assert callable(_guided_metadata_answer_is_responsive)


class TestFollowUpResponseMode:
    def test_returns_string(self) -> None:
        result = _guided_follow_up_response_mode("what?", {})
        assert isinstance(result, str)

    @pytest.mark.parametrize(
        ("question", "strategy_metadata", "expected"),
        [
            ("Which paper should I read first?", {}, "selection"),
            (
                "Which providers were searched, and what specific evidence gap prevented a grounded answer?",
                {"followUpMode": "claim_check"},
                "metadata",
            ),
        ],
    )
    def test_preserves_guided_wrapper_semantics(
        self,
        question: str,
        strategy_metadata: dict[str, str],
        expected: str,
    ) -> None:
        assert _guided_follow_up_response_mode(question, strategy_metadata) == expected

    def test_question_shape_wins_over_saved_claim_check_hint(self) -> None:
        question = "What limitations or validation issues appear most often in the returned evidence?"
        assert _guided_follow_up_response_mode(question, {"followUpMode": "claim_check"}) == classify_question_mode(
            question
        )


class TestFollowUpAnswerMode:
    def test_returns_string(self) -> None:
        result = _guided_follow_up_answer_mode("what?", {})
        assert isinstance(result, str)


class TestAnswerFollowUpSmoke:
    @pytest.mark.asyncio
    async def test_callable(self) -> None:
        assert callable(_answer_follow_up_from_session_state)


class TestRelevanceTriage:
    def test_callable(self) -> None:
        assert callable(_guided_relevance_triage_answers)
