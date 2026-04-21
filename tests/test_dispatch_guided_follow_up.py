"""TDD tests for dispatch/guided/follow_up extraction."""

from __future__ import annotations

import pytest

from paper_chaser_mcp.dispatch.guided import follow_up as follow_up_mod


class TestIntrospectionFacets:
    def test_empty_question_returns_empty_set(self) -> None:
        assert follow_up_mod._guided_follow_up_introspection_facets("") == set()

    def test_returns_set_instance(self) -> None:
        result = follow_up_mod._guided_follow_up_introspection_facets("what is the author?")
        assert isinstance(result, set)


class TestIsUsableAnswerText:
    def test_none_returns_false(self) -> None:
        assert follow_up_mod._guided_is_usable_answer_text(None) is False

    def test_blank_returns_false(self) -> None:
        assert follow_up_mod._guided_is_usable_answer_text("   ") is False

    def test_non_string_returns_false(self) -> None:
        assert follow_up_mod._guided_is_usable_answer_text(42) is True

    def test_real_string_returns_true(self) -> None:
        assert follow_up_mod._guided_is_usable_answer_text("a real answer") is True


class TestSourceMetadataAnswers:
    def test_empty_sources_returns_empty_list(self) -> None:
        assert follow_up_mod._guided_source_metadata_answers("who?", []) == []

    def test_returns_list(self) -> None:
        result = follow_up_mod._guided_source_metadata_answers(
            "what year?",
            [{"year": "2020", "title": "Paper"}],
        )
        assert isinstance(result, list)


class TestRequestedMetadataFacets:
    def test_empty_question_returns_empty_set(self) -> None:
        assert follow_up_mod._guided_requested_metadata_facets("") == set()

    def test_returns_set(self) -> None:
        result = follow_up_mod._guided_requested_metadata_facets("who wrote this?")
        assert isinstance(result, set)


class TestMetadataAnswerIsResponsive:
    def test_callable(self) -> None:
        assert callable(follow_up_mod._guided_metadata_answer_is_responsive)


class TestFollowUpResponseMode:
    def test_returns_string(self) -> None:
        result = follow_up_mod._guided_follow_up_response_mode("what?", {})
        assert isinstance(result, str)


class TestFollowUpAnswerMode:
    def test_returns_string(self) -> None:
        result = follow_up_mod._guided_follow_up_answer_mode("what?", {})
        assert isinstance(result, str)


class TestAnswerFollowUpSmoke:
    @pytest.mark.asyncio
    async def test_callable(self) -> None:
        assert callable(follow_up_mod._answer_follow_up_from_session_state)


class TestRelevanceTriage:
    def test_callable(self) -> None:
        assert callable(follow_up_mod._guided_relevance_triage_answers)
