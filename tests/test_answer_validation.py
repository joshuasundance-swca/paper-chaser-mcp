"""Tests for answer-status post-validation: deterministic refusal detection and LLM validation."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from paper_chaser_mcp.agentic.provider_helpers import AnswerStatusValidation
from paper_chaser_mcp.guided_semantic import classify_answerability

# ---------------------------------------------------------------------------
# 1. AnswerStatusValidation Pydantic schema
# ---------------------------------------------------------------------------


class TestAnswerStatusValidationSchema:
    def test_accepts_answered(self) -> None:
        v = AnswerStatusValidation(classification="answered", reasoning="Substantive response.")
        assert v.classification == "answered"

    def test_accepts_abstained(self) -> None:
        v = AnswerStatusValidation(classification="abstained", reasoning="Explicit inability.")
        assert v.classification == "abstained"

    def test_accepts_insufficient_evidence(self) -> None:
        v = AnswerStatusValidation(classification="insufficient_evidence", reasoning="Hedging detected.")
        assert v.classification == "insufficient_evidence"

    def test_rejects_invalid_classification(self) -> None:
        with pytest.raises(Exception):
            AnswerStatusValidation(classification="grounded", reasoning="bad")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. Deterministic refusal detection in classify_answerability
# ---------------------------------------------------------------------------


class TestClassifyAnswerabilityRefusalDetection:
    """classify_answerability with answer_text should detect refusal patterns."""

    _EVIDENCE = [{"sourceId": "s1", "title": "Paper A"}]
    _NO_LEADS: list[dict[str, Any]] = []
    _NO_GAPS: list[str] = []

    def test_grounded_without_refusal(self) -> None:
        result = classify_answerability(
            status="succeeded",
            evidence=self._EVIDENCE,
            leads=self._NO_LEADS,
            evidence_gaps=self._NO_GAPS,
            answer_text="The study found significant effects on population dynamics.",
        )
        assert result == "grounded"

    def test_grounded_downgraded_on_cannot_determine(self) -> None:
        result = classify_answerability(
            status="succeeded",
            evidence=self._EVIDENCE,
            leads=self._NO_LEADS,
            evidence_gaps=self._NO_GAPS,
            answer_text="I cannot determine the answer based on available evidence.",
        )
        assert result == "insufficient"

    def test_grounded_downgraded_on_insufficient_evidence(self) -> None:
        result = classify_answerability(
            status="answered",
            evidence=self._EVIDENCE,
            leads=self._NO_LEADS,
            evidence_gaps=self._NO_GAPS,
            answer_text="There is insufficient evidence to draw a conclusion.",
        )
        assert result == "insufficient"

    def test_grounded_downgraded_on_unable_to_provide(self) -> None:
        result = classify_answerability(
            status="succeeded",
            evidence=self._EVIDENCE,
            leads=self._NO_LEADS,
            evidence_gaps=self._NO_GAPS,
            answer_text="I am unable to provide a definitive answer to this question.",
        )
        assert result == "insufficient"

    def test_grounded_downgraded_on_could_not_find(self) -> None:
        result = classify_answerability(
            status="succeeded",
            evidence=self._EVIDENCE,
            leads=self._NO_LEADS,
            evidence_gaps=self._NO_GAPS,
            answer_text="I could not find relevant studies on this topic.",
        )
        assert result == "insufficient"

    def test_grounded_downgraded_on_no_relevant(self) -> None:
        result = classify_answerability(
            status="succeeded",
            evidence=self._EVIDENCE,
            leads=self._NO_LEADS,
            evidence_gaps=self._NO_GAPS,
            answer_text="There are no relevant sources available for this query.",
        )
        assert result == "insufficient"

    def test_grounded_downgraded_on_dont_have_enough(self) -> None:
        result = classify_answerability(
            status="answered",
            evidence=self._EVIDENCE,
            leads=self._NO_LEADS,
            evidence_gaps=self._NO_GAPS,
            answer_text="I don't have enough information to answer this.",
        )
        assert result == "insufficient"

    def test_grounded_downgraded_on_no_evidence(self) -> None:
        result = classify_answerability(
            status="succeeded",
            evidence=self._EVIDENCE,
            leads=self._NO_LEADS,
            evidence_gaps=self._NO_GAPS,
            answer_text="Based on the available evidence, there is no clear answer.",
        )
        assert result == "insufficient"

    def test_no_downgrade_without_answer_text(self) -> None:
        """When answer_text is empty, no refusal detection fires."""
        result = classify_answerability(
            status="succeeded",
            evidence=self._EVIDENCE,
            leads=self._NO_LEADS,
            evidence_gaps=self._NO_GAPS,
            answer_text="",
        )
        assert result == "grounded"

    def test_no_downgrade_when_already_insufficient(self) -> None:
        """Refusal detection only applies to would-be 'grounded' status."""
        result = classify_answerability(
            status="failed",
            evidence=[],
            leads=self._NO_LEADS,
            evidence_gaps=self._NO_GAPS,
            answer_text="I cannot determine the answer.",
        )
        assert result == "insufficient"

    def test_backward_compatible_without_answer_text(self) -> None:
        """Calling without answer_text preserves old behavior."""
        result = classify_answerability(
            status="succeeded",
            evidence=self._EVIDENCE,
            leads=self._NO_LEADS,
            evidence_gaps=self._NO_GAPS,
        )
        assert result == "grounded"


# ---------------------------------------------------------------------------
# 3. LLM-powered avalidate_answer_status on ModelProviderBundle
# ---------------------------------------------------------------------------


class TestModelProviderBundleValidateAnswerStatus:
    """DeterministicProviderBundle.avalidate_answer_status should return None (no LLM)."""

    @pytest.mark.asyncio
    async def test_deterministic_bundle_returns_none(self) -> None:
        from paper_chaser_mcp.agentic.config import AgenticConfig
        from paper_chaser_mcp.agentic.provider_base import DeterministicProviderBundle

        config = AgenticConfig(
            enabled=True,
            provider="deterministic",
            planner_model="deterministic",
            synthesis_model="deterministic",
            embedding_model="deterministic",
            index_backend="memory",
            session_ttl_seconds=300,
            enable_trace_log=False,
        )
        bundle = DeterministicProviderBundle(config)
        result = await bundle.avalidate_answer_status(
            query="What is the effect of noise on whales?",
            answer_text="I cannot determine the answer.",
            evidence_count=3,
        )
        assert result is None


class TestOpenAIProviderBundleValidateAnswerStatus:
    """OpenAI provider should call structured parse and return AnswerStatusValidation."""

    @pytest.mark.asyncio
    async def test_openai_returns_validation_on_success(self) -> None:
        from paper_chaser_mcp.agentic.provider_openai import OpenAIProviderBundle

        mock_result = AnswerStatusValidation(
            classification="abstained",
            reasoning="The answer explicitly states inability to answer.",
        )
        bundle = OpenAIProviderBundle.__new__(OpenAIProviderBundle)
        bundle._provider_name = "openai"
        bundle._last_effective_provider_name = "openai"
        bundle.synthesis_model_name = "gpt-4o-mini"

        with patch.object(bundle, "_aresponses_parse", new_callable=AsyncMock, return_value=mock_result):
            result = await bundle.avalidate_answer_status(
                query="What is X?",
                answer_text="I cannot determine.",
                evidence_count=2,
            )
        assert result is not None
        assert result.classification == "abstained"

    @pytest.mark.asyncio
    async def test_openai_returns_none_on_failure(self) -> None:
        from paper_chaser_mcp.agentic.provider_openai import OpenAIProviderBundle

        bundle = OpenAIProviderBundle.__new__(OpenAIProviderBundle)
        bundle._provider_name = "openai"
        bundle._last_effective_provider_name = "openai"
        bundle.synthesis_model_name = "gpt-4o-mini"

        with patch.object(bundle, "_aresponses_parse", new_callable=AsyncMock, return_value=None):
            result = await bundle.avalidate_answer_status(
                query="What is X?",
                answer_text="The answer is clear.",
                evidence_count=5,
            )
        assert result is None


class TestLangChainProviderBundleValidateAnswerStatus:
    """LangChain provider should call structured_async and return AnswerStatusValidation."""

    @pytest.mark.asyncio
    async def test_langchain_returns_validation_on_success(self) -> None:
        from paper_chaser_mcp.agentic.provider_langchain import LangChainChatProviderBundle

        mock_result = AnswerStatusValidation(
            classification="answered",
            reasoning="Substantive response provided.",
        )
        bundle = LangChainChatProviderBundle.__new__(LangChainChatProviderBundle)
        bundle._provider_name = "langchain"
        bundle._last_effective_provider_name = "langchain"
        bundle.synthesis_model_name = "gpt-4o-mini"

        with (
            patch.object(bundle, "_load_models", return_value=(None, "mock_model")),
            patch.object(bundle, "_structured_async", new_callable=AsyncMock, return_value=mock_result),
        ):
            result = await bundle.avalidate_answer_status(
                query="What is X?",
                answer_text="The study found clear effects.",
                evidence_count=5,
            )
        assert result is not None
        assert result.classification == "answered"

    @pytest.mark.asyncio
    async def test_langchain_returns_none_on_failure(self) -> None:
        from paper_chaser_mcp.agentic.provider_langchain import LangChainChatProviderBundle

        bundle = LangChainChatProviderBundle.__new__(LangChainChatProviderBundle)
        bundle._provider_name = "langchain"
        bundle._last_effective_provider_name = "langchain"
        bundle.synthesis_model_name = "gpt-4o-mini"

        with (
            patch.object(bundle, "_load_models", return_value=(None, "mock_model")),
            patch.object(bundle, "_structured_async", new_callable=AsyncMock, return_value=None),
        ):
            result = await bundle.avalidate_answer_status(
                query="What is X?",
                answer_text="Some text.",
                evidence_count=1,
            )
        assert result is None


# ---------------------------------------------------------------------------
# 4. Integration: LLM override in _guided_contract_fields
# ---------------------------------------------------------------------------


class TestGuidedContractFieldsLLMOverride:
    """_guided_contract_fields should use LLM validation to override answerability when available."""

    _SOURCES = [
        {
            "sourceId": "s1",
            "title": "Paper A",
            "sourceAlias": "a1",
            "verificationStatus": "verified_primary_source",
            "topicalRelevance": "on_topic",
        }
    ]

    @pytest.mark.asyncio
    async def test_llm_override_downgrades_answerability(self) -> None:
        """When LLM says 'abstained', answerability should be overridden from grounded."""
        from paper_chaser_mcp.dispatch import _guided_contract_fields

        mock_bundle = AsyncMock()
        mock_bundle.avalidate_answer_status = AsyncMock(
            return_value=AnswerStatusValidation(
                classification="abstained",
                reasoning="Answer is non-substantive.",
            )
        )

        fields = await _guided_contract_fields(
            query="What is the effect of X on Y?",
            intent="discovery",
            status="succeeded",
            sources=self._SOURCES,
            unverified_leads=[],
            evidence_gaps=[],
            coverage_summary=None,
            strategy_metadata=None,
            answer_text="I cannot determine the answer.",
            provider_bundle=mock_bundle,
        )

        assert fields["answerability"] == "insufficient"

    @pytest.mark.asyncio
    async def test_no_override_when_llm_agrees(self) -> None:
        """When LLM says 'answered', answerability stays grounded."""
        from paper_chaser_mcp.dispatch import _guided_contract_fields

        mock_bundle = AsyncMock()
        mock_bundle.avalidate_answer_status = AsyncMock(
            return_value=AnswerStatusValidation(
                classification="answered",
                reasoning="Substantive response.",
            )
        )

        fields = await _guided_contract_fields(
            query="What is the effect of X on Y?",
            intent="discovery",
            status="succeeded",
            sources=self._SOURCES,
            unverified_leads=[],
            evidence_gaps=[],
            coverage_summary=None,
            strategy_metadata=None,
            answer_text="The study found significant effects.",
            provider_bundle=mock_bundle,
        )

        assert fields["answerability"] == "grounded"

    @pytest.mark.asyncio
    async def test_no_override_without_bundle(self) -> None:
        """When no provider_bundle, answerability is purely deterministic."""
        from paper_chaser_mcp.dispatch import _guided_contract_fields

        fields = await _guided_contract_fields(
            query="What is the effect of X on Y?",
            intent="discovery",
            status="succeeded",
            sources=self._SOURCES,
            unverified_leads=[],
            evidence_gaps=[],
            coverage_summary=None,
            strategy_metadata=None,
            answer_text="The study found significant effects.",
        )

        assert fields["answerability"] == "grounded"

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_deterministic(self) -> None:
        """When LLM returns None (failure), deterministic result is used."""
        from paper_chaser_mcp.dispatch import _guided_contract_fields

        mock_bundle = AsyncMock()
        mock_bundle.avalidate_answer_status = AsyncMock(return_value=None)

        fields = await _guided_contract_fields(
            query="What is the effect of X on Y?",
            intent="discovery",
            status="succeeded",
            sources=self._SOURCES,
            unverified_leads=[],
            evidence_gaps=[],
            coverage_summary=None,
            strategy_metadata=None,
            answer_text="I cannot determine the answer.",
            provider_bundle=mock_bundle,
        )

        # Deterministic refusal detection should still catch this
        assert fields["answerability"] == "insufficient"

    @pytest.mark.asyncio
    async def test_llm_insufficient_evidence_maps_correctly(self) -> None:
        """LLM insufficient_evidence should map to 'insufficient'."""
        from paper_chaser_mcp.dispatch import _guided_contract_fields

        mock_bundle = AsyncMock()
        mock_bundle.avalidate_answer_status = AsyncMock(
            return_value=AnswerStatusValidation(
                classification="insufficient_evidence",
                reasoning="Hedging and vague.",
            )
        )

        fields = await _guided_contract_fields(
            query="What is the effect of X on Y?",
            intent="discovery",
            status="succeeded",
            sources=self._SOURCES,
            unverified_leads=[],
            evidence_gaps=[],
            coverage_summary=None,
            strategy_metadata=None,
            answer_text="Some vague response that hedges.",
            provider_bundle=mock_bundle,
        )

        assert fields["answerability"] == "insufficient"
