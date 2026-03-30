"""LangChain-based provider bundles for the additive smart research layer."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from pydantic import SecretStr

from ..provider_runtime import ProviderDiagnosticsRegistry
from .config import AgenticConfig
from .models import ExpansionCandidate, PlannerDecision
from .provider_base import DeterministicProviderBundle
from .provider_helpers import (
    _AnswerSchema,
    _build_answer_payload,
    _build_theme_label_payload,
    _build_theme_summary_payload,
    _coerce_langchain_structured_response,
    _ExpansionListSchema,
    _filter_expansion_candidates,
    _langchain_message_text,
    _normalize_theme_label_output,
    _PlannerResponseSchema,
    _ResponseModelT,
)

logger = logging.getLogger("paper-chaser-mcp")

__all__ = [
    "AnthropicProviderBundle",
    "GoogleProviderBundle",
    "LangChainChatProviderBundle",
]


def _execute_provider_call_sync(**kwargs: Any) -> Any:
    from . import providers as providers_facade

    return providers_facade.execute_provider_call_sync(**kwargs)


async def _execute_provider_call(**kwargs: Any) -> Any:
    from . import providers as providers_facade

    return await providers_facade.execute_provider_call(**kwargs)


class LangChainChatProviderBundle(DeterministicProviderBundle):
    """Chat-only provider bundle built on LangChain structured outputs."""

    def __init__(
        self,
        config: AgenticConfig,
        *,
        provider_name: str,
        api_key: str | None,
        provider_registry: ProviderDiagnosticsRegistry | None = None,
        structured_output_method: str | None = None,
    ) -> None:
        super().__init__(config)
        self._provider_name = provider_name
        self._api_key = api_key
        self._provider_registry = provider_registry
        self._structured_output_method = structured_output_method
        self._timeout_seconds = config.openai_timeout_seconds
        self.planner_model_name = config.planner_model
        self.synthesis_model_name = config.synthesis_model
        self.embedding_model_name = config.embedding_model
        self._planner: Any | None = None
        self._synthesizer: Any | None = None

    def supports_embeddings(self) -> bool:
        return False

    def _create_chat_model(self, model_name: str) -> Any:
        raise NotImplementedError

    def _load_models(self) -> tuple[Any | None, Any | None]:
        if self._planner is not None and self._synthesizer is not None:
            return self._planner, self._synthesizer
        if not self._api_key:
            return None, None
        try:
            self._planner = self._create_chat_model(self.planner_model_name)
            self._synthesizer = self._create_chat_model(self.synthesis_model_name)
        except ImportError:
            logger.info(
                "%s dependencies are not installed; falling back to deterministic smart planning.",
                self._provider_name,
            )
            return None, None
        except Exception:
            logger.exception("%s model initialization failed.", self._provider_name)
            return None, None
        return self._planner, self._synthesizer

    def _wrap_structured_model(self, model: Any, response_model: type[_ResponseModelT]) -> Any:
        if self._structured_output_method is None:
            return model.with_structured_output(response_model)
        return model.with_structured_output(response_model, method=self._structured_output_method)

    @staticmethod
    def _messages(system_prompt: str, payload: dict[str, Any]) -> list[tuple[str, str]]:
        return [
            ("system", system_prompt),
            ("human", json.dumps(payload)),
        ]

    async def _ainvoke(self, model: Any, messages: list[tuple[str, str]]) -> Any:
        if hasattr(model, "ainvoke"):
            return await model.ainvoke(messages)
        return await asyncio.to_thread(model.invoke, messages)

    async def _ainvoke_structured(self, structured: Any, messages: list[tuple[str, str]]) -> Any:
        if hasattr(structured, "ainvoke"):
            return await structured.ainvoke(messages)
        return await asyncio.to_thread(structured.invoke, messages)

    def _structured_sync(
        self,
        *,
        endpoint: str,
        model: Any,
        response_model: type[_ResponseModelT],
        system_prompt: str,
        payload: dict[str, Any],
    ) -> _ResponseModelT | None:
        if model is None:
            return None

        response = _execute_provider_call_sync(
            provider=self._provider_name,
            endpoint=endpoint,
            operation=lambda: self._wrap_structured_model(model, response_model).invoke(
                self._messages(system_prompt, payload)
            ),
            registry=self._provider_registry,
        )
        if response.payload is None:
            return None
        try:
            return _coerce_langchain_structured_response(response.payload, response_model)
        except Exception:
            logger.exception("%s structured parse failed for %s.", self._provider_name, endpoint)
            return None

    async def _structured_async(
        self,
        *,
        endpoint: str,
        model: Any,
        response_model: type[_ResponseModelT],
        system_prompt: str,
        payload: dict[str, Any],
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> _ResponseModelT | None:
        if model is None:
            return None

        async def _call() -> Any:
            structured = self._wrap_structured_model(model, response_model)
            return await self._ainvoke_structured(structured, self._messages(system_prompt, payload))

        response = await _execute_provider_call(
            provider=self._provider_name,
            endpoint=endpoint,
            operation=_call,
            registry=self._provider_registry,
            request_outcomes=request_outcomes,
            request_id=request_id,
        )
        if response.payload is None:
            return None
        try:
            return _coerce_langchain_structured_response(response.payload, response_model)
        except Exception:
            logger.exception("%s structured parse failed for %s.", self._provider_name, endpoint)
            return None

    def _text_sync(
        self,
        *,
        endpoint: str,
        model: Any,
        system_prompt: str,
        payload: dict[str, Any],
    ) -> str | None:
        if model is None:
            return None

        response = _execute_provider_call_sync(
            provider=self._provider_name,
            endpoint=endpoint,
            operation=lambda: model.invoke(self._messages(system_prompt, payload)),
            registry=self._provider_registry,
        )
        if response.payload is None:
            return None
        text = _langchain_message_text(response.payload)
        return text or None

    async def _text_async(
        self,
        *,
        endpoint: str,
        model: Any,
        system_prompt: str,
        payload: dict[str, Any],
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> str | None:
        if model is None:
            return None

        response = await _execute_provider_call(
            provider=self._provider_name,
            endpoint=endpoint,
            operation=lambda: self._ainvoke(model, self._messages(system_prompt, payload)),
            registry=self._provider_registry,
            request_outcomes=request_outcomes,
            request_id=request_id,
        )
        if response.payload is None:
            return None
        text = _langchain_message_text(response.payload)
        return text or None

    def _structured_from_text_sync(
        self,
        *,
        endpoint: str,
        model: Any,
        response_model: type[_ResponseModelT],
        system_prompt: str,
        payload: dict[str, Any],
    ) -> _ResponseModelT | None:
        if model is None:
            return None

        response = _execute_provider_call_sync(
            provider=self._provider_name,
            endpoint=endpoint,
            operation=lambda: model.invoke(self._messages(system_prompt, payload)),
            registry=self._provider_registry,
        )
        if response.payload is None:
            return None
        try:
            return _coerce_langchain_structured_response(response.payload, response_model)
        except Exception:
            logger.exception("%s text-to-structured parse failed for %s.", self._provider_name, endpoint)
            return None

    async def _structured_from_text_async(
        self,
        *,
        endpoint: str,
        model: Any,
        response_model: type[_ResponseModelT],
        system_prompt: str,
        payload: dict[str, Any],
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> _ResponseModelT | None:
        if model is None:
            return None

        response = await _execute_provider_call(
            provider=self._provider_name,
            endpoint=endpoint,
            operation=lambda: self._ainvoke(model, self._messages(system_prompt, payload)),
            registry=self._provider_registry,
            request_outcomes=request_outcomes,
            request_id=request_id,
        )
        if response.payload is None:
            return None
        try:
            return _coerce_langchain_structured_response(response.payload, response_model)
        except Exception:
            logger.exception("%s text-to-structured parse failed for %s.", self._provider_name, endpoint)
            return None

    def plan_search(
        self,
        *,
        query: str,
        mode: str,
        year: str | None = None,
        venue: str | None = None,
        focus: str | None = None,
    ) -> PlannerDecision:
        planner, _ = self._load_models()
        try:
            direct = self._structured_sync(
                endpoint="structured:planner",
                model=planner,
                response_model=_PlannerResponseSchema,
                system_prompt=(
                    "Plan a grounded literature-search workflow. Keep providerPlan "
                    "limited to semantic_scholar, openalex, scholarapi, core, and arxiv. "
                    "Return compact structured output only."
                ),
                payload={
                    "query": query,
                    "mode": mode,
                    "year": year,
                    "venue": venue,
                    "focus": focus,
                },
            )
            if direct is not None:
                self._mark_provider_used()
                return direct.to_planner_decision()
        except Exception:
            logger.exception("%s planner failed; falling back to deterministic planning.", self._provider_name)
        self._mark_deterministic_fallback()
        return super().plan_search(query=query, mode=mode, year=year, venue=venue, focus=focus)

    async def aplan_search(
        self,
        *,
        query: str,
        mode: str,
        year: str | None = None,
        venue: str | None = None,
        focus: str | None = None,
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> PlannerDecision:
        planner, _ = self._load_models()
        try:
            direct = await self._structured_async(
                endpoint="structured:planner",
                model=planner,
                response_model=_PlannerResponseSchema,
                system_prompt=(
                    "Plan a grounded literature-search workflow. Keep providerPlan "
                    "limited to semantic_scholar, openalex, scholarapi, core, and arxiv. "
                    "Return compact structured output only."
                ),
                payload={
                    "query": query,
                    "mode": mode,
                    "year": year,
                    "venue": venue,
                    "focus": focus,
                },
                request_outcomes=request_outcomes,
                request_id=request_id,
            )
            if direct is not None:
                self._mark_provider_used()
                return direct.to_planner_decision()
        except Exception:
            logger.exception("%s planner failed; falling back to deterministic planning.", self._provider_name)
        self._mark_deterministic_fallback()
        return super().plan_search(query=query, mode=mode, year=year, venue=venue, focus=focus)

    def suggest_speculative_expansions(
        self,
        *,
        query: str,
        evidence_texts: list[str],
        max_variants: int,
    ) -> list[ExpansionCandidate]:
        planner, _ = self._load_models()
        try:
            direct = self._structured_sync(
                endpoint="structured:expansions",
                model=planner,
                response_model=_ExpansionListSchema,
                system_prompt=(
                    "Suggest at most three short literature-search expansions. "
                    "Each expansion must preserve the user's research intent, "
                    "must not add unrelated domains, and must avoid stopwords, "
                    "generic verbs, or filler terms. Label each expansion as "
                    "from_input, from_retrieved_evidence, or speculative."
                ),
                payload={
                    "query": query,
                    "evidence": evidence_texts[:5],
                    "max_variants": max_variants,
                },
            )
            if direct is None:
                self._mark_deterministic_fallback()
                return super().suggest_speculative_expansions(
                    query=query,
                    evidence_texts=evidence_texts,
                    max_variants=max_variants,
                )
            self._mark_provider_used()
            return _filter_expansion_candidates(
                query,
                direct.expansions,
                max_variants=max_variants,
            )
        except Exception:
            logger.exception(
                "%s expansion generation failed; falling back to deterministic expansions.",
                self._provider_name,
            )
            self._mark_deterministic_fallback()
            return super().suggest_speculative_expansions(
                query=query,
                evidence_texts=evidence_texts,
                max_variants=max_variants,
            )

    async def asuggest_speculative_expansions(
        self,
        *,
        query: str,
        evidence_texts: list[str],
        max_variants: int,
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> list[ExpansionCandidate]:
        planner, _ = self._load_models()
        try:
            direct = await self._structured_async(
                endpoint="structured:expansions",
                model=planner,
                response_model=_ExpansionListSchema,
                system_prompt=(
                    "Suggest at most three short literature-search expansions. "
                    "Each expansion must preserve the user's research intent, "
                    "must not add unrelated domains, and must avoid stopwords, "
                    "generic verbs, or filler terms. Label each expansion as "
                    "from_input, from_retrieved_evidence, or speculative."
                ),
                payload={
                    "query": query,
                    "evidence": evidence_texts[:5],
                    "max_variants": max_variants,
                },
                request_outcomes=request_outcomes,
                request_id=request_id,
            )
            if direct is None:
                self._mark_deterministic_fallback()
                return super().suggest_speculative_expansions(
                    query=query,
                    evidence_texts=evidence_texts,
                    max_variants=max_variants,
                )
            self._mark_provider_used()
            return _filter_expansion_candidates(
                query,
                direct.expansions,
                max_variants=max_variants,
            )
        except Exception:
            logger.exception(
                "%s expansion generation failed; falling back to deterministic expansions.",
                self._provider_name,
            )
            self._mark_deterministic_fallback()
            return super().suggest_speculative_expansions(
                query=query,
                evidence_texts=evidence_texts,
                max_variants=max_variants,
            )

    def label_theme(
        self,
        *,
        seed_terms: list[str],
        papers: list[dict[str, Any]],
    ) -> str:
        _, synthesizer = self._load_models()
        try:
            direct = self._text_sync(
                endpoint="text:label_theme",
                model=synthesizer,
                system_prompt=(
                    "Write a very short literature theme label. Return plain text only, "
                    "no markdown, no bullets, no explanation."
                ),
                payload=_build_theme_label_payload(seed_terms, papers),
            )
            if direct:
                label = _normalize_theme_label_output(direct)
                if label:
                    self._mark_provider_used()
                    return label
        except Exception:
            logger.exception(
                "%s theme labeling failed; falling back to deterministic theme labels.",
                self._provider_name,
            )
        self._mark_deterministic_fallback()
        return super().label_theme(seed_terms=seed_terms, papers=papers)

    async def alabel_theme(
        self,
        *,
        seed_terms: list[str],
        papers: list[dict[str, Any]],
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> str:
        _, synthesizer = self._load_models()
        try:
            direct = await self._text_async(
                endpoint="text:label_theme",
                model=synthesizer,
                system_prompt=(
                    "Write a very short literature theme label. Return plain text only, "
                    "no markdown, no bullets, no explanation."
                ),
                payload=_build_theme_label_payload(seed_terms, papers),
                request_outcomes=request_outcomes,
                request_id=request_id,
            )
            if direct:
                label = _normalize_theme_label_output(direct)
                if label:
                    self._mark_provider_used()
                    return label
        except Exception:
            logger.exception(
                "%s theme labeling failed; falling back to deterministic theme labels.",
                self._provider_name,
            )
        self._mark_deterministic_fallback()
        return super().label_theme(seed_terms=seed_terms, papers=papers)

    def summarize_theme(
        self,
        *,
        title: str,
        papers: list[dict[str, Any]],
    ) -> str:
        _, synthesizer = self._load_models()
        try:
            direct = self._text_sync(
                endpoint="text:summarize_theme",
                model=synthesizer,
                system_prompt="Summarize one literature cluster in two sentences. Return plain text only.",
                payload=_build_theme_summary_payload(title, papers),
            )
            if direct:
                self._mark_provider_used()
                return direct
        except Exception:
            logger.exception(
                "%s theme summarization failed; falling back to deterministic summaries.",
                self._provider_name,
            )
        self._mark_deterministic_fallback()
        return super().summarize_theme(title=title, papers=papers)

    async def asummarize_theme(
        self,
        *,
        title: str,
        papers: list[dict[str, Any]],
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> str:
        _, synthesizer = self._load_models()
        try:
            direct = await self._text_async(
                endpoint="text:summarize_theme",
                model=synthesizer,
                system_prompt="Summarize one literature cluster in two sentences. Return plain text only.",
                payload=_build_theme_summary_payload(title, papers),
                request_outcomes=request_outcomes,
                request_id=request_id,
            )
            if direct:
                self._mark_provider_used()
                return direct
        except Exception:
            logger.exception(
                "%s theme summarization failed; falling back to deterministic summaries.",
                self._provider_name,
            )
        self._mark_deterministic_fallback()
        return super().summarize_theme(title=title, papers=papers)

    def answer_question(
        self,
        *,
        question: str,
        evidence_papers: list[dict[str, Any]],
        answer_mode: str,
    ) -> dict[str, Any]:
        _, synthesizer = self._load_models()
        try:
            direct = self._structured_sync(
                endpoint="structured:answer",
                model=synthesizer,
                response_model=_AnswerSchema,
                system_prompt=(
                    "Answer only from the supplied papers. If evidence is weak, "
                    "say so. Confidence must be exactly one of: high, medium, low."
                ),
                payload=_build_answer_payload(question, answer_mode, evidence_papers),
            )
            if direct is not None:
                parsed = direct.model_dump()
                parsed["confidence"] = self.normalize_confidence(parsed.get("confidence"))
                self._mark_provider_used()
                return parsed
            text_fallback = self._structured_from_text_sync(
                endpoint="text:answer",
                model=synthesizer,
                response_model=_AnswerSchema,
                system_prompt=(
                    "Answer only from the supplied papers. If evidence is weak, say so. "
                    "Return only JSON with keys answer, unsupportedAsks, followUpQuestions, confidence. "
                    "Confidence must be exactly one of: high, medium, low."
                ),
                payload=_build_answer_payload(question, answer_mode, evidence_papers),
            )
            if text_fallback is not None:
                parsed = text_fallback.model_dump()
                parsed["confidence"] = self.normalize_confidence(parsed.get("confidence"))
                self._mark_provider_used()
                return parsed
        except Exception:
            logger.exception(
                "%s synthesis failed; falling back to deterministic answer generation.",
                self._provider_name,
            )
        self._mark_deterministic_fallback()
        return super().answer_question(
            question=question,
            evidence_papers=evidence_papers,
            answer_mode=answer_mode,
        )

    async def aanswer_question(
        self,
        *,
        question: str,
        evidence_papers: list[dict[str, Any]],
        answer_mode: str,
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        _, synthesizer = self._load_models()
        try:
            direct = await self._structured_async(
                endpoint="structured:answer",
                model=synthesizer,
                response_model=_AnswerSchema,
                system_prompt=(
                    "Answer only from the supplied papers. If evidence is weak, "
                    "say so. Confidence must be exactly one of: high, medium, low."
                ),
                payload=_build_answer_payload(question, answer_mode, evidence_papers),
                request_outcomes=request_outcomes,
                request_id=request_id,
            )
            if direct is not None:
                parsed = direct.model_dump()
                parsed["confidence"] = self.normalize_confidence(parsed.get("confidence"))
                self._mark_provider_used()
                return parsed
            text_fallback = await self._structured_from_text_async(
                endpoint="text:answer",
                model=synthesizer,
                response_model=_AnswerSchema,
                system_prompt=(
                    "Answer only from the supplied papers. If evidence is weak, say so. "
                    "Return only JSON with keys answer, unsupportedAsks, followUpQuestions, confidence. "
                    "Confidence must be exactly one of: high, medium, low."
                ),
                payload=_build_answer_payload(question, answer_mode, evidence_papers),
                request_outcomes=request_outcomes,
                request_id=request_id,
            )
            if text_fallback is not None:
                parsed = text_fallback.model_dump()
                parsed["confidence"] = self.normalize_confidence(parsed.get("confidence"))
                self._mark_provider_used()
                return parsed
        except Exception:
            logger.exception(
                "%s synthesis failed; falling back to deterministic answer generation.",
                self._provider_name,
            )
        self._mark_deterministic_fallback()
        return super().answer_question(
            question=question,
            evidence_papers=evidence_papers,
            answer_mode=answer_mode,
        )


class AnthropicProviderBundle(LangChainChatProviderBundle):
    """Anthropic smart-layer adapter via LangChain."""

    def __init__(
        self,
        config: AgenticConfig,
        api_key: str | None,
        *,
        provider_registry: ProviderDiagnosticsRegistry | None = None,
    ) -> None:
        super().__init__(
            config,
            provider_name="anthropic",
            api_key=api_key,
            provider_registry=provider_registry,
        )

    def _create_chat_model(self, model_name: str) -> Any:
        from langchain_anthropic import ChatAnthropic

        chat_anthropic: Any = ChatAnthropic
        kwargs = {
            "model_name": model_name,
            "api_key": SecretStr(self._api_key or ""),
            "stop": None,
            "temperature": 0,
            "timeout": self._timeout_seconds,
            "max_retries": 0,
        }
        return chat_anthropic(**kwargs)


class GoogleProviderBundle(LangChainChatProviderBundle):
    """Google Gemini smart-layer adapter via LangChain."""

    def __init__(
        self,
        config: AgenticConfig,
        api_key: str | None,
        *,
        provider_registry: ProviderDiagnosticsRegistry | None = None,
    ) -> None:
        super().__init__(
            config,
            provider_name="google",
            api_key=api_key,
            provider_registry=provider_registry,
            structured_output_method="json_schema",
        )

    def _create_chat_model(self, model_name: str) -> Any:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self._api_key,
            temperature=0,
        )
