"""LangChain-based provider bundles for the additive smart research layer."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal, cast

from pydantic import SecretStr

from ..provider_runtime import ProviderDiagnosticsRegistry
from .config import AgenticConfig
from .models import ExpansionCandidate, PlannerDecision
from .provider_base import DeterministicProviderBundle
from .provider_helpers import (
    _AdequacyJudgmentSchema,
    _AnswerSchema,
    _build_answer_payload,
    _build_theme_label_payload,
    _build_theme_summary_payload,
    _coerce_langchain_structured_response,
    _ExpansionListSchema,
    _filter_expansion_candidates,
    _langchain_message_text,
    _normalize_answer_schema_output,
    _normalize_theme_label_output,
    _PlannerResponseSchema,
    _RelevanceBatchSchema,
    _ResponseModelT,
    _ReviseStrategySchema,
    _sanitize_provider_plan,
)

logger = logging.getLogger("paper-chaser-mcp")

__all__ = [
    "AnthropicProviderBundle",
    "GoogleProviderBundle",
    "HuggingFaceProviderBundle",
    "LangChainChatProviderBundle",
    "MistralProviderBundle",
    "NvidiaProviderBundle",
    "OpenRouterProviderBundle",
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
        self._configured_provider = config.provider
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

    @staticmethod
    def _json_only_system_prompt(
        base_prompt: str,
        *,
        json_shape: str,
    ) -> str:
        return (
            f"{base_prompt} Return only JSON matching this shape: {json_shape}. "
            "Do not wrap the JSON in markdown fences and do not add commentary."
        )

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
            base_prompt = (
                "You are a scientific literature search planner. Classify the query and design the optimal "
                "retrieval strategy.\n"
                "\n"
                "INTENT CLASSIFICATION RULES:\n"
                "- discovery: any broad conceptual, multi-factor, or 'what does the literature say' question. "
                "Default for most queries. Use even when a year or title-like wording appears.\n"
                "- review: explicit requests for literature reviews, systematic reviews, or evidence synthesis.\n"
                "- known_item: ONLY when a hard bibliographic identifier is present (DOI, arXiv ID, or exact "
                "title lookup with no broader research question attached).\n"
                "- author: query is specifically about a person's publication list.\n"
                "- citation: query is repairing or verifying a specific citation reference string.\n"
                "- regulatory: query explicitly references a CFR section, rulemaking, or federal register notice.\n"
                "\n"
                "OUTPUT REQUIREMENTS:\n"
                "- querySpecificity: high|medium|low\n"
                "- ambiguityLevel: low|medium|high\n"
                "- queryType: broad_concept|known_item|citation_repair|regulatory|author|review\n"
                "- breadthEstimate: integer 1-4 where 1 is narrow and 4 is very broad\n"
                "- searchAngles: 2-4 distinct retrieval angles or reformulations for broader discovery asks\n"
                "- firstPassMode: targeted|broad|mixed\n"
                "- uncertaintyFlags: list any ambiguities or competing interpretations\n"
                "- retrievalHypotheses: concrete evidence expectations the search should try to satisfy\n"
                "\n"
                "PROVIDER SELECTION (only from: semantic_scholar, openalex, scholarapi, core, arxiv, ecos, "
                "federal_register, govinfo, tavily, perplexity). Prefer semantic_scholar and openalex for "
                "peer-reviewed literature. Add tavily or perplexity only for grey literature needs.\n"
                "\n"
                "candidateConcepts: list 3-6 key concepts/noun phrases that should anchor retrieval.\n"
                "successCriteria: list 2-3 concrete conditions that would make this search successful.\n"
                "Return compact structured output only."
            )
            direct = self._structured_sync(
                endpoint="structured:planner",
                model=planner,
                response_model=_PlannerResponseSchema,
                system_prompt=base_prompt,
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
            text_fallback = self._structured_from_text_sync(
                endpoint="text:planner",
                model=planner,
                response_model=_PlannerResponseSchema,
                system_prompt=self._json_only_system_prompt(
                    base_prompt,
                    json_shape=(
                        '{"intent":"discovery|review|known_item|author|citation|regulatory",'
                        '"querySpecificity":"high|medium|low","ambiguityLevel":"low|medium|high",'
                        '"queryType":"broad_concept|known_item|citation_repair|regulatory|author|review",'
                        '"breadthEstimate":2,"searchAngles":["..."],'
                        '"uncertaintyFlags":["..."],"firstPassMode":"targeted|broad|mixed",'
                        '"retrievalHypotheses":["..."],'
                        '"constraints":{"year":"optional","venue":"optional","focus":"optional"},'
                        '"seedIdentifiers":["..."],"candidateConcepts":["..."],'
                        '"providerPlan":["semantic_scholar"],"authorityFirst":true,'
                        '"anchorType":"optional","anchorValue":"optional",'
                        '"requiredPrimarySources":["optional"],"successCriteria":["optional"],'
                        '"followUpMode":"qa|claim_check|comparison"}'
                    ),
                ),
                payload={
                    "query": query,
                    "mode": mode,
                    "year": year,
                    "venue": venue,
                    "focus": focus,
                },
            )
            if text_fallback is not None:
                self._mark_provider_used()
                return text_fallback.to_planner_decision()
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
            base_prompt = (
                "You are a scientific literature search planner. Classify the query and design the optimal "
                "retrieval strategy.\n"
                "\n"
                "INTENT CLASSIFICATION RULES:\n"
                "- discovery: any broad conceptual, multi-factor, or 'what does the literature say' question. "
                "Default for most queries. Use even when a year or title-like wording appears.\n"
                "- review: explicit requests for literature reviews, systematic reviews, or evidence synthesis.\n"
                "- known_item: ONLY when a hard bibliographic identifier is present (DOI, arXiv ID, or exact "
                "title lookup with no broader research question attached).\n"
                "- author: query is specifically about a person's publication list.\n"
                "- citation: query is repairing or verifying a specific citation reference string.\n"
                "- regulatory: query explicitly references a CFR section, rulemaking, or federal register notice.\n"
                "\n"
                "OUTPUT REQUIREMENTS:\n"
                "- querySpecificity: high|medium|low\n"
                "- ambiguityLevel: low|medium|high\n"
                "- queryType: broad_concept|known_item|citation_repair|regulatory|author|review\n"
                "- breadthEstimate: integer 1-4 where 1 is narrow and 4 is very broad\n"
                "- searchAngles: 2-4 distinct retrieval angles or reformulations for broader discovery asks\n"
                "- firstPassMode: targeted|broad|mixed\n"
                "- uncertaintyFlags: list any ambiguities or competing interpretations\n"
                "- retrievalHypotheses: concrete evidence expectations the search should try to satisfy\n"
                "\n"
                "PROVIDER SELECTION (only from: semantic_scholar, openalex, scholarapi, core, arxiv, ecos, "
                "federal_register, govinfo, tavily, perplexity). Prefer semantic_scholar and openalex for "
                "peer-reviewed literature. Add tavily or perplexity only for grey literature needs.\n"
                "\n"
                "candidateConcepts: list 3-6 key concepts/noun phrases that should anchor retrieval.\n"
                "successCriteria: list 2-3 concrete conditions that would make this search successful.\n"
                "Return compact structured output only."
            )
            direct = await self._structured_async(
                endpoint="structured:planner",
                model=planner,
                response_model=_PlannerResponseSchema,
                system_prompt=base_prompt,
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
            text_fallback = await self._structured_from_text_async(
                endpoint="text:planner",
                model=planner,
                response_model=_PlannerResponseSchema,
                system_prompt=self._json_only_system_prompt(
                    base_prompt,
                    json_shape=(
                        '{"intent":"discovery|review|known_item|author|citation|regulatory",'
                        '"querySpecificity":"high|medium|low","ambiguityLevel":"low|medium|high",'
                        '"queryType":"broad_concept|known_item|citation_repair|regulatory|author|review",'
                        '"breadthEstimate":2,"searchAngles":["..."],'
                        '"uncertaintyFlags":["..."],"firstPassMode":"targeted|broad|mixed",'
                        '"retrievalHypotheses":["..."],'
                        '"constraints":{"year":"optional","venue":"optional","focus":"optional"},'
                        '"seedIdentifiers":["..."],"candidateConcepts":["..."],'
                        '"providerPlan":["semantic_scholar"],"authorityFirst":true,'
                        '"anchorType":"optional","anchorValue":"optional",'
                        '"requiredPrimarySources":["optional"],"successCriteria":["optional"],'
                        '"followUpMode":"qa|claim_check|comparison"}'
                    ),
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
            if text_fallback is not None:
                self._mark_provider_used()
                return text_fallback.to_planner_decision()
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
            base_prompt = (
                "Suggest at most three short literature-search expansions. "
                "Each expansion must preserve the user's research intent, "
                "must not add unrelated domains, and must avoid stopwords, "
                "generic verbs, or filler terms. Label each expansion as "
                "from_input, from_retrieved_evidence, or speculative."
            )
            direct = self._structured_sync(
                endpoint="structured:expansions",
                model=planner,
                response_model=_ExpansionListSchema,
                system_prompt=base_prompt,
                payload={
                    "query": query,
                    "evidence": evidence_texts[:5],
                    "max_variants": max_variants,
                },
            )
            if direct is None:
                direct = self._structured_from_text_sync(
                    endpoint="text:expansions",
                    model=planner,
                    response_model=_ExpansionListSchema,
                    system_prompt=self._json_only_system_prompt(
                        base_prompt,
                        json_shape='{"expansions":[{"variant":"...","source":"from_input|from_retrieved_evidence|speculative","rationale":"..."}]}',
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
            base_prompt = (
                "Suggest at most three short literature-search expansions. "
                "Each expansion must preserve the user's research intent, "
                "must not add unrelated domains, and must avoid stopwords, "
                "generic verbs, or filler terms. Label each expansion as "
                "from_input, from_retrieved_evidence, or speculative."
            )
            direct = await self._structured_async(
                endpoint="structured:expansions",
                model=planner,
                response_model=_ExpansionListSchema,
                system_prompt=base_prompt,
                payload={
                    "query": query,
                    "evidence": evidence_texts[:5],
                    "max_variants": max_variants,
                },
                request_outcomes=request_outcomes,
                request_id=request_id,
            )
            if direct is None:
                direct = await self._structured_from_text_async(
                    endpoint="text:expansions",
                    model=planner,
                    response_model=_ExpansionListSchema,
                    system_prompt=self._json_only_system_prompt(
                        base_prompt,
                        json_shape='{"expansions":[{"variant":"...","source":"from_input|from_retrieved_evidence|speculative","rationale":"..."}]}',
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

    async def asuggest_grounded_expansions(
        self,
        *,
        query: str,
        papers: list[dict[str, Any]],
        max_variants: int,
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> list[ExpansionCandidate]:
        planner, _ = self._load_models()
        paper_payload = [
            {
                "paperId": paper.get("paperId") or paper.get("sourceId") or paper.get("canonicalId"),
                "title": paper.get("title"),
                "abstract": str(paper.get("abstract") or "")[:1000] or None,
            }
            for paper in papers[:8]
        ]
        try:
            base_prompt = (
                "Given the original literature query and the first-pass retrieved papers, suggest only the "
                "missing retrieval angles or concepts that would broaden coverage without drifting off topic. "
                "Return at most three grounded expansions. Label each expansion as from_retrieved_evidence "
                "or hypothesis, and explain briefly why it is missing from the original query."
            )
            direct = await self._structured_async(
                endpoint="structured:grounded_expansions",
                model=planner,
                response_model=_ExpansionListSchema,
                system_prompt=base_prompt,
                payload={
                    "query": query,
                    "papers": paper_payload,
                    "max_variants": max_variants,
                },
                request_outcomes=request_outcomes,
                request_id=request_id,
            )
            if direct is None:
                direct = await self._structured_from_text_async(
                    endpoint="text:grounded_expansions",
                    model=planner,
                    response_model=_ExpansionListSchema,
                    system_prompt=self._json_only_system_prompt(
                        base_prompt,
                        json_shape='{"expansions":[{"variant":"...","source":"from_retrieved_evidence|hypothesis","rationale":"..."}]}',
                    ),
                    payload={
                        "query": query,
                        "papers": paper_payload,
                        "max_variants": max_variants,
                    },
                    request_outcomes=request_outcomes,
                    request_id=request_id,
                )
            if direct is None:
                self._mark_deterministic_fallback()
                return super().suggest_grounded_expansions(
                    query=query,
                    papers=papers,
                    max_variants=max_variants,
                )
            self._mark_provider_used()
            return _filter_expansion_candidates(query, direct.expansions, max_variants=max_variants)
        except Exception:
            logger.exception(
                "%s grounded expansion generation failed; falling back to deterministic grounded expansions.",
                self._provider_name,
            )
            self._mark_deterministic_fallback()
            return super().suggest_grounded_expansions(
                query=query,
                papers=papers,
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
                    "You are a scientific literature analyst. Answer the user's question using ONLY "
                    "the supplied paper abstracts. Follow these rules exactly:\n"
                    "1. For every factual claim, cite the supporting paper by title and year in "
                    "parentheses, e.g. (Smith et al., 2022).\n"
                    "2. If papers report different findings or disagree, describe the disagreement "
                    "explicitly — do not synthesize conflicting results into a single claim.\n"
                    "3. If the question asks about a mechanism, rate, quantity, or causal pathway, "
                    "describe what each relevant paper actually reports about it.\n"
                    "4. Your answer must be at least 3 substantive sentences for any question that "
                    "has supporting evidence. Do not produce keyword lists or title enumerations.\n"
                    "5. If the supplied papers do not adequately address the question, state clearly "
                    "at the end what evidence is missing or would be needed.\n"
                    "6. Confidence must be exactly one of: high, medium, low — based on the "
                    "directness and consistency of the evidence, not topic breadth."
                ),
                payload=_build_answer_payload(question, answer_mode, evidence_papers),
            )
            if direct is not None:
                parsed = _normalize_answer_schema_output(
                    parsed_answer=direct,
                    evidence_papers=evidence_papers,
                    confidence_normalizer=self.normalize_confidence,
                )
                self._mark_provider_used()
                return parsed
            text_fallback = self._structured_from_text_sync(
                endpoint="text:answer",
                model=synthesizer,
                response_model=_AnswerSchema,
                system_prompt=(
                    "You are a scientific literature analyst. Answer the user's question using ONLY "
                    "the supplied paper abstracts. Cite each claim by paper title and year. "
                    "Describe disagreements between papers explicitly. Provide at least 3 substantive "
                    "sentences for any answerable question. State missing evidence at the end if "
                    "the papers are insufficient. Confidence must be exactly one of: high, medium, low. "
                    "Return only JSON with keys: answer, unsupportedAsks, followUpQuestions, confidence, "
                    "answerability, selectedEvidenceIds, selectedLeadIds, citedPaperIds, "
                    "evidenceSummary, missingEvidenceDescription."
                ),
                payload=_build_answer_payload(question, answer_mode, evidence_papers),
            )
            if text_fallback is not None:
                parsed = _normalize_answer_schema_output(
                    parsed_answer=text_fallback,
                    evidence_papers=evidence_papers,
                    confidence_normalizer=self.normalize_confidence,
                )
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
                    "You are a scientific literature analyst. Answer the user's question using ONLY "
                    "the supplied paper abstracts. Follow these rules exactly:\n"
                    "1. For every factual claim, cite the supporting paper by title and year in "
                    "parentheses, e.g. (Smith et al., 2022).\n"
                    "2. If papers report different findings or disagree, describe the disagreement "
                    "explicitly — do not synthesize conflicting results into a single claim.\n"
                    "3. If the question asks about a mechanism, rate, quantity, or causal pathway, "
                    "describe what each relevant paper actually reports about it.\n"
                    "4. Your answer must be at least 3 substantive sentences for any question that "
                    "has supporting evidence. Do not produce keyword lists or title enumerations.\n"
                    "5. If the supplied papers do not adequately address the question, state clearly "
                    "at the end what evidence is missing or would be needed.\n"
                    "6. Confidence must be exactly one of: high, medium, low — based on the "
                    "directness and consistency of the evidence, not topic breadth."
                ),
                payload=_build_answer_payload(question, answer_mode, evidence_papers),
                request_outcomes=request_outcomes,
                request_id=request_id,
            )
            if direct is not None:
                parsed = _normalize_answer_schema_output(
                    parsed_answer=direct,
                    evidence_papers=evidence_papers,
                    confidence_normalizer=self.normalize_confidence,
                )
                self._mark_provider_used()
                return parsed
            text_fallback = await self._structured_from_text_async(
                endpoint="text:answer",
                model=synthesizer,
                response_model=_AnswerSchema,
                system_prompt=(
                    "You are a scientific literature analyst. Answer the user's question using ONLY "
                    "the supplied paper abstracts. Cite each claim by paper title and year. "
                    "Describe disagreements between papers explicitly. Provide at least 3 substantive "
                    "sentences for any answerable question. State missing evidence at the end if "
                    "the papers are insufficient. Confidence must be exactly one of: high, medium, low. "
                    "Return only JSON with keys: answer, unsupportedAsks, followUpQuestions, confidence, "
                    "answerability, selectedEvidenceIds, selectedLeadIds, citedPaperIds, "
                    "evidenceSummary, missingEvidenceDescription."
                ),
                payload=_build_answer_payload(question, answer_mode, evidence_papers),
                request_outcomes=request_outcomes,
                request_id=request_id,
            )
            if text_fallback is not None:
                parsed = _normalize_answer_schema_output(
                    parsed_answer=text_fallback,
                    evidence_papers=evidence_papers,
                    confidence_normalizer=self.normalize_confidence,
                )
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

    async def arevise_search_strategy(
        self,
        *,
        original_query: str,
        original_intent: str,
        tried_providers: list[str],
        result_summary: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        planner, _ = self._load_models()
        try:
            system_prompt = (
                "You are a search strategy advisor. The initial retrieval attempt returned no useful results. "
                "Recommend a revised retrieval strategy.\n"
                "\n"
                "REVISED INTENT options: discovery, review, known_item, author, citation, regulatory\n"
                "REVISED PROVIDERS (only from): semantic_scholar, openalex, scholarapi, core, arxiv, "
                "ecos, federal_register, govinfo, tavily, perplexity\n"
                "\n"
                "Consider: Is the query better served by a different intent? Should the query be rephrased "
                "to be more specific or broader? Which providers are more likely to have this content?\n"
                "Return revisedQuery, revisedIntent, revisedProviders (list), and rationale."
            )
            result = await self._structured_async(
                endpoint="structured:revise_strategy",
                model=planner,
                response_model=_ReviseStrategySchema,
                system_prompt=system_prompt,
                payload={
                    "originalQuery": original_query,
                    "originalIntent": original_intent,
                    "triedProviders": tried_providers,
                    "resultSummary": result_summary,
                },
                request_id=request_id,
            )
            if result is not None:
                self._mark_provider_used()
                providers = _sanitize_provider_plan(
                    intent=result.revised_intent,
                    provider_plan=result.revised_providers,
                )
                return {
                    "revisedQuery": result.revised_query or original_query,
                    "revisedIntent": result.revised_intent,
                    "revisedProviders": providers,
                    "rationale": result.rationale,
                }
        except Exception:
            logger.exception("Strategy revision call failed; using deterministic fallback.")
        return await super().arevise_search_strategy(
            original_query=original_query,
            original_intent=original_intent,
            tried_providers=tried_providers,
            result_summary=result_summary,
            request_id=request_id,
        )

    async def aclassify_relevance_batch(
        self,
        *,
        query: str,
        papers: list[dict[str, Any]],
        request_id: str | None = None,
    ) -> dict[str, dict[str, str]]:
        if not papers:
            return {}
        _, answer_model = self._load_models()
        try:
            paper_items = [
                {
                    "paperId": str(paper.get("paperId") or paper.get("paper_id") or f"paper-{i}"),
                    "title": str(paper.get("title") or ""),
                    "abstract": str(paper.get("abstract") or "")[:500],
                }
                for i, paper in enumerate(papers)
            ]
            system_prompt = (
                "You are a scientific literature relevance classifier. For each paper in the list, "
                "classify its relevance to the research query as:\n"
                "- on_topic: the paper directly addresses the query's core research question\n"
                "- weak_match: the paper is related but does not directly address the query\n"
                "- off_topic: the paper is not relevant to the query\n"
                "Base classification on the paper title and abstract. For each paper, also provide a one-sentence "
                "rationale explaining why it is on-topic, weak_match, or off_topic. Return a classification and "
                "rationale for every paper."
            )
            result = await self._structured_async(
                endpoint="structured:relevance_batch",
                model=answer_model,
                response_model=_RelevanceBatchSchema,
                system_prompt=system_prompt,
                payload={"query": query, "papers": paper_items},
                request_id=request_id,
            )
            if result is not None:
                self._mark_provider_used()
                return {
                    item.paper_id: {
                        "classification": cast(Literal["on_topic", "weak_match", "off_topic"], item.classification),
                        "rationale": str(item.rationale or "").strip(),
                    }
                    for item in result.classifications
                    if item.paper_id
                }
        except Exception:
            logger.exception("Relevance batch classification failed; falling back to weak_match.")
        return {
            str(paper.get("paperId") or paper.get("paper_id") or f"paper-{i}"): {
                "classification": "weak_match",
                "rationale": "Fallback relevance classification used because the model call failed.",
            }
            for i, paper in enumerate(papers)
        }

    async def aassess_result_adequacy(
        self,
        *,
        query: str,
        intent: str,
        verified_sources: list[dict[str, Any]],
        evidence_gaps: list[str],
        request_id: str | None = None,
    ) -> dict[str, str]:
        _, answer_model = self._load_models()
        try:
            result = await self._structured_async(
                endpoint="structured:adequacy_judgment",
                model=answer_model,
                response_model=_AdequacyJudgmentSchema,
                system_prompt=(
                    "You assess whether a research discovery result is adequate. Given the user query, intent, "
                    "the verified sources, and known evidence gaps, return exactly one adequacy label: "
                    "succeeded, partial, or insufficient. Promote to succeeded only when the verified sources "
                    "substantively cover the core query. Return a short reason."
                ),
                payload={
                    "query": query,
                    "intent": intent,
                    "verifiedSources": [
                        {
                            "sourceId": source.get("sourceId") or source.get("sourceAlias"),
                            "title": source.get("title"),
                            "note": source.get("note"),
                            "topicalRelevance": source.get("topicalRelevance"),
                        }
                        for source in verified_sources[:8]
                    ],
                    "evidenceGaps": evidence_gaps[:5],
                },
                request_id=request_id,
            )
            if result is not None:
                self._mark_provider_used()
                return {"adequacy": result.adequacy, "reason": result.reason}
        except Exception:
            logger.exception("Adequacy judgment failed; using deterministic adequacy fallback.")
        return await super().aassess_result_adequacy(
            query=query,
            intent=intent,
            verified_sources=verified_sources,
            evidence_gaps=evidence_gaps,
            request_id=request_id,
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


class NvidiaProviderBundle(LangChainChatProviderBundle):
    """NVIDIA NIM smart-layer adapter via LangChain."""

    def __init__(
        self,
        config: AgenticConfig,
        api_key: str | None,
        *,
        base_url: str | None = None,
        provider_registry: ProviderDiagnosticsRegistry | None = None,
    ) -> None:
        super().__init__(
            config,
            provider_name="nvidia",
            api_key=api_key,
            provider_registry=provider_registry,
        )
        self._base_url = base_url

    def _create_chat_model(self, model_name: str) -> Any:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        kwargs: dict[str, Any] = {
            "model": model_name,
            "api_key": self._api_key,
            "temperature": 0,
        }
        if self._base_url:
            kwargs["base_url"] = self._base_url
        model = ChatNVIDIA(**kwargs)
        client = getattr(model, "_client", None)
        if client is not None and hasattr(client, "timeout"):
            client.timeout = self._timeout_seconds
        return model


class MistralProviderBundle(LangChainChatProviderBundle):
    """Mistral smart-layer adapter via LangChain."""

    def __init__(
        self,
        config: AgenticConfig,
        api_key: str | None,
        *,
        provider_registry: ProviderDiagnosticsRegistry | None = None,
    ) -> None:
        super().__init__(
            config,
            provider_name="mistral",
            api_key=api_key,
            provider_registry=provider_registry,
            structured_output_method="json_schema",
        )

    def _create_chat_model(self, model_name: str) -> Any:
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(
            model_name=model_name,
            api_key=SecretStr(self._api_key or ""),
            temperature=0,
            max_retries=0,
            timeout=int(self._timeout_seconds),
        )


class HuggingFaceProviderBundle(LangChainChatProviderBundle):
    """Hugging Face router smart-layer adapter via OpenAI-compatible chat completions."""

    def __init__(
        self,
        config: AgenticConfig,
        api_key: str | None,
        *,
        base_url: str = "https://router.huggingface.co/v1",
        provider_registry: ProviderDiagnosticsRegistry | None = None,
    ) -> None:
        super().__init__(
            config,
            provider_name="huggingface",
            api_key=api_key,
            provider_registry=provider_registry,
        )
        self._base_url = base_url

    def _create_chat_model(self, model_name: str) -> Any:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            api_key=SecretStr(self._api_key or ""),
            base_url=self._base_url,
            temperature=0,
            max_retries=0,
            timeout=self._timeout_seconds,
        )


class OpenRouterProviderBundle(LangChainChatProviderBundle):
    """OpenRouter smart-layer adapter via OpenAI-compatible chat completions."""

    def __init__(
        self,
        config: AgenticConfig,
        api_key: str | None,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        http_referer: str | None = None,
        title: str | None = None,
        provider_registry: ProviderDiagnosticsRegistry | None = None,
    ) -> None:
        super().__init__(
            config,
            provider_name="openrouter",
            api_key=api_key,
            provider_registry=provider_registry,
            structured_output_method="json_schema",
        )
        self._base_url = base_url
        self._http_referer = (http_referer or "").strip() or None
        self._title = (title or "").strip() or None

    def _create_chat_model(self, model_name: str) -> Any:
        from langchain_openai import ChatOpenAI

        default_headers: dict[str, str] = {}
        if self._http_referer:
            default_headers["HTTP-Referer"] = self._http_referer
        if self._title:
            default_headers["X-OpenRouter-Title"] = self._title

        kwargs: dict[str, Any] = {
            "model": model_name,
            "api_key": SecretStr(self._api_key or ""),
            "base_url": self._base_url,
            "temperature": 0,
            "max_retries": 0,
            "timeout": self._timeout_seconds,
            "extra_body": {"provider": {"require_parameters": True}},
        }
        if default_headers:
            kwargs["default_headers"] = default_headers
        return ChatOpenAI(**kwargs)
