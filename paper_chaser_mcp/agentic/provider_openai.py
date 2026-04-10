"""OpenAI-compatible provider bundles for the additive smart research layer."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal, cast

from pydantic import SecretStr

from ..provider_runtime import ProviderDiagnosticsRegistry
from ..transport import maybe_close_async_resource
from .config import AgenticConfig
from .models import ExpansionCandidate, PlannerDecision
from .provider_base import DeterministicProviderBundle
from .provider_helpers import (
    _AdequacyJudgmentSchema,
    _AnswerSchema,
    _build_answer_payload,
    _build_theme_label_payload,
    _build_theme_summary_payload,
    _cosine_similarity,
    _ExpansionListSchema,
    _extract_json_object,
    _filter_expansion_candidates,
    _normalize_answer_schema_output,
    _normalize_theme_label_output,
    _normalized_embedding_text,
    _PlannerResponseSchema,
    _RelevanceBatchSchema,
    _ResponseModelT,
    _ReviseStrategySchema,
    _sanitize_provider_plan,
)

logger = logging.getLogger("paper-chaser-mcp")

__all__ = [
    "AzureOpenAIProviderBundle",
    "OpenAIProviderBundle",
]


def _execute_provider_call_sync(**kwargs: Any) -> Any:
    from . import providers as providers_facade

    return providers_facade.execute_provider_call_sync(**kwargs)


async def _execute_provider_call(**kwargs: Any) -> Any:
    from . import providers as providers_facade

    return await providers_facade.execute_provider_call(**kwargs)


class OpenAIProviderBundle(DeterministicProviderBundle):
    """Best-effort LangChain-backed OpenAI adapter with deterministic fallback."""

    _ANSWER_TEXT_FALLBACK_MAX_OUTPUT_TOKENS = 400

    def __init__(
        self,
        config: AgenticConfig,
        api_key: str | None,
        *,
        provider_registry: ProviderDiagnosticsRegistry | None = None,
    ) -> None:
        super().__init__(config)
        self._configured_provider = config.provider
        self._provider_name = "openai"
        self.planner_model_name = config.planner_model
        self.synthesis_model_name = config.synthesis_model
        self.embedding_model_name = config.embedding_model
        self._disable_embeddings = config.disable_embeddings
        self._timeout_seconds = config.openai_timeout_seconds
        self._api_key = api_key
        self._provider_registry = provider_registry
        self._openai_client: Any | None = None
        self._async_openai_client: Any | None = None
        self._planner: Any | None = None
        self._synthesizer: Any | None = None
        self._embeddings: Any | None = None
        self._embedding_cache: dict[str, tuple[float, ...]] = {}

    def supports_embeddings(self) -> bool:
        return (not self._disable_embeddings) and bool(self._api_key)

    def _allow_langchain_chat_fallback(self) -> bool:
        """Whether sync methods may fall back to LangChain chat/completions calls."""
        return True

    def _load_openai_client(self) -> Any | None:
        if self._openai_client is not None:
            return self._openai_client
        if not self._api_key:
            return None
        try:
            from openai import OpenAI
        except ImportError:
            logger.info("openai is not installed; falling back to LangChain and deterministic smart-provider adapters.")
            return None
        self._openai_client = OpenAI(
            api_key=self._api_key,
            timeout=self._timeout_seconds,
            max_retries=0,
        )
        return self._openai_client

    def _load_async_openai_client(self) -> Any | None:
        if self._async_openai_client is not None:
            return self._async_openai_client
        if not self._api_key:
            return None
        try:
            from openai import AsyncOpenAI
        except ImportError:
            logger.info("openai is not installed; falling back to deterministic smart-provider adapters.")
            return None
        self._async_openai_client = AsyncOpenAI(
            api_key=self._api_key,
            timeout=self._timeout_seconds,
            max_retries=0,
        )
        return self._async_openai_client

    def _load_models(self) -> tuple[Any | None, Any | None]:
        if self._planner is not None and self._synthesizer is not None:
            return self._planner, self._synthesizer
        if not self._api_key:
            return None, None
        try:
            from langchain.chat_models import init_chat_model
        except ImportError:
            logger.info(
                "LangChain v1 chat model helpers are not installed; falling back to deterministic smart planning."
            )
            return None, None
        self._planner = init_chat_model(
            model=self.planner_model_name,
            model_provider="openai",
            api_key=self._api_key,
            temperature=0,
        )
        self._synthesizer = init_chat_model(
            model=self.synthesis_model_name,
            model_provider="openai",
            api_key=self._api_key,
            temperature=0,
        )
        return self._planner, self._synthesizer

    def _load_embeddings(self) -> Any | None:
        if self._embeddings is not None:
            return self._embeddings
        if self._disable_embeddings:
            return None
        if not self._api_key:
            return None
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            logger.info("langchain-openai is not installed; falling back to lexical similarity for smart ranking.")
            return None

        self._embeddings = OpenAIEmbeddings(
            model=self.embedding_model_name,
            api_key=SecretStr(self._api_key),
            max_retries=0,
        )
        return self._embeddings

    def _cache_embedding(self, text: str, vector: list[float]) -> tuple[float, ...]:
        normalized = _normalized_embedding_text(text)
        cached = tuple(float(value) for value in vector)
        self._embedding_cache[normalized] = cached
        return cached

    async def aclose(self) -> None:
        """Close any lazily created OpenAI clients."""
        async_client, self._async_openai_client = self._async_openai_client, None
        sync_client, self._openai_client = self._openai_client, None
        await maybe_close_async_resource(async_client)
        await maybe_close_async_resource(sync_client)

    async def _await_with_total_timeout(
        self,
        awaitable: Any,
        *,
        operation_name: str,
    ) -> Any:
        try:
            return await asyncio.wait_for(awaitable, timeout=self._timeout_seconds)
        except TimeoutError as exc:
            raise TimeoutError(
                f"OpenAI {operation_name} exceeded total timeout of {self._timeout_seconds:.1f}s"
            ) from exc

    @staticmethod
    def _responses_input(
        system_prompt: str,
        payload: dict[str, Any],
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload)},
        ]

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        for item in getattr(response, "output", None) or []:
            for content in getattr(item, "content", None) or []:
                text = getattr(content, "text", None)
                if isinstance(text, str) and text.strip():
                    return text.strip()
        return ""

    @staticmethod
    def _extract_response_parsed(
        response: Any,
        response_model: type[_ResponseModelT],
    ) -> _ResponseModelT:
        parsed = getattr(response, "output_parsed", None)
        if parsed is not None:
            if isinstance(parsed, response_model):
                return parsed
            return response_model.model_validate(parsed)
        for item in getattr(response, "output", None) or []:
            for content in getattr(item, "content", None) or []:
                parsed_content = getattr(content, "parsed", None)
                if parsed_content is not None:
                    if isinstance(parsed_content, response_model):
                        return parsed_content
                    return response_model.model_validate(parsed_content)
                text = getattr(content, "text", None)
                if isinstance(text, str) and text.strip():
                    return response_model.model_validate_json(text)
        text = OpenAIProviderBundle._extract_response_text(response)
        if text:
            return response_model.model_validate_json(text)
        raise ValueError("OpenAI Responses payload did not include structured output.")

    def _responses_parse(
        self,
        *,
        endpoint: str,
        model_name: str,
        response_model: type[_ResponseModelT],
        system_prompt: str,
        payload: dict[str, Any],
        previous_response_id: str | None = None,
    ) -> _ResponseModelT | None:
        client = self._load_openai_client()
        if client is None or not hasattr(client, "responses"):
            return None
        if not hasattr(client.responses, "parse"):
            return None

        def _call() -> Any:
            kwargs: dict[str, Any] = {
                "model": model_name,
                "input": self._responses_input(system_prompt, payload),
                "text_format": response_model,
            }
            if previous_response_id:
                kwargs["previous_response_id"] = previous_response_id
            return client.responses.parse(**kwargs)

        response = _execute_provider_call_sync(
            provider=self._provider_name,
            endpoint=endpoint,
            operation=_call,
            registry=self._provider_registry,
        )
        if response.payload is None:
            logger.warning(
                "OpenAI Responses structured call failed for %s: %s",
                endpoint,
                response.outcome.error or response.outcome.fallback_reason,
            )
            return None
        try:
            return self._extract_response_parsed(response.payload, response_model)
        except Exception:
            logger.exception("OpenAI Responses parse failed for %s.", endpoint)
            return None

    def _responses_text(
        self,
        *,
        endpoint: str,
        model_name: str,
        system_prompt: str,
        payload: dict[str, Any],
        max_output_tokens: int | None = None,
        previous_response_id: str | None = None,
    ) -> str | None:
        client = self._load_openai_client()
        if client is None or not hasattr(client, "responses"):
            return None

        def _call() -> Any:
            kwargs: dict[str, Any] = {
                "model": model_name,
                "input": self._responses_input(system_prompt, payload),
            }
            if max_output_tokens is not None:
                kwargs["max_output_tokens"] = max_output_tokens
            if previous_response_id:
                kwargs["previous_response_id"] = previous_response_id
            return client.responses.create(**kwargs)

        response = _execute_provider_call_sync(
            provider=self._provider_name,
            endpoint=endpoint,
            operation=_call,
            registry=self._provider_registry,
        )
        if response.payload is None:
            return None
        text = self._extract_response_text(response.payload)
        return text or None

    async def _aresponses_parse(
        self,
        *,
        endpoint: str,
        model_name: str,
        response_model: type[_ResponseModelT],
        system_prompt: str,
        payload: dict[str, Any],
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
        previous_response_id: str | None = None,
    ) -> _ResponseModelT | None:
        client = self._load_async_openai_client()
        if client is None or not hasattr(client, "responses"):
            return None
        if not hasattr(client.responses, "parse"):
            return None

        async def _call() -> Any:
            kwargs: dict[str, Any] = {
                "model": model_name,
                "input": self._responses_input(system_prompt, payload),
                "text_format": response_model,
            }
            if previous_response_id:
                kwargs["previous_response_id"] = previous_response_id
            return await self._await_with_total_timeout(
                client.responses.parse(**kwargs),
                operation_name=endpoint,
            )

        response = await _execute_provider_call(
            provider=self._provider_name,
            endpoint=endpoint,
            operation=_call,
            registry=self._provider_registry,
            request_outcomes=request_outcomes,
            request_id=request_id,
        )
        if response.payload is None:
            logger.warning(
                "OpenAI Responses structured call failed for %s: %s",
                endpoint,
                response.outcome.error or response.outcome.fallback_reason,
            )
            return None
        try:
            return self._extract_response_parsed(response.payload, response_model)
        except Exception:
            logger.exception("OpenAI Responses parse failed for %s.", endpoint)
            return None

    async def _aresponses_text(
        self,
        *,
        endpoint: str,
        model_name: str,
        system_prompt: str,
        payload: dict[str, Any],
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
        max_output_tokens: int | None = None,
        previous_response_id: str | None = None,
    ) -> str | None:
        client = self._load_async_openai_client()
        if client is None or not hasattr(client, "responses"):
            return None

        async def _call() -> Any:
            kwargs: dict[str, Any] = {
                "model": model_name,
                "input": self._responses_input(system_prompt, payload),
            }
            if max_output_tokens is not None:
                kwargs["max_output_tokens"] = max_output_tokens
            if previous_response_id:
                kwargs["previous_response_id"] = previous_response_id
            return await self._await_with_total_timeout(
                client.responses.create(**kwargs),
                operation_name=endpoint,
            )

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
        text = self._extract_response_text(response.payload)
        return text or None

    def _parse_answer_text_fallback(
        self,
        text_fallback: str,
        *,
        evidence_papers: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        json_payload = _extract_json_object(text_fallback) or text_fallback.strip()
        try:
            parsed_answer = _AnswerSchema.model_validate_json(json_payload)
        except Exception as exc:
            logger.warning(
                "OpenAI answer text fallback returned invalid JSON; using deterministic fallback: %s: %s",
                type(exc).__name__,
                exc,
            )
            return None
        return _normalize_answer_schema_output(
            parsed_answer=parsed_answer,
            evidence_papers=evidence_papers,
            confidence_normalizer=self.normalize_confidence,
        )

    @staticmethod
    def _embedding_vectors(payload: Any) -> list[list[float]]:
        data = getattr(payload, "data", None)
        if data is None and isinstance(payload, dict):
            data = payload.get("data")
        vectors: list[list[float]] = []
        for item in data or []:
            embedding = getattr(item, "embedding", None)
            if embedding is None and isinstance(item, dict):
                embedding = item.get("embedding")
            if isinstance(embedding, list):
                vectors.append([float(value) for value in embedding])
        return vectors

    def _log_embedding_batch_start(
        self,
        *,
        request_id: str | None,
        total_texts: int,
        uncached_texts: int,
    ) -> None:
        if request_id is None:
            return
        logger.info(
            "embedding-batch[%s] model=%s total_texts=%s uncached_texts=%s timeout_s=%s",
            request_id,
            self.embedding_model_name,
            total_texts,
            uncached_texts,
            self._timeout_seconds,
        )

    def _log_embedding_batch_failure(
        self,
        *,
        request_id: str | None,
        total_texts: int,
        uncached_texts: int,
        status_bucket: str,
        reason: str | None,
    ) -> None:
        if request_id is None:
            return
        logger.warning(
            "embedding-batch[%s] model=%s total_texts=%s uncached_texts=%s timeout_s=%s status=%s reason=%s",
            request_id,
            self.embedding_model_name,
            total_texts,
            uncached_texts,
            self._timeout_seconds,
            status_bucket,
            reason or "unknown",
        )

    def embed_query(self, text: str) -> tuple[float, ...] | None:
        if self._disable_embeddings:
            return None
        normalized = _normalized_embedding_text(text)
        if not normalized:
            return None
        cached = self._embedding_cache.get(normalized)
        if cached is not None:
            return cached
        client = self._load_openai_client()
        if client is not None and hasattr(client, "embeddings"):
            try:
                response = _execute_provider_call_sync(
                    provider=self._provider_name,
                    endpoint="embeddings.create",
                    operation=lambda: client.embeddings.create(
                        model=self.embedding_model_name,
                        input=normalized,
                    ),
                    registry=self._provider_registry,
                )
                vectors = self._embedding_vectors(response.payload)
                if vectors:
                    return self._cache_embedding(normalized, vectors[0])
            except Exception:
                logger.exception("OpenAI embeddings failed; falling back to LangChain or lexical similarity.")
        embeddings = self._load_embeddings()
        if embeddings is None:
            return None
        try:
            vector = embeddings.embed_query(normalized)
        except Exception:
            logger.exception("OpenAI embeddings failed; falling back to lexical similarity.")
            return None
        return self._cache_embedding(normalized, vector)

    def embed_texts(self, texts: list[str]) -> list[tuple[float, ...] | None]:
        if self._disable_embeddings:
            return [None for _ in texts]
        normalized_texts = [_normalized_embedding_text(text) for text in texts]
        client = self._load_openai_client()
        pending = [text for text in normalized_texts if text and text not in self._embedding_cache]
        if client is not None and hasattr(client, "embeddings") and pending:
            try:
                response = _execute_provider_call_sync(
                    provider=self._provider_name,
                    endpoint="embeddings.create",
                    operation=lambda: client.embeddings.create(
                        model=self.embedding_model_name,
                        input=pending,
                    ),
                    registry=self._provider_registry,
                )
                vectors = self._embedding_vectors(response.payload)
                for text, vector in zip(pending, vectors):
                    self._cache_embedding(text, vector)
            except Exception:
                logger.exception("Batched OpenAI embeddings failed; falling back to LangChain or lexical similarity.")
        embeddings = self._load_embeddings()
        if embeddings is None:
            return [self.embed_query(text) for text in normalized_texts]

        pending = [text for text in normalized_texts if text and text not in self._embedding_cache]
        if pending:
            try:
                vectors = embeddings.embed_documents(pending)
            except Exception:
                logger.exception("Batched OpenAI embeddings failed; falling back to per-text lexical similarity.")
                return [self.embed_query(text) for text in normalized_texts]
            for text, vector in zip(pending, vectors):
                self._cache_embedding(text, vector)
        return [self._embedding_cache.get(text) if text else None for text in normalized_texts]

    def similarity(self, left: str, right: str) -> float:
        lexical = super().similarity(left, right)
        left_embedding = self.embed_query(left)
        right_embedding = self.embed_query(right)
        if left_embedding is None or right_embedding is None:
            return lexical
        semantic = _cosine_similarity(left_embedding, right_embedding)
        return max(0.0, min(1.0, 0.55 * semantic + 0.45 * lexical))

    def batched_similarity(self, query: str, texts: list[str]) -> list[float]:
        _parent_similarity = super().similarity
        lexical_scores = [_parent_similarity(query, text) for text in texts]
        query_embedding = self.embed_query(query)
        if query_embedding is None:
            return lexical_scores

        text_embeddings = self.embed_texts(texts)
        scores: list[float] = []
        for lexical, embedding in zip(lexical_scores, text_embeddings):
            if embedding is None:
                scores.append(lexical)
                continue
            semantic = _cosine_similarity(query_embedding, embedding)
            scores.append(max(0.0, min(1.0, 0.55 * semantic + 0.45 * lexical)))
        return scores

    async def aembed_query(
        self,
        text: str,
        *,
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> tuple[float, ...] | None:
        embeddings = await self.aembed_texts(
            [text],
            request_outcomes=request_outcomes,
            request_id=request_id,
        )
        return embeddings[0] if embeddings else None

    async def aembed_texts(
        self,
        texts: list[str],
        *,
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> list[tuple[float, ...] | None]:
        if self._disable_embeddings:
            return [None for _ in texts]
        normalized_texts = [_normalized_embedding_text(text) for text in texts]
        pending = [text for text in normalized_texts if text and text not in self._embedding_cache]
        client = self._load_async_openai_client()
        if client is not None and hasattr(client, "embeddings") and pending:
            self._log_embedding_batch_start(
                request_id=request_id,
                total_texts=len(normalized_texts),
                uncached_texts=len(pending),
            )
            response = await _execute_provider_call(
                provider=self._provider_name,
                endpoint="embeddings.create",
                operation=lambda: self._await_with_total_timeout(
                    client.embeddings.create(
                        model=self.embedding_model_name,
                        input=pending,
                    ),
                    operation_name="embeddings.create",
                ),
                registry=self._provider_registry,
                request_outcomes=request_outcomes,
                request_id=request_id,
            )
            if response.payload is not None:
                vectors = self._embedding_vectors(response.payload)
                for text, vector in zip(pending, vectors):
                    self._cache_embedding(text, vector)
            else:
                self._log_embedding_batch_failure(
                    request_id=request_id,
                    total_texts=len(normalized_texts),
                    uncached_texts=len(pending),
                    status_bucket=response.outcome.status_bucket,
                    reason=response.outcome.error or response.outcome.fallback_reason,
                )
                logger.warning(
                    "Async OpenAI embeddings failed for %s: %s",
                    request_id or "request",
                    response.outcome.error or response.outcome.fallback_reason,
                )
        return [self._embedding_cache.get(text) if text else None for text in normalized_texts]

    async def abatched_similarity(
        self,
        query: str,
        texts: list[str],
        *,
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> list[float]:
        _parent_similarity = super().similarity
        lexical_scores = [_parent_similarity(query, text) for text in texts]
        query_embedding = await self.aembed_query(
            query,
            request_outcomes=request_outcomes,
            request_id=request_id,
        )
        if query_embedding is None:
            return lexical_scores

        text_embeddings = await self.aembed_texts(
            texts,
            request_outcomes=request_outcomes,
            request_id=request_id,
        )
        scores: list[float] = []
        for lexical, embedding in zip(lexical_scores, text_embeddings):
            if embedding is None:
                scores.append(lexical)
                continue
            semantic = _cosine_similarity(query_embedding, embedding)
            scores.append(max(0.0, min(1.0, 0.55 * semantic + 0.45 * lexical)))
        return scores

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
        try:
            direct = await self._aresponses_parse(
                endpoint="responses.parse:planner",
                model_name=self.planner_model_name,
                response_model=_PlannerResponseSchema,
                system_prompt=(
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
            logger.exception("Async OpenAI planner failed; falling back to deterministic planning.")
        self._mark_deterministic_fallback()
        return super().plan_search(
            query=query,
            mode=mode,
            year=year,
            venue=venue,
            focus=focus,
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
        try:
            response = await self._aresponses_parse(
                endpoint="responses.parse:expansions",
                model_name=self.planner_model_name,
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
            if response is None:
                self._mark_deterministic_fallback()
                return super().suggest_speculative_expansions(
                    query=query,
                    evidence_texts=evidence_texts,
                    max_variants=max_variants,
                )
            self._mark_provider_used()
            return _filter_expansion_candidates(
                query,
                response.expansions,
                max_variants=max_variants,
            )
        except Exception:
            logger.exception("Async OpenAI variant generation failed; falling back to deterministic expansions.")
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
        try:
            response = await self._aresponses_parse(
                endpoint="responses.parse:grounded_expansions",
                model_name=self.planner_model_name,
                response_model=_ExpansionListSchema,
                system_prompt=(
                    "Given the original literature query and the first-pass retrieved papers, suggest only the "
                    "missing retrieval angles or concepts that would broaden coverage without drifting off topic. "
                    "Return at most three grounded expansions. Label each expansion as from_retrieved_evidence "
                    "or hypothesis, and explain briefly why it is missing from the original query."
                ),
                payload={
                    "query": query,
                    "papers": [
                        {
                            "paperId": paper.get("paperId") or paper.get("sourceId") or paper.get("canonicalId"),
                            "title": paper.get("title"),
                            "abstract": str(paper.get("abstract") or "")[:1000] or None,
                        }
                        for paper in papers[:8]
                    ],
                    "max_variants": max_variants,
                },
                request_outcomes=request_outcomes,
                request_id=request_id,
            )
            if response is None:
                self._mark_deterministic_fallback()
                return super().suggest_grounded_expansions(
                    query=query,
                    papers=papers,
                    max_variants=max_variants,
                )
            self._mark_provider_used()
            return _filter_expansion_candidates(query, response.expansions, max_variants=max_variants)
        except Exception:
            logger.exception(
                "Async OpenAI grounded expansion generation failed; falling back to deterministic expansions."
            )
            self._mark_deterministic_fallback()
            return super().suggest_grounded_expansions(
                query=query,
                papers=papers,
                max_variants=max_variants,
            )

    async def alabel_theme(
        self,
        *,
        seed_terms: list[str],
        papers: list[dict[str, Any]],
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> str:
        try:
            direct = await self._aresponses_text(
                endpoint="responses.create:label_theme",
                model_name=self.synthesis_model_name,
                system_prompt="Write a very short literature theme label.",
                payload=_build_theme_label_payload(seed_terms, papers),
                request_outcomes=request_outcomes,
                request_id=request_id,
                max_output_tokens=40,
            )
            if direct:
                self._mark_provider_used()
                return _normalize_theme_label_output(direct)
        except Exception:
            logger.exception("Async OpenAI theme labeling failed; falling back to deterministic theme labels.")
        self._mark_deterministic_fallback()
        return super().label_theme(seed_terms=seed_terms, papers=papers)

    async def asummarize_theme(
        self,
        *,
        title: str,
        papers: list[dict[str, Any]],
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> str:
        try:
            direct = await self._aresponses_text(
                endpoint="responses.create:summarize_theme",
                model_name=self.synthesis_model_name,
                system_prompt="Summarize one literature cluster in two sentences.",
                payload=_build_theme_summary_payload(title, papers),
                request_outcomes=request_outcomes,
                request_id=request_id,
                max_output_tokens=180,
            )
            if direct:
                self._mark_provider_used()
                return direct
        except Exception:
            logger.exception("Async OpenAI theme summarization failed; falling back to deterministic summaries.")
        self._mark_deterministic_fallback()
        return super().summarize_theme(title=title, papers=papers)

    async def aanswer_question(
        self,
        *,
        question: str,
        evidence_papers: list[dict[str, Any]],
        answer_mode: str,
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            direct = await self._aresponses_parse(
                endpoint="responses.parse:answer",
                model_name=self.synthesis_model_name,
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
            text_fallback = await self._aresponses_text(
                endpoint="responses.create:answer",
                model_name=self.synthesis_model_name,
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
                max_output_tokens=self._ANSWER_TEXT_FALLBACK_MAX_OUTPUT_TOKENS,
            )
            if text_fallback:
                fallback_parsed = self._parse_answer_text_fallback(
                    text_fallback,
                    evidence_papers=evidence_papers,
                )
                if fallback_parsed is not None:
                    self._mark_provider_used()
                    return fallback_parsed
        except Exception:
            logger.exception("Async OpenAI synthesis failed; falling back to deterministic answer generation.")
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
        try:
            result = await self._aresponses_parse(
                endpoint="responses.parse:revise_strategy",
                model_name=self.planner_model_name,
                response_model=_ReviseStrategySchema,
                system_prompt=(
                    "You are a search strategy advisor. The initial retrieval attempt returned no useful results. "
                    "Recommend a revised retrieval strategy. Return revisedQuery, revisedIntent, "
                    "revisedProviders, and rationale."
                ),
                payload={
                    "originalQuery": original_query,
                    "originalIntent": original_intent,
                    "triedProviders": tried_providers,
                    "resultSummary": result_summary,
                },
                request_outcomes=None,
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
            logger.exception("Async OpenAI strategy revision failed; using deterministic fallback.")
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
        try:
            result = await self._aresponses_parse(
                endpoint="responses.parse:relevance_batch",
                model_name=self.synthesis_model_name,
                response_model=_RelevanceBatchSchema,
                system_prompt=(
                    "You are a scientific literature relevance classifier. For each paper in the list, classify "
                    "its relevance to the research query as on_topic, weak_match, or off_topic, and provide "
                    "a one-sentence rationale for the classification."
                ),
                payload={
                    "query": query,
                    "papers": [
                        {
                            "paperId": str(paper.get("paperId") or paper.get("paper_id") or f"paper-{i}"),
                            "title": str(paper.get("title") or ""),
                            "abstract": str(paper.get("abstract") or "")[:500],
                        }
                        for i, paper in enumerate(papers)
                    ],
                },
                request_outcomes=None,
                request_id=request_id,
            )
            if result is not None:
                self._mark_provider_used()
                return {
                    item.paper_id: {
                        "classification": cast(
                            Literal["on_topic", "weak_match", "off_topic"],
                            item.classification,
                        ),
                        "rationale": str(item.rationale or "").strip(),
                    }
                    for item in result.classifications
                    if item.paper_id
                }
        except Exception:
            logger.exception("Async OpenAI relevance batch failed; falling back to weak_match.")
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
        try:
            result = await self._aresponses_parse(
                endpoint="responses.parse:adequacy_judgment",
                model_name=self.synthesis_model_name,
                response_model=_AdequacyJudgmentSchema,
                system_prompt=(
                    "You assess whether a research discovery result is adequate. Given the user query, intent, "
                    "the verified sources, and known evidence gaps, return exactly one adequacy label: "
                    "succeeded, partial, or insufficient, plus a short reason."
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
                request_outcomes=None,
                request_id=request_id,
            )
            if result is not None:
                self._mark_provider_used()
                return {"adequacy": result.adequacy, "reason": result.reason}
        except Exception:
            logger.exception("Async OpenAI adequacy judgment failed; using deterministic fallback.")
        return await super().aassess_result_adequacy(
            query=query,
            intent=intent,
            verified_sources=verified_sources,
            evidence_gaps=evidence_gaps,
            request_id=request_id,
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
            direct = self._responses_parse(
                endpoint="responses.parse:planner",
                model_name=self.planner_model_name,
                response_model=_PlannerResponseSchema,
                system_prompt=(
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
            if not self._allow_langchain_chat_fallback():
                self._mark_deterministic_fallback()
                return super().plan_search(
                    query=query,
                    mode=mode,
                    year=year,
                    venue=venue,
                    focus=focus,
                )
            if planner is None:
                self._mark_deterministic_fallback()
                return super().plan_search(
                    query=query,
                    mode=mode,
                    year=year,
                    venue=venue,
                    focus=focus,
                )

            structured = planner.with_structured_output(
                _PlannerResponseSchema,
                method="function_calling",
            )
            response = structured.invoke(
                [
                    (
                        "system",
                        "You are a scientific literature search planner. Classify the query and design the optimal "
                        "retrieval strategy.\n"
                        "\n"
                        "INTENT CLASSIFICATION RULES:\n"
                        "- discovery: any broad conceptual, multi-factor, or 'what does the literature say' "
                        "question. "
                        "Default for most queries. Use even when a year or title-like wording appears.\n"
                        "- review: explicit requests for literature reviews, systematic reviews, or evidence "
                        "synthesis.\n"
                        "- known_item: ONLY when a hard bibliographic identifier is present (DOI, arXiv ID, or exact "
                        "title lookup with no broader research question attached).\n"
                        "- author: query is specifically about a person's publication list.\n"
                        "- citation: query is repairing or verifying a specific citation reference string.\n"
                        "- regulatory: query explicitly references a CFR section, rulemaking, or federal register "
                        "notice.\n"
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
                        "PROVIDER SELECTION (only from: semantic_scholar, openalex, scholarapi, core, arxiv, "
                        "ecos, federal_register, govinfo, tavily, perplexity). Prefer semantic_scholar and "
                        "openalex for peer-reviewed literature. Add tavily or perplexity only for grey "
                        "literature needs.\n"
                        "\n"
                        "candidateConcepts: list 3-6 key concepts/noun phrases that should anchor retrieval.\n"
                        "successCriteria: list 2-3 concrete conditions that would make this search successful.\n"
                        "Return compact structured output only.",
                    ),
                    (
                        "human",
                        json.dumps(
                            {
                                "query": query,
                                "mode": mode,
                                "year": year,
                                "venue": venue,
                                "focus": focus,
                            }
                        ),
                    ),
                ]
            )
            self._mark_provider_used()
            return response.to_planner_decision()
        except Exception:
            logger.exception("OpenAI planner failed; falling back to deterministic planning.")
            self._mark_deterministic_fallback()
            return super().plan_search(
                query=query,
                mode=mode,
                year=year,
                venue=venue,
                focus=focus,
            )

    def suggest_speculative_expansions(
        self,
        *,
        query: str,
        evidence_texts: list[str],
        max_variants: int,
    ) -> list[ExpansionCandidate]:
        planner, _ = self._load_models()
        try:
            direct = self._responses_parse(
                endpoint="responses.parse:expansions",
                model_name=self.planner_model_name,
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
            if direct is not None:
                response = direct
            else:
                if not self._allow_langchain_chat_fallback():
                    self._mark_deterministic_fallback()
                    return super().suggest_speculative_expansions(
                        query=query,
                        evidence_texts=evidence_texts,
                        max_variants=max_variants,
                    )
                if planner is None:
                    self._mark_deterministic_fallback()
                    return super().suggest_speculative_expansions(
                        query=query,
                        evidence_texts=evidence_texts,
                        max_variants=max_variants,
                    )
                structured = planner.with_structured_output(
                    _ExpansionListSchema,
                    method="function_calling",
                )
                response = structured.invoke(
                    [
                        (
                            "system",
                            "Suggest at most three short literature-search expansions. "
                            "Each expansion must preserve the user's research intent, "
                            "must not add unrelated domains, and must avoid stopwords, "
                            "generic verbs, or filler terms. Label each expansion as "
                            "from_input, from_retrieved_evidence, or speculative.",
                        ),
                        (
                            "human",
                            json.dumps(
                                {
                                    "query": query,
                                    "evidence": evidence_texts[:5],
                                    "max_variants": max_variants,
                                }
                            ),
                        ),
                    ]
                )
            self._mark_provider_used()
            return _filter_expansion_candidates(
                query,
                response.expansions,
                max_variants=max_variants,
            )
        except Exception:
            logger.exception("OpenAI variant generation failed; falling back to deterministic expansions.")
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
        direct = self._responses_text(
            endpoint="responses.create:label_theme",
            model_name=self.synthesis_model_name,
            system_prompt="Write a very short literature theme label.",
            payload=_build_theme_label_payload(seed_terms, papers),
            max_output_tokens=40,
        )
        if direct:
            self._mark_provider_used()
            return _normalize_theme_label_output(direct)
        if not self._allow_langchain_chat_fallback():
            self._mark_deterministic_fallback()
            return super().label_theme(seed_terms=seed_terms, papers=papers)
        _, synthesizer = self._load_models()
        if synthesizer is None:
            self._mark_deterministic_fallback()
            return super().label_theme(seed_terms=seed_terms, papers=papers)
        try:
            response = synthesizer.invoke(
                [
                    ("system", "Write a very short literature theme label."),
                    (
                        "human",
                        json.dumps(_build_theme_label_payload(seed_terms, papers)),
                    ),
                ]
            )
            label = _normalize_theme_label_output(str(response.content))
            if label:
                self._mark_provider_used()
                return label
            self._mark_deterministic_fallback()
            return super().label_theme(seed_terms=seed_terms, papers=papers)
        except Exception:
            self._mark_deterministic_fallback()
            return super().label_theme(seed_terms=seed_terms, papers=papers)

    def summarize_theme(
        self,
        *,
        title: str,
        papers: list[dict[str, Any]],
    ) -> str:
        direct = self._responses_text(
            endpoint="responses.create:summarize_theme",
            model_name=self.synthesis_model_name,
            system_prompt="Summarize one literature cluster in two sentences.",
            payload=_build_theme_summary_payload(title, papers),
            max_output_tokens=180,
        )
        if direct:
            self._mark_provider_used()
            return direct
        if not self._allow_langchain_chat_fallback():
            self._mark_deterministic_fallback()
            return super().summarize_theme(title=title, papers=papers)
        _, synthesizer = self._load_models()
        if synthesizer is None:
            self._mark_deterministic_fallback()
            return super().summarize_theme(title=title, papers=papers)
        try:
            response = synthesizer.invoke(
                [
                    ("system", "Summarize one literature cluster in two sentences."),
                    (
                        "human",
                        json.dumps(_build_theme_summary_payload(title, papers)),
                    ),
                ]
            )
            content = str(response.content).strip()
            if content:
                self._mark_provider_used()
                return content
            self._mark_deterministic_fallback()
            return super().summarize_theme(title=title, papers=papers)
        except Exception:
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
            direct = self._responses_parse(
                endpoint="responses.parse:answer",
                model_name=self.synthesis_model_name,
                response_model=_AnswerSchema,
                system_prompt=(
                    "Answer only from the supplied papers. If evidence is weak, "
                    "say so. Confidence must be exactly one of: high, medium, low."
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
            else:
                text_fallback = self._responses_text(
                    endpoint="responses.create:answer",
                    model_name=self.synthesis_model_name,
                    system_prompt=(
                        "Answer only from the supplied papers. If evidence is weak, say so. "
                        "Return only JSON with keys answer, unsupportedAsks, followUpQuestions, confidence, "
                        "answerability, selectedEvidenceIds, and selectedLeadIds. "
                        "Confidence must be exactly one of: high, medium, low."
                    ),
                    payload=_build_answer_payload(question, answer_mode, evidence_papers),
                    max_output_tokens=self._ANSWER_TEXT_FALLBACK_MAX_OUTPUT_TOKENS,
                )
                if text_fallback:
                    fallback_parsed = self._parse_answer_text_fallback(
                        text_fallback,
                        evidence_papers=evidence_papers,
                    )
                    if fallback_parsed is not None:
                        self._mark_provider_used()
                        return fallback_parsed
                if not self._allow_langchain_chat_fallback():
                    self._mark_deterministic_fallback()
                    return super().answer_question(
                        question=question,
                        evidence_papers=evidence_papers,
                        answer_mode=answer_mode,
                    )
                if synthesizer is None:
                    self._mark_deterministic_fallback()
                    return super().answer_question(
                        question=question,
                        evidence_papers=evidence_papers,
                        answer_mode=answer_mode,
                    )
                structured = synthesizer.with_structured_output(
                    _AnswerSchema,
                    method="function_calling",
                )
                response = structured.invoke(
                    [
                        (
                            "system",
                            "Answer only from the supplied papers. If evidence is "
                            "weak, say so. Confidence must be exactly one of: high, "
                            "medium, low. Include answerability, selectedEvidenceIds, "
                            "and selectedLeadIds in the structured response.",
                        ),
                        (
                            "human",
                            json.dumps(_build_answer_payload(question, answer_mode, evidence_papers)),
                        ),
                    ]
                )
                parsed = _normalize_answer_schema_output(
                    parsed_answer=response,
                    evidence_papers=evidence_papers,
                    confidence_normalizer=self.normalize_confidence,
                )
                self._mark_provider_used()
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            logger.exception("OpenAI synthesis failed; falling back to deterministic answer generation.")
        self._mark_deterministic_fallback()
        return super().answer_question(
            question=question,
            evidence_papers=evidence_papers,
            answer_mode=answer_mode,
        )


class AzureOpenAIProviderBundle(OpenAIProviderBundle):
    """Azure OpenAI adapter that reuses the OpenAI-compatible response path."""

    def __init__(
        self,
        config: AgenticConfig,
        api_key: str | None,
        azure_endpoint: str | None,
        api_version: str | None,
        *,
        azure_planner_deployment: str | None = None,
        azure_synthesis_deployment: str | None = None,
        provider_registry: ProviderDiagnosticsRegistry | None = None,
    ) -> None:
        super().__init__(config, api_key, provider_registry=provider_registry)
        self._provider_name = "azure-openai"
        self._azure_endpoint = azure_endpoint
        self._api_version = api_version
        self._azure_planner_deployment = azure_planner_deployment
        self._azure_synthesis_deployment = azure_synthesis_deployment
        if azure_planner_deployment:
            self.planner_model_name = azure_planner_deployment
        if azure_synthesis_deployment:
            self.synthesis_model_name = azure_synthesis_deployment

    def supports_embeddings(self) -> bool:
        return (not self._disable_embeddings) and bool(self._api_key and self._azure_endpoint)

    def _allow_langchain_chat_fallback(self) -> bool:
        """Azure sync parity should prefer responses and then deterministic fallback."""
        return False

    def _load_openai_client(self) -> Any | None:
        if self._openai_client is not None:
            return self._openai_client
        if not self._api_key or not self._azure_endpoint:
            return None
        try:
            from openai import AzureOpenAI
        except ImportError:
            logger.info("openai is not installed; falling back to LangChain and deterministic smart-provider adapters.")
            return None
        self._openai_client = AzureOpenAI(
            api_key=self._api_key,
            azure_endpoint=self._azure_endpoint,
            api_version=self._api_version,
            timeout=self._timeout_seconds,
            max_retries=0,
        )
        return self._openai_client

    def _load_async_openai_client(self) -> Any | None:
        if self._async_openai_client is not None:
            return self._async_openai_client
        if not self._api_key or not self._azure_endpoint:
            return None
        try:
            from openai import AsyncAzureOpenAI
        except ImportError:
            logger.info("openai is not installed; falling back to deterministic smart-provider adapters.")
            return None
        self._async_openai_client = AsyncAzureOpenAI(
            api_key=self._api_key,
            azure_endpoint=self._azure_endpoint,
            api_version=self._api_version,
            timeout=self._timeout_seconds,
            max_retries=0,
        )
        return self._async_openai_client

    def _load_models(self) -> tuple[Any | None, Any | None]:
        if self._planner is not None and self._synthesizer is not None:
            return self._planner, self._synthesizer
        if not self._api_key or not self._azure_endpoint:
            return None, None
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError:
            logger.info("langchain-openai is not installed; falling back to deterministic smart planning.")
            return None, None

        azure_chat_model: Any = AzureChatOpenAI
        planner_kwargs = {
            "azure_endpoint": self._azure_endpoint,
            "api_key": SecretStr(self._api_key),
            "api_version": self._api_version,
            "azure_deployment": self.planner_model_name,
            "temperature": 0,
            "timeout": self._timeout_seconds,
            "max_retries": 0,
        }
        synthesis_kwargs = {
            "azure_endpoint": self._azure_endpoint,
            "api_key": SecretStr(self._api_key),
            "api_version": self._api_version,
            "azure_deployment": self.synthesis_model_name,
            "temperature": 0,
            "timeout": self._timeout_seconds,
            "max_retries": 0,
        }
        self._planner = azure_chat_model(**planner_kwargs)
        self._synthesizer = azure_chat_model(**synthesis_kwargs)
        return self._planner, self._synthesizer

    def _load_embeddings(self) -> Any | None:
        if self._embeddings is not None:
            return self._embeddings
        if self._disable_embeddings:
            return None
        if not self._api_key or not self._azure_endpoint:
            return None
        try:
            from langchain_openai import AzureOpenAIEmbeddings
        except ImportError:
            logger.info(
                "langchain-openai is not installed; falling back to lexical similarity for Azure OpenAI smart ranking."
            )
            return None

        self._embeddings = AzureOpenAIEmbeddings(
            model=self.embedding_model_name,
            azure_deployment=self.embedding_model_name,
            api_key=SecretStr(self._api_key),
            azure_endpoint=self._azure_endpoint,
            api_version=self._api_version,
            max_retries=0,
            timeout=self._timeout_seconds,
        )
        return self._embeddings
