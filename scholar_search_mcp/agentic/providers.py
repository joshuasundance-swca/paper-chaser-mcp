"""Model-provider adapters for the additive smart research layer."""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr

from ..provider_runtime import (
    ProviderDiagnosticsRegistry,
    execute_provider_call_sync,
)
from .config import AgenticConfig
from .models import ExpansionCandidate, PlannerDecision

logger = logging.getLogger("scholar-search-mcp")

TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
COMMON_QUERY_WORDS = {
    "paper",
    "papers",
    "research",
    "study",
    "studies",
    "review",
    "recent",
    "latest",
    "work",
    "works",
}
MAX_EMBED_TEXT_LENGTH = 6_000


class _PlannerConstraintsSchema(BaseModel):
    """OpenAI Structured Outputs-compatible planner constraints."""

    year: str | None = None
    venue: str | None = None
    focus: str | None = None


class _PlannerResponseSchema(BaseModel):
    """Structured planner response that avoids free-form object maps."""

    intent: Literal[
        "discovery",
        "review",
        "known_item",
        "author",
        "citation",
    ] = "discovery"
    constraints: _PlannerConstraintsSchema = Field(
        default_factory=_PlannerConstraintsSchema
    )
    seedIdentifiers: list[str] = Field(default_factory=list)
    candidateConcepts: list[str] = Field(default_factory=list)
    providerPlan: list[str] = Field(default_factory=list)
    followUpMode: Literal["qa", "claim_check", "comparison"] = "qa"

    def to_planner_decision(self) -> PlannerDecision:
        return PlannerDecision(
            intent=self.intent,
            constraints={
                key: value
                for key, value in self.constraints.model_dump(exclude_none=True).items()
                if value
            },
            seedIdentifiers=self.seedIdentifiers,
            candidateConcepts=self.candidateConcepts,
            providerPlan=self.providerPlan,
            followUpMode=self.followUpMode,
        )


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _normalized_embedding_text(text: str) -> str:
    return " ".join(text.split())[:MAX_EMBED_TEXT_LENGTH]


def _top_terms(texts: list[str], *, limit: int = 8) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(
            token for token in _tokenize(text) if token not in COMMON_QUERY_WORDS
        )
    return [term for term, _ in counts.most_common(limit)]


def _lexical_similarity(left: str, right: str) -> float:
    left_tokens: Counter[str] = Counter(_tokenize(left))
    right_tokens: Counter[str] = Counter(_tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = set(left_tokens) & set(right_tokens)
    numerator = sum(left_tokens[token] * right_tokens[token] for token in intersection)
    left_norm = math.sqrt(sum(value * value for value in left_tokens.values()))
    right_norm = math.sqrt(sum(value * value for value in right_tokens.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    try:
        import numpy as np
    except ImportError:
        numerator = sum(
            left_value * right_value for left_value, right_value in zip(left, right)
        )
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    left_array = np.array(left)
    right_array = np.array(right)
    left_norm = float(np.linalg.norm(left_array))
    right_norm = float(np.linalg.norm(right_array))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(np.dot(left_array, right_array) / (left_norm * right_norm))


def _normalize_confidence_label(value: Any) -> Literal["high", "medium", "low"]:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"high", "medium", "low"}:
            return normalized  # type: ignore[return-value]
        if normalized in {"strong", "very_high", "very high"}:
            return "high"
        if normalized in {"moderate", "mid", "mixed"}:
            return "medium"
        if normalized in {"weak", "uncertain", "insufficient"}:
            return "low"
        try:
            numeric = float(normalized)
        except ValueError:
            numeric = None
        if numeric is not None:
            if numeric >= 0.8:
                return "high"
            if numeric >= 0.5:
                return "medium"
            return "low"
    if isinstance(value, (int, float)):
        if value >= 0.8:
            return "high"
        if value >= 0.5:
            return "medium"
        return "low"
    return "medium"


class ModelProviderBundle:
    """Provider bundle with planner, synthesis, and embeddings roles."""

    planner_model_name: str
    synthesis_model_name: str
    embedding_model_name: str

    def plan_search(
        self,
        *,
        query: str,
        mode: str,
        year: str | None = None,
        venue: str | None = None,
        focus: str | None = None,
    ) -> PlannerDecision:
        raise NotImplementedError

    def suggest_speculative_expansions(
        self,
        *,
        query: str,
        evidence_texts: list[str],
        max_variants: int,
    ) -> list[ExpansionCandidate]:
        return []

    def label_theme(
        self,
        *,
        seed_terms: list[str],
        papers: list[dict[str, Any]],
    ) -> str:
        raise NotImplementedError

    def summarize_theme(
        self,
        *,
        title: str,
        papers: list[dict[str, Any]],
    ) -> str:
        raise NotImplementedError

    def answer_question(
        self,
        *,
        question: str,
        evidence_papers: list[dict[str, Any]],
        answer_mode: str,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def embed_query(self, text: str) -> tuple[float, ...] | None:
        return None

    def embed_texts(self, texts: list[str]) -> list[tuple[float, ...] | None]:
        return [self.embed_query(text) for text in texts]

    def similarity(self, left: str, right: str) -> float:
        return _lexical_similarity(left, right)

    def batched_similarity(self, query: str, texts: list[str]) -> list[float]:
        return [self.similarity(query, text) for text in texts]

    def normalize_confidence(self, value: Any) -> Literal["high", "medium", "low"]:
        return _normalize_confidence_label(value)


class DeterministicProviderBundle(ModelProviderBundle):
    """Pure-Python fallback planner/synthesizer for tests and offline use."""

    def __init__(self, config: AgenticConfig) -> None:
        self.planner_model_name = f"{config.provider}:deterministic-planner"
        self.synthesis_model_name = f"{config.provider}:deterministic-synthesizer"
        self.embedding_model_name = f"{config.provider}:deterministic-lexical"

    def plan_search(
        self,
        *,
        query: str,
        mode: str,
        year: str | None = None,
        venue: str | None = None,
        focus: str | None = None,
    ) -> PlannerDecision:
        normalized = query.strip()
        lowered = normalized.lower()
        inferred_intent = mode if mode != "auto" else "discovery"
        if any(
            marker in lowered for marker in ("doi", "arxiv:", "https://", "http://")
        ):
            inferred_intent = "known_item"
        elif "author" in lowered and mode == "auto":
            inferred_intent = "author"
        elif (
            any(marker in lowered for marker in ("citation", "cites", "cited by"))
            and mode == "auto"
        ):
            inferred_intent = "citation"
        elif (
            any(marker in lowered for marker in ("survey", "review", "landscape"))
            and mode == "auto"
        ):
            inferred_intent = "review"

        concept_text = " ".join(
            part for part in [normalized, focus or ""] if isinstance(part, str) and part
        )
        candidate_concepts = [
            token
            for token in _tokenize(concept_text)
            if token not in COMMON_QUERY_WORDS
        ][:8]
        constraints = {
            key: value
            for key, value in {"year": year, "venue": venue, "focus": focus}.items()
            if value
        }
        return PlannerDecision(
            intent=inferred_intent,  # type: ignore[arg-type]
            constraints=constraints,
            seedIdentifiers=_extract_seed_identifiers(normalized),
            candidateConcepts=candidate_concepts,
            providerPlan=["semantic_scholar", "openalex", "core", "arxiv"],
            followUpMode="claim_check" if inferred_intent == "review" else "qa",
        )

    def suggest_speculative_expansions(
        self,
        *,
        query: str,
        evidence_texts: list[str],
        max_variants: int,
    ) -> list[ExpansionCandidate]:
        variants: list[ExpansionCandidate] = []
        top_terms = _top_terms(evidence_texts, limit=max_variants * 2)
        for term in top_terms:
            if term in query.lower():
                continue
            variants.append(
                ExpansionCandidate(
                    variant=f"{query} {term}",
                    source="from_retrieved_evidence",
                    rationale=(
                        f"Common evidence term '{term}' appears across top results."
                    ),
                )
            )
            if len(variants) >= max_variants:
                break
        return variants

    def label_theme(
        self,
        *,
        seed_terms: list[str],
        papers: list[dict[str, Any]],
    ) -> str:
        if seed_terms:
            return " / ".join(seed_terms[:2]).title()
        if papers and papers[0].get("venue"):
            return f"{papers[0]['venue']} cluster"
        return "General theme"

    def summarize_theme(
        self,
        *,
        title: str,
        papers: list[dict[str, Any]],
    ) -> str:
        if not papers:
            return f"{title}: no papers were available to summarize."

        venues = sorted(
            [
                str(paper["venue"])
                for paper in papers
                if isinstance(paper.get("venue"), str) and paper.get("venue")
            ]
        )
        years = sorted(
            [paper["year"] for paper in papers if isinstance(paper.get("year"), int)]
        )
        venue_text = f" across {', '.join(venues[:2])}" if venues else ""
        if years:
            year_text = (
                f" spanning {years[0]}-{years[-1]}"
                if len(years) > 1
                else f" in {years[0]}"
            )
        else:
            year_text = ""
        return (
            f"{title} groups {len(papers)} papers{venue_text}{year_text}. "
            "These papers share overlapping terms in their titles and abstracts."
        )

    def answer_question(
        self,
        *,
        question: str,
        evidence_papers: list[dict[str, Any]],
        answer_mode: str,
    ) -> dict[str, Any]:
        if not evidence_papers:
            return {
                "answer": (
                    "The saved result set does not contain enough evidence to answer "
                    "this confidently."
                ),
                "unsupportedAsks": [question],
                "followUpQuestions": [
                    "Try a broader search query.",
                    "Add a method, venue, or year constraint.",
                ],
                "confidence": "low",
            }

        titles: list[str] = [
            str(paper.get("title") or paper.get("paperId") or "Untitled")
            for paper in evidence_papers[:4]
        ]
        if answer_mode == "claim_check":
            status = (
                "supported" if len(evidence_papers) >= 2 else "insufficient_evidence"
            )
            return {
                "answer": (
                    f"Claim check status: {status}. The strongest supporting "
                    f"evidence in this result set comes from {', '.join(titles[:2])}."
                ),
                "unsupportedAsks": (
                    [] if status != "insufficient_evidence" else [question]
                ),
                "followUpQuestions": [
                    "Would you like a comparison of the top evidence papers?"
                ],
                "confidence": "medium" if status == "supported" else "low",
            }
        if answer_mode == "comparison":
            comparison_lines = [
                (
                    f"- {paper.get('title') or paper.get('paperId')}: "
                    f"{paper.get('venue') or 'venue unknown'}, "
                    f"{paper.get('year') or 'year unknown'}"
                )
                for paper in evidence_papers[:4]
            ]
            return {
                "answer": "Comparison grounded in the saved result set:\n"
                + "\n".join(comparison_lines),
                "unsupportedAsks": [],
                "followUpQuestions": [
                    "Which comparison dimension matters most: method, data, or recency?"
                ],
                "confidence": "medium",
            }
        return {
            "answer": (
                f"The saved result set most directly answers this through "
                f"{', '.join(titles[:3])}. These papers are the closest matches "
                "within the current workspace."
            ),
            "unsupportedAsks": [],
            "followUpQuestions": [
                "Would you like a claim check?",
                "Should I map the themes in this result set?",
            ],
            "confidence": "medium",
        }


class OpenAIProviderBundle(DeterministicProviderBundle):
    """Best-effort LangChain-backed OpenAI adapter with deterministic fallback."""

    def __init__(
        self,
        config: AgenticConfig,
        api_key: str | None,
        *,
        provider_registry: ProviderDiagnosticsRegistry | None = None,
    ) -> None:
        super().__init__(config)
        self.planner_model_name = config.planner_model
        self.synthesis_model_name = config.synthesis_model
        self.embedding_model_name = config.embedding_model
        self._api_key = api_key
        self._provider_registry = provider_registry
        self._openai_client: Any | None = None
        self._planner: Any | None = None
        self._synthesizer: Any | None = None
        self._embeddings: Any | None = None
        self._embedding_cache: dict[str, tuple[float, ...]] = {}

    def _load_openai_client(self) -> Any | None:
        if self._openai_client is not None:
            return self._openai_client
        if not self._api_key:
            return None
        try:
            from openai import OpenAI
        except ImportError:
            logger.info(
                "openai is not installed; falling back to LangChain and "
                "deterministic smart-provider adapters."
            )
            return None
        self._openai_client = OpenAI(api_key=self._api_key)
        return self._openai_client

    def _load_models(self) -> tuple[Any | None, Any | None]:
        if self._planner is not None or self._synthesizer is not None:
            return self._planner, self._synthesizer
        if not self._api_key:
            return None, None
        try:
            from langchain.chat_models import init_chat_model
        except ImportError:
            logger.info(
                "LangChain v1 chat model helpers are not installed; falling "
                "back to deterministic smart planning."
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
        if not self._api_key:
            return None
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            logger.info(
                "langchain-openai is not installed; falling back to lexical "
                "similarity for smart ranking."
            )
            return None

        self._embeddings = OpenAIEmbeddings(
            model=self.embedding_model_name,
            api_key=SecretStr(self._api_key),
        )
        return self._embeddings

    def _cache_embedding(self, text: str, vector: list[float]) -> tuple[float, ...]:
        normalized = _normalized_embedding_text(text)
        cached = tuple(float(value) for value in vector)
        self._embedding_cache[normalized] = cached
        return cached

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
        response_model: type[BaseModel],
    ) -> BaseModel:
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
        response_model: type[BaseModel],
        system_prompt: str,
        payload: dict[str, Any],
        previous_response_id: str | None = None,
    ) -> BaseModel | None:
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

        response = execute_provider_call_sync(
            provider="openai",
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

        response = execute_provider_call_sync(
            provider="openai",
            endpoint=endpoint,
            operation=_call,
            registry=self._provider_registry,
        )
        if response.payload is None:
            return None
        text = self._extract_response_text(response.payload)
        return text or None

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

    def embed_query(self, text: str) -> tuple[float, ...] | None:
        normalized = _normalized_embedding_text(text)
        if not normalized:
            return None
        cached = self._embedding_cache.get(normalized)
        if cached is not None:
            return cached
        client = self._load_openai_client()
        if client is not None and hasattr(client, "embeddings"):
            try:
                response = execute_provider_call_sync(
                    provider="openai",
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
                logger.exception(
                    "OpenAI embeddings failed; falling back to LangChain "
                    "or lexical similarity."
                )
        embeddings = self._load_embeddings()
        if embeddings is None:
            return None
        try:
            vector = embeddings.embed_query(normalized)
        except Exception:
            logger.exception(
                "OpenAI embeddings failed; falling back to lexical similarity."
            )
            return None
        return self._cache_embedding(normalized, vector)

    def embed_texts(self, texts: list[str]) -> list[tuple[float, ...] | None]:
        normalized_texts = [_normalized_embedding_text(text) for text in texts]
        client = self._load_openai_client()
        pending = [
            text
            for text in normalized_texts
            if text and text not in self._embedding_cache
        ]
        if client is not None and hasattr(client, "embeddings") and pending:
            try:
                response = execute_provider_call_sync(
                    provider="openai",
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
                logger.exception(
                    "Batched OpenAI embeddings failed; falling back to "
                    "LangChain or lexical similarity."
                )
        embeddings = self._load_embeddings()
        if embeddings is None:
            return [self.embed_query(text) for text in normalized_texts]

        pending = [
            text
            for text in normalized_texts
            if text and text not in self._embedding_cache
        ]
        if pending:
            try:
                vectors = embeddings.embed_documents(pending)
            except Exception:
                logger.exception(
                    "Batched OpenAI embeddings failed; falling back to per-text "
                    "lexical similarity."
                )
                return [self.embed_query(text) for text in normalized_texts]
            for text, vector in zip(pending, vectors):
                self._cache_embedding(text, vector)
        return [
            self._embedding_cache.get(text) if text else None
            for text in normalized_texts
        ]

    def similarity(self, left: str, right: str) -> float:
        lexical = super().similarity(left, right)
        left_embedding = self.embed_query(left)
        right_embedding = self.embed_query(right)
        if left_embedding is None or right_embedding is None:
            return lexical
        semantic = _cosine_similarity(left_embedding, right_embedding)
        return max(0.0, min(1.0, 0.55 * semantic + 0.45 * lexical))

    def batched_similarity(self, query: str, texts: list[str]) -> list[float]:
        lexical_scores = [super().similarity(query, text) for text in texts]
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
                    "Plan a grounded literature-search workflow. Keep providerPlan "
                    "limited to semantic_scholar, openalex, core, and arxiv. "
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
                return direct.to_planner_decision()
            if planner is None:
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
                        "Plan a grounded literature-search workflow. Keep "
                        "providerPlan limited to semantic_scholar, openalex, "
                        "core, and arxiv. Return compact structured output only.",
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
            return response.to_planner_decision()
        except Exception:
            logger.exception(
                "OpenAI planner failed; falling back to deterministic planning."
            )
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
            from pydantic import BaseModel, Field

            class ExpansionSchema(BaseModel):
                variant: str
                source: str = Field(default="speculative")
                rationale: str = Field(default="")

            class ExpansionListSchema(BaseModel):
                expansions: list[ExpansionSchema] = Field(default_factory=list)

            direct = self._responses_parse(
                endpoint="responses.parse:expansions",
                model_name=self.planner_model_name,
                response_model=ExpansionListSchema,
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
                if planner is None:
                    return super().suggest_speculative_expansions(
                        query=query,
                        evidence_texts=evidence_texts,
                        max_variants=max_variants,
                    )
                structured = planner.with_structured_output(
                    ExpansionListSchema,
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
            variants: list[ExpansionCandidate] = []
            for item in response.expansions[:max_variants]:
                variant = item.variant.strip()
                if not variant:
                    continue
                new_tokens = [
                    token
                    for token in _tokenize(variant)
                    if token not in set(_tokenize(query))
                ]
                if not new_tokens or all(
                    token in COMMON_QUERY_WORDS for token in new_tokens
                ):
                    continue
                variants.append(ExpansionCandidate.model_validate(item.model_dump()))
            return variants
        except Exception:
            logger.exception(
                "OpenAI variant generation failed; falling back to "
                "deterministic expansions."
            )
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
            payload={
                "seed_terms": seed_terms,
                "titles": [paper.get("title") for paper in papers[:6]],
            },
            max_output_tokens=40,
        )
        if direct:
            return direct.strip().strip('"')
        _, synthesizer = self._load_models()
        if synthesizer is None:
            return super().label_theme(seed_terms=seed_terms, papers=papers)
        try:
            response = synthesizer.invoke(
                [
                    ("system", "Write a very short literature theme label."),
                    (
                        "human",
                        json.dumps(
                            {
                                "seed_terms": seed_terms,
                                "titles": [paper.get("title") for paper in papers[:6]],
                            }
                        ),
                    ),
                ]
            )
            label = str(response.content).strip().strip('"')
            return label or super().label_theme(seed_terms=seed_terms, papers=papers)
        except Exception:
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
            payload={
                "title": title,
                "papers": [
                    {
                        "title": paper.get("title"),
                        "abstract": paper.get("abstract"),
                        "venue": paper.get("venue"),
                        "year": paper.get("year"),
                    }
                    for paper in papers[:5]
                ],
            },
            max_output_tokens=180,
        )
        if direct:
            return direct
        _, synthesizer = self._load_models()
        if synthesizer is None:
            return super().summarize_theme(title=title, papers=papers)
        try:
            response = synthesizer.invoke(
                [
                    ("system", "Summarize one literature cluster in two sentences."),
                    (
                        "human",
                        json.dumps(
                            {
                                "title": title,
                                "papers": [
                                    {
                                        "title": paper.get("title"),
                                        "abstract": paper.get("abstract"),
                                        "venue": paper.get("venue"),
                                        "year": paper.get("year"),
                                    }
                                    for paper in papers[:5]
                                ],
                            }
                        ),
                    ),
                ]
            )
            content = str(response.content).strip()
            return content or super().summarize_theme(title=title, papers=papers)
        except Exception:
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
            from pydantic import BaseModel, Field

            class AnswerSchema(BaseModel):
                answer: str = Field(default="")
                unsupportedAsks: list[str] = Field(default_factory=list)
                followUpQuestions: list[str] = Field(default_factory=list)
                confidence: Literal["high", "medium", "low"] = "medium"

            direct = self._responses_parse(
                endpoint="responses.parse:answer",
                model_name=self.synthesis_model_name,
                response_model=AnswerSchema,
                system_prompt=(
                    "Answer only from the supplied papers. If evidence is weak, "
                    "say so. Confidence must be exactly one of: high, medium, low."
                ),
                payload={
                    "question": question,
                    "answer_mode": answer_mode,
                    "evidence": [
                        {
                            "title": paper.get("title"),
                            "abstract": paper.get("abstract"),
                            "venue": paper.get("venue"),
                            "year": paper.get("year"),
                        }
                        for paper in evidence_papers[:6]
                    ],
                },
            )
            if direct is not None:
                parsed = direct.model_dump()
            else:
                if synthesizer is None:
                    return super().answer_question(
                        question=question,
                        evidence_papers=evidence_papers,
                        answer_mode=answer_mode,
                    )
                structured = synthesizer.with_structured_output(
                    AnswerSchema,
                    method="function_calling",
                )
                response = structured.invoke(
                    [
                        (
                            "system",
                            "Answer only from the supplied papers. If evidence is "
                            "weak, say so. Confidence must be exactly one of: high, "
                            "medium, low.",
                        ),
                        (
                            "human",
                            json.dumps(
                                {
                                    "question": question,
                                    "answer_mode": answer_mode,
                                    "evidence": [
                                        {
                                            "title": paper.get("title"),
                                            "abstract": paper.get("abstract"),
                                            "venue": paper.get("venue"),
                                            "year": paper.get("year"),
                                        }
                                        for paper in evidence_papers[:6]
                                    ],
                                }
                            ),
                        ),
                    ]
                )
                parsed = response.model_dump()
            parsed["confidence"] = self.normalize_confidence(parsed.get("confidence"))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            logger.exception(
                "OpenAI synthesis failed; falling back to deterministic answer "
                "generation."
            )
        return super().answer_question(
            question=question,
            evidence_papers=evidence_papers,
            answer_mode=answer_mode,
        )


def resolve_provider_bundle(
    config: AgenticConfig,
    *,
    openai_api_key: str | None,
    provider_registry: ProviderDiagnosticsRegistry | None = None,
) -> ModelProviderBundle:
    """Resolve the configured provider bundle with deterministic fallback."""
    if config.provider == "deterministic":
        return DeterministicProviderBundle(config)
    return OpenAIProviderBundle(
        config,
        openai_api_key,
        provider_registry=provider_registry,
    )


def _extract_seed_identifiers(query: str) -> list[str]:
    identifiers: list[str] = []
    for pattern in (
        r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)",
        r"(arxiv:\d{4}\.\d{4,5}(?:v\d+)?)",
        r"((?:https?://)[^\s]+)",
        r"(\d{4}\.\d{4,5}(?:v\d+)?)",
    ):
        for match in re.findall(pattern, query, flags=re.IGNORECASE):
            identifiers.append(str(match))
    seen: set[str] = set()
    deduped: list[str] = []
    for identifier in identifiers:
        if identifier in seen:
            continue
        seen.add(identifier)
        deduped.append(identifier)
    return deduped
