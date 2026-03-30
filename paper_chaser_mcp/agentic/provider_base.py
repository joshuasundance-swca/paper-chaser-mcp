"""Base provider bundles for the additive smart research layer."""

from __future__ import annotations

from typing import Any, Literal

from .config import AgenticConfig
from .models import ExpansionCandidate, PlannerDecision
from .provider_helpers import (
    COMMON_QUERY_WORDS,
    GAP_QUESTION_MARKERS,
    _compact_theme_label,
    _deterministic_comparison_answer,
    _deterministic_gap_insights,
    _deterministic_theme_summary,
    _extract_seed_identifiers,
    _lexical_similarity,
    _normalize_confidence_label,
    _tokenize,
    _top_terms,
)

__all__ = [
    "DeterministicProviderBundle",
    "ModelProviderBundle",
]


class ModelProviderBundle:
    """Provider bundle with planner, synthesis, and embeddings roles."""

    planner_model_name: str
    synthesis_model_name: str
    embedding_model_name: str

    def configured_provider_name(self) -> str:
        return str(getattr(self, "_configured_provider", getattr(self, "_provider_name", "deterministic")))

    def active_provider_name(self) -> str:
        return str(getattr(self, "_last_effective_provider_name", getattr(self, "_provider_name", "deterministic")))

    def _mark_provider_used(self, provider_name: str | None = None) -> None:
        self._last_effective_provider_name = provider_name or getattr(self, "_provider_name", "deterministic")

    def _mark_deterministic_fallback(self) -> None:
        self._last_effective_provider_name = "deterministic"

    def supports_embeddings(self) -> bool:
        return False

    def selection_metadata(self) -> dict[str, Any]:
        configured_provider = self.configured_provider_name()
        active_provider = self.active_provider_name()
        config = getattr(self, "_config", None)
        planner_source = getattr(config, "planner_model_source", "configured")
        synthesis_source = getattr(config, "synthesis_model_source", "configured")
        planner_model = self.planner_model_name
        synthesis_model = self.synthesis_model_name
        if active_provider == "deterministic":
            planner_source = "deterministic"
            synthesis_source = "deterministic"
            planner_model = f"{configured_provider}:deterministic-planner"
            synthesis_model = f"{configured_provider}:deterministic-synthesizer"
        return {
            "configuredSmartProvider": str(configured_provider),
            "activeSmartProvider": str(active_provider),
            "plannerModel": planner_model,
            "plannerModelSource": planner_source,
            "synthesisModel": synthesis_model,
            "synthesisModelSource": synthesis_source,
        }

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
        del request_outcomes, request_id
        return self.plan_search(
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
        return []

    async def asuggest_speculative_expansions(
        self,
        *,
        query: str,
        evidence_texts: list[str],
        max_variants: int,
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> list[ExpansionCandidate]:
        del request_outcomes, request_id
        return self.suggest_speculative_expansions(
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
        raise NotImplementedError

    async def alabel_theme(
        self,
        *,
        seed_terms: list[str],
        papers: list[dict[str, Any]],
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> str:
        del request_outcomes, request_id
        return self.label_theme(seed_terms=seed_terms, papers=papers)

    def summarize_theme(
        self,
        *,
        title: str,
        papers: list[dict[str, Any]],
    ) -> str:
        raise NotImplementedError

    async def asummarize_theme(
        self,
        *,
        title: str,
        papers: list[dict[str, Any]],
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> str:
        del request_outcomes, request_id
        return self.summarize_theme(title=title, papers=papers)

    def answer_question(
        self,
        *,
        question: str,
        evidence_papers: list[dict[str, Any]],
        answer_mode: str,
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def aanswer_question(
        self,
        *,
        question: str,
        evidence_papers: list[dict[str, Any]],
        answer_mode: str,
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        del request_outcomes, request_id
        return self.answer_question(
            question=question,
            evidence_papers=evidence_papers,
            answer_mode=answer_mode,
        )

    def embed_query(self, text: str) -> tuple[float, ...] | None:
        return None

    async def aembed_query(
        self,
        text: str,
        *,
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> tuple[float, ...] | None:
        del request_outcomes, request_id
        return self.embed_query(text)

    def embed_texts(self, texts: list[str]) -> list[tuple[float, ...] | None]:
        return [self.embed_query(text) for text in texts]

    async def aembed_texts(
        self,
        texts: list[str],
        *,
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> list[tuple[float, ...] | None]:
        del request_outcomes, request_id
        return self.embed_texts(texts)

    def similarity(self, left: str, right: str) -> float:
        return _lexical_similarity(left, right)

    def batched_similarity(self, query: str, texts: list[str]) -> list[float]:
        return [self.similarity(query, text) for text in texts]

    async def abatched_similarity(
        self,
        query: str,
        texts: list[str],
        *,
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> list[float]:
        del request_outcomes, request_id
        return self.batched_similarity(query, texts)

    def normalize_confidence(self, value: Any) -> Literal["high", "medium", "low"]:
        return _normalize_confidence_label(value)

    async def aclose(self) -> None:
        """Close any provider-specific resources."""


class DeterministicProviderBundle(ModelProviderBundle):
    """Pure-Python fallback planner/synthesizer for tests and offline use."""

    def __init__(self, config: AgenticConfig) -> None:
        self._config = config
        self._configured_provider = config.provider
        self._provider_name = "deterministic"
        self._last_effective_provider_name = "deterministic"
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
        if any(marker in lowered for marker in ("doi", "arxiv:", "https://", "http://")):
            inferred_intent = "known_item"
        elif "author" in lowered and mode == "auto":
            inferred_intent = "author"
        elif any(marker in lowered for marker in ("citation", "cites", "cited by")) and mode == "auto":
            inferred_intent = "citation"
        elif any(marker in lowered for marker in ("survey", "review", "landscape")) and mode == "auto":
            inferred_intent = "review"

        concept_text = " ".join(part for part in [normalized, focus or ""] if isinstance(part, str) and part)
        candidate_concepts = [token for token in _tokenize(concept_text) if token not in COMMON_QUERY_WORDS][:8]
        constraints = {key: value for key, value in {"year": year, "venue": venue, "focus": focus}.items() if value}
        return PlannerDecision(
            intent=inferred_intent,  # type: ignore[arg-type]
            constraints=constraints,
            seedIdentifiers=_extract_seed_identifiers(normalized),
            candidateConcepts=candidate_concepts,
            providerPlan=["semantic_scholar", "openalex", "scholarapi", "core", "arxiv"],
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
                    rationale=(f"Common evidence term '{term}' appears across top results."),
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
        return _compact_theme_label(seed_terms, papers)

    def summarize_theme(
        self,
        *,
        title: str,
        papers: list[dict[str, Any]],
    ) -> str:
        return _deterministic_theme_summary(title, papers)

    def answer_question(
        self,
        *,
        question: str,
        evidence_papers: list[dict[str, Any]],
        answer_mode: str,
    ) -> dict[str, Any]:
        if not evidence_papers:
            return {
                "answer": ("The saved result set does not contain enough evidence to answer this confidently."),
                "unsupportedAsks": [question],
                "followUpQuestions": [
                    "Try a broader search query.",
                    "Add a method, venue, or year constraint.",
                ],
                "confidence": "low",
            }

        titles: list[str] = [
            str(paper.get("title") or paper.get("paperId") or "Untitled") for paper in evidence_papers[:4]
        ]
        if answer_mode == "claim_check":
            status = "supported" if len(evidence_papers) >= 2 else "insufficient_evidence"
            return {
                "answer": (
                    f"Claim check status: {status}. The strongest supporting "
                    f"evidence in this result set comes from {', '.join(titles[:2])}."
                ),
                "unsupportedAsks": ([] if status != "insufficient_evidence" else [question]),
                "followUpQuestions": ["Would you like a comparison of the top evidence papers?"],
                "confidence": "medium" if status == "supported" else "low",
            }
        if answer_mode == "comparison":
            return {
                "answer": _deterministic_comparison_answer(question, evidence_papers),
                "unsupportedAsks": [],
                "followUpQuestions": ["Which comparison dimension matters most: method, data, or recency?"],
                "confidence": "medium",
            }
        question_tokens = set(_tokenize(question))
        if question_tokens & GAP_QUESTION_MARKERS:
            insights = _deterministic_gap_insights(evidence_papers)
            answer_text = (
                "Across these papers, the main recurring knowledge gaps are " + "; ".join(insights[:4]) + "."
                if insights
                else (
                    "Across these papers, the evidence still looks uneven, so the "
                    "remaining gaps are best framed as broader taxonomic or "
                    "geographic coverage, stronger long-term outcome tracking, and "
                    "clearer links from observed responses to population or fitness "
                    "consequences."
                )
            )
            return {
                "answer": answer_text,
                "unsupportedAsks": [],
                "followUpQuestions": [
                    "Should I map which papers support each gap?",
                    "Do you want a deeper landscape summary for this result set?",
                ],
                "confidence": "medium" if insights else "low",
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