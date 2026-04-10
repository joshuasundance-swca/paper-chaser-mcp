"""Base provider bundles for the additive smart research layer."""

from __future__ import annotations

from typing import Any, Literal, cast

from .config import AgenticConfig
from .models import (
    RETRIEVAL_MODE_BROAD,
    RETRIEVAL_MODE_MIXED,
    RETRIEVAL_MODE_TARGETED,
    ExpansionCandidate,
    PlannerDecision,
)
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


def _default_selected_evidence_ids(evidence_papers: list[dict[str, Any]], *, limit: int = 3) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for paper in evidence_papers[:limit]:
        if not isinstance(paper, dict):
            continue
        for key in ("paperId", "sourceId", "canonicalId"):
            value = str(paper.get(key) or "").strip()
            if value and value not in seen:
                seen.add(value)
                selected.append(value)
                break
    return selected


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

    def suggest_grounded_expansions(
        self,
        *,
        query: str,
        papers: list[dict[str, Any]],
        max_variants: int,
    ) -> list[ExpansionCandidate]:
        return []

    async def asuggest_grounded_expansions(
        self,
        *,
        query: str,
        papers: list[dict[str, Any]],
        max_variants: int,
        request_outcomes: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> list[ExpansionCandidate]:
        del request_outcomes, request_id
        return self.suggest_grounded_expansions(
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

    async def arevise_search_strategy(
        self,
        *,
        original_query: str,
        original_intent: str,
        tried_providers: list[str],
        result_summary: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Recommend a revised retrieval strategy when the first pass failed.

        Returns dict with revisedQuery, revisedIntent, revisedProviders, rationale.
        Default fallback: prefer known_item for title/citation-like queries,
        otherwise review for regulatory and discovery for other intents.
        """
        # Late import to avoid circular dependency (planner → providers → provider_base).
        from .planner import looks_like_citation_query, looks_like_exact_title  # noqa: PLC0415

        if looks_like_exact_title(original_query) or looks_like_citation_query(original_query):
            return {
                "revisedQuery": original_query,
                "revisedIntent": "known_item",
                "revisedProviders": ["semantic_scholar"],
                "rationale": (
                    "Deterministic fallback: query looks like a paper title or citation;"
                    " retried with semantic known-item recovery."
                ),
            }
        return {
            "revisedQuery": original_query,
            "revisedIntent": "review" if original_intent == "regulatory" else "discovery",
            "revisedProviders": ["semantic_scholar", "openalex"],
            "rationale": "Deterministic fallback: retry as review/discovery.",
        }

    async def aclose(self) -> None:
        """Close any provider-specific resources."""

    async def aclassify_relevance_batch(
        self,
        *,
        query: str,
        papers: list[dict[str, Any]],
        request_id: str | None = None,
    ) -> dict[str, dict[str, str]]:
        """Return a mapping of paperId -> classification/rationale for the batch."""
        del query, request_id
        return {
            str(paper.get("paperId") or paper.get("paper_id") or ""): {
                "classification": "weak_match",
                "rationale": "Deterministic fallback did not run semantic relevance classification.",
            }
            for paper in papers
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
        del query, request_id
        verified_count = len(verified_sources)
        if verified_count >= 5 and not evidence_gaps:
            return {"adequacy": "succeeded", "reason": "Five or more verified sources covered the query."}
        if verified_count == 0:
            return {"adequacy": "insufficient", "reason": "No verified sources were available."}
        if intent in {"discovery", "review"} and verified_count >= 3:
            return {"adequacy": "partial", "reason": "Some verified evidence exists, but coverage is incomplete."}
        return {"adequacy": "partial", "reason": "Deterministic fallback could not promote adequacy beyond partial."}


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
        from .planner import (  # noqa: PLC0415
            _looks_broad_concept_query,
            detect_regulatory_intent,
            looks_like_citation_query,
            looks_like_exact_title,
            query_facets,
        )

        normalized = query.strip()
        lowered = normalized.lower()
        inferred_intent = mode if mode != "auto" else "discovery"
        query_type = "broad_concept"
        if any(marker in lowered for marker in ("doi", "arxiv:", "https://", "http://")):
            inferred_intent = "known_item"
            query_type = "known_item"
        elif detect_regulatory_intent(normalized, focus) and mode == "auto":
            inferred_intent = "regulatory"
            query_type = "regulatory"
        elif "author" in lowered and mode == "auto":
            inferred_intent = "author"
            query_type = "author"
        elif any(marker in lowered for marker in ("citation", "cites", "cited by")) and mode == "auto":
            inferred_intent = "citation"
            query_type = "citation_repair"
        elif looks_like_citation_query(normalized) and mode == "auto":
            inferred_intent = "known_item"
            query_type = "citation_repair"
        elif looks_like_exact_title(normalized) and mode == "auto":
            inferred_intent = "known_item"
            query_type = "known_item"
        elif any(marker in lowered for marker in ("survey", "review", "landscape")) and mode == "auto":
            inferred_intent = "review"
            query_type = "review"

        concept_text = " ".join(part for part in [normalized, focus or ""] if isinstance(part, str) and part)
        candidate_concepts = [
            concept
            for concept in [*query_facets(concept_text), *[token for token in _tokenize(concept_text)]]
            if concept not in COMMON_QUERY_WORDS
        ][:8]
        broad_concept_signal = _looks_broad_concept_query(
            normalized_query=normalized,
            focus=focus,
            year=year,
            venue=venue,
        )
        if inferred_intent in {"known_item", "author", "citation", "regulatory"}:
            query_specificity = "high"
            ambiguity_level = "low"
            breadth_estimate = 1
            first_pass_mode = RETRIEVAL_MODE_TARGETED
        elif broad_concept_signal:
            query_specificity = "low"
            ambiguity_level = "high"
            breadth_estimate = 4
            first_pass_mode = RETRIEVAL_MODE_BROAD
        elif inferred_intent == "review":
            query_specificity = "medium"
            ambiguity_level = "medium"
            breadth_estimate = 3
            first_pass_mode = RETRIEVAL_MODE_MIXED
        else:
            query_specificity = "medium"
            ambiguity_level = "medium"
            breadth_estimate = 2
            first_pass_mode = RETRIEVAL_MODE_MIXED

        search_angles: list[str] = []
        if breadth_estimate > 1:
            if len(candidate_concepts) >= 2:
                search_angles.append(" ".join(candidate_concepts[:2]))
            if candidate_concepts:
                search_angles.append(str(candidate_concepts[0]))
            if len(candidate_concepts) >= 3:
                search_angles.append(" ".join(candidate_concepts[1:3]))
        deduped_search_angles: list[str] = []
        seen_angles: set[str] = {normalized.lower()}
        for angle in search_angles:
            cleaned = str(angle).strip()
            lowered_angle = cleaned.lower()
            if not cleaned or lowered_angle in seen_angles:
                continue
            seen_angles.add(lowered_angle)
            deduped_search_angles.append(cleaned)

        uncertainty_flags: list[str] = []
        if broad_concept_signal:
            uncertainty_flags.append("broad_or_multi_factor_query")
        if year and inferred_intent == "discovery":
            uncertainty_flags.append("year_constraint_may_or_may_not_be_central")
        if len(candidate_concepts) >= 4:
            uncertainty_flags.append("multiple_competing_concepts")

        retrieval_hypotheses = list(deduped_search_angles) or ([normalized] if normalized else [])
        constraints = {key: value for key, value in {"year": year, "venue": venue, "focus": focus}.items() if value}
        return PlannerDecision(
            intent=inferred_intent,  # type: ignore[arg-type]
            querySpecificity=cast(Any, query_specificity),
            ambiguityLevel=cast(Any, ambiguity_level),
            queryType=cast(Any, query_type),
            breadthEstimate=breadth_estimate,
            searchAngles=deduped_search_angles,
            uncertaintyFlags=uncertainty_flags,
            firstPassMode=cast(Any, first_pass_mode),
            retrievalHypotheses=retrieval_hypotheses,
            intentRationale=(
                f"Deterministic planner routed the query as {inferred_intent} using lightweight local heuristics."
            ),
            constraints=constraints,
            seedIdentifiers=_extract_seed_identifiers(normalized),
            candidateConcepts=candidate_concepts,
            providerPlan=(
                ["ecos", "federal_register", "govinfo"]
                if inferred_intent == "regulatory"
                else ["semantic_scholar", "openalex", "scholarapi", "core", "arxiv"]
            ),
            followUpMode="claim_check" if inferred_intent == "review" else "qa",
        )

    def suggest_grounded_expansions(
        self,
        *,
        query: str,
        papers: list[dict[str, Any]],
        max_variants: int,
    ) -> list[ExpansionCandidate]:
        evidence_texts = [
            " ".join(part for part in [str(paper.get("title") or ""), str(paper.get("abstract") or "")] if part)
            for paper in papers[:8]
        ]
        variants: list[ExpansionCandidate] = []
        for term in _top_terms(evidence_texts, limit=max_variants * 2):
            if term in query.lower():
                continue
            variants.append(
                ExpansionCandidate(
                    variant=f"{query} {term}",
                    source="from_retrieved_evidence",
                    rationale=f"Deterministic grounded fallback expanded the query with evidence term '{term}'.",
                )
            )
            if len(variants) >= max_variants:
                break
        return variants

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
                "answerability": "insufficient",
                "selectedEvidenceIds": [],
                "selectedLeadIds": [],
                "confidence": "low",
            }

        selected_evidence_ids = _default_selected_evidence_ids(evidence_papers)
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
                "answerability": ("grounded" if status == "supported" else "limited"),
                "selectedEvidenceIds": selected_evidence_ids[:2],
                "selectedLeadIds": [],
                "confidence": "medium" if status == "supported" else "low",
            }
        if answer_mode == "comparison":
            return {
                "answer": _deterministic_comparison_answer(question, evidence_papers),
                "unsupportedAsks": [],
                "followUpQuestions": ["Which comparison dimension matters most: method, data, or recency?"],
                "answerability": "grounded",
                "selectedEvidenceIds": selected_evidence_ids,
                "selectedLeadIds": [],
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
                "answerability": ("grounded" if insights else "limited"),
                "selectedEvidenceIds": selected_evidence_ids,
                "selectedLeadIds": [],
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
            "answerability": "grounded",
            "selectedEvidenceIds": selected_evidence_ids,
            "selectedLeadIds": [],
            "confidence": "medium",
        }
