"""Base provider bundles for the additive smart research layer."""

from __future__ import annotations

import re
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
    AnswerStatusValidation,
    _compact_theme_label,
    _deterministic_comparison_answer,
    _deterministic_gap_insights,
    _deterministic_theme_summary,
    _extract_seed_identifiers,
    _lexical_similarity,
    _normalize_confidence_label,
    _tokenize,
    _top_terms,
    generate_evidence_gaps_without_llm,
)

__all__ = [
    "DeterministicProviderBundle",
    "ModelProviderBundle",
]

_FALLBACK_RELEVANCE_STOPWORDS = COMMON_QUERY_WORDS | {
    "current",
    "evidence",
    "federal",
    "history",
    "latest",
    "literature",
    "peer",
    "recent",
    "regulatory",
    "reports",
    "review",
    "scientific",
    "scholarship",
    "studies",
    "study",
    "systematic",
}


def _fallback_query_terms(query: str) -> list[str]:
    return [token for token in _tokenize(query) if token not in _FALLBACK_RELEVANCE_STOPWORDS and len(token) >= 4]


def _fallback_query_facets(query: str) -> list[list[str]]:
    lowered = str(query or "").lower()
    segments = re.split(r"\b(?:for|in|on|about|into|within|across|via|through|regarding|around|under)\b", lowered)
    facets: list[list[str]] = []
    for segment in segments:
        tokens = [
            token for token in _tokenize(segment) if token not in _FALLBACK_RELEVANCE_STOPWORDS and len(token) >= 4
        ]
        if len(tokens) >= 2:
            facets.append(tokens[:3])
    return facets[:3]


def classify_relevance_without_llm(
    *,
    query: str,
    paper: dict[str, Any],
) -> dict[str, Any]:
    title = str(paper.get("title") or "")
    abstract = str(paper.get("abstract") or "")
    paper_text = " ".join(part for part in [title, abstract] if part)
    similarity = _lexical_similarity(query, paper_text)
    query_terms = _fallback_query_terms(query)
    title_tokens = set(_tokenize(title))
    body_tokens = set(_tokenize(paper_text))
    title_hits = [term for term in query_terms if term in title_tokens]
    body_hits = [term for term in query_terms if term in body_tokens]
    facets = _fallback_query_facets(query)

    facet_title_hits = 0
    facet_body_hits = 0
    for facet in facets:
        required = len(facet) if len(facet) <= 2 else 2
        if sum(token in title_tokens for token in facet) >= required:
            facet_title_hits += 1
        if sum(token in body_tokens for token in facet) >= required:
            facet_body_hits += 1

    title_anchor_coverage = (len(title_hits) / len(query_terms)) if query_terms else 0.0
    title_facet_coverage = (facet_title_hits / len(facets)) if facets else 0.0
    body_facet_coverage = (facet_body_hits / len(facets)) if facets else 0.0

    if (title_facet_coverage > 0.0 or title_anchor_coverage >= 0.34) and similarity >= 0.2:
        classification = "on_topic"
        rationale = "Deterministic fallback found strong title or facet overlap with the query."
    elif similarity < 0.08 or (not body_hits and body_facet_coverage == 0.0):
        classification = "off_topic"
        rationale = "Deterministic fallback found little direct overlap beyond generic research language."
    else:
        classification = "weak_match"
        rationale = (
            "Deterministic fallback found partial semantic overlap without enough direct title or facet support."
        )

    return {
        "classification": classification,
        "rationale": rationale,
        "fallback": True,
        "provenance": "deterministic_fallback",
    }


def relevance_paper_identifier(paper: dict[str, Any], index: int | None = None) -> str:
    return str(
        paper.get("paperId")
        or paper.get("paper_id")
        or paper.get("canonicalId")
        or paper.get("sourceId")
        or (f"paper-{index}" if index is not None else "")
    ).strip()


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

    def is_available(self) -> bool:
        return True

    @property
    def is_deterministic(self) -> bool:
        """Whether this bundle is the deterministic / offline shim.

        Concrete LLM-backed bundles keep the default (``False``). Only
        :class:`DeterministicProviderBundle` overrides this to ``True`` so that
        provenance consumers (e.g. :func:`subject_grounding.resolve_subject_card`)
        can stamp ``source="deterministic_fallback"`` correctly even though the
        deterministic shim still fills the same planner contract.
        """
        return False

    def configured_provider_name(self) -> str:
        return str(getattr(self, "_configured_provider", getattr(self, "_provider_name", "deterministic")))

    def active_provider_name(self) -> str:
        if not self.is_available():
            return "deterministic"
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
    ) -> dict[str, dict[str, Any]]:
        """Return a mapping of paperId -> classification/rationale for the batch."""
        del request_id
        from .relevance_fallback import classify_batch_deterministic

        return classify_batch_deterministic(
            query=query,
            papers=papers,
            reason="deterministic_provider",
        )

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

    async def avalidate_answer_status(
        self,
        *,
        query: str,
        answer_text: str,
        evidence_count: int,
        request_id: str | None = None,
    ) -> AnswerStatusValidation | None:
        """Validate whether an answer is substantive using LLM judgment.

        Returns None when no model is available (deterministic bundles).
        """
        del query, answer_text, evidence_count, request_id
        return None

    async def aclassify_answer_mode(
        self,
        *,
        question: str,
        modes: tuple[str, ...],
        request_id: str | None = None,
    ) -> str | None:
        """Classify a follow-up ``question`` into one of ``modes``.

        Returns ``None`` when this bundle has no LLM capability to route the
        classification (e.g. the deterministic shim), in which case callers
        should fall back to :func:`answer_modes._classify_question_mode_keyword`.
        Concrete LLM-backed bundles override this to perform a lightweight
        structured call. The wrapper returns ``None`` rather than raising so
        that classification hiccups never block the rest of ``ask_result_set``.
        """
        del question, modes, request_id
        return None

    async def agenerate_evidence_gaps(
        self,
        *,
        query: str,
        intent: str,
        sources: list[dict[str, Any]],
        evidence_gaps: list[str],
        retrieval_hypotheses: list[str],
        coverage_summary: dict[str, Any] | None,
        timeline: dict[str, Any] | None,
        anchor_type: str | None,
        request_id: str | None = None,
    ) -> list[str]:
        del request_id
        return generate_evidence_gaps_without_llm(
            query=query,
            intent=intent,
            sources=sources,
            evidence_gaps=evidence_gaps,
            retrieval_hypotheses=retrieval_hypotheses,
            coverage_summary=coverage_summary,
            timeline=timeline,
            anchor_type=anchor_type,
        )


class DeterministicProviderBundle(ModelProviderBundle):
    """Pure-Python fallback planner/synthesizer for tests and offline use."""

    @property
    def is_deterministic(self) -> bool:
        return True

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
            plannerSource="deterministic",
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
        sufficient_default = len(evidence_papers) >= 2
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
            "answerability": "grounded" if sufficient_default else "limited",
            "selectedEvidenceIds": selected_evidence_ids if sufficient_default else [],
            "selectedLeadIds": [],
            "confidence": "medium" if sufficient_default else "low",
        }
