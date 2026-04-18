"""P1-2: tests for top_recommendation on comparative/selection follow-ups."""

from __future__ import annotations

from typing import Any, Literal, cast

import pytest

from paper_chaser_mcp.agentic.answer_modes import (
    build_evidence_use_plan,
    selection_anchor_candidate_ids,
)
from paper_chaser_mcp.agentic.models import EvidenceItem, StructuredSourceRecord
from paper_chaser_mcp.agentic.selection_scoring import (
    infer_comparative_axis,
    score_papers_for_comparative_axis,
)
from paper_chaser_mcp.models.common import Paper
from tests.test_smart_tools import (  # type: ignore[import-untyped]
    RecordingOpenAlexClient,
    RecordingSemanticClient,
    _deterministic_runtime,
)

# ---------------------------------------------------------------------------
# infer_comparative_axis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Which paper is the most recent on PFAS?", "recency"),
        ("Which is the most authoritative source?", "authority"),
        ("Which is the most cited paper?", "authority"),
        ("Which is the most beginner-friendly paper?", "beginner"),
        ("What is the best starting point for this topic?", "beginner"),
        ("Which source should I start with first?", "beginner"),
        ("Which source should I read first?", "beginner"),
        ("Which paper is the most comprehensive review?", "coverage"),
    ],
)
def test_infer_comparative_axis_routes_known_markers(question: str, expected: str) -> None:
    assert infer_comparative_axis(question) == expected


def test_infer_comparative_axis_defaults_to_relevance_fallback() -> None:
    assert infer_comparative_axis("Which one should I read?") == "relevance_fallback"


# ---------------------------------------------------------------------------
# score_papers_for_comparative_axis
# ---------------------------------------------------------------------------


def _paper(
    paper_id: str,
    *,
    title: str = "",
    abstract: str = "",
    year: int | None = None,
    citations: int | None = None,
) -> Paper:
    return Paper(
        paperId=paper_id,
        title=title or f"Paper {paper_id}",
        abstract=abstract,
        year=year,
        citationCount=citations,
    )


def test_scoring_authority_favors_higher_citation_count() -> None:
    papers = [
        _paper("p1", citations=10),
        _paper("p2", citations=1000),
        _paper("p3", citations=None),
    ]
    scores = score_papers_for_comparative_axis(papers, "most cited", "authority")
    assert scores["p2"] > scores["p1"] > 0.0
    assert scores["p3"] == 0.0


def test_scoring_recency_favors_newer_year() -> None:
    papers = [
        _paper("old", year=2010),
        _paper("new", year=2024),
        _paper("unknown", year=None),
    ]
    scores = score_papers_for_comparative_axis(papers, "most recent", "recency")
    assert scores["new"] == 1.0
    assert scores["old"] == 0.0
    assert scores["unknown"] == 0.0


def test_scoring_beginner_prefers_survey_titles() -> None:
    papers = [
        _paper("survey", title="A Survey of Retrieval-Augmented Agents"),
        _paper("primary", title="Formal Analysis of Token-Level Attention"),
        _paper("review-abstract", title="Novel Method", abstract="We review recent advances."),
    ]
    scores = score_papers_for_comparative_axis(papers, "beginner-friendly", "beginner")
    assert scores["survey"] == 1.0
    assert scores["review-abstract"] > scores["primary"]


def test_scoring_coverage_counts_keyword_overlap() -> None:
    papers = [
        _paper(
            "broad",
            title="PFAS removal via adsorption and membranes",
            abstract="We cover adsorption, membranes, oxidation methods.",
        ),
        _paper(
            "narrow",
            title="Signal processing in radar systems",
            abstract="Radar signal theory.",
        ),
    ]
    scores = score_papers_for_comparative_axis(papers, "PFAS removal adsorption membranes", "coverage")
    assert scores["broad"] > scores["narrow"]


def test_scoring_relevance_fallback_uses_supplied_scores() -> None:
    papers = [_paper("a"), _paper("b")]
    scores = score_papers_for_comparative_axis(
        papers,
        "which one should I read",
        "relevance_fallback",
        relevance_scores={"a": 0.9, "b": 0.2},
    )
    assert scores == {"a": 0.9, "b": 0.2}


# ---------------------------------------------------------------------------
# ask_result_set integration
# ---------------------------------------------------------------------------


def test_selection_mode_allows_unique_current_text_anchor() -> None:
    evidence = [
        EvidenceItem(
            evidenceId="50 CFR 17.95",
            paper=Paper(
                paperId="50 CFR 17.95",
                title="50 CFR 17.95 — California condor critical habitat",
            ),
            excerpt="Current codified critical habitat text.",
            whyRelevant="",
            relevanceScore=0.9,
        )
    ]
    source_records = [
        StructuredSourceRecord(
            sourceId="50 CFR 17.95",
            title="50 CFR 17.95 — California condor critical habitat",
            sourceType="primary_regulatory",
            verificationStatus="verified_primary_source",
            accessStatus="url_verified",
            isPrimarySource=True,
            topicalRelevance=cast(Literal["on_topic", "weak_match", "off_topic"], "on_topic"),
            citationText="50 CFR 17.95",
        )
    ]
    question = "Which returned source should I start with for the current codified habitat text under 50 CFR 17.95?"

    assert selection_anchor_candidate_ids(question, evidence, source_records) == ["50 CFR 17.95"]
    plan = build_evidence_use_plan(
        question=question,
        answer_mode="comparison",
        evidence=evidence,
        source_records=source_records,
        unsupported_asks=[],
        llm_relevance={"50 CFR 17.95": {"classification": "on_topic", "fallback": False}},
        question_mode="selection",
    )

    assert plan["sufficient"] is True
    assert plan["anchoredSelectionSourceIds"] == ["50 CFR 17.95"]
    assert "exact strong source" in plan["rationale"]


def test_selection_mode_keeps_ambiguous_current_text_anchor_insufficient() -> None:
    evidence = [
        EvidenceItem(
            evidenceId="50 CFR 17.95",
            paper=Paper(paperId="50 CFR 17.95", title="50 CFR 17.95 — California condor critical habitat"),
            excerpt="Current codified text.",
            whyRelevant="",
            relevanceScore=0.9,
        ),
        EvidenceItem(
            evidenceId="50 CFR 17.95-app",
            paper=Paper(paperId="50 CFR 17.95-app", title="Appendix text for 50 CFR 17.95"),
            excerpt="Supplementary codified text.",
            whyRelevant="",
            relevanceScore=0.1,
        ),
    ]
    source_records = [
        StructuredSourceRecord(
            sourceId="50 CFR 17.95",
            title="50 CFR 17.95 — California condor critical habitat",
            sourceType="primary_regulatory",
            verificationStatus="verified_primary_source",
            accessStatus="url_verified",
            isPrimarySource=True,
            topicalRelevance=cast(Literal["on_topic", "weak_match", "off_topic"], "on_topic"),
            citationText="50 CFR 17.95",
        ),
        StructuredSourceRecord(
            sourceId="50 CFR 17.95-app",
            title="Appendix text for 50 CFR 17.95",
            sourceType="primary_regulatory",
            verificationStatus="verified_primary_source",
            accessStatus="url_verified",
            isPrimarySource=True,
            topicalRelevance=cast(Literal["on_topic", "weak_match", "off_topic"], "on_topic"),
            citationText="50 CFR 17.95",
        ),
    ]
    question = "Which returned source should I start with for the current codified habitat text under 50 CFR 17.95?"

    plan = build_evidence_use_plan(
        question=question,
        answer_mode="comparison",
        evidence=evidence,
        source_records=source_records,
        unsupported_asks=[],
        llm_relevance={},
        question_mode="selection",
    )

    assert plan["sufficient"] is False
    assert plan["anchoredSelectionSourceIds"] == ["50 CFR 17.95", "50 CFR 17.95-app"]


def _grounded_answer_stub(selected: list[str]) -> Any:
    async def _answer_question(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "answer": (
                "The saved papers each cover retrieval-augmented agents, with varying emphasis on "
                "evidence grounding, citation tracing, and tool-use delegation patterns."
            ),
            "unsupportedAsks": [],
            "followUpQuestions": [],
            "confidence": "medium",
            "answerability": "grounded",
            "selectedEvidenceIds": selected,
            "selectedLeadIds": [],
            "citedPaperIds": selected,
            "evidenceSummary": "Two strong on-topic sources.",
            "missingEvidenceDescription": "",
        }

    return _answer_question


async def _high_similarity(query: str, texts: list[str], **kwargs: object) -> list[float]:
    del query, kwargs
    return [0.82 for _ in texts]


def _attach_stubs(runtime: Any, papers_ids: list[str]) -> None:
    answer_stub = _grounded_answer_stub(papers_ids)
    setattr(runtime._provider_bundle, "aanswer_question", answer_stub)
    setattr(runtime._deterministic_bundle, "aanswer_question", answer_stub)
    setattr(runtime._provider_bundle, "abatched_similarity", _high_similarity)
    setattr(runtime._deterministic_bundle, "abatched_similarity", _high_similarity)


@pytest.mark.asyncio
async def test_ask_result_set_populates_top_recommendation_for_selection_mode() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    registry, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)

    record = registry.save_result_set(
        source_tool="search_papers_smart",
        query="retrieval-augmented agents",
        payload={
            "data": [
                {
                    "paperId": "paper-old",
                    "title": "Retrieval-Augmented Agents: An Early Study",
                    "abstract": "Agents combine retrieval with generation to ground answers.",
                    "source": "semantic_scholar",
                    "year": 2019,
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "full_text_verified",
                },
                {
                    "paperId": "paper-new",
                    "title": "Retrieval-Augmented Agents: Recent Advances",
                    "abstract": "Agents combine retrieval with generation to ground answers.",
                    "source": "semantic_scholar",
                    "year": 2024,
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "full_text_verified",
                },
            ]
        },
    )
    _attach_stubs(runtime, ["paper-old", "paper-new"])

    ask = await runtime.ask_result_set(
        search_session_id=record.search_session_id,
        question="Which is the most recent paper on retrieval-augmented agents?",
        top_k=2,
        answer_mode="comparison",
    )

    assert ask["answerStatus"] == "answered"
    top = ask["topRecommendation"]
    assert top is not None
    assert top["sourceId"] == "paper-new"
    assert top["comparativeAxis"] == "recency"
    assert top["recommendationReason"]
    assert top["axisScore"] > 0.0
    assert isinstance(top["alternativeRecommendations"], list)
    assert top["axis"] == "recency"
    assert top["rationale"] == top["recommendationReason"]


@pytest.mark.parametrize(
    "question",
    [
        "Which source should I start with first?",
        "Which source should I read first?",
    ],
)
@pytest.mark.asyncio
async def test_ask_result_set_populates_beginner_recommendation_for_start_phrasing(question: str) -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    registry, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)

    record = registry.save_result_set(
        source_tool="search_papers_smart",
        query="retrieval-augmented agents",
        payload={
            "data": [
                {
                    "paperId": "survey",
                    "title": "A Survey of Retrieval-Augmented Agents",
                    "abstract": "A review of retrieval-grounded agent architectures.",
                    "source": "semantic_scholar",
                    "year": 2024,
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "full_text_verified",
                },
                {
                    "paperId": "technical",
                    "title": "Dense Retrieval Calibration for Agent Planning",
                    "abstract": "Technical treatment of retrieval calibration.",
                    "source": "semantic_scholar",
                    "year": 2023,
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "full_text_verified",
                },
            ]
        },
    )
    _attach_stubs(runtime, ["survey", "technical"])

    ask = await runtime.ask_result_set(
        search_session_id=record.search_session_id,
        question=question,
        top_k=2,
        answer_mode="comparison",
    )

    assert ask["answerStatus"] == "answered"
    top = ask["topRecommendation"]
    assert top is not None
    assert top["sourceId"] == "survey"
    assert top["comparativeAxis"] == "beginner_friendly"
    assert "beginner-friendly" in top["recommendationReason"].lower()


@pytest.mark.asyncio
async def test_ask_result_set_populates_regulatory_start_recommendation() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    registry, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)

    record = registry.save_result_set(
        source_tool="search_papers_smart",
        query="california condor recovery rulemaking",
        payload={
            "data": [
                {
                    "paperId": "condor-overview",
                    "title": "Overview of California Condor Recovery Actions",
                    "abstract": "Introductory overview of recovery and rulemaking milestones.",
                    "source": "federal_register",
                    "sourceType": "primary_regulatory",
                    "year": 2024,
                    "verificationStatus": "verified_primary_source",
                    "accessStatus": "full_text_verified",
                    "isPrimarySource": True,
                },
                {
                    "paperId": "condor-rule",
                    "title": "Final Rule Revising Critical Habitat for California Condor",
                    "abstract": "Detailed final rule text for the California condor.",
                    "source": "federal_register",
                    "sourceType": "primary_regulatory",
                    "year": 2023,
                    "verificationStatus": "verified_primary_source",
                    "accessStatus": "full_text_verified",
                    "isPrimarySource": True,
                },
            ],
            "strategyMetadata": {"intent": "regulatory"},
        },
    )
    _attach_stubs(runtime, ["condor-overview", "condor-rule"])

    ask = await runtime.ask_result_set(
        search_session_id=record.search_session_id,
        question="Which source should I start with first?",
        top_k=2,
        answer_mode="comparison",
    )

    assert ask["answerStatus"] == "answered"
    top = ask["topRecommendation"]
    assert top is not None
    assert top["sourceId"] == "condor-overview"
    assert top["comparativeAxis"] == "beginner_friendly"
    assert top["recommendationReason"]


@pytest.mark.asyncio
async def test_ask_result_set_answers_unique_current_text_selection_from_top_recommendation() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    registry, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)

    record = registry.save_result_set(
        source_tool="search_papers_smart",
        query="california condor critical habitat",
        payload={
            "data": [
                {
                    "paperId": "50 CFR 17.95",
                    "title": "50 CFR 17.95 — California condor critical habitat",
                    "abstract": "",
                    "source": "govinfo",
                    "sourceType": "primary_regulatory",
                    "verificationStatus": "verified_primary_source",
                    "accessStatus": "url_verified",
                    "isPrimarySource": True,
                }
            ]
        },
    )

    async def _selection_stub(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "answer": "",
            "unsupportedAsks": [],
            "followUpQuestions": [],
            "confidence": "medium",
            "answerability": "limited",
            "selectedEvidenceIds": [],
            "selectedLeadIds": [],
            "citedPaperIds": [],
            "evidenceSummary": "One exact primary source.",
            "missingEvidenceDescription": "",
        }

    setattr(runtime._provider_bundle, "aanswer_question", _selection_stub)
    setattr(runtime._deterministic_bundle, "aanswer_question", _selection_stub)
    setattr(runtime._provider_bundle, "abatched_similarity", _high_similarity)
    setattr(runtime._deterministic_bundle, "abatched_similarity", _high_similarity)

    ask = await runtime.ask_result_set(
        search_session_id=record.search_session_id,
        question="Which returned source should I start with for the current codified habitat text under 50 CFR 17.95?",
        top_k=1,
        answer_mode="comparison",
    )

    assert ask["answerStatus"] == "answered"
    assert ask["answerability"] == "limited"
    assert ask["selectedEvidenceIds"] == ["50 CFR 17.95"]
    assert "Start with" in ask["answer"]
    top = ask["topRecommendation"]
    assert top is not None
    assert top["sourceId"] == "50 CFR 17.95"
    assert top["comparativeAxis"] == "authority"
    assert "exact verified primary source" in top["recommendationReason"]


@pytest.mark.asyncio
async def test_ask_result_set_omits_top_recommendation_for_non_selection_modes() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    registry, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)

    record = registry.save_result_set(
        source_tool="search_papers_smart",
        payload={
            "data": [
                {
                    "paperId": "p1",
                    "title": "Retrieval-Augmented Agents",
                    "abstract": "Topic text.",
                    "source": "semantic_scholar",
                    "year": 2024,
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "full_text_verified",
                },
                {
                    "paperId": "p2",
                    "title": "Retrieval-Augmented Agents In Practice",
                    "abstract": "Topic text.",
                    "source": "semantic_scholar",
                    "year": 2023,
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "full_text_verified",
                },
            ]
        },
    )
    _attach_stubs(runtime, ["p1", "p2"])

    ask = await runtime.ask_result_set(
        search_session_id=record.search_session_id,
        question="What do these papers say about retrieval grounding?",
        top_k=2,
        answer_mode="synthesis",
    )

    assert ask["topRecommendation"] is None


@pytest.mark.asyncio
async def test_ask_result_set_omits_top_recommendation_when_evidence_is_weak() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    registry, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)

    # Only one verified paper -> strong_evidence_ids has < 2 items, the gate
    # must refuse to recommend.
    record = registry.save_result_set(
        source_tool="search_papers_smart",
        payload={
            "data": [
                {
                    "paperId": "only",
                    "title": "Lone verified source",
                    "abstract": "Some abstract.",
                    "source": "semantic_scholar",
                    "year": 2022,
                    "verificationStatus": "verified_metadata",
                    "accessStatus": "full_text_verified",
                }
            ]
        },
    )
    _attach_stubs(runtime, ["only"])

    ask = await runtime.ask_result_set(
        search_session_id=record.search_session_id,
        question="Which is the most recent paper?",
        top_k=2,
        answer_mode="comparison",
    )

    assert ask["topRecommendation"] is None


def test_top_recommendation_field_exists_on_response_model() -> None:
    from paper_chaser_mcp.agentic.models import AskResultSetResponse

    fields = AskResultSetResponse.model_fields
    assert "top_recommendation" in fields
    info = fields["top_recommendation"]
    assert info.alias == "topRecommendation"


# ---------------------------------------------------------------------------
# Non-regression: selection mode + insufficient evidence still abstains.
# ---------------------------------------------------------------------------


def test_selection_mode_insufficient_evidence_still_insufficient() -> None:
    evidence = [
        EvidenceItem(
            evidenceId="only",
            paper=Paper(paperId="only", title="Lone paper"),
            excerpt="",
            whyRelevant="",
            relevanceScore=0.4,
        )
    ]
    source_records = [
        StructuredSourceRecord(
            sourceId="only",
            topicalRelevance=cast(Literal["on_topic", "weak_match", "off_topic"], "on_topic"),
        )
    ]

    plan = build_evidence_use_plan(
        question="Which is the most recent paper here?",
        answer_mode="comparison",
        evidence=evidence,
        source_records=source_records,
        unsupported_asks=[],
        llm_relevance={"only": {"classification": "on_topic", "fallback": False}},
        question_mode="selection",
    )
    # With only one on-topic paper, a selection synthesis must not be
    # reported as sufficient.
    assert plan["sufficient"] is False
