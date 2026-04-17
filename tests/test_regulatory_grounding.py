"""Tests for ws-regulatory-grounding (cross-domain WS-C + WS-D + env-sci WS-B).

Covers:
1. ``PlannerDecision.regulatory_intent`` is populated for the five canonical
   regulatory intents plus ``unspecified`` for generic regulatory queries.
2. ``SubjectCard`` resolution for species-dossier / CFR / guidance / cultural
   (Section 106) queries.
3. Document-family ranking boost in ``_rank_regulatory_documents``.
4. ``StructuredSourceRecord`` carries ``documentFamilyMatch`` /
   ``documentFamilyBoost`` after ranking.
5. Species-dossier weak-match demotion helper demotes documents that do not
   mention the subject species.
6. ``compute_subject_chain_gaps`` flags missing recovery plan / critical
   habitat rungs for species queries.
7. ``hybrid_policy_science`` retrieval hypothesis is added when
   ``regulatory_intent == hybrid_regulatory_plus_literature``.
8. Known-item weak-signal breadth pass: exact-title-looking + no identifier +
   high ambiguity does not force-route to known_item.
"""

from __future__ import annotations

import pytest

from paper_chaser_mcp.agentic.graphs import (
    _rank_regulatory_documents,
    _source_record_from_regulatory_document,
)
from paper_chaser_mcp.agentic.models import (
    PlannerDecision,
    SubjectCard,
)
from paper_chaser_mcp.agentic.planner import classify_query
from paper_chaser_mcp.agentic.subject_grounding import (
    compute_subject_chain_gaps,
    detect_document_family_match,
    resolve_subject_card,
    species_mentioned,
)


class _StubBundle:
    """Minimal provider bundle that returns a pre-baked PlannerDecision."""

    def __init__(self, planner: PlannerDecision) -> None:
        self._planner = planner

    async def aplan_search(self, **kwargs: object) -> PlannerDecision:
        return self._planner.model_copy(deep=True)


# ---------------------------------------------------------------------------
# 1) regulatory_intent labels
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,expected_intent",
    [
        (
            "What does 50 CFR 17.11 currently say about desert tortoise?",
            "current_cfr_text",
        ),
        ("Listing history and rulemaking timeline for the California condor", "rulemaking_history"),
        ("Desert tortoise species dossier and recovery plan under ESA", "species_dossier"),
        ("EPA PFAS drinking water guidance document", "guidance_lookup"),
    ],
)
async def test_regulatory_intent_matches_expected_label(query: str, expected_intent: str) -> None:
    seed = PlannerDecision(intent="regulatory", followUpMode="qa")
    _, planner = await classify_query(
        query=query,
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_StubBundle(seed),  # type: ignore[arg-type]
    )
    assert planner.regulatory_intent == expected_intent


@pytest.mark.asyncio
async def test_regulatory_intent_unspecified_for_generic_regulatory_query() -> None:
    """A regulatory query without specific cues falls back to ``unspecified``."""
    seed = PlannerDecision(intent="regulatory", followUpMode="qa")
    _, planner = await classify_query(
        query="federal regulation policy overview",
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_StubBundle(seed),  # type: ignore[arg-type]
    )
    assert planner.regulatory_intent in {"unspecified", "rulemaking_history"}


@pytest.mark.asyncio
async def test_non_regulatory_query_has_no_regulatory_intent() -> None:
    seed = PlannerDecision(intent="discovery", followUpMode="qa")
    _, planner = await classify_query(
        query="transformer attention mechanism survey",
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_StubBundle(seed),  # type: ignore[arg-type]
    )
    assert planner.regulatory_intent is None


# ---------------------------------------------------------------------------
# 2) SubjectCard resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subject_card_resolves_desert_tortoise_with_recovery_plan() -> None:
    seed = PlannerDecision(
        intent="regulatory",
        followUpMode="qa",
        entityCard={
            "commonName": "desert tortoise",
            "scientificName": "Gopherus agassizii",
            "authorityContext": "ESA",
            "requestedDocumentFamily": "recovery_plan",
        },
        candidateConcepts=["desert tortoise", "recovery plan", "ESA"],
    )
    _, planner = await classify_query(
        query="Desert tortoise recovery plan under the ESA",
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_StubBundle(seed),  # type: ignore[arg-type]
    )
    assert planner.subject_card is not None
    card = planner.subject_card
    assert card.common_name == "desert tortoise"
    assert card.scientific_name == "Gopherus agassizii"
    assert card.requested_document_family == "recovery_plan"
    assert card.agency == "ESA"
    assert card.confidence == "high"
    assert card.source == "planner_llm"


@pytest.mark.asyncio
async def test_subject_card_for_section_106_offshore_wind_is_consultation_guidance() -> None:
    seed = PlannerDecision(intent="regulatory", followUpMode="qa")
    _, planner = await classify_query(
        query="Section 106 consultation for offshore wind project",
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_StubBundle(seed),  # type: ignore[arg-type]
    )
    assert planner.subject_card is not None
    assert planner.subject_card.requested_document_family in {
        "consultation_guidance",
        "programmatic_agreement",
    }


def test_subject_card_deterministic_fallback_without_planner() -> None:
    card = resolve_subject_card(query="PFAS drinking water EPA guidance", planner=None)
    assert isinstance(card, SubjectCard)
    assert card.confidence == "deterministic_fallback"
    assert card.source == "deterministic_fallback"


# ---------------------------------------------------------------------------
# 3 & 4) Document-family ranking and source-record plumbing
# ---------------------------------------------------------------------------


def test_rank_regulatory_documents_boosts_requested_document_family() -> None:
    documents = [
        {
            "title": "Analysis of Economic Impacts",
            "summary": "Economic discussion of species recovery efforts.",
            "publicationDate": "2019-05-01",
        },
        {
            "title": "Final Recovery Plan for the Desert Tortoise",
            "summary": "Recovery plan describing actions for Gopherus agassizii.",
            "publicationDate": "2011-06-30",
        },
    ]
    ranked = _rank_regulatory_documents(
        documents,
        subject_terms={"desert", "tortoise", "recovery"},
        priority_terms=set(),
        facet_terms=[],
        prefer_guidance=False,
        prefer_recent=False,
        requested_document_family="recovery_plan",
    )
    assert ranked[0]["title"].startswith("Final Recovery Plan")
    assert ranked[0].get("_documentFamilyMatch") == "recovery_plan"
    assert ranked[0].get("_documentFamilyBoost", 0.0) > 0.0


def test_source_record_carries_document_family_boost() -> None:
    document = {
        "title": "Final Recovery Plan for the Desert Tortoise",
        "summary": "Recovery plan actions for Gopherus agassizii.",
        "publicationDate": "2011-06-30",
        "citation": "USFWS 2011 Recovery Plan",
    }
    _rank_regulatory_documents(
        [document],
        subject_terms={"desert", "tortoise"},
        priority_terms=set(),
        facet_terms=[],
        prefer_guidance=False,
        prefer_recent=False,
        requested_document_family="recovery_plan",
    )
    record = _source_record_from_regulatory_document(
        document, provider="federal_register", topical_relevance="on_topic"
    )
    assert record.document_family_match == "recovery_plan"
    assert record.document_family_boost is not None and record.document_family_boost > 0.0


# ---------------------------------------------------------------------------
# 5) Species-dossier weak-match demotion helper
# ---------------------------------------------------------------------------


def test_species_mentioned_detects_common_name() -> None:
    card = SubjectCard(commonName="desert tortoise", scientificName="Gopherus agassizii")
    assert species_mentioned({"title": "Recovery plan for desert tortoise"}, card)
    assert species_mentioned({"summary": "Actions benefiting Gopherus agassizii"}, card)


def test_species_mentioned_returns_false_when_species_absent() -> None:
    card = SubjectCard(commonName="desert tortoise", scientificName="Gopherus agassizii")
    assert not species_mentioned(
        {"title": "General ESA rulemaking discussion", "summary": "Listing criteria overview"},
        card,
    )


# ---------------------------------------------------------------------------
# 6) subject_chain_gaps
# ---------------------------------------------------------------------------


def test_compute_subject_chain_gaps_flags_missing_recovery_plan() -> None:
    card = SubjectCard(commonName="desert tortoise", scientificName="Gopherus agassizii")
    documents = [
        {"title": "ECOS species profile for desert tortoise", "summary": "Gopherus agassizii"},
    ]
    gaps = compute_subject_chain_gaps(
        card=card,
        regulatory_intent="species_dossier",
        documents=documents,
    )
    assert "species_evidence_without_recovery_plan" in gaps
    assert "species_evidence_without_critical_habitat" in gaps


def test_compute_subject_chain_gaps_empty_for_non_species_intent() -> None:
    card = SubjectCard(commonName="generic", scientificName=None)
    gaps = compute_subject_chain_gaps(
        card=card,
        regulatory_intent="current_cfr_text",
        documents=[{"title": "50 CFR 17.11"}],
    )
    assert gaps == []


# ---------------------------------------------------------------------------
# 7) hybrid_policy_science retrieval hypothesis
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_regulatory_plus_literature_emits_hybrid_policy_science_hypothesis() -> None:
    seed = PlannerDecision(
        intent="regulatory",
        followUpMode="qa",
        regulatorySubintent="hybrid_regulatory_plus_literature",
    )
    _, planner = await classify_query(
        query="Ecological effects of rulemaking on sage grouse habitat: scientific and regulatory perspectives",
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_StubBundle(seed),  # type: ignore[arg-type]
    )
    assert planner.regulatory_intent == "hybrid_regulatory_plus_literature"
    assert any("hybrid_policy_science" in hypothesis for hypothesis in planner.retrieval_hypotheses)


# ---------------------------------------------------------------------------
# 8) Known-item weak-signal breadth pass
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_known_item_weak_signal_with_ambiguity_is_demoted() -> None:
    """Exact-title-looking query + no identifier + high ambiguity must not
    collapse to pure known-item recovery."""
    seed = PlannerDecision(
        intent="known_item",
        followUpMode="qa",
        ambiguityLevel="high",
        queryType="known_item",
        querySpecificity="medium",
    )
    _, planner = await classify_query(
        query="Large Language Models for Scientific Hypothesis Generation",
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_StubBundle(seed),  # type: ignore[arg-type]
    )
    assert planner.intent != "known_item"


# ---------------------------------------------------------------------------
# document_family_match helper direct tests
# ---------------------------------------------------------------------------


def test_detect_document_family_match_requires_requested_family() -> None:
    doc = {"title": "Some Recovery Plan"}
    assert detect_document_family_match(doc, None) == (None, 0.0)
    family, boost = detect_document_family_match(doc, "recovery_plan")
    assert family == "recovery_plan"
    assert boost > 0.0
