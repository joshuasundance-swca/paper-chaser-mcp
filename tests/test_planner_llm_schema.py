"""Tests for ws-planner-llm-schema: native planner-LLM emission of
``regulatoryIntent`` and ``subjectCard`` with a deterministic fallback path.

The planner's LLM JSON schema now exposes two optional additive fields
(``regulatoryIntent`` + ``subjectCard``). When the model populates them, the
post-hoc derivation in :func:`classify_query` must respect those values and
stop running its deterministic extractors. When the model omits them (or
returns invalid enum values), the deterministic fallback must still fire so
the planner output stays internally consistent.
"""

from __future__ import annotations

import logging

import pytest

from paper_chaser_mcp.agentic.config import AgenticConfig
from paper_chaser_mcp.agentic.models import PlannerDecision, SubjectCard
from paper_chaser_mcp.agentic.planner import classify_query
from paper_chaser_mcp.agentic.provider_base import DeterministicProviderBundle
from paper_chaser_mcp.agentic.provider_helpers import (
    _PlannerResponseSchema,
    _PlannerSubjectCardSchema,
)


class _StubBundle:
    """Minimal provider bundle that returns a pre-baked PlannerDecision."""

    def __init__(self, planner: PlannerDecision) -> None:
        self._planner = planner

    async def aplan_search(self, **kwargs: object) -> PlannerDecision:
        return self._planner.model_copy(deep=True)


def _llm_planner_decision(**overrides: object) -> PlannerDecision:
    """Build a ``PlannerDecision`` the way an LLM planner would via the schema."""

    schema = _PlannerResponseSchema(intent="regulatory", **overrides)  # type: ignore[arg-type]
    return schema.to_planner_decision()


# ---------------------------------------------------------------------------
# 1) LLM populates both fields natively -> flows through unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_native_regulatory_intent_and_subject_card_pass_through() -> None:
    seed = _llm_planner_decision(
        regulatoryIntent="species_dossier",
        subjectCard=_PlannerSubjectCardSchema(
            commonName="desert tortoise",
            scientificName="Gopherus agassizii",
            agency="USFWS",
            requestedDocumentFamily="recovery_plan",
            subjectTerms=["desert tortoise", "recovery plan"],
            confidence="high",
        ),
    )
    assert seed.regulatory_intent == "species_dossier"
    assert isinstance(seed.subject_card, SubjectCard)
    assert seed.subject_card.source == "planner_llm"
    assert seed.subject_card.confidence == "high"

    _, planner = await classify_query(
        query="Desert tortoise recovery plan under the ESA",
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_StubBundle(seed),  # type: ignore[arg-type]
    )

    # LLM-native values must not be clobbered by the deterministic fallback.
    assert planner.regulatory_intent == "species_dossier"
    assert planner.subject_card is not None
    assert planner.subject_card.common_name == "desert tortoise"
    assert planner.subject_card.scientific_name == "Gopherus agassizii"
    assert planner.subject_card.requested_document_family == "recovery_plan"
    assert planner.subject_card.agency == "USFWS"
    assert planner.subject_card.confidence == "high"
    assert planner.subject_card.source == "planner_llm"


# ---------------------------------------------------------------------------
# 2) LLM omits both fields -> deterministic fallback fires
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_llm_response_without_new_fields_triggers_fallback() -> None:
    seed = _llm_planner_decision()  # no regulatoryIntent / subjectCard supplied
    assert seed.regulatory_intent is None
    assert seed.subject_card is None

    _, planner = await classify_query(
        query="Desert tortoise recovery plan under the ESA",
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_StubBundle(seed),  # type: ignore[arg-type]
    )

    # Deterministic derivation should have kicked in.
    assert planner.regulatory_intent is not None
    assert planner.regulatory_intent in {
        "species_dossier",
        "rulemaking_history",
        "unspecified",
        "hybrid_regulatory_plus_literature",
    }
    assert planner.subject_card is not None
    # Provenance contract: an LLM bundle was available (the stub provided a
    # PlannerDecision via aplan_search). Even though the LLM itself omitted
    # ``subjectCard``, the downstream resolver derives the card from the
    # LLM-backed planner signals (entity_card / candidate_concepts / planner
    # context). Per the ws-subject-grounding fix (commit df918df) the stamp
    # must therefore be ``planner_llm`` — the deterministic-fallback label is
    # reserved for the case where no LLM bundle is available at all (see the
    # DeterministicProviderBundle test in test_phase4_signal_integration.py).
    # Pin strictly so a future regression that confuses these two paths fails
    # loudly rather than masking behind an `in {...}` set.
    assert planner.subject_card.source == "planner_llm", (
        f"With an LLM bundle present (StubBundle), the post-hoc subject-card "
        f"resolver must stamp source='planner_llm'; got {planner.subject_card.source!r}"
    )


# ---------------------------------------------------------------------------
# 3) LLM returns invalid enum values -> logged + fallback fires
# ---------------------------------------------------------------------------


def test_invalid_regulatory_intent_enum_is_rejected_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="paper_chaser_mcp.agentic.provider_helpers")
    schema = _PlannerResponseSchema(
        intent="regulatory",
        regulatoryIntent="species_dossier_v2",  # not in the enum
        subjectCard=_PlannerSubjectCardSchema(
            commonName="desert tortoise",
            requestedDocumentFamily="not_a_family",  # invalid family
            confidence="super-high",  # invalid confidence
        ),
    )
    decision = schema.to_planner_decision()
    assert decision.regulatory_intent is None
    assert decision.subject_card is not None
    assert decision.subject_card.common_name == "desert tortoise"
    assert decision.subject_card.requested_document_family is None
    assert decision.subject_card.confidence == "medium"
    assert decision.subject_card.source == "planner_llm"

    joined = " ".join(record.message for record in caplog.records)
    assert "species_dossier_v2" in joined
    assert "not_a_family" in joined
    assert "super-high" in joined


@pytest.mark.asyncio
async def test_classify_query_falls_back_when_llm_regulatory_intent_is_invalid(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="paper_chaser_mcp.agentic.provider_helpers")
    seed_schema = _PlannerResponseSchema(
        intent="regulatory",
        regulatoryIntent="not_a_real_intent",
    )
    seed = seed_schema.to_planner_decision()
    assert seed.regulatory_intent is None  # dropped by schema coercion

    _, planner = await classify_query(
        query="Desert tortoise recovery plan under the ESA",
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=_StubBundle(seed),  # type: ignore[arg-type]
    )
    # Deterministic derivation recovered a regulatory intent.
    assert planner.regulatory_intent is not None


# ---------------------------------------------------------------------------
# 4) DeterministicProviderBundle end-to-end -> fallback flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deterministic_provider_bundle_uses_fallback_subject_card() -> None:
    config = AgenticConfig(
        enabled=True,
        provider="deterministic",
        planner_model="deterministic-planner",
        synthesis_model="deterministic-synthesizer",
        embedding_model="deterministic-lexical",
        index_backend="memory",
        session_ttl_seconds=1800,
        enable_trace_log=False,
    )
    bundle = DeterministicProviderBundle(config)
    _, planner = await classify_query(
        query="Desert tortoise recovery plan under the ESA",
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=bundle,
    )
    # The deterministic bundle never emits the new LLM-native schema fields;
    # the post-hoc fallback chain in classify_query must still populate both
    # so downstream code that consumes ``regulatoryIntent`` / ``subjectCard``
    # keeps working when the smart-layer planner is in fallback mode.
    assert planner.regulatory_intent is not None
    assert planner.subject_card is not None
    assert planner.subject_card.common_name == "desert tortoise"
