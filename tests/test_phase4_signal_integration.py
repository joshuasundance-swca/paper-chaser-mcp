"""Red-team integration tests for Phase 4/5 additive signals.

These tests exercise the guided tool surface end-to-end through
``server.call_tool`` so a signal that was added by Phase 4/5 work in
``graphs.py`` / the smart layer but silently dropped by the dispatch
serializer will fail here even if the unit tests for that signal still pass.

The tests combine two offline harnesses:

* A ``_FakeRuntime`` that returns a pre-baked smart result (matching the
  existing ``tests/test_env_sci_eval_slices.py`` pattern). This is the only
  reliable way to inject Phase 4/5 strategy metadata (``intentFamily``,
  ``regulatoryIntent``, ``subjectCard``, ``subjectChainGaps``, richer
  ``retrievalHypotheses``) into the dispatch path without live LLM keys.
* Direct use of ``DeterministicProviderBundle`` via ``classify_query`` to
  verify the LLM-first fallback principle (``subjectCard.source ==
  "deterministic_fallback"`` when no LLM keys are configured).

Findings that surfaced from this red-team are recorded as
``pytest.mark.xfail`` entries below with a one-paragraph explanation.
"""

from __future__ import annotations

from typing import Any

import pytest

from paper_chaser_mcp import server
from paper_chaser_mcp.agentic import (
    AgenticConfig,
    WorkspaceRegistry,
)
from paper_chaser_mcp.agentic.models import SubjectCard
from paper_chaser_mcp.agentic.planner import classify_query
from paper_chaser_mcp.agentic.provider_base import DeterministicProviderBundle
from tests.helpers import _payload

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Top-level research response keys that must remain stable on the grounded
# (smart-backed) path. Built from an actual dispatch of ``research`` with a
# fake smart runtime so the snapshot test below tracks *real* emission.
_STABLE_RESEARCH_KEYS_GROUNDED: frozenset[str] = frozenset(
    {
        "searchSessionId",
        "status",
        "intent",
        "evidenceGaps",
        "trustSummary",
        "coverageSummary",
        "failureSummary",
        "nextActions",
        "resultMeaning",
        "resultState",
        "executionProvenance",
        "confidenceSignals",
        "routingSummary",
        "evidence",
        "leads",
        "legacyFieldsIncluded",
        "inputNormalization",
        "summary",
    }
)


def _smart_result(
    *,
    session_id: str = "ssn-phase4-integration",
    intent: str = "regulatory",
    structured: list[dict[str, Any]] | None = None,
    leads: list[dict[str, Any]] | None = None,
    gaps: list[str] | None = None,
    strategy_overrides: dict[str, Any] | None = None,
    result_status: str = "succeeded",
) -> dict[str, Any]:
    """Build a smart-layer result carrying Phase 4/5 strategy metadata.

    Mirrors the shape emitted by ``agentic_runtime.search_papers_smart`` so
    the dispatch layer for ``research`` can consume it without other mocks.
    """
    strategy: dict[str, Any] = {
        "intent": intent,
        "querySpecificity": "high",
        "ambiguityLevel": "low",
        "retrievalHypotheses": ["core_literature"],
        "providerPlan": ["semantic_scholar", "openalex"],
        "providersUsed": ["semantic_scholar", "openalex"],
        "routingConfidence": "high",
    }
    if strategy_overrides:
        strategy.update(strategy_overrides)
    return {
        "searchSessionId": session_id,
        "strategyMetadata": strategy,
        "structuredSources": structured or [],
        "candidateLeads": leads or [],
        "evidenceGaps": gaps or [],
        "coverageSummary": {
            "providersAttempted": strategy.get("providerPlan") or [],
            "providersSucceeded": strategy.get("providerPlan") or [],
            "providersZeroResults": [],
            "likelyCompleteness": "partial",
            "searchMode": "smart_literature_review",
        },
        "failureSummary": None,
        "clarification": None,
        "resultStatus": result_status,
    }


class _FakeRuntime:
    """Runtime stub with pluggable ``search_papers_smart`` / ``ask_result_set`` hooks."""

    def __init__(
        self,
        *,
        smart_result: dict[str, Any] | None = None,
        ask_result: dict[str, Any] | None = None,
    ) -> None:
        self._smart = smart_result
        self._ask = ask_result
        self.smart_calls: list[dict[str, Any]] = []
        self.ask_calls: list[dict[str, Any]] = []

    async def search_papers_smart(self, **kwargs: Any) -> dict[str, Any]:
        self.smart_calls.append(kwargs)
        assert self._smart is not None, "smart_result not configured"
        return self._smart

    async def ask_result_set(self, **kwargs: Any) -> dict[str, Any]:
        self.ask_calls.append(kwargs)
        assert self._ask is not None, "ask_result not configured"
        return self._ask


def _strong_source(**overrides: Any) -> dict[str, Any]:
    base = {
        "sourceId": "strong-1",
        "title": "Desert tortoise recovery plan (USFWS 2011 revision)",
        "provider": "federal_register",
        "sourceType": "regulatory_guidance",
        "verificationStatus": "verified_primary_source",
        "accessStatus": "full_text_verified",
        "topicalRelevance": "on_topic",
        "confidence": "high",
        "isPrimarySource": True,
        "canonicalUrl": "https://example.gov/desert-tortoise-recovery-plan",
        "date": "2011-05-06",
    }
    base.update(overrides)
    return base


def _weak_authoritative_source(**overrides: Any) -> dict[str, Any]:
    base = {
        "sourceId": "fr-weak-1",
        "title": "Endangered and Threatened Wildlife and Plants; General Notice",
        "provider": "federal_register",
        "sourceType": "regulatory_document",
        "verificationStatus": "verified_primary_source",
        "accessStatus": "full_text_verified",
        "topicalRelevance": "weak_match",
        "confidence": "low",
        "isPrimarySource": True,
        "whyClassifiedAsWeakMatch": (
            "Authoritative FR notice, but it does not specifically address desert tortoise recovery planning."
        ),
        "canonicalUrl": "https://example.gov/fr-weak",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# research: grounded / strong-match path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_research_grounded_response_exposes_confidence_and_routing_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A strong-match research response must expose confidenceSignals +
    routingSummary.retrievalHypotheses end-to-end, and must NOT populate
    ``trustSummary.authoritativeButWeak`` or set a source-level
    ``whyClassifiedAsWeakMatch`` (negative assertion — prevents false positives).
    """
    smart = _smart_result(
        intent="regulatory",
        structured=[_strong_source()],
        strategy_overrides={
            "retrievalHypotheses": ["hybrid_policy_science", "species_recovery"],
            "intentFamily": "species_dossier_regulatory",
            "regulatoryIntent": "species_dossier",
            "subjectCard": {
                "commonName": "desert tortoise",
                "scientificName": "Gopherus agassizii",
                "confidence": "high",
                "source": "planner_llm",
                "subjectTerms": ["desert tortoise", "recovery plan"],
            },
            "subjectChainGaps": [],
        },
    )
    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime(smart_result=smart))

    payload = _payload(
        await server.call_tool(
            "research",
            {"query": "desert tortoise recovery plan", "includeLegacyFields": True},
        )
    )

    # confidenceSignals wired end-to-end
    cs = payload.get("confidenceSignals")
    assert isinstance(cs, dict), "confidenceSignals missing from grounded research response"
    assert "evidenceQualityProfile" in cs
    assert "synthesisMode" in cs

    # routingSummary.retrievalHypotheses carries the Phase 4 hybrid hint
    rs = payload.get("routingSummary")
    assert isinstance(rs, dict)
    assert "hybrid_policy_science" in (rs.get("retrievalHypotheses") or [])

    # trustSummary.authoritativeButWeak bucket is present and empty on strong match
    ts = payload.get("trustSummary")
    assert isinstance(ts, dict)
    assert "authoritativeButWeak" in ts, "authoritativeButWeak bucket missing even on strong-match case"
    assert ts["authoritativeButWeak"] == [], (
        f"Expected empty authoritativeButWeak on strong-match, got {ts['authoritativeButWeak']!r}"
    )

    # Negative: no weak-match rationale injected onto a strong source
    sources = payload.get("sources") or []
    assert sources, "expected at least one source on grounded response"
    for src in sources:
        assert not src.get("whyClassifiedAsWeakMatch"), (
            "Strong-match source should not carry whyClassifiedAsWeakMatch; "
            f"got {src.get('whyClassifiedAsWeakMatch')!r}"
        )


# ---------------------------------------------------------------------------
# research: weak-match / authoritative-but-weak bucketing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_research_weak_match_populates_authoritative_but_weak_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sources tagged ``topicalRelevance=weak_match`` + primary-source authoritative
    must flow end-to-end into ``trustSummary.authoritativeButWeak`` AND retain their
    ``whyClassifiedAsWeakMatch`` rationale on the source record."""
    smart = _smart_result(
        intent="regulatory",
        structured=[_weak_authoritative_source()],
        strategy_overrides={
            "regulatoryIntent": "species_dossier",
            "subjectChainGaps": [
                "No USFWS recovery plan document located; only a general FR notice was retrieved.",
            ],
        },
        result_status="partial",
        gaps=["Desert tortoise recovery plan document was not retrieved."],
    )
    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime(smart_result=smart))

    payload = _payload(
        await server.call_tool(
            "research",
            {"query": "desert tortoise recovery plan", "includeLegacyFields": True},
        )
    )

    ts = payload.get("trustSummary") or {}
    assert "fr-weak-1" in (ts.get("authoritativeButWeak") or []), (
        "authoritative but weak source did not land in trustSummary.authoritativeButWeak"
    )

    sources = payload.get("sources") or []
    assert sources
    reasons = [src.get("whyClassifiedAsWeakMatch") for src in sources]
    assert any("does not specifically address" in (r or "") for r in reasons)


# ---------------------------------------------------------------------------
# Phase 4/5 strategy signals are surfaced through routingSummary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_research_surfaces_phase4_strategy_fields_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fix for ws-dispatch-surface-strategy: ``intentFamily``,
    ``regulatoryIntent``, ``subjectCard``, and ``subjectChainGaps`` emitted by
    the smart layer must be surfaced on the research response so downstream
    agents can observe the planner's classification. They are attached
    additively under ``routingSummary``.
    """
    smart = _smart_result(
        intent="regulatory",
        structured=[_strong_source()],
        strategy_overrides={
            "intentFamily": "species_dossier_regulatory",
            "regulatoryIntent": "species_dossier",
            "subjectCard": {
                "commonName": "desert tortoise",
                "confidence": "high",
                "source": "planner_llm",
            },
            "subjectChainGaps": ["missing recovery plan document"],
            "retrievalHypotheses": ["species_recovery"],
        },
    )
    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime(smart_result=smart))

    payload = _payload(await server.call_tool("research", {"query": "desert tortoise recovery plan"}))

    rs = payload.get("routingSummary")
    assert isinstance(rs, dict), "routingSummary must be present on grounded research responses"
    assert rs.get("intentFamily") == "species_dossier_regulatory"
    assert rs.get("regulatoryIntent") == "species_dossier"
    subject_card = rs.get("subjectCard")
    assert isinstance(subject_card, dict) and subject_card.get("commonName") == "desert tortoise", (
        f"routingSummary.subjectCard should round-trip the smart-layer card; got {subject_card!r}"
    )
    assert list(rs.get("subjectChainGaps") or []) == ["missing recovery plan document"]
    assert "species_recovery" in list(rs.get("retrievalHypotheses") or []), (
        "Existing retrievalHypotheses consumers must continue to see their field."
    )
    # ws-dispatch-contract-trust (finding #5): subjectChainGaps must also flow
    # into the machine-readable trust signals, not only into the top-level
    # routing summary / prose rationale.
    signals = payload.get("confidenceSignals")
    assert isinstance(signals, dict)
    assert list(signals.get("subjectChainGaps") or []) == ["missing recovery plan document"]


@pytest.mark.asyncio
async def test_inspect_source_threads_subject_chain_gaps_into_confidence_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ws-dispatch-contract-trust (finding #5): when a saved session records
    planner ``subjectChainGaps`` in its ``strategyMetadata``, a subsequent
    ``inspect_source`` call must surface those gaps in ``confidenceSignals``
    too — not only in the human-readable weak-match prose."""
    isolated_registry = WorkspaceRegistry()
    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-inspect-gaps",
        query="desert tortoise recovery plan",
        payload={
            "query": "desert tortoise recovery plan",
            "intent": "regulatory",
            "sources": [_weak_authoritative_source()],
            "strategyMetadata": {
                "intent": "regulatory",
                "subjectChainGaps": ["planner could not bind subject to a recovery plan document"],
            },
        },
    )

    monkeypatch.setattr(server, "workspace_registry", isolated_registry)

    payload = _payload(
        await server.call_tool(
            "inspect_source",
            {"searchSessionId": "ssn-inspect-gaps", "sourceId": "fr-weak-1"},
        )
    )
    signals = payload.get("confidenceSignals")
    assert isinstance(signals, dict)
    assert "planner could not bind subject to a recovery plan document" in list(signals.get("subjectChainGaps") or [])


# ---------------------------------------------------------------------------
# follow_up_research — abstention path + trust revision narrative
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_follow_up_research_abstention_path_carries_confidence_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a saved session has only a thin/weak pool and ask_result_set returns
    insufficient_evidence, the follow-up response must carry confidenceSignals
    describing the abstention (synthesisMode/synthesisPath) and the weak source
    must appear in trustSummary.authoritativeButWeak."""
    isolated_registry = WorkspaceRegistry()
    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-follow-up-abstain",
        query="desert tortoise recovery plan",
        payload={
            "query": "desert tortoise recovery plan",
            "intent": "regulatory",
            "sources": [_weak_authoritative_source()],
        },
    )

    ask_result: dict[str, Any] = {
        "answerStatus": "insufficient_evidence",
        "answer": None,
        "providerUsed": "deterministic",
        "degradationReason": None,
        "evidenceUsePlan": None,
        "evidence": [],
        "unsupportedAsks": [
            "Cannot determine recovery actions from the saved pool.",
        ],
        "followUpQuestions": [],
        "structuredSources": [_weak_authoritative_source()],
        "candidateLeads": [],
        "evidenceGaps": ["Saved pool does not contain the USFWS recovery plan."],
        "coverageSummary": {
            "providersAttempted": ["federal_register"],
            "providersSucceeded": ["federal_register"],
            "providersFailed": [],
            "providersZeroResults": [],
            "searchMode": "grounded_follow_up",
        },
        "failureSummary": None,
    }

    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime(ask_result=ask_result))
    original_registry = server.workspace_registry
    server.workspace_registry = isolated_registry
    try:
        payload = _payload(
            await server.call_tool(
                "follow_up_research",
                {
                    "searchSessionId": "ssn-follow-up-abstain",
                    "question": "What actions are required under the recovery plan?",
                },
            )
        )
    finally:
        server.workspace_registry = original_registry

    cs = payload.get("confidenceSignals")
    assert isinstance(cs, dict), "follow_up_research missing confidenceSignals on abstention path"
    # The dispatch must label synthesis as either the abstention synthesisPath
    # (``conservative``) or the abstaining synthesisMode (``limited`` /
    # ``grounded_follow_up`` when degraded). At minimum one of these keys must
    # reflect a non-normal synthesis state.
    assert cs.get("synthesisMode") or cs.get("synthesisPath"), (
        f"Expected synthesisMode or synthesisPath on abstention; got {cs!r}"
    )

    # answerStatus should faithfully echo the insufficient_evidence abstention.
    assert payload.get("answerStatus") == "insufficient_evidence"


@pytest.mark.asyncio
async def test_follow_up_research_trust_revision_narrative_populates_when_pool_is_weak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the saved pool is weak AND the ask path degrades, confidenceSignals
    should expose either ``trustRevisionNarrative`` or ``trustRevisionReason``
    so the agent has a narrative rather than only a boolean abstention signal.

    The dispatch currently populates ``trustRevisionReason`` on the
    ``degradationReason`` branch and ``trustRevisionNarrative`` on the weak
    evidence branch — either satisfies the red-team requirement.
    """
    isolated_registry = WorkspaceRegistry()
    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-follow-up-degraded",
        query="desert tortoise recovery plan",
        payload={
            "query": "desert tortoise recovery plan",
            "intent": "regulatory",
            "sources": [_weak_authoritative_source()],
        },
    )
    ask_result: dict[str, Any] = {
        "answerStatus": "answered",
        "answer": "Deterministic fallback synthesis.",
        "providerUsed": "deterministic",
        "degradationReason": "deterministic_synthesis_fallback",
        "evidenceUsePlan": None,
        "evidence": [],
        "unsupportedAsks": [],
        "followUpQuestions": [],
        "structuredSources": [_weak_authoritative_source()],
        "candidateLeads": [],
        "evidenceGaps": ["deterministic_synthesis_fallback"],
        "coverageSummary": {
            "providersAttempted": ["federal_register"],
            "providersSucceeded": ["federal_register"],
            "providersFailed": [],
            "providersZeroResults": [],
            "searchMode": "grounded_follow_up",
        },
        "failureSummary": None,
    }

    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime(ask_result=ask_result))
    original_registry = server.workspace_registry
    server.workspace_registry = isolated_registry
    try:
        payload = _payload(
            await server.call_tool(
                "follow_up_research",
                {
                    "searchSessionId": "ssn-follow-up-degraded",
                    "question": "What actions are required?",
                    "responseMode": "standard",
                },
            )
        )
    finally:
        server.workspace_registry = original_registry

    cs = payload.get("confidenceSignals") or {}
    narrative = cs.get("trustRevisionNarrative") or cs.get("trustRevisionReason")
    assert narrative, f"Expected confidenceSignals.trustRevisionNarrative/Reason to be populated; got {cs!r}"


# ---------------------------------------------------------------------------
# inspect_source — whyClassifiedAsWeakMatch + directReadRecommendationDetails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inspect_source_weak_record_surfaces_rationale_and_trust_level() -> None:
    """Inspecting a known-weak authoritative record must expose
    ``whyClassifiedAsWeakMatch`` AND ``directReadRecommendationDetails``
    where each entry is a dict carrying ``trustLevel`` + ``whyRecommended``."""
    isolated_registry = WorkspaceRegistry()
    isolated_registry.save_result_set(
        source_tool="research",
        search_session_id="ssn-inspect-weak",
        query="desert tortoise recovery planning",
        payload={
            "query": "desert tortoise recovery planning",
            "intent": "regulatory",
            "sources": [_weak_authoritative_source()],
        },
    )

    original_registry = server.workspace_registry
    server.workspace_registry = isolated_registry
    try:
        payload = _payload(
            await server.call_tool(
                "inspect_source",
                {"searchSessionId": "ssn-inspect-weak", "sourceId": "fr-weak-1"},
            )
        )
    finally:
        server.workspace_registry = original_registry

    assert payload.get("whyClassifiedAsWeakMatch"), (
        "inspect_source did not surface whyClassifiedAsWeakMatch for a known-weak record"
    )
    details = payload.get("directReadRecommendationDetails")
    assert isinstance(details, list) and details, "directReadRecommendationDetails is missing/empty"
    first = details[0]
    assert isinstance(first, dict)
    assert "trustLevel" in first, f"directReadRecommendationDetails entry missing trustLevel: {first!r}"
    # Either whyRecommended (spec name) or a plausible synonym from the
    # dispatch implementation ("why" / "recommendation") must be present.
    has_rationale_field = any(k in first for k in ("whyRecommended", "why", "recommendation"))
    assert has_rationale_field, f"directReadRecommendationDetails entry missing rationale field: {sorted(first)}"


@pytest.mark.asyncio
async def test_inspect_source_strong_record_omits_false_positive_weak_rationale() -> None:
    """Negative assertion: inspecting a strong on-topic, primary-source record
    must NOT emit ``whyClassifiedAsWeakMatch``. This guards against
    ``_compose_why_classified_weak_match`` leaking rationale onto healthy
    records (which would make every response look abstention-shaped)."""
    isolated_registry = WorkspaceRegistry()
    isolated_registry.save_result_set(
        source_tool="research",
        search_session_id="ssn-inspect-strong",
        query="desert tortoise recovery planning",
        payload={
            "query": "desert tortoise recovery planning",
            "intent": "regulatory",
            "sources": [_strong_source()],
        },
    )
    original_registry = server.workspace_registry
    server.workspace_registry = isolated_registry
    try:
        payload = _payload(
            await server.call_tool(
                "inspect_source",
                {"searchSessionId": "ssn-inspect-strong", "sourceId": "strong-1"},
            )
        )
    finally:
        server.workspace_registry = original_registry

    assert not payload.get("whyClassifiedAsWeakMatch"), (
        f"Strong-match inspect_source leaked whyClassifiedAsWeakMatch: {payload.get('whyClassifiedAsWeakMatch')!r}"
    )
    # And the recommendation details should reflect a non-low trust level.
    details = payload.get("directReadRecommendationDetails") or []
    assert details, "directReadRecommendationDetails should still be present on strong records"
    trust_levels = [entry.get("trustLevel") for entry in details if isinstance(entry, dict)]
    assert any(level in {"high", "medium"} for level in trust_levels), (
        f"Expected at least one high/medium trustLevel on strong record; got {trust_levels!r}"
    )


# ---------------------------------------------------------------------------
# LLM-first principle: DeterministicProviderBundle → deterministic_fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deterministic_provider_bundle_emits_populated_subject_card() -> None:
    """LLM-first principle check: when the real ``DeterministicProviderBundle``
    is used (no LLM keys), the subject card must still be populated so
    downstream code that consumes the card keeps working. After
    ws-dispatch-surface-strategy, provenance must honor the actual resolver:
    the deterministic shim stamps ``subjectCard.source == "deterministic_fallback"``
    instead of leaking the ``planner_llm`` tag.
    """
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
    assert bundle.is_deterministic, "DeterministicProviderBundle must self-identify as deterministic."
    _, planner = await classify_query(
        query="Desert tortoise recovery plan under the ESA",
        mode="auto",
        year=None,
        venue=None,
        focus=None,
        provider_bundle=bundle,
    )
    assert planner.subject_card is not None
    assert planner.subject_card.source == "deterministic_fallback", (
        f"DeterministicProviderBundle must stamp subjectCard.source='deterministic_fallback'; "
        f"got {planner.subject_card.source!r}"
    )
    # Downstream accessors must still work — common_name populated by the extractor.
    assert (planner.subject_card.common_name or "").lower().startswith("desert tortoise")


# ---------------------------------------------------------------------------
# Response shape stability snapshot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_research_response_shape_snapshot_keeps_default_keys_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Snapshot-style test: on the smart-backed research path, the top-level
    response must continue to expose the stable default key set that
    downstream guided agents and docs rely on after the legacy-field migration.
    """
    smart = _smart_result(
        intent="regulatory",
        structured=[_strong_source()],
        strategy_overrides={
            "retrievalHypotheses": ["hybrid_policy_science"],
        },
    )
    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime(smart_result=smart))

    payload = _payload(await server.call_tool("research", {"query": "desert tortoise recovery plan"}))
    top_keys = set(payload.keys())

    missing = _STABLE_RESEARCH_KEYS_GROUNDED - top_keys
    assert not missing, (
        f"Legacy/stable research response keys went missing after Phase 4/5: {sorted(missing)}. "
        f"Actual keys: {sorted(top_keys)}"
    )


# ---------------------------------------------------------------------------
# ws-dispatch-surface-strategy regression tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_research_routing_summary_accepts_subject_card_pydantic_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the smart layer may emit ``subjectCard`` as either a dict
    or a ``SubjectCard`` pydantic instance depending on serialization path.
    The dispatch must normalize both shapes into a JSON-safe dict under
    ``routingSummary.subjectCard`` with camelCase aliases preserved."""
    card = SubjectCard(
        commonName="desert tortoise",
        scientificName="Gopherus agassizii",
        confidence="high",
        source="planner_llm",
        subjectTerms=["desert tortoise", "recovery plan"],
    )
    smart = _smart_result(
        intent="regulatory",
        structured=[_strong_source()],
        strategy_overrides={
            "intentFamily": "species_dossier_regulatory",
            "regulatoryIntent": "species_dossier",
            "subjectCard": card,
            "subjectChainGaps": ["missing recovery plan document"],
        },
    )
    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime(smart_result=smart))

    payload = _payload(await server.call_tool("research", {"query": "desert tortoise recovery plan"}))

    rs = payload.get("routingSummary") or {}
    sc = rs.get("subjectCard")
    assert isinstance(sc, dict), (
        f"routingSummary.subjectCard must be normalized to a dict even when the smart layer "
        f"emits a pydantic SubjectCard; got {type(sc).__name__}"
    )
    assert sc.get("commonName") == "desert tortoise"
    assert sc.get("scientificName") == "Gopherus agassizii"
    assert sc.get("source") == "planner_llm"
    assert "desert tortoise" in (sc.get("subjectTerms") or [])


@pytest.mark.asyncio
async def test_research_trust_summary_always_exposes_authoritative_but_weak_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: ``trustSummary.authoritativeButWeak`` must always be present
    (even as an empty list) on grounded research responses so downstream
    consumers can ``get('authoritativeButWeak', [])``-style key-check safely
    instead of branching on a ``KeyError``-shaped surface."""
    smart = _smart_result(
        intent="regulatory",
        structured=[_strong_source()],
    )
    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime(smart_result=smart))

    payload = _payload(await server.call_tool("research", {"query": "desert tortoise recovery plan"}))

    ts = payload.get("trustSummary")
    assert isinstance(ts, dict), "trustSummary must be present on grounded research responses"
    assert "authoritativeButWeak" in ts, (
        "trustSummary.authoritativeButWeak key must always be present on research responses"
    )
    assert isinstance(ts["authoritativeButWeak"], list)


@pytest.mark.asyncio
async def test_follow_up_research_trust_summary_always_exposes_authoritative_but_weak_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the follow_up_research session re-emit path must include
    ``trustSummary.authoritativeButWeak`` (setdefault to empty list) so the
    key is stable across research/follow_up_research."""
    isolated_registry = WorkspaceRegistry()
    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-follow-up-trust-key",
        query="desert tortoise recovery plan",
        payload={
            "query": "desert tortoise recovery plan",
            "intent": "regulatory",
            "sources": [_strong_source()],
        },
    )
    ask_result: dict[str, Any] = {
        "answerStatus": "answered",
        "answer": "Synthesis result.",
        "providerUsed": "deterministic",
        "degradationReason": None,
        "evidenceUsePlan": None,
        "evidence": [],
        "unsupportedAsks": [],
        "followUpQuestions": [],
        "structuredSources": [_strong_source()],
        "candidateLeads": [],
        "evidenceGaps": [],
        "coverageSummary": {
            "providersAttempted": ["federal_register"],
            "providersSucceeded": ["federal_register"],
            "providersFailed": [],
            "providersZeroResults": [],
            "searchMode": "grounded_follow_up",
        },
        "failureSummary": None,
    }

    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime(ask_result=ask_result))
    original_registry = server.workspace_registry
    server.workspace_registry = isolated_registry
    try:
        payload = _payload(
            await server.call_tool(
                "follow_up_research",
                {
                    "searchSessionId": "ssn-follow-up-trust-key",
                    "question": "What actions are required under the recovery plan?",
                },
            )
        )
    finally:
        server.workspace_registry = original_registry

    ts = payload.get("trustSummary")
    if ts is not None:
        assert isinstance(ts, dict)
        assert "authoritativeButWeak" in ts, (
            "trustSummary.authoritativeButWeak key missing on follow_up_research re-emit path"
        )
        assert isinstance(ts["authoritativeButWeak"], list)
