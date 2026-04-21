"""Phase 3 TDD tests for ``dispatch/guided/strategy_metadata.py``."""

from __future__ import annotations

import pytest

from paper_chaser_mcp.dispatch.guided import strategy_metadata as sm


def test__guided_provider_budget_payload_returns_bool_shape() -> None:
    assert sm._guided_provider_budget_payload(allow_paid_providers=True) == {"allowPaidProviders": True}
    assert sm._guided_provider_budget_payload(allow_paid_providers=False) == {"allowPaidProviders": False}


def test__guided_execution_provenance_payload_minimum() -> None:
    payload = sm._guided_execution_provenance_payload(execution_mode="guided")
    assert payload["executionMode"] == "guided"
    assert payload["serverPolicyApplied"] == "quality_first"
    assert payload["passesRun"] == 0


def test__guided_execution_provenance_payload_deterministic_fallback_flag() -> None:
    payload = sm._guided_execution_provenance_payload(
        execution_mode="guided",
        strategy_metadata={
            "configuredSmartProvider": "openai",
            "activeSmartProvider": "deterministic",
        },
    )
    assert payload["deterministicFallbackUsed"] is True


def test__guided_live_strategy_metadata_preserves_user_metadata() -> None:
    out = sm._guided_live_strategy_metadata(
        agentic_runtime=None,
        strategy_metadata={"intent": "discovery"},
        latency_profile="quality",
    )
    assert out["intent"] == "discovery"
    assert out["latencyProfile"] == "quality"


def test__guided_abstention_details_payload_returns_none_for_settled_status() -> None:
    assert (
        sm._guided_abstention_details_payload(
            status="settled",
            sources=[],
            evidence_gaps=[],
            trust_summary={},
        )
        is None
    )


def test__guided_abstention_details_payload_emits_for_abstained() -> None:
    payload = sm._guided_abstention_details_payload(
        status="abstained",
        sources=[],
        evidence_gaps=[],
        trust_summary={},
    )
    assert payload is not None
    assert "category" in payload
    assert "refinementHints" in payload


def test__guided_strategy_metadata_from_runs_merges_first_wins() -> None:
    runs = [
        {"strategyMetadata": {"intent": "discovery", "providersUsed": ["a"]}},
        {"strategyMetadata": {"intent": "review", "providersUsed": ["b"]}},
    ]
    out = sm._guided_strategy_metadata_from_runs(runs)
    assert out["intent"] == "discovery"
    assert out["providersUsed"] == ["a", "b"]


def test__guided_strategy_metadata_from_runs_handles_empty() -> None:
    assert sm._guided_strategy_metadata_from_runs([]) == {}


def test__guided_is_agency_guidance_query() -> None:
    assert sm._guided_is_agency_guidance_query("EPA guidance on stormwater") is True
    assert sm._guided_is_agency_guidance_query("systematic review") is False


def test__guided_is_known_item_query_detects_doi() -> None:
    assert sm._guided_is_known_item_query("10.1234/abcd") is True
    assert sm._guided_is_known_item_query("foo") is False


def test__guided_mentions_literature() -> None:
    assert sm._guided_mentions_literature("systematic review of X") is True
    assert sm._guided_mentions_literature("recipe for bread") is False


def test__guided_is_mixed_intent_query_respects_planner() -> None:
    assert (
        sm._guided_is_mixed_intent_query(
            "something",
            planner_regulatory_intent="hybrid_regulatory_plus_literature",
        )
        is True
    )


def test__guided_reference_signal_words_strips_generics() -> None:
    # ``paper about X`` should surface only the meaningful tokens. The
    # exact generic-word lists live in _core; we assert the stopword
    # "a"/"the" at minimum gets dropped.
    out = sm._guided_reference_signal_words("The paper")
    assert "the" not in out


def test__guided_should_escalate_research_disallows_when_sources_present() -> None:
    assert (
        sm._guided_should_escalate_research(
            intent="discovery",
            status="abstained",
            sources=[{"sourceId": "s"}],
            verified_findings=[],
            clarification=None,
            pass_modes=[],
            max_passes=2,
        )
        is False
    )


def test__guided_should_escalate_research_escalates_empty_abstained() -> None:
    assert (
        sm._guided_should_escalate_research(
            intent="discovery",
            status="abstained",
            sources=[],
            verified_findings=[],
            clarification=None,
            pass_modes=["smart"],
            max_passes=2,
        )
        is True
    )


_EXPECTED_EXPORTS = (
    "_guided_execution_provenance_payload",
    "_guided_live_strategy_metadata",
    "_guided_abstention_details_payload",
    "_guided_provider_budget_payload",
    "_guided_strategy_metadata_from_runs",
    "_guided_should_add_review_pass",
    "_guided_review_pass_overrides",
    "_guided_is_agency_guidance_query",
    "_guided_should_escalate_research",
    "_guided_is_known_item_query",
    "_guided_mentions_literature",
    "_guided_is_mixed_intent_query",
    "_guided_reference_signal_words",
    "_guided_merge_coverage_summaries",
    "_guided_merge_failure_summaries",
)


@pytest.mark.parametrize("name", _EXPECTED_EXPORTS)
def test_strategy_metadata_submodule_exports(name: str) -> None:
    assert hasattr(sm, name)
