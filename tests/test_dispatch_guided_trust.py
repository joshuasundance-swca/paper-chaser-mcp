"""Phase 3 TDD tests for ``dispatch/guided/trust.py``."""

from __future__ import annotations

import pytest

from paper_chaser_mcp.dispatch.guided import trust as trust_mod


def test__guided_sources_all_off_topic_empty_false() -> None:
    assert trust_mod._guided_sources_all_off_topic([]) is False


def test__guided_sources_all_off_topic_all_off_topic_true() -> None:
    assert (
        trust_mod._guided_sources_all_off_topic([{"topicalRelevance": "off_topic"}, {"topicalRelevance": "off_topic"}])
        is True
    )


def test__guided_sources_all_off_topic_mixed_false() -> None:
    assert (
        trust_mod._guided_sources_all_off_topic([{"topicalRelevance": "off_topic"}, {"topicalRelevance": "on_topic"}])
        is False
    )


def test__guided_trust_summary_empty_returns_schema_shape() -> None:
    out = trust_mod._guided_trust_summary([], [])
    assert out["verifiedSourceCount"] == 0
    assert out["verifiedPrimarySourceCount"] == 0
    assert out["evidenceGapCount"] == 0
    assert "strengthExplanation" in out
    assert "trustRationale" in out


def test__guided_trust_summary_counts_primary_sources() -> None:
    sources = [
        {
            "isPrimarySource": True,
            "verificationStatus": "verified_primary_source",
            "topicalRelevance": "on_topic",
        },
        {
            "isPrimarySource": True,
            "verificationStatus": "verified_metadata",
            "topicalRelevance": "on_topic",
        },
    ]
    out = trust_mod._guided_trust_summary(sources, [])
    assert out["verifiedPrimarySourceCount"] == 2
    assert out["fullTextVerifiedPrimarySourceCount"] == 1


def test__guided_trust_summary_subject_chain_gaps() -> None:
    out = trust_mod._guided_trust_summary([], [], subject_chain_gaps=["missing regulator"])
    assert out["subjectChainGaps"] == ["missing regulator"]


def test__guided_missing_evidence_type_off_topic_only() -> None:
    category = trust_mod._guided_missing_evidence_type(
        status="abstained",
        evidence_gaps=[],
        sources=[{"topicalRelevance": "off_topic"}],
    )
    assert isinstance(category, str)


def test__guided_failure_summary_returns_payload_when_not_failing() -> None:
    out = trust_mod._guided_failure_summary(
        failure_summary=None,
        status="succeeded",
        sources=[{"topicalRelevance": "on_topic"}],
        evidence_gaps=[],
        all_sources_off_topic=False,
    )
    assert isinstance(out, dict)
    assert out.get("outcome") == "no_failure"


def test__guided_deterministic_fallback_used_returns_false_for_none() -> None:
    assert trust_mod._guided_deterministic_fallback_used(None) is False


def test__guided_partial_recovery_possible_returns_bool() -> None:
    result = trust_mod._guided_partial_recovery_possible(coverage_summary=None, failure_summary=None)
    assert isinstance(result, bool)


def test__guided_follow_up_status_respects_allowed_values() -> None:
    assert trust_mod._guided_follow_up_status("succeeded") == "succeeded"
    assert trust_mod._guided_follow_up_status("abstained") == "abstained"
    assert trust_mod._guided_follow_up_status(None)  # some default string


def test__guided_next_actions_returns_list() -> None:
    out = trust_mod._guided_next_actions(
        search_session_id="s1",
        status="succeeded",
        has_sources=True,
        all_sources_off_topic=False,
    )
    assert isinstance(out, list)


def test__guided_result_meaning_returns_string() -> None:
    out = trust_mod._guided_result_meaning(
        status="succeeded",
        verified_findings=[],
        evidence_gaps=[],
        coverage=None,
        failure_summary={},
        source_count=0,
        all_sources_off_topic=False,
    )
    assert isinstance(out, str)


def test__guided_summary_returns_string() -> None:
    out = trust_mod._guided_summary(
        intent="discovery",
        status="succeeded",
        findings=[],
        sources=[],
    )
    assert isinstance(out, str)


_EXPECTED_EXPORTS = (
    "_guided_trust_summary",
    "_guided_confidence_signals",
    "_guided_sources_all_off_topic",
    "_guided_failure_summary",
    "_guided_result_meaning",
    "_guided_deterministic_fallback_used",
    "_guided_partial_recovery_possible",
    "_guided_research_status",
    "_guided_deterministic_evidence_gaps",
    "_guided_generate_evidence_gaps",
    "_guided_machine_failure_payload",
    "_guided_summary",
    "_guided_next_actions",
    "_guided_missing_evidence_type",
    "_guided_best_next_internal_action",
    "_guided_result_state",
    "_guided_record_source_candidates",
    "_guided_follow_up_status",
)


@pytest.mark.parametrize("name", _EXPECTED_EXPORTS)
def test_trust_submodule_exports(name: str) -> None:
    assert hasattr(trust_mod, name)
