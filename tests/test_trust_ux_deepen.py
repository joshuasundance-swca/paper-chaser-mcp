"""Tests for Workstream C residual (ws-trust-ux-deepen).

Covers additive fields exposed by :mod:`paper_chaser_mcp.dispatch`:

* top-level ``whyClassifiedAsWeakMatch`` on ``inspect_source``
* ``confidenceSignals.evidenceProfileDetail`` / ``synthesisPath`` /
  ``trustRevisionNarrative``
* ``trustSummary.authoritativeButWeak`` bucket
* quality-aware ``directReadRecommendationDetails``
* improved trust-summary prose
"""

from __future__ import annotations

from typing import Any

import pytest

import paper_chaser_mcp.dispatch as dispatch_module
from paper_chaser_mcp import server
from paper_chaser_mcp.agentic import WorkspaceRegistry
from tests.helpers import _payload


def _fr_weak_source() -> dict[str, Any]:
    return {
        "sourceId": "fr-weak",
        "title": "Endangered and Threatened Wildlife and Plants; General Notice",
        "provider": "federal_register",
        "sourceType": "regulatory_document",
        "verificationStatus": "verified_primary_source",
        "accessStatus": "full_text_verified",
        "topicalRelevance": "weak_match",
        "classificationRationale": "Notice mentions endangered species broadly, not desert tortoise.",
        "whyClassifiedAsWeakMatch": (
            "Authoritative notice, but it does not specifically address the desert tortoise dossier or recovery plan."
        ),
        "canonicalUrl": "https://example.com/fr-weak",
        "note": "Authoritative notice, but not species-specific enough.",
        "isPrimarySource": True,
    }


def _on_topic_primary_source() -> dict[str, Any]:
    return {
        "sourceId": "fr-strong",
        "title": "Desert Tortoise Recovery Plan, Final Rule",
        "provider": "federal_register",
        "sourceType": "federal_register_rule",
        "verificationStatus": "verified_primary_source",
        "accessStatus": "full_text_verified",
        "topicalRelevance": "on_topic",
        "isPrimarySource": True,
        "canonicalUrl": "https://example.com/fr-strong",
    }


def _off_topic_source() -> dict[str, Any]:
    return {
        "sourceId": "paper-off",
        "title": "Machine Learning for Chess",
        "provider": "openalex",
        "sourceType": "scholarly_article",
        "verificationStatus": "verified_metadata",
        "topicalRelevance": "off_topic",
        "classificationRationale": "No query-term overlap with desert tortoise.",
    }


def test_evidence_quality_detail_classifies_strong_on_topic() -> None:
    profile = dispatch_module._evidence_quality_detail([_on_topic_primary_source()])
    assert profile == "strong_on_topic"


def test_evidence_quality_detail_weak_authoritative_only() -> None:
    profile = dispatch_module._evidence_quality_detail([_fr_weak_source()])
    assert profile == "weak_authoritative_only"


def test_evidence_quality_detail_mixed() -> None:
    profile = dispatch_module._evidence_quality_detail([_on_topic_primary_source(), _fr_weak_source()])
    assert profile == "mixed"


def test_evidence_quality_detail_off_topic() -> None:
    profile = dispatch_module._evidence_quality_detail([_off_topic_source()])
    assert profile == "off_topic"


def test_evidence_quality_detail_insufficient_for_empty() -> None:
    assert dispatch_module._evidence_quality_detail([]) == "insufficient"


def test_synthesis_path_direct_when_primary_on_topic() -> None:
    path = dispatch_module._synthesis_path(
        status="succeeded",
        sources=[_on_topic_primary_source()],
        evidence_gaps=[],
        synthesis_mode="grounded",
    )
    assert path == "direct"


def test_synthesis_path_metadata_only_for_source_audit() -> None:
    path = dispatch_module._synthesis_path(
        status="succeeded",
        sources=[_fr_weak_source()],
        evidence_gaps=[],
        synthesis_mode="source_audit",
    )
    assert path == "metadata_only"


def test_synthesis_path_abstained_when_no_sources() -> None:
    path = dispatch_module._synthesis_path(
        status="abstained",
        sources=[],
        evidence_gaps=["no evidence"],
        synthesis_mode=None,
    )
    assert path == "abstained"


def test_synthesis_path_conservative_when_only_weak() -> None:
    path = dispatch_module._synthesis_path(
        status="succeeded",
        sources=[_fr_weak_source()],
        evidence_gaps=["species-specific evidence missing"],
        synthesis_mode=None,
    )
    assert path == "conservative"


def test_authoritative_but_weak_surfaces_in_trust_summary() -> None:
    sources = [_fr_weak_source(), _on_topic_primary_source(), _off_topic_source()]
    summary = dispatch_module._guided_trust_summary(sources, evidence_gaps=[])
    assert "authoritativeButWeak" in summary
    assert summary["authoritativeButWeak"] == ["fr-weak"]


def test_trust_summary_prose_differentiates_strong_and_weak_authority() -> None:
    sources = [_fr_weak_source(), _on_topic_primary_source(), _off_topic_source()]
    summary = dispatch_module._guided_trust_summary(sources, evidence_gaps=[])
    prose = summary["strengthExplanation"]
    assert "strong on-topic" in prose
    assert "authoritative but weak-match" in prose
    assert "off-target" in prose


def test_trust_summary_backcompat_when_all_on_topic() -> None:
    summary = dispatch_module._guided_trust_summary([_on_topic_primary_source()], evidence_gaps=[])
    assert summary["authoritativeButWeak"] == []
    assert summary["trustRationale"] == summary["strengthExplanation"]


def test_compose_why_classified_weak_match_combines_fragments() -> None:
    source = _fr_weak_source()
    sentence = dispatch_module._compose_why_classified_weak_match(source)
    assert sentence
    assert len(sentence) <= 200
    # prior-work classificationRationale should be the primary fragment
    assert sentence.lower().startswith("notice mentions endangered species")


def test_compose_why_classified_weak_match_graceful_without_subject_card() -> None:
    # No strategy metadata, no whyWeak, but classificationRationale is present
    source = {
        "topicalRelevance": "weak_match",
        "classificationRationale": "Partial facet coverage.",
    }
    sentence = dispatch_module._compose_why_classified_weak_match(source)
    assert sentence == "Partial facet coverage."


def test_compose_why_classified_weak_match_uses_subject_chain_gaps() -> None:
    source = {
        "topicalRelevance": "weak_match",
        "classificationRationale": "Authoritative but generic.",
    }
    strategy_metadata = {"subjectChainGaps": ["species-specific evidence missing"]}
    sentence = dispatch_module._compose_why_classified_weak_match(source, strategy_metadata=strategy_metadata)
    assert sentence is not None
    assert "species-specific evidence missing" in sentence


def test_compose_why_classified_weak_match_returns_none_for_on_topic() -> None:
    source = {"topicalRelevance": "on_topic", "classificationRationale": "great"}
    assert dispatch_module._compose_why_classified_weak_match(source) is None


def test_direct_read_recommendation_details_low_authoritative_but_weak() -> None:
    details = dispatch_module._direct_read_recommendation_details(_fr_weak_source(), tool_profile="guided")
    assert details
    first = details[0]
    assert first["trustLevel"] == "low_authoritative_but_weak"
    assert "authoritative" in first["whyRecommended"].lower()
    assert any("authoritative" in c.lower() for c in first["cautions"])


def test_direct_read_recommendation_details_high_for_on_topic_primary() -> None:
    details = dispatch_module._direct_read_recommendation_details(_on_topic_primary_source(), tool_profile="guided")
    assert details[0]["trustLevel"] == "high"


def test_direct_read_recommendations_stringlist_unchanged() -> None:
    # Backward-compat: list[str] shape is preserved
    recs = dispatch_module._direct_read_recommendations(_fr_weak_source(), tool_profile="guided")
    assert all(isinstance(item, str) for item in recs)
    assert recs[0].startswith("This source is only a weak match")


@pytest.mark.asyncio
async def test_inspect_source_emits_why_classified_weak_match_and_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure server runtime (workspace_registry etc.) is initialized before swap
    # so monkeypatch restores a real registry rather than ``None``.
    server._initialize_runtime()
    isolated_registry = WorkspaceRegistry()
    isolated_registry.save_result_set(
        source_tool="research",
        search_session_id="ssn-trust-ux-weak",
        query="desert tortoise recovery planning",
        payload={
            "query": "desert tortoise recovery planning",
            "intent": "regulatory",
            "sources": [_fr_weak_source()],
            "strategyMetadata": {"subjectChainGaps": ["species-specific dossier missing"]},
        },
    )

    monkeypatch.setattr(server, "workspace_registry", isolated_registry)
    payload = _payload(
        await server.call_tool(
            "inspect_source",
            {"searchSessionId": "ssn-trust-ux-weak", "sourceId": "fr-weak"},
        )
    )

    assert "whyClassifiedAsWeakMatch" in payload
    sentence = payload["whyClassifiedAsWeakMatch"]
    assert isinstance(sentence, str) and sentence
    assert len(sentence) <= 200

    signals = payload["confidenceSignals"]
    assert signals["evidenceProfileDetail"] == "weak_authoritative_only"
    # inspect_source ends up running contract-field synthesis so the actual
    # shipped synthesisPath is "conservative" (no on-topic evidence).
    assert signals["synthesisPath"] in {"metadata_only", "conservative"}
    assert signals.get("trustRevisionNarrative")

    details = payload["directReadRecommendationDetails"]
    assert isinstance(details, list) and details
    assert details[0]["trustLevel"] == "low_authoritative_but_weak"
    # Order of parallel lists must match the strings
    assert payload["directReadRecommendations"][0] == details[0]["recommendation"]


@pytest.mark.asyncio
async def test_inspect_source_graceful_without_subject_chain_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When upstream ws-regulatory-grounding fields are absent, the new
    surfaces still work on prior-shape payloads."""
    server._initialize_runtime()
    isolated_registry = WorkspaceRegistry()
    minimal_source = {
        "sourceId": "paper-1",
        "title": "Desert Tortoise Habitat Study",
        "provider": "openalex",
        "sourceType": "scholarly_article",
        "topicalRelevance": "on_topic",
        "verificationStatus": "verified_metadata",
    }
    isolated_registry.save_result_set(
        source_tool="research",
        search_session_id="ssn-trust-ux-min",
        query="desert tortoise habitat",
        payload={"query": "desert tortoise habitat", "sources": [minimal_source]},
    )

    monkeypatch.setattr(server, "workspace_registry", isolated_registry)
    payload = _payload(
        await server.call_tool(
            "inspect_source",
            {"searchSessionId": "ssn-trust-ux-min", "sourceId": "paper-1"},
        )
    )

    # No weak-match rationale expected on on-topic source
    assert "whyClassifiedAsWeakMatch" not in payload
    signals = payload["confidenceSignals"]
    assert signals["evidenceProfileDetail"] in {"strong_on_topic", "mixed"}
    assert "trustRevisionNarrative" not in signals or signals["trustRevisionNarrative"]


def test_confidence_signals_preserves_existing_fields() -> None:
    # Backward-compatibility: existing evidenceQualityProfile/synthesisMode still present.
    signals = dispatch_module._guided_confidence_signals(
        status="succeeded",
        sources=[_on_topic_primary_source()],
        evidence_gaps=[],
    )
    assert signals["evidenceQualityProfile"] == "high"
    assert signals["synthesisMode"] == "grounded"
    assert signals["evidenceProfileDetail"] == "strong_on_topic"
    assert signals["synthesisPath"] == "direct"
