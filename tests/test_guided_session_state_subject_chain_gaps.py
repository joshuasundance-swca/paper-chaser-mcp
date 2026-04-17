"""Regression: ``_guided_session_state`` must forward ``subjectChainGaps``.

Finding 4 of the Phase 6 rubber-duck review: the saved-session trust summary
rebuild in :func:`paper_chaser_mcp.dispatch._guided_session_state` was
reconstructing ``trustSummary`` via :func:`_guided_trust_summary` without
forwarding ``payload.strategyMetadata.subjectChainGaps``. Session follow-ups
(``follow_up_research`` / ``inspect_source`` / ``ask_result_set``) therefore
silently lost the subject-chain-gap trust signal even though the helper
supports it.
"""

from __future__ import annotations

from typing import Any

from paper_chaser_mcp.agentic import WorkspaceRegistry
from paper_chaser_mcp.dispatch import _guided_session_state


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
            "Authoritative notice, but it does not specifically address the desert tortoise dossier."
        ),
        "canonicalUrl": "https://example.com/fr-weak",
        "note": "Authoritative notice, but not species-specific enough.",
        "isPrimarySource": True,
    }


def test_guided_session_state_forwards_subject_chain_gaps() -> None:
    registry = WorkspaceRegistry()
    registry.save_result_set(
        source_tool="research",
        search_session_id="ssn-subject-chain-gap",
        query="desert tortoise recovery plan",
        payload={
            "query": "desert tortoise recovery plan",
            "intent": "regulatory",
            "sources": [_fr_weak_source()],
            "strategyMetadata": {
                "subjectChainGaps": [
                    "species-specific dossier missing",
                    "no recovery-plan full text verified",
                ],
            },
        },
    )

    session_state = _guided_session_state(
        workspace_registry=registry,
        search_session_id="ssn-subject-chain-gap",
    )
    assert session_state is not None
    trust_summary = session_state["trustSummary"]
    assert "subjectChainGaps" in trust_summary, (
        "Rebuilt trust summary must forward strategyMetadata.subjectChainGaps"
    )
    assert trust_summary["subjectChainGaps"] == [
        "species-specific dossier missing",
        "no recovery-plan full text verified",
    ]


def test_guided_session_state_omits_subject_chain_gaps_when_none() -> None:
    registry = WorkspaceRegistry()
    registry.save_result_set(
        source_tool="research",
        search_session_id="ssn-no-gaps",
        query="desert tortoise",
        payload={
            "query": "desert tortoise",
            "intent": "regulatory",
            "sources": [_fr_weak_source()],
            "strategyMetadata": {},
        },
    )

    session_state = _guided_session_state(
        workspace_registry=registry,
        search_session_id="ssn-no-gaps",
    )
    assert session_state is not None
    # Absence is the contract: callers rely on ``subjectChainGaps`` only
    # appearing when upstream recorded it.
    assert "subjectChainGaps" not in session_state["trustSummary"]
