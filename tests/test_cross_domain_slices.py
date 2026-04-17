"""Behavioral regression harness for Workstream G cross-domain slices.

Each test exercises a durable end-to-end behavior for one domain slice using
the `_FakeRuntime` mock pattern established in ``tests/test_dispatch.py``.
The slice fixtures themselves live in ``tests/fixtures/ux_prompt_corpus.json``
under the ``cross_domain_slices`` key; the tests below mirror the named
scenarios in ``docs/cross-domain-remediation-plan.md`` Workstream G.

Guiding invariants enforced here:

* Off-topic ``candidateLeads`` must surface under ``payload["leads"]`` with
  a ``whyNotVerified`` reason, never silently re-enter ``payload["evidence"]``.
* Regulatory / natural-science positive controls must preserve intent and
  provider-plan metadata through ``routingSummary``.
* Known-item exact-title recoveries must report a canonical resolution
  distinguishable from discovery or abstention.
* Anchored prompts must keep routing metadata consistent with the mocked
  anchor (e.g. species / CFR citation).
"""

from __future__ import annotations

from typing import Any

import pytest

from paper_chaser_mcp import server
from tests.helpers import RecordingSemanticClient, _payload


def _base_smart_result(
    *,
    session_id: str,
    intent: str,
    structured: list[dict[str, Any]],
    leads: list[dict[str, Any]],
    gaps: list[str],
    result_status: str = "succeeded",
    anchor_type: str | None = None,
    anchor_subject: str | None = None,
    provider_plan: list[str] | None = None,
    search_mode: str = "smart_literature_review",
) -> dict[str, Any]:
    strategy: dict[str, Any] = {
        "intent": intent,
        "querySpecificity": "medium",
        "ambiguityLevel": "low",
        "retrievalHypotheses": [
            "Topic should be recoverable from peer-reviewed evidence.",
        ],
    }
    if anchor_type:
        strategy["anchorType"] = anchor_type
    if anchor_subject:
        strategy["anchoredSubject"] = anchor_subject
    if provider_plan:
        strategy["providerPlan"] = provider_plan
        strategy["providersUsed"] = list(provider_plan)
        strategy["routingConfidence"] = "high"
    return {
        "searchSessionId": session_id,
        "strategyMetadata": strategy,
        "structuredSources": structured,
        "candidateLeads": leads,
        "evidenceGaps": gaps,
        "coverageSummary": {
            "providersAttempted": provider_plan or ["semantic_scholar", "openalex"],
            "providersSucceeded": provider_plan or ["semantic_scholar", "openalex"],
            "providersZeroResults": [],
            "likelyCompleteness": "partial",
            "searchMode": search_mode,
        },
        "failureSummary": None,
        "clarification": None,
        "resultStatus": result_status,
    }


def _make_fake_runtime(result: dict[str, Any]) -> Any:
    class _FakeRuntime:
        async def search_papers_smart(self, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            return result

    return _FakeRuntime()


def _routing_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the first available routing-metadata map.

    Accepts both the new top-level ``routingSummary`` contract and the legacy
    ``strategyMetadata`` surface so the harness is resilient while sibling
    dispatch-layer changes land.
    """
    summary = payload.get("routingSummary")
    if isinstance(summary, dict) and summary:
        return summary
    strategy = payload.get("strategyMetadata")
    if isinstance(strategy, dict):
        return strategy
    return {}


@pytest.mark.asyncio
async def test_natural_science_pfas_positive_control_is_grounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PFAS drinking-water positive control: literature must ground the answer."""

    structured = [
        {
            "sourceId": "pfas-review-2024",
            "title": "Per- and polyfluoroalkyl substances in U.S. drinking water: a synthesis",
            "provider": "openalex",
            "sourceType": "peer_reviewed_article",
            "verificationStatus": "verified_metadata",
            "accessStatus": "full_text_verified",
            "topicalRelevance": "on_topic",
            "confidence": "high",
            "isPrimarySource": False,
            "canonicalUrl": "https://example.org/pfas-review",
            "citation": {
                "title": "PFAS in U.S. drinking water: a synthesis",
                "url": "https://example.org/pfas-review",
                "sourceType": "peer_reviewed_article",
            },
            "date": "2024-05-10",
        }
    ]
    result = _base_smart_result(
        session_id="ssn-cross-natural-pfas",
        intent="discovery",
        structured=structured,
        leads=[],
        gaps=[],
        result_status="succeeded",
        provider_plan=["semantic_scholar", "openalex", "core"],
        search_mode="smart_literature_review",
    )
    monkeypatch.setattr(server, "agentic_runtime", _make_fake_runtime(result))

    query = "What does the literature say about PFAS bioaccumulation in freshwater fish across U.S. watersheds?"
    payload = _payload(
        await server.call_tool(
            "research",
            {"query": query},
        )
    )

    assert payload["resultStatus"] in {"succeeded", "partial"}
    assert payload.get("answerability") in {"grounded", "limited"}
    evidence_ids = [item.get("evidenceId") for item in payload.get("evidence", [])]
    assert "pfas-review-2024" in evidence_ids
    assert not payload.get("leads"), "Positive control must not produce lead-only output"


@pytest.mark.asyncio
async def test_human_dimensions_ambiguous_prompt_preserves_routing_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ambiguous human-dimensions prompt must keep retrieval hypotheses visible."""

    result = _base_smart_result(
        session_id="ssn-cross-human-dim-ambig",
        intent="discovery",
        structured=[],
        leads=[],
        gaps=["Query is ambiguous — consider narrowing geography or intervention."],
        result_status="partial",
        provider_plan=["openalex", "semantic_scholar"],
    )
    # Mark the prompt as high-ambiguity so dispatch can surface it downstream.
    result["strategyMetadata"]["ambiguityLevel"] = "high"
    result["strategyMetadata"]["querySpecificity"] = "low"
    result["strategyMetadata"]["retrievalHypotheses"] = [
        "Prompt may refer to federal co-stewardship, tribal co-management, or research governance.",
        "Intent could be policy review or empirical-outcomes discovery.",
    ]

    monkeypatch.setattr(server, "agentic_runtime", _make_fake_runtime(result))

    payload = _payload(
        await server.call_tool(
            "research",
            {"query": "How effective is tribal consultation?"},
        )
    )

    summary = _routing_summary(payload)
    assert summary, "Routing metadata must be surfaced for ambiguous prompts"
    hypotheses = summary.get("retrievalHypotheses") or []
    assert hypotheses, "Ambiguous prompts must expose retrieval hypotheses"
    assert len(hypotheses) >= 2
    assert payload["resultStatus"] in {"partial", "needs_disambiguation", "abstained", "succeeded"}


@pytest.mark.asyncio
async def test_heritage_archaeology_off_topic_noise_stays_in_leads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unrelated Medicare-billing noise must land in leads, never in evidence."""

    structured = [
        {
            "sourceId": "wildfire-archaeology-2023",
            "title": "Wildfire effects on surface lithic assemblages in the Intermountain West",
            "provider": "semantic_scholar",
            "sourceType": "peer_reviewed_article",
            "verificationStatus": "verified_metadata",
            "accessStatus": "full_text_verified",
            "topicalRelevance": "on_topic",
            "confidence": "high",
            "isPrimarySource": False,
            "canonicalUrl": "https://example.org/wildfire-archaeology",
            "citation": {
                "title": "Wildfire effects on surface lithic assemblages",
                "url": "https://example.org/wildfire-archaeology",
                "sourceType": "peer_reviewed_article",
            },
            "date": "2023-08-01",
        }
    ]
    leads = [
        {
            "sourceId": "medicare-billing-unrelated",
            "title": "Medicare reimbursement schedule update for outpatient services",
            "provider": "openalex",
            "sourceType": "policy_document",
            "verificationStatus": "verified_metadata",
            "accessStatus": "access_unverified",
            "topicalRelevance": "off_topic",
            "confidence": "low",
            "isPrimarySource": False,
            "canonicalUrl": "https://example.org/medicare",
            "date": "2024-01-01",
            "note": "Filtered because topic is unrelated to wildfire effects on archaeology.",
        },
        {
            "sourceId": "desert-species-notice",
            "title": "Federal Register notice: unrelated species critical habitat",
            "provider": "federal_register",
            "sourceType": "primary_regulatory",
            "verificationStatus": "verified_metadata",
            "accessStatus": "access_unverified",
            "topicalRelevance": "off_topic",
            "confidence": "low",
            "isPrimarySource": True,
            "canonicalUrl": "https://example.org/species-notice",
            "date": "2024-03-01",
            "note": "Filtered because species notice is unrelated to archaeological impacts.",
        },
    ]
    result = _base_smart_result(
        session_id="ssn-cross-heritage-archaeology",
        intent="discovery",
        structured=structured,
        leads=leads,
        gaps=[],
        result_status="succeeded",
        provider_plan=["semantic_scholar", "openalex"],
    )
    monkeypatch.setattr(server, "agentic_runtime", _make_fake_runtime(result))

    payload = _payload(
        await server.call_tool(
            "research",
            {
                "query": (
                    "Post-wildfire effects on lithic and ceramic surface archaeological sites in the Intermountain West"
                )
            },
        )
    )

    evidence_ids = [item.get("evidenceId") for item in payload.get("evidence", [])]
    lead_ids = [item.get("evidenceId") for item in payload.get("leads", [])]
    assert "wildfire-archaeology-2023" in evidence_ids
    assert "medicare-billing-unrelated" not in evidence_ids
    assert "desert-species-notice" not in evidence_ids
    assert "medicare-billing-unrelated" in lead_ids
    assert "desert-species-notice" in lead_ids
    for lead in payload.get("leads", []):
        assert lead.get("whyNotVerified"), f"Lead {lead.get('evidenceId')} missing whyNotVerified"


@pytest.mark.asyncio
async def test_regulation_cfr_positive_control_preserves_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CFR-anchored query must keep the anchored subject visible in routing metadata."""

    structured = [
        {
            "sourceId": "50-cfr-17-95",
            "title": "50 CFR 17.95 — Critical habitat designation: lesser prairie chicken",
            "provider": "govinfo",
            "sourceType": "primary_regulatory",
            "verificationStatus": "verified_primary_source",
            "accessStatus": "full_text_verified",
            "topicalRelevance": "on_topic",
            "confidence": "high",
            "isPrimarySource": True,
            "canonicalUrl": "https://www.govinfo.gov/app/collection/cfr",
            "citation": {
                "title": "50 CFR 17.95",
                "url": "https://www.govinfo.gov/app/collection/cfr",
                "sourceType": "primary_regulatory",
            },
            "date": "2024-11-01",
        }
    ]
    result = _base_smart_result(
        session_id="ssn-cross-reg-lpc",
        intent="regulatory",
        structured=structured,
        leads=[],
        gaps=[],
        result_status="succeeded",
        anchor_type="cfr_citation",
        anchor_subject="Lesser prairie chicken",
        provider_plan=["govinfo", "federal_register", "ecos"],
        search_mode="regulatory_primary_source",
    )
    monkeypatch.setattr(server, "agentic_runtime", _make_fake_runtime(result))

    payload = _payload(
        await server.call_tool(
            "research",
            {"query": "Trace critical habitat designations for the lesser prairie chicken under 50 CFR 17.95"},
        )
    )

    summary = _routing_summary(payload)
    assert summary.get("intent") == "regulatory"
    anchor_value = summary.get("anchorValue") or summary.get("anchoredSubject")
    assert anchor_value == "Lesser prairie chicken"
    plan = summary.get("providerPlan")
    assert plan and "govinfo" in plan
    coverage = payload.get("coverageSummary") or {}
    assert coverage.get("searchMode") == "regulatory_primary_source"
    evidence_ids = [item.get("evidenceId") for item in payload.get("evidence", [])]
    assert "50-cfr-17-95" in evidence_ids


@pytest.mark.asyncio
async def test_known_item_recovery_vaswani_reports_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact arXiv identifier for 'Attention Is All You Need' must resolve canonically."""

    semantic = RecordingSemanticClient()
    monkeypatch.setattr(server, "client", semantic)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_openalex", False)

    payload = _payload(await server.call_tool("resolve_reference", {"reference": "arXiv:1706.03762"}))

    assert payload.get("status") == "resolved"
    assert payload.get("resolutionType") == "paper_identifier"
    # Known-item recovery must not drop into discovery or abstention paths.
    assert payload.get("status") not in {"abstained", "insufficient_evidence", "needs_disambiguation"}
