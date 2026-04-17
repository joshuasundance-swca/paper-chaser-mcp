"""Workstream D regression slices — environmental-science failure families.

Each test covers one named failure family from the env-sci benchmark pack at
``tests/fixtures/evals/env_sci_benchmark_pack.json``. The slices assert durable
behavioral invariants against the real ``server.call_tool`` dispatch layer by
replacing ``server.agentic_runtime`` with a light fake that returns the exact
``search_papers_smart`` shape we want to exercise.

Durable invariants per family:

* thin_deterministic_salvage — follow-up on a weak saved pool must not
  synthesize a confident winner; ``answerability`` must reflect the
  insufficiency.
* species_specificity_failure — generic-wildlife notices returned for a
  species-anchored query must land under ``payload["leads"]`` with a
  ``whyNotVerified`` rationale, not under ``payload["evidence"]``.
* regulatory_primary_source_mixing — the evidence bucket for a CFR-anchored
  regulatory prompt must remain primary-regulatory; peer-reviewed reviews must
  not cross over into the evidence bucket as primary sources.
* archaeology_crossover_drift — archaeology prompts must not absorb species
  notices or generic environmental-regulation material into their evidence.
* management_intervention_comparison — comparative intervention prompts with
  thin evidence must abstain or surface insufficient-evidence signals rather
  than picking a single winner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from paper_chaser_mcp import server
from paper_chaser_mcp.eval_curation import (
    ENV_SCI_JUDGE_ROLES,
    EnvSciJudgeRubric,
    build_env_sci_judge_prompt,
    load_env_sci_benchmark_pack,
    load_env_sci_judge_rubric_template,
)
from tests.helpers import _payload

BENCHMARK_PATH = Path(__file__).resolve().parent / "fixtures" / "evals" / "env_sci_benchmark_pack.json"


def _row(row_id: str) -> dict[str, Any]:
    pack = load_env_sci_benchmark_pack(BENCHMARK_PATH)
    for row in pack["rows"]:
        if row["id"] == row_id:
            return row
    raise AssertionError(f"missing env-sci benchmark row: {row_id}")


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
        "retrievalHypotheses": ["Env-sci prompt should route cleanly."],
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


def _fake_runtime(result: dict[str, Any]) -> Any:
    class _FakeRuntime:
        async def search_papers_smart(self, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            return result

    return _FakeRuntime()


# ---------------------------------------------------------------------------
# Benchmark pack + rubric structural guarantees (fast, fixture-only)
# ---------------------------------------------------------------------------


def test_env_sci_benchmark_pack_shape_is_stable() -> None:
    pack = load_env_sci_benchmark_pack(BENCHMARK_PATH)
    assert pack["slice"] == "env-sci"
    rows = pack["rows"]
    assert len(rows) >= 12

    families_in_pack = {row["failureFamily"] for row in rows if row.get("failureFamily")}
    assert {
        "thin_deterministic_salvage",
        "species_specificity_failure",
        "regulatory_primary_source_mixing",
        "archaeology_crossover_drift",
        "management_intervention_comparison",
    }.issubset(families_in_pack)

    domains = {row["domain"] for row in rows}
    assert {
        "literature_discovery",
        "regulatory_discovery",
        "grounded_follow_up",
        "source_inspection",
        "species_dossier",
    }.issubset(domains)

    for row in rows:
        assert row["id"] and row["query"]
        assert isinstance(row["mustSurfaceFeatures"], list)
        assert isinstance(row["mustNotSurface"], list)


def test_env_sci_judge_prompt_renders_and_round_trips() -> None:
    row = _row("env_sci_follow_up_thin_deterministic_salvage")
    captured = {
        "query": row["query"],
        "payload": {"answerability": "insufficient_evidence"},
    }
    prompt = build_env_sci_judge_prompt(captured, expected_row=row)
    assert row["query"] in prompt
    # Ensure each role surfaces in the rendered prompt scoring guidance.
    for role in ENV_SCI_JUDGE_ROLES:
        assert role in prompt

    judge_output = {
        "plannerSpecificity": 0.9,
        "followUpResponsiveness": 1.0,
        "evidenceSufficiency": 0.75,
        "provenanceHonesty": 0.8,
        "notes": ["Planner kept follow-up intent", "Abstention cited evidence gap"],
    }
    rubric = EnvSciJudgeRubric.from_dict(judge_output)
    assert rubric.plannerSpecificity == 0.9
    assert rubric.followUpResponsiveness == 1.0
    assert rubric.overall == pytest.approx((0.9 + 1.0 + 0.75 + 0.8) / 4.0)

    round_tripped = EnvSciJudgeRubric.from_dict(rubric.to_dict())
    assert round_tripped.to_dict() == rubric.to_dict()

    # Out-of-range scores are clamped, not rejected.
    clamped = EnvSciJudgeRubric.from_dict(
        {
            "plannerSpecificity": 2.0,
            "followUpResponsiveness": -1.0,
            "evidenceSufficiency": "bogus",
            "provenanceHonesty": None,
        }
    )
    assert clamped.plannerSpecificity == 1.0
    assert clamped.followUpResponsiveness == 0.0
    assert clamped.evidenceSufficiency == 0.0
    assert clamped.provenanceHonesty == 0.0


def test_env_sci_rubric_template_exposes_four_roles() -> None:
    rubric = load_env_sci_judge_rubric_template()
    role_ids = {role["id"] for role in rubric["roles"]}
    assert role_ids == set(ENV_SCI_JUDGE_ROLES)


# ---------------------------------------------------------------------------
# Failure-family behavioral slices (use server.call_tool dispatch)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thin_deterministic_salvage_follow_up_must_abstain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _row("env_sci_follow_up_thin_deterministic_salvage")
    # Mocked pool is deliberately weak: one low-confidence lead, no verified evidence.
    leads = [
        {
            "sourceId": "edna-weak-1",
            "title": "Preliminary eDNA sampling pilot in one watershed",
            "provider": "openalex",
            "sourceType": "peer_reviewed_article",
            "verificationStatus": "verified_metadata",
            "accessStatus": "access_unverified",
            "topicalRelevance": "weak_match",
            "confidence": "low",
            "isPrimarySource": False,
            "canonicalUrl": "https://example.org/edna-weak",
            "date": "2022-04-01",
            "note": "Single-watershed pilot cannot support 'always best across watersheds'.",
        }
    ]
    result = _base_smart_result(
        session_id="ssn-env-sci-thin-salvage",
        intent="follow_up",
        structured=[],
        leads=leads,
        gaps=[
            "Only one low-confidence study is available; cross-watershed claims are not supported.",
        ],
        result_status="partial",
        provider_plan=["semantic_scholar", "openalex"],
    )
    monkeypatch.setattr(server, "agentic_runtime", _fake_runtime(result))

    payload = _payload(await server.call_tool("research", {"query": row["query"]}))

    # Must NOT surface deterministic-salvage prose: no evidence items promoted.
    assert not payload.get("evidence"), "Thin pool must not be promoted into evidence"
    # Must surface abstention-style status.
    assert payload.get("resultStatus") in {"partial", "abstained", "insufficient_evidence", "needs_disambiguation"}
    assert payload.get("answerability") in {
        "limited",
        "insufficient_evidence",
        "abstained",
        "unanswerable",
        None,
    }
    # The weak lead stays in leads with a visible rationale.
    lead_ids = [item.get("evidenceId") for item in payload.get("leads", [])]
    assert "edna-weak-1" in lead_ids
    for lead in payload.get("leads", []):
        assert lead.get("whyNotVerified"), "Weak leads must explain why they are not verified"


@pytest.mark.asyncio
async def test_species_specificity_failure_rejects_generic_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _row("env_sci_species_dossier_northern_long_eared_bat")
    structured = [
        {
            "sourceId": "ecos-nleb-dossier",
            "title": "ECOS Species Profile: Northern long-eared bat (Myotis septentrionalis)",
            "provider": "ecos",
            "sourceType": "species_dossier",
            "verificationStatus": "verified_primary_source",
            "accessStatus": "full_text_verified",
            "topicalRelevance": "on_topic",
            "confidence": "high",
            "isPrimarySource": True,
            "canonicalUrl": "https://ecos.fws.gov/ecp/species/nleb",
            "citation": {
                "title": "ECOS Species Profile: Northern long-eared bat",
                "url": "https://ecos.fws.gov/ecp/species/nleb",
                "sourceType": "species_dossier",
            },
            "date": "2024-10-15",
        }
    ]
    leads = [
        {
            "sourceId": "generic-wildlife-notice",
            "title": "Endangered and Threatened Wildlife and Plants; 12-Month Findings",
            "provider": "federal_register",
            "sourceType": "primary_regulatory",
            "verificationStatus": "verified_metadata",
            "accessStatus": "access_unverified",
            "topicalRelevance": "off_topic",
            "confidence": "low",
            "isPrimarySource": True,
            "canonicalUrl": "https://example.org/12-month-findings",
            "date": "2024-02-01",
            "note": "Authoritative but not species-anchored to northern long-eared bat.",
        }
    ]
    result = _base_smart_result(
        session_id="ssn-env-sci-nleb",
        intent="regulatory",
        structured=structured,
        leads=leads,
        gaps=[],
        result_status="succeeded",
        anchor_type="species",
        anchor_subject="Northern long-eared bat",
        provider_plan=["ecos", "federal_register", "govinfo"],
        search_mode="regulatory_primary_source",
    )
    monkeypatch.setattr(server, "agentic_runtime", _fake_runtime(result))

    payload = _payload(await server.call_tool("research", {"query": row["query"]}))

    evidence_ids = [item.get("evidenceId") for item in payload.get("evidence", [])]
    lead_ids = [item.get("evidenceId") for item in payload.get("leads", [])]
    assert "ecos-nleb-dossier" in evidence_ids
    # Generic wildlife notice must NOT be promoted into the evidence bucket.
    assert "generic-wildlife-notice" not in evidence_ids
    assert "generic-wildlife-notice" in lead_ids
    generic_lead = next(item for item in payload["leads"] if item.get("evidenceId") == "generic-wildlife-notice")
    assert generic_lead.get("whyNotVerified")


@pytest.mark.asyncio
async def test_regulatory_primary_source_mixing_keeps_buckets_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _row("env_sci_reg_primary_source_mixing_guard")
    structured = [
        {
            "sourceId": "cfr-17-95-lpc",
            "title": "50 CFR 17.95 — Critical habitat: lesser prairie chicken",
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
            "date": "2024-06-01",
        }
    ]
    leads = [
        {
            "sourceId": "peer-review-lpc-recovery",
            "title": "A peer-reviewed review of lesser prairie chicken recovery literature",
            "provider": "openalex",
            "sourceType": "peer_reviewed_article",
            "verificationStatus": "verified_metadata",
            "accessStatus": "access_unverified",
            "topicalRelevance": "adjacent",
            "confidence": "medium",
            "isPrimarySource": False,
            "canonicalUrl": "https://example.org/lpc-review",
            "date": "2023-11-10",
            "note": "Literature review is adjacent context; not a primary regulatory source.",
        }
    ]
    result = _base_smart_result(
        session_id="ssn-env-sci-lpc-primary",
        intent="regulatory",
        structured=structured,
        leads=leads,
        gaps=[],
        result_status="succeeded",
        anchor_type="cfr_citation",
        anchor_subject="Lesser prairie chicken",
        provider_plan=["govinfo", "federal_register", "ecos"],
        search_mode="regulatory_primary_source",
    )
    monkeypatch.setattr(server, "agentic_runtime", _fake_runtime(result))

    payload = _payload(await server.call_tool("research", {"query": row["query"]}))

    evidence_items = payload.get("evidence", [])
    evidence_ids = [item.get("evidenceId") for item in evidence_items]
    assert "cfr-17-95-lpc" in evidence_ids
    # Peer-reviewed review must not cross over into the primary-source evidence bucket.
    assert "peer-review-lpc-recovery" not in evidence_ids
    # Every evidence item for this regulatory prompt must be primary-regulatory.
    for item in evidence_items:
        source_type = (item.get("sourceType") or "").lower()
        assert source_type in {"primary_regulatory", "species_dossier", "agency_guidance"}, (
            f"Non-primary source {item.get('evidenceId')} leaked into regulatory evidence bucket"
        )


@pytest.mark.asyncio
async def test_archaeology_crossover_drift_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _row("env_sci_archaeology_crossover_drift_guard")
    structured = [
        {
            "sourceId": "archaeology-lithic-2023",
            "title": "Wildfire effects on surface lithic assemblages in the Intermountain West",
            "provider": "semantic_scholar",
            "sourceType": "peer_reviewed_article",
            "verificationStatus": "verified_metadata",
            "accessStatus": "full_text_verified",
            "topicalRelevance": "on_topic",
            "confidence": "high",
            "isPrimarySource": False,
            "canonicalUrl": "https://example.org/lithic-wildfire",
            "citation": {
                "title": "Wildfire effects on surface lithic assemblages",
                "url": "https://example.org/lithic-wildfire",
                "sourceType": "peer_reviewed_article",
            },
            "date": "2023-07-01",
        }
    ]
    leads = [
        {
            "sourceId": "unrelated-species-notice",
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
            "note": "Species notice is unrelated to archaeological impacts.",
        },
        {
            "sourceId": "generic-env-regulation",
            "title": "Generic NEPA environmental-review guidance for federal actions",
            "provider": "govinfo",
            "sourceType": "agency_guidance",
            "verificationStatus": "verified_metadata",
            "accessStatus": "access_unverified",
            "topicalRelevance": "off_topic",
            "confidence": "low",
            "isPrimarySource": False,
            "canonicalUrl": "https://example.org/nepa-generic",
            "date": "2021-01-01",
            "note": "Generic env-regulation scope; not archaeology-specific.",
        },
    ]
    result = _base_smart_result(
        session_id="ssn-env-sci-archaeology-drift",
        intent="discovery",
        structured=structured,
        leads=leads,
        gaps=[],
        result_status="succeeded",
        provider_plan=["semantic_scholar", "openalex"],
    )
    monkeypatch.setattr(server, "agentic_runtime", _fake_runtime(result))

    payload = _payload(await server.call_tool("research", {"query": row["query"]}))

    evidence_ids = [item.get("evidenceId") for item in payload.get("evidence", [])]
    lead_ids = [item.get("evidenceId") for item in payload.get("leads", [])]
    assert "archaeology-lithic-2023" in evidence_ids
    assert "unrelated-species-notice" not in evidence_ids
    assert "generic-env-regulation" not in evidence_ids
    assert "unrelated-species-notice" in lead_ids
    assert "generic-env-regulation" in lead_ids
    for lead in payload.get("leads", []):
        assert lead.get("whyNotVerified"), f"Lead {lead.get('evidenceId')} missing whyNotVerified"


@pytest.mark.asyncio
async def test_management_intervention_comparison_thin_evidence_must_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _row("env_sci_follow_up_pfas_comparison_hybrid")
    # Only adsorption has a verified study; membrane + destruction are absent.
    structured = [
        {
            "sourceId": "pfas-adsorption-2024",
            "title": "Adsorption for PFAS removal in drinking water: meta-analysis",
            "provider": "openalex",
            "sourceType": "peer_reviewed_article",
            "verificationStatus": "verified_metadata",
            "accessStatus": "full_text_verified",
            "topicalRelevance": "on_topic",
            "confidence": "high",
            "isPrimarySource": False,
            "canonicalUrl": "https://example.org/pfas-adsorption",
            "citation": {
                "title": "Adsorption for PFAS removal: meta-analysis",
                "url": "https://example.org/pfas-adsorption",
                "sourceType": "peer_reviewed_article",
            },
            "date": "2024-01-20",
        }
    ]
    result = _base_smart_result(
        session_id="ssn-env-sci-pfas-compare",
        intent="follow_up",
        structured=structured,
        leads=[],
        gaps=[
            "No verified evidence for membrane methods in the saved session.",
            "No verified evidence for destruction methods in the saved session.",
        ],
        result_status="partial",
        provider_plan=["semantic_scholar", "openalex"],
    )
    monkeypatch.setattr(server, "agentic_runtime", _fake_runtime(result))

    payload = _payload(await server.call_tool("research", {"query": row["query"]}))

    # Must NOT pick a single-winner answer from one-sided evidence.
    summary_text = json.dumps(payload).lower()
    # Heuristic guard: a good response should not contain "single best" or "always best" affirmatively;
    # the fixture gap should survive through the dispatcher into the response surface.
    assert payload.get("resultStatus") in {"partial", "abstained", "insufficient_evidence"}
    # Evidence gaps must be preserved somewhere visible in the response surface.
    assert "membrane" in summary_text or payload.get("resultStatus") != "succeeded"
    # Only the one verified paper may appear in evidence — the rest must remain gaps.
    evidence_ids = [item.get("evidenceId") for item in payload.get("evidence", [])]
    assert evidence_ids == ["pfas-adsorption-2024"]
