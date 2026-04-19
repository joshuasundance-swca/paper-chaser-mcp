"""Payload efficiency tests for guided response contracts.

Phase 4 of the stress test remediation: reduce follow-up response bloat,
strip null fields from evidence records, and clean empty fields from source
records.
"""

from __future__ import annotations

import json

import pytest

from paper_chaser_mcp import server
from paper_chaser_mcp.agentic.workspace import WorkspaceRegistry
from paper_chaser_mcp.dispatch import _guided_source_record_from_paper
from paper_chaser_mcp.guided_semantic import build_evidence_records, strip_null_fields
from tests.helpers import _payload

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source(
    *,
    source_id: str = "src-1",
    source_alias: str = "source-1",
    topical_relevance: str = "on_topic",
    verification_status: str = "verified_metadata",
    is_primary_source: bool = False,
    title: str = "Test Paper",
    provider: str = "semantic_scholar",
    source_type: str = "scholarly_article",
    canonical_url: str | None = "https://example.com/paper",
    retrieved_url: str | None = None,
    citation: dict | None = None,
    date: str | None = None,
    note: str | None = None,
    **extra: object,
) -> dict:
    base: dict[str, object] = {
        "sourceId": source_id,
        "sourceAlias": source_alias,
        "topicalRelevance": topical_relevance,
        "verificationStatus": verification_status,
        "isPrimarySource": is_primary_source,
        "title": title,
        "provider": provider,
        "sourceType": source_type,
        "canonicalUrl": canonical_url,
        "retrievedUrl": retrieved_url,
        "citation": citation,
        "date": date,
        "note": note,
        "accessStatus": "access_unverified",
        "fullTextUrlFound": False,
        "abstractObserved": False,
        "openAccessRoute": None,
        "citationText": None,
        "whyClassifiedAsWeakMatch": None,
        "confidence": "high",
    }
    base.update(extra)
    return base


def _make_session_state(
    *,
    sources: list[dict],
    search_session_id: str = "sess-abc",
    evidence_gaps: list[str] | None = None,
) -> dict:
    """Build a minimal session state dict for follow-up testing."""
    return {
        "searchSessionId": search_session_id,
        "sources": sources,
        "unverifiedLeads": [],
        "verifiedFindings": [
            {"claim": s["title"], "sourceId": s["sourceId"]} for s in sources if s.get("topicalRelevance") == "on_topic"
        ],
        "evidenceGaps": evidence_gaps or [],
        "trustSummary": {"verifiedSourceCount": len(sources)},
        "coverage": {"providersAttempted": ["semantic_scholar"]},
        "failureSummary": {"outcome": "no_failure"},
        "resultMeaning": "Test session.",
        "nextActions": ["inspect_source"],
        "intent": "discovery",
        "query": "test query",
    }


# ---------------------------------------------------------------------------
# 1. Follow-up response source filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_follow_up_session_answer_only_includes_referenced_sources() -> None:
    """Follow-up from session state should only include sources matching selectedEvidenceIds."""
    from paper_chaser_mcp.dispatch import _answer_follow_up_from_session_state

    sources = [_make_source(source_id=f"src-{i}", source_alias=f"source-{i}", title=f"Paper {i}") for i in range(1, 6)]
    session_state = _make_session_state(sources=sources)

    # "how many verified" triggers the trust_summary facet
    result = await _answer_follow_up_from_session_state(
        question="How many verified sources were found?",
        session_state=session_state,
        response_mode="metadata",
    )

    assert result is not None, "Should have produced a session answer"
    response_sources = result.get("sources", [])

    # The response should NOT re-serialize all 5 session sources
    assert len(response_sources) < len(sources), (
        f"Follow-up included {len(response_sources)} sources but session had {len(sources)}; "
        "expected only referenced sources"
    )


@pytest.mark.asyncio
async def test_follow_up_overview_only_includes_selected_sources() -> None:
    """When a source overview question is asked, only the selected subset should appear."""
    from paper_chaser_mcp.dispatch import _answer_follow_up_from_session_state

    sources = [_make_source(source_id=f"src-{i}", source_alias=f"source-{i}", title=f"Paper {i}") for i in range(1, 8)]
    session_state = _make_session_state(sources=sources)

    # "what sources" triggers source_overview facet
    result = await _answer_follow_up_from_session_state(
        question="What sources were found in this session?",
        session_state=session_state,
        response_mode="metadata",
    )

    assert result is not None
    response_sources = result.get("sources", [])
    assert len(response_sources) < len(sources), (
        f"Source overview included all {len(sources)} sources; expected filtered subset"
    )


# ---------------------------------------------------------------------------
# 2. Null field stripping from evidence records
# ---------------------------------------------------------------------------


def test_evidence_records_strip_null_fields() -> None:
    """Evidence records should not contain None/null fields after serialization."""
    sources = [
        _make_source(
            source_id="src-1",
            canonical_url=None,
            retrieved_url=None,
            citation=None,
            date=None,
            note=None,
        ),
    ]

    evidence, leads = build_evidence_records(sources=sources, leads=[])

    for record in evidence + leads:
        null_keys = [k for k, v in record.items() if v is None]
        assert not null_keys, f"Evidence record contains null fields that should be stripped: {null_keys}"


def test_evidence_records_strip_empty_lists() -> None:
    """Evidence records should not contain empty list fields."""
    sources = [
        _make_source(source_id="src-1"),
    ]

    evidence, leads = build_evidence_records(sources=sources, leads=[])

    for record in evidence + leads:
        empty_list_keys = [k for k, v in record.items() if isinstance(v, list) and not v]
        assert not empty_list_keys, (
            f"Evidence record contains empty list fields that should be stripped: {empty_list_keys}"
        )


def test_strip_null_fields_utility() -> None:
    """strip_null_fields should remove None, empty strings, and empty lists."""
    record: dict[str, object] = {
        "evidenceId": "src-1",
        "title": "Real Title",
        "provider": None,
        "canonicalUrl": None,
        "retrievedUrl": "",
        "citation": None,
        "date": None,
        "whyNotVerified": None,
        "tags": [],
    }

    cleaned = strip_null_fields(record)

    assert "evidenceId" in cleaned
    assert "title" in cleaned
    assert "provider" not in cleaned, "None field should be stripped"
    assert "canonicalUrl" not in cleaned, "None field should be stripped"
    assert "retrievedUrl" not in cleaned, "Empty string should be stripped"
    assert "citation" not in cleaned, "None field should be stripped"
    assert "date" not in cleaned, "None field should be stripped"
    assert "whyNotVerified" not in cleaned, "None field should be stripped"
    assert "tags" not in cleaned, "Empty list should be stripped"


def test_strip_null_fields_preserves_false_and_zero() -> None:
    """strip_null_fields should keep False booleans and zero integers."""
    record = {
        "evidenceId": "src-1",
        "isPrimarySource": False,
        "count": 0,
        "name": None,
    }

    cleaned = strip_null_fields(record)

    assert cleaned["isPrimarySource"] is False
    assert cleaned["count"] == 0
    assert "name" not in cleaned


# ---------------------------------------------------------------------------
# 3. Source record cleanup
# ---------------------------------------------------------------------------


def test_source_record_strips_null_fields() -> None:
    """Source records from _guided_source_record_from_paper should not contain None/empty values."""
    paper = {
        "title": "Test Paper",
        "source": "semantic_scholar",
        "paperId": "abc123",
    }

    record = _guided_source_record_from_paper("test query", paper, index=1)

    null_keys = [k for k, v in record.items() if v is None]
    assert not null_keys, f"Source record contains null fields that should be stripped: {null_keys}"

    empty_str_keys = [k for k, v in record.items() if v == ""]
    assert not empty_str_keys, f"Source record contains empty string fields that should be stripped: {empty_str_keys}"


def test_source_record_preserves_meaningful_values() -> None:
    """Source records should keep all fields that have meaningful values."""
    paper = {
        "title": "Important Paper",
        "source": "openalex",
        "paperId": "def456",
        "url": "https://example.com/paper",
        "year": "2024",
        "venue": "Nature",
        "isPrimarySource": False,
        "fullTextUrlFound": True,
        "abstractObserved": True,
    }

    record = _guided_source_record_from_paper("test query", paper, index=1)

    assert record["title"] == "Important Paper"
    assert record["canonicalUrl"] == "https://example.com/paper"
    assert record["date"] == "2024"
    assert record["note"] == "Nature"
    assert record["fullTextUrlFound"] is True
    assert record["isPrimarySource"] is False  # False is meaningful, must be kept


def test_source_record_many_null_fields_minimal() -> None:
    """A paper with minimal data should produce a source record with no null/empty fields."""
    paper = {
        "title": "Minimal Paper",
        "paperId": "min-1",
    }

    record = _guided_source_record_from_paper("test query", paper, index=1)

    null_or_empty = {k: v for k, v in record.items() if v is None or v == ""}
    assert not null_or_empty, (
        f"Source record has null/empty fields that should be stripped: {list(null_or_empty.keys())}"
    )


# ---------------------------------------------------------------------------
# 4. Payload size regression coverage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_follow_up_insufficient_evidence_payload_is_compact(monkeypatch: pytest.MonkeyPatch) -> None:
    isolated_registry = WorkspaceRegistry()
    isolated_registry.save_result_set(
        source_tool="search_papers_smart",
        search_session_id="ssn-payload-follow-up",
        query="PFAS remediation in groundwater",
        payload={"sources": [{"sourceId": "paper-a", "title": "PFAS adsorption review"}]},
    )

    class _FakeRuntime:
        async def ask_result_set(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {
                "answerStatus": "insufficient_evidence",
                "answer": None,
                "structuredSources": [
                    {
                        "sourceId": f"source-{index}",
                        "title": f"PFAS paper {index}",
                        "provider": "openalex",
                        "sourceType": "scholarly_article",
                        "verificationStatus": "verified_metadata",
                        "accessStatus": "abstract_only",
                        "topicalRelevance": "on_topic",
                        "confidence": "medium",
                    }
                    for index in range(1, 6)
                ],
                "candidateLeads": [
                    {
                        "sourceId": f"lead-{index}",
                        "title": f"PFAS lead {index}",
                        "provider": "openalex",
                        "sourceType": "scholarly_article",
                        "verificationStatus": "unverified",
                        "accessStatus": "access_unverified",
                        "topicalRelevance": "weak_match",
                        "confidence": "low",
                    }
                    for index in range(1, 6)
                ],
                "evidence": [
                    {
                        "evidenceId": f"source-{index}",
                        "paper": {"paperId": f"source-{index}", "title": f"PFAS paper {index}"},
                        "excerpt": "Short evidence excerpt.",
                    }
                    for index in range(1, 4)
                ],
                "unsupportedAsks": ["Compare adsorption and membranes and name a winner."],
                "followUpQuestions": ["Would you like a narrower comparison by mechanism instead?"],
                "evidenceGaps": [
                    "Directly comparable PFAS treatment studies were too sparse for a winner-take-all answer."
                ],
                "coverageSummary": {
                    "providersAttempted": ["openalex"],
                    "providersSucceeded": ["openalex"],
                    "providersFailed": [],
                    "providersZeroResults": [],
                    "searchMode": "grounded_follow_up",
                },
            }

    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime())
    original_registry = server.workspace_registry
    server.workspace_registry = isolated_registry
    try:
        payload = _payload(
            await server.call_tool(
                "follow_up_research",
                {
                    "searchSessionId": "ssn-payload-follow-up",
                    "question": "Compare adsorption and membranes and name a winner.",
                },
            )
        )
    finally:
        server.workspace_registry = original_registry

    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    assert len(encoded) < 2200
    assert payload["sourcesSuppressed"] is True
    assert payload["legacyFieldsIncluded"] is False
    assert "evidence" not in payload
    assert "sources" not in payload
    assert "verifiedFindings" not in payload
    assert "unverifiedLeads" not in payload
    assert "trustSummary" not in payload
    assert "coverage" not in payload


@pytest.mark.asyncio
async def test_research_abstention_payload_is_compact(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeRuntime:
        def __init__(self) -> None:
            self._provider_bundle = None

        async def search_papers_smart(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {
                "searchSessionId": "ssn-payload-research",
                "strategyMetadata": {
                    "intent": "review",
                    "configuredSmartProvider": "openai",
                    "activeSmartProvider": "deterministic",
                },
                "structuredSources": [],
                "candidateLeads": [],
                "evidenceGaps": ["No trustworthy on-topic evidence was recovered for the query."],
                "coverageSummary": {
                    "providersAttempted": ["semantic_scholar", "openalex"],
                    "providersSucceeded": [],
                    "providersFailed": [],
                    "providersZeroResults": ["semantic_scholar", "openalex"],
                    "searchMode": "smart_literature_review",
                },
                "failureSummary": {
                    "outcome": "partial_success",
                    "fallbackAttempted": True,
                    "fallbackMode": "deterministic",
                },
                "clarification": None,
            }

    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime())
    payload = _payload(
        await server.call_tool(
            "research",
            {"query": "Do healing crystals reverse ocean acidification?"},
        )
    )

    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    assert len(encoded) < 3200
    assert payload["sourcesSuppressed"] is True
    assert payload["legacyFieldsIncluded"] is False
    assert payload["resultStatus"] == "abstained"
    assert payload.get("sources") in (None, [])
    assert "verifiedFindings" not in payload
    assert "unverifiedLeads" not in payload
    assert "trustSummary" not in payload


@pytest.mark.asyncio
async def test_inspect_source_payload_keeps_only_the_selected_source_context() -> None:
    isolated_registry = WorkspaceRegistry()
    isolated_registry.save_result_set(
        source_tool="research",
        search_session_id="ssn-compact-inspect",
        query="test query",
        payload={
            "query": "test query",
            "intent": "discovery",
            "coverageSummary": {
                "searchMode": "grounded_follow_up",
                "providersAttempted": ["semantic_scholar", "openalex", "crossref"],
                "providersSucceeded": ["semantic_scholar"],
                "providersFailed": ["openalex"],
                "totalSources": 3,
                "byAccessStatus": {"body_text_embedded": 1, "url_verified": 2},
            },
            "sources": [
                _make_source(source_id="src-1", title="Paper 1", note="keep"),
                _make_source(source_id="src-2", title="Paper 2", note="drop me"),
                _make_source(source_id="src-3", title="Paper 3", note="drop me too"),
            ],
            "unverifiedLeads": [
                _make_source(
                    source_id="lead-1",
                    source_alias="lead-1",
                    title="Lead 1",
                    topical_relevance="weak_match",
                    verification_status="verified_metadata",
                )
            ],
        },
    )

    original_registry = server.workspace_registry
    server.workspace_registry = isolated_registry
    try:
        payload = _payload(
            await server.call_tool(
                "inspect_source",
                {"searchSessionId": "ssn-compact-inspect", "sourceId": "src-1"},
            )
        )
    finally:
        server.workspace_registry = original_registry

    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    assert len(encoded) < 4500
    assert payload["source"]["sourceId"] == "src-1"
    assert [entry["evidenceId"] for entry in payload["evidence"]] == ["src-1"]
    assert "leads" not in payload
    assert set(payload["coverageSummary"]) <= {"searchMode", "totalSources", "byAccessStatus"}


# ---------------------------------------------------------------------------
# P1-1: Payload slimdown & response_mode for follow_up_research
# ---------------------------------------------------------------------------

from paper_chaser_mcp.dispatch import (  # noqa: E402
    _apply_follow_up_response_mode,
    _guided_finalize_response,
)
from paper_chaser_mcp.models.tools import FollowUpResearchArgs  # noqa: E402


def _grounded_follow_up_response() -> dict:
    """Representative grounded follow-up response used by P1-1 tests."""
    return {
        "searchSessionId": "ssn-1",
        "answerStatus": "answered",
        "answer": "Ragas supports automated RAG evaluation.",
        "evidence": [{"evidenceId": "ev-1", "claim": "x"}],
        "sources": [{"sourceId": "src-1", "title": "Ragas"}],
        "structuredSources": [
            {"sourceId": "src-1", "title": "Ragas"},
            {"sourceId": "src-2", "title": "Another"},
        ],
        "verifiedFindings": [{"claim": "Ragas supports RAG", "sourceId": "src-1"}],
        "unverifiedLeads": [{"sourceId": "lead-1"}],
        "coverage": {
            "totalSources": 2,
            "byAccessStatus": {"open_access": 1, "access_unverified": 1},
            "byProvider": {"semantic_scholar": 2},
            "byPrimarySource": {"false": 2},
        },
        "failureSummary": {"outcome": "answered"},
        "confidenceSignals": {"trustRevisionReason": "deterministic_synthesis_fallback"},
        "trustSummary": {"authoritative": 1},
        "telemetry": {"latencyMs": 42},
        "evidenceUsePlan": {"plan": "use ev-1"},
        "providerUsed": "deterministic",
        "degradationReason": "deterministic_synthesis_fallback",
        "classificationProvenance": {"source": "llm"},
        "degradedClassification": False,
        "topRecommendation": {"sourceId": "src-1", "reason": "best match"},
        "evidenceGaps": [],
        "nextActions": [],
        "resultMeaning": "answered_with_evidence",
        "resultState": "ok",
        "answerability": "full",
        "selectedEvidenceIds": ["ev-1"],
        "selectedLeadIds": [],
        "agentHints": {"next": "inspect_source"},
        "sessionResolution": {"status": "ok"},
        "inputNormalization": {"changed": False},
        "executionProvenance": {"path": "grounded"},
        "unsupportedAsks": [],
        "followUpQuestions": [],
        "abstentionDetails": None,
    }


def test_follow_up_default_is_compact() -> None:
    args = FollowUpResearchArgs.model_validate({"question": "q"})
    assert args.response_mode == "compact"
    assert args.include_legacy_fields is False


def test_follow_up_default_compact_drops_legacy_and_trust_fields() -> None:
    shaped = _apply_follow_up_response_mode(
        _grounded_follow_up_response(),
        response_mode="compact",
        include_legacy_fields=False,
    )
    assert "evidence" not in shaped
    assert "sources" not in shaped
    assert "verifiedFindings" not in shaped
    assert "unverifiedLeads" not in shaped
    for key in (
        "failureSummary",
        "confidenceSignals",
        "trustSummary",
        "telemetry",
        "evidenceUsePlan",
        "providerUsed",
        "degradationReason",
        "classificationProvenance",
        "degradedClassification",
    ):
        assert key not in shaped, key
    assert shaped["legacyFieldsIncluded"] is False
    assert shaped["responseMode"] == "compact"


def test_follow_up_default_compact_prefers_selected_ids_over_source_arrays() -> None:
    shaped = _apply_follow_up_response_mode(
        _grounded_follow_up_response(),
        response_mode="compact",
        include_legacy_fields=False,
    )
    assert "structuredSources" not in shaped
    assert "structuredSourceIds" not in shaped
    assert shaped["selectedEvidenceIds"] == ["ev-1"]


def test_follow_up_default_compact_uses_source_ids_when_no_selected_ids() -> None:
    payload = _grounded_follow_up_response()
    payload["selectedEvidenceIds"] = []
    shaped = _apply_follow_up_response_mode(
        payload,
        response_mode="compact",
        include_legacy_fields=False,
    )
    assert shaped["structuredSourceIds"] == ["src-1", "src-2"]


def test_follow_up_default_compact_collapses_coverage() -> None:
    shaped = _apply_follow_up_response_mode(
        _grounded_follow_up_response(),
        response_mode="compact",
        include_legacy_fields=False,
    )
    assert shaped["coverage"] == {
        "totalSources": 2,
        "byAccessStatus": {"open_access": 1, "access_unverified": 1},
    }


def test_follow_up_default_compact_omits_none_and_empty_fields() -> None:
    payload = _grounded_follow_up_response()
    payload["abstentionDetails"] = None
    payload["evidenceGaps"] = []
    payload["unsupportedAsks"] = []
    shaped = _apply_follow_up_response_mode(
        payload,
        response_mode="compact",
        include_legacy_fields=False,
    )
    assert "abstentionDetails" not in shaped
    assert "evidenceGaps" not in shaped
    assert "unsupportedAsks" not in shaped


def test_follow_up_default_compact_flags_source_suppression_without_legacy_lists() -> None:
    payload = _grounded_follow_up_response()
    payload["verifiedFindings"] = []
    payload["unverifiedLeads"] = []
    shaped = _apply_follow_up_response_mode(
        payload,
        response_mode="compact",
        include_legacy_fields=False,
    )
    assert shaped["sourcesSuppressed"] is True


def test_follow_up_compact_reports_sources_suppressed_int_count() -> None:
    shaped = _apply_follow_up_response_mode(
        _grounded_follow_up_response(),
        response_mode="compact",
        include_legacy_fields=False,
    )
    assert shaped["sourcesSuppressed"] == 2


def test_follow_up_compact_preserves_top_recommendation() -> None:
    shaped = _apply_follow_up_response_mode(
        _grounded_follow_up_response(),
        response_mode="compact",
        include_legacy_fields=False,
    )
    assert shaped["topRecommendation"] == {"sourceId": "src-1", "reason": "best match"}


def test_follow_up_include_legacy_fields_restores_legacy() -> None:
    shaped = _apply_follow_up_response_mode(
        _grounded_follow_up_response(),
        response_mode="compact",
        include_legacy_fields=True,
    )
    assert shaped["verifiedFindings"][0]["claim"] == "Ragas supports RAG"
    assert shaped["unverifiedLeads"][0]["sourceId"] == "lead-1"
    assert shaped["legacyFieldsIncluded"] is True


def test_follow_up_standard_restores_full_shape_except_empties() -> None:
    payload = _grounded_follow_up_response()
    payload["abstentionDetails"] = None
    shaped = _apply_follow_up_response_mode(
        payload,
        response_mode="standard",
        include_legacy_fields=False,
    )
    assert "verifiedFindings" in shaped
    assert "unverifiedLeads" in shaped
    assert "failureSummary" in shaped
    assert "confidenceSignals" in shaped
    assert "structuredSources" in shaped
    assert "structuredSourceIds" not in shaped
    assert "abstentionDetails" not in shaped
    assert shaped["responseMode"] == "standard"
    assert shaped["topRecommendation"]["sourceId"] == "src-1"


def test_follow_up_debug_includes_everything_unfiltered() -> None:
    payload = _grounded_follow_up_response()
    payload["abstentionDetails"] = None
    shaped = _apply_follow_up_response_mode(
        payload,
        response_mode="debug",
        include_legacy_fields=False,
    )
    assert shaped["abstentionDetails"] is None
    assert "verifiedFindings" in shaped
    assert "confidenceSignals" in shaped
    assert "failureSummary" in shaped
    assert shaped["responseMode"] == "debug"
    assert shaped["topRecommendation"]["sourceId"] == "src-1"


def test_finalize_routes_follow_up_to_compact_by_default() -> None:
    shaped = _guided_finalize_response(
        tool_name="follow_up_research",
        response=_grounded_follow_up_response(),
        response_mode="compact",
        include_legacy_fields=False,
    )
    assert shaped["responseMode"] == "compact"
    assert "evidence" not in shaped
    assert "sources" not in shaped
    assert "verifiedFindings" not in shaped
    assert "structuredSourceIds" not in shaped


def test_finalize_standard_preserves_legacy_and_trust() -> None:
    shaped = _guided_finalize_response(
        tool_name="follow_up_research",
        response=_grounded_follow_up_response(),
        response_mode="standard",
        include_legacy_fields=False,
    )
    assert shaped["responseMode"] == "standard"
    assert "verifiedFindings" in shaped
    assert "confidenceSignals" in shaped


def test_follow_up_insufficient_evidence_still_abstains_compactly() -> None:
    abstention = {
        "searchSessionId": "ssn-1",
        "answerStatus": "insufficient_evidence",
        "answer": None,
        "nextActions": ["broaden_search"],
        "evidenceGaps": ["No primary source covers this question."],
        "abstentionDetails": {"reason": "weak_pool"},
        "verifiedFindings": [{"claim": "should be dropped"}],
        "unverifiedLeads": [{"sourceId": "lead-x"}],
        "evidence": [{"evidenceId": "ev-1"}],
    }
    shaped = _guided_finalize_response(
        tool_name="follow_up_research",
        response=abstention,
        response_mode="compact",
        include_legacy_fields=False,
    )
    assert shaped["answerStatus"] == "insufficient_evidence"
    assert shaped.get("sourcesSuppressed") is True
    assert shaped.get("legacyFieldsIncluded") is False
    assert shaped["abstentionDetails"]["reason"] == "weak_pool"
    assert shaped["nextActions"] == ["broaden_search"]


def test_follow_up_args_rejects_invalid_response_mode() -> None:
    with pytest.raises(Exception):
        FollowUpResearchArgs.model_validate({"question": "q", "responseMode": "tiny"})


def test_follow_up_args_accepts_camelcase_aliases() -> None:
    args = FollowUpResearchArgs.model_validate(
        {"question": "q", "responseMode": "debug", "includeLegacyFields": True},
    )
    assert args.response_mode == "debug"
    assert args.include_legacy_fields is True
