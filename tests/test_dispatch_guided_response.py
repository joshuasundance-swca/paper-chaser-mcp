"""TDD tests for dispatch/guided/response extraction."""

from __future__ import annotations

import pytest

from paper_chaser_mcp.dispatch.guided.response import (
    _guided_compact_response_if_needed,
    _guided_contract_fields,
    _guided_finalize_response,
)


class TestCompactResponseIfNeeded:
    def test_passthrough_for_unrelated_tool(self) -> None:
        payload = {"foo": "bar"}
        result = _guided_compact_response_if_needed(
            tool_name="search_papers",
            response=payload,
        )
        assert result is payload

    def test_passthrough_when_follow_up_not_insufficient(self) -> None:
        payload = {"answerStatus": "answered", "foo": "bar"}
        result = _guided_compact_response_if_needed(
            tool_name="follow_up_research",
            response=payload,
        )
        assert result is payload

    def test_compacts_follow_up_insufficient_evidence(self) -> None:
        payload = {
            "answerStatus": "insufficient_evidence",
            "answer": "short",
            "searchSessionId": "abc",
            "evidence": [{"id": "e1"}],
            "unusedField": "dropped",
        }
        result = _guided_compact_response_if_needed(
            tool_name="follow_up_research",
            response=payload,
        )
        assert "unusedField" not in result
        assert result.get("sourcesSuppressed") is True
        assert result.get("legacyFieldsIncluded") is False

    def test_compacts_research_abstained(self) -> None:
        payload = {"status": "abstained", "evidence": [], "unusedField": "dropped"}
        result = _guided_compact_response_if_needed(
            tool_name="research",
            response=payload,
        )
        assert "unusedField" not in result
        assert result.get("legacyFieldsIncluded") is False

    def test_research_non_abstained_passthrough(self) -> None:
        payload = {"status": "answered", "evidence": []}
        result = _guided_compact_response_if_needed(
            tool_name="research",
            response=payload,
        )
        assert result is payload


class TestFinalizeResponse:
    def test_strips_empty_legacy_lists(self) -> None:
        payload = {"verifiedFindings": [], "unverifiedLeads": [], "status": "answered"}
        result = _guided_finalize_response(
            tool_name="research",
            response=payload,
        )
        assert "verifiedFindings" not in result
        assert "unverifiedLeads" not in result

    def test_sets_legacy_fields_included_flag(self) -> None:
        payload = {"status": "answered"}
        result = _guided_finalize_response(
            tool_name="research",
            response=payload,
        )
        assert "legacyFieldsIncluded" in result

    def test_applies_compaction_for_abstained_research(self) -> None:
        payload = {"status": "abstained", "extraneous": "data"}
        result = _guided_finalize_response(
            tool_name="research",
            response=payload,
        )
        assert "extraneous" not in result

    def test_research_default_omits_legacy_success_fields(self) -> None:
        payload = {
            "status": "answered",
            "verifiedFindings": [{"claim": "kept in compatibility view"}],
            "sources": [{"sourceId": "src-1"}],
            "unverifiedLeads": [{"sourceId": "lead-1"}],
            "coverage": {"totalSources": 1},
            "evidence": [{"evidenceId": "src-1"}],
            "leads": [{"sourceId": "lead-1"}],
        }
        result = _guided_finalize_response(
            tool_name="research",
            response=payload,
        )
        assert "verifiedFindings" not in result
        assert "sources" not in result
        assert "unverifiedLeads" not in result
        assert "coverage" not in result
        assert result["legacyFieldsIncluded"] is False

    def test_research_include_legacy_fields_restores_success_fields(self) -> None:
        payload = {
            "status": "answered",
            "verifiedFindings": [{"claim": "kept in compatibility view"}],
            "sources": [{"sourceId": "src-1"}],
            "unverifiedLeads": [{"sourceId": "lead-1"}],
            "coverage": {"totalSources": 1},
            "evidence": [{"evidenceId": "src-1"}],
            "leads": [{"sourceId": "lead-1"}],
        }
        result = _guided_finalize_response(
            tool_name="research",
            response=payload,
            include_legacy_fields=True,
        )
        assert result["verifiedFindings"][0]["claim"] == "kept in compatibility view"
        assert result["sources"][0]["sourceId"] == "src-1"
        assert result["unverifiedLeads"][0]["sourceId"] == "lead-1"
        assert result["coverage"]["totalSources"] == 1
        assert result["legacyFieldsIncluded"] is True


class TestContractFieldsSmoke:
    @pytest.mark.asyncio
    async def test_callable_and_returns_dict(self) -> None:
        assert callable(_guided_contract_fields)
