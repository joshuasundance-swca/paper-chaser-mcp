from __future__ import annotations

import pytest

from paper_chaser_mcp import server
from tests.helpers import _payload


@pytest.mark.asyncio
async def test_guided_research_marks_deterministic_fallback_in_execution_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeBundle:
        def selection_metadata(self) -> dict[str, object]:
            return {
                "configuredSmartProvider": "openai",
                "activeSmartProvider": "deterministic",
            }

    class _FakeRuntime:
        _provider_bundle = _FakeBundle()

        async def search_papers_smart(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {
                "searchSessionId": "ssn-det-research",
                "strategyMetadata": {
                    "intent": "review",
                    "configuredSmartProvider": "openai",
                    "activeSmartProvider": "deterministic",
                },
                "structuredSources": [
                    {
                        "sourceId": "paper-a",
                        "title": "A deterministic fallback source",
                        "provider": "semantic_scholar",
                        "sourceType": "scholarly_article",
                        "verificationStatus": "verified_metadata",
                        "accessStatus": "abstract_only",
                        "topicalRelevance": "on_topic",
                        "confidence": "medium",
                    }
                ],
                "candidateLeads": [],
                "evidenceGaps": ["deterministic_synthesis_fallback: live model routing was unavailable."],
                "coverageSummary": {
                    "providersAttempted": ["semantic_scholar"],
                    "providersSucceeded": ["semantic_scholar"],
                    "providersFailed": [],
                    "providersZeroResults": [],
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

    payload = _payload(await server.call_tool("research", {"query": "retrieval evaluation methods"}))

    assert payload["status"] in {"succeeded", "partial"}
    assert payload["executionProvenance"]["configuredSmartProvider"] == "openai"
    assert payload["executionProvenance"]["activeSmartProvider"] == "deterministic"
    assert payload["executionProvenance"]["deterministicFallbackUsed"] is True


@pytest.mark.asyncio
async def test_get_runtime_status_reports_deterministic_fallback_provider_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeBundle:
        def selection_metadata(self) -> dict[str, object]:
            return {
                "configuredSmartProvider": "openai",
                "activeSmartProvider": "deterministic",
                "plannerModel": "openai:gpt-5-mini",
                "plannerModelSource": "deterministic",
                "synthesisModel": "openai:gpt-5-mini",
                "synthesisModelSource": "deterministic",
            }

    class _FakeRuntime:
        _provider_bundle = _FakeBundle()

        @staticmethod
        def smart_provider_diagnostics() -> tuple[dict[str, bool], list[str]]:
            return (
                {
                    "openai": False,
                    "azure-openai": False,
                    "anthropic": False,
                    "nvidia": False,
                    "google": False,
                    "mistral": False,
                    "huggingface": False,
                    "openrouter": False,
                },
                ["openai", "azure-openai", "anthropic", "nvidia", "google", "mistral", "huggingface", "openrouter"],
            )

    monkeypatch.setattr(server, "agentic_runtime", _FakeRuntime())

    payload = _payload(await server.call_tool("get_runtime_status", {}))

    assert payload["runtimeSummary"]["configuredSmartProvider"] == "openai"
    assert payload["runtimeSummary"]["activeSmartProvider"] == "deterministic"
