from __future__ import annotations

import os

import pytest

from paper_chaser_mcp import server
from tests.helpers import _payload


def _has_live_smart_key() -> bool:
    return any(
        os.getenv(name)
        for name in (
            "OPENAI_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "MISTRAL_API_KEY",
            "NVIDIA_API_KEY",
            "HUGGINGFACE_API_KEY",
        )
    )


pytestmark = pytest.mark.skipif(
    not _has_live_smart_key(),
    reason="Requires a configured live smart-provider API key.",
)


@pytest.mark.asyncio
async def test_live_guided_research_round_trip() -> None:
    payload = _payload(
        await server.call_tool(
            "research",
            {"query": "retrieval-augmented generation for coding agents", "limit": 5},
        )
    )

    assert payload["status"] in {"succeeded", "partial", "abstained", "needs_disambiguation"}
    assert payload["resultStatus"] in {"succeeded", "partial", "abstained", "needs_disambiguation"}
    assert "executionProvenance" in payload
    assert payload.get("searchSessionId") is not None


@pytest.mark.asyncio
async def test_live_resolve_reference_known_doi() -> None:
    payload = _payload(await server.call_tool("resolve_reference", {"reference": "10.48550/arXiv.1706.03762"}))

    assert payload["status"] in {"resolved", "multiple_candidates"}
    assert payload["resolutionConfidence"] in {"high", "medium", "low"}
    assert payload.get("bestMatch") or payload.get("alternatives")
