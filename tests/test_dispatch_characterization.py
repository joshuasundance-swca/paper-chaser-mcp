"""Guided-tool characterization harness.

Pins the *structural* shape of dispatch responses for the five guided tools
plus a pair of expert-profile representatives so later structural refactors
cannot silently drop, rename, or retype top-level contract fields.

Design
------

* Assertions are **structural** (key presence, types, enum membership,
  order-insensitive list keys, presence of next-step hints), not byte-equal.
* One normalized JSONL fixture per tool lives under
  ``tests/fixtures/characterization/`` as a human-readable contract
  reference. Volatile fields (session ids, UUIDs, timestamps) are replaced
  with canonical placeholders by :func:`_normalize_payload`.
* Invoked tool paths go through ``server.call_tool`` (the public thin
  wrapper around ``paper_chaser_mcp.dispatch.dispatch_tool``), matching the
  dominant test idiom in this repository and reusing real runtime wiring.
* Provider nondeterminism is neutralized with the
  ``DeterministicProviderBundle`` (via ``_deterministic_runtime``) and
  ``Recording*`` client stubs, so no network or LLM calls are made.
* Set ``PAPER_CHASER_CHAR_REGEN=1`` to regenerate every fixture in one run.
  The harness also auto-writes a fixture the first time it is missing.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from paper_chaser_mcp import server
from tests.helpers import (
    RecordingOpenAlexClient,
    RecordingSemanticClient,
    _payload,
)
from tests.test_smart_tools import _deterministic_runtime

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "characterization"
_REGEN = os.environ.get("PAPER_CHASER_CHAR_REGEN") == "1"

# ---------------------------------------------------------------------------
# Normalization + fixture I/O
# ---------------------------------------------------------------------------

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_SESSION_ID_RE = re.compile(r"^(?:ssn|sess|smart|wss|trc)[_-][A-Za-z0-9_-]+$")
_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?$",
)

_SESSION_ID_KEYS = frozenset(
    {
        "searchSessionId",
        "sessionId",
        "workspaceSessionId",
        "parentSessionId",
        "sourceSessionId",
    },
)
_TIMESTAMP_KEYS = frozenset(
    {"timestamp", "createdAt", "updatedAt", "expiresAt", "savedAt", "recordedAt"},
)
_UUID_KEYS = frozenset({"traceId", "requestId", "runId", "batchId"})


def _normalize_payload(payload: Any) -> Any:
    """Replace volatile ids and timestamps with canonical placeholders.

    Also used at fixture-generation time so fixtures and live normalized
    output use the same convention.
    """

    if isinstance(payload, dict):
        result: dict[str, Any] = {}
        for key, value in payload.items():
            if key in _SESSION_ID_KEYS and isinstance(value, str) and value:
                result[key] = "<session-id>"
            elif key in _TIMESTAMP_KEYS and isinstance(value, str) and value:
                result[key] = "<timestamp>"
            elif key in _UUID_KEYS and isinstance(value, str) and value:
                result[key] = "<uuid>"
            else:
                result[key] = _normalize_payload(value)
        return result
    if isinstance(payload, list):
        return [_normalize_payload(item) for item in payload]
    if isinstance(payload, str):
        if _UUID_RE.match(payload):
            return "<uuid>"
        if _SESSION_ID_RE.match(payload):
            return "<session-id>"
        if _TIMESTAMP_RE.match(payload):
            return "<timestamp>"
    return payload


def _fixture_path(name: str) -> Path:
    return FIXTURES_DIR / f"{name}.jsonl"


def _write_fixture(name: str, normalized: dict[str, Any]) -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    path = _fixture_path(name)
    path.write_text(
        json.dumps(normalized, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_fixture(name: str) -> dict[str, Any] | None:
    path = _fixture_path(name)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        line = handle.readline()
    if not line.strip():
        return None
    parsed = json.loads(line)
    if not isinstance(parsed, dict):
        raise TypeError(f"fixture {name!r} is not a JSON object")
    return parsed


def _assert_structural_contract(name: str, live: dict[str, Any]) -> dict[str, Any]:
    """Pin the structural contract against the fixture.

    - All top-level keys present in the fixture must remain present live.
    - Top-level types must match (lists stay lists, dicts stay dicts, etc.).
    - If no fixture exists yet, generate one (this is the bootstrap path and
      also what ``PAPER_CHASER_CHAR_REGEN=1`` forces).
    """

    normalized = _normalize_payload(live)
    existing = _read_fixture(name)
    if _REGEN or existing is None:
        _write_fixture(name, normalized)
        existing = _read_fixture(name)
    assert existing is not None, f"fixture {name!r} could not be loaded"

    missing = sorted(set(existing) - set(normalized))
    assert not missing, f"{name}: pinned top-level keys disappeared from response: {missing}"

    for key, reference_value in existing.items():
        live_value = normalized[key]
        # ``None`` is contract-stable: a field that can be null must stay
        # capable of being null, and a field that is currently non-null may
        # become null only if both sides saw null on this run. We only check
        # the concrete types here so optional fields stay permissive.
        if reference_value is None or live_value is None:
            continue
        assert type(live_value) is type(reference_value), (
            f"{name}: type mismatch on {key!r}: {type(live_value).__name__} vs {type(reference_value).__name__}"
        )
    return normalized


# ---------------------------------------------------------------------------
# Deterministic runtime setup helpers
# ---------------------------------------------------------------------------


def _install_deterministic_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[RecordingSemanticClient, RecordingOpenAlexClient]:
    """Wire the server module to a DeterministicProviderBundle-backed runtime.

    Recording clients stand in for live providers so no network traffic is
    generated and planner/synthesis routing falls through the deterministic
    bundle. This matches the ``tests/test_smart_tools.py`` pattern and keeps
    characterization output reproducible across runs.
    """

    monkeypatch.setenv("PAPER_CHASER_TOOL_PROFILE", "expert")
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    registry, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)
    monkeypatch.setattr(server, "agentic_runtime", runtime)
    monkeypatch.setattr(server, "workspace_registry", registry)
    monkeypatch.setattr(server, "client", semantic)
    monkeypatch.setattr(server, "openalex_client", openalex)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_openalex", True)
    return semantic, openalex


# ---------------------------------------------------------------------------
# Guided tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_characterization_get_runtime_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_runtime(monkeypatch)

    payload = _payload(await server.call_tool("get_runtime_status", {}))
    normalized = _assert_structural_contract("get_runtime_status", payload)

    required = {"status", "runtimeSummary", "providerOrder", "providers", "warnings"}
    assert required.issubset(normalized), f"missing required keys: {sorted(required - set(normalized))}"
    assert isinstance(normalized["providerOrder"], list)
    assert isinstance(normalized["providers"], list)
    assert isinstance(normalized["warnings"], list)
    # Provider rows are order-significant elsewhere but for characterization we
    # only pin that the set of provider names is stable.
    provider_names = {row.get("provider") for row in normalized["providers"]}
    assert provider_names == set(normalized["providerOrder"]), "provider rows drifted away from providerOrder"


@pytest.mark.asyncio
async def test_characterization_research(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_runtime(monkeypatch)

    payload = _payload(
        await server.call_tool("research", {"query": "transformers"}),
    )
    normalized = _assert_structural_contract("research", payload)

    required = {
        "status",
        "resultStatus",
        "answerability",
        "intent",
        "summary",
        "evidenceGaps",
        "nextActions",
        "resultState",
        "executionProvenance",
        "routingSummary",
        "searchSessionId",
    }
    assert required.issubset(normalized), f"missing required keys: {sorted(required - set(normalized))}"
    assert normalized["status"] in {
        "succeeded",
        "partial",
        "abstained",
        "needs_disambiguation",
        "insufficient_evidence",
        "failed",
    }
    assert normalized["answerability"] in {"grounded", "limited", "insufficient_evidence"}
    assert isinstance(normalized["nextActions"], list)
    assert normalized["nextActions"], "research must always emit next-step guidance"
    assert isinstance(normalized["evidenceGaps"], list)
    assert isinstance(normalized["routingSummary"], dict)
    assert isinstance(normalized["executionProvenance"], dict)
    assert isinstance(normalized["resultState"], dict)
    # searchSessionId is the stable handle for chaining into follow-up tools.
    assert normalized["searchSessionId"] in {"<session-id>", None}


@pytest.mark.asyncio
async def test_characterization_follow_up_research(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_runtime(monkeypatch)

    initial = _payload(
        await server.call_tool("research", {"query": "transformers"}),
    )
    session_id = initial.get("searchSessionId")
    if not session_id:
        pytest.skip(
            "deterministic research did not yield a searchSessionId; "
            "follow_up_research cannot be characterized this run",
        )

    payload = _payload(
        await server.call_tool(
            "follow_up_research",
            {
                "searchSessionId": session_id,
                "question": "What were the key findings in the saved result set?",
            },
        ),
    )
    normalized = _assert_structural_contract("follow_up_research", payload)

    required = {
        "answerStatus",
        "answer",
        "searchSessionId",
        "evidenceGaps",
        "nextActions",
        "executionProvenance",
        "sessionResolution",
        "sourcesSuppressed",
    }
    assert required.issubset(normalized), f"missing required keys: {sorted(required - set(normalized))}"
    assert normalized["answerStatus"] in {
        "answered",
        "abstained",
        "insufficient_evidence",
        "needs_disambiguation",
        "failed",
    }
    assert normalized["searchSessionId"] == "<session-id>"
    assert isinstance(normalized["evidenceGaps"], list)
    assert isinstance(normalized["nextActions"], list)
    assert isinstance(normalized["sessionResolution"], dict)


@pytest.mark.asyncio
async def test_characterization_resolve_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_runtime(monkeypatch)
    # resolve_reference is a pure identifier path for arXiv, so the smart
    # runtime is not needed but installing it keeps the env consistent.

    payload = _payload(
        await server.call_tool(
            "resolve_reference",
            {"reference": "arXiv:1706.03762"},
        ),
    )
    normalized = _assert_structural_contract("resolve_reference", payload)

    required = {
        "status",
        "resolutionType",
        "resolutionConfidence",
        "bestMatch",
        "alternatives",
        "nextActions",
        "knownItemResolutionState",
    }
    assert required.issubset(normalized), f"missing required keys: {sorted(required - set(normalized))}"
    assert normalized["status"] in {
        "resolved",
        "no_match",
        "multiple_candidates",
        "needs_disambiguation",
        "regulatory_primary_source",
    }
    assert normalized["resolutionType"] in {
        "paper_identifier",
        "title_fragment",
        "citation_repair",
        "regulatory_reference",
    }
    assert normalized["resolutionConfidence"] in {"high", "medium", "low"}
    assert isinstance(normalized["alternatives"], list)
    assert isinstance(normalized["nextActions"], list)
    assert normalized["nextActions"], "resolve_reference must emit next-step guidance"


@pytest.mark.asyncio
async def test_characterization_inspect_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_runtime(monkeypatch)

    initial = _payload(
        await server.call_tool("research", {"query": "transformers"}),
    )
    session_id = initial.get("searchSessionId")
    sources = initial.get("sources") or initial.get("structuredSources") or []
    if not session_id or not sources:
        pytest.skip(
            "deterministic research produced no inspectable sources; inspect_source cannot be characterized this run",
        )
    source_id = sources[0].get("sourceId")
    if not source_id:
        pytest.skip("deterministic research returned a source without sourceId")

    payload = _payload(
        await server.call_tool(
            "inspect_source",
            {"searchSessionId": session_id, "sourceId": source_id},
        ),
    )
    normalized = _assert_structural_contract("inspect_source", payload)

    required = {
        "resultStatus",
        "answerability",
        "source",
        "searchSessionId",
        "evidenceId",
        "nextActions",
        "sourceResolution",
        "sessionResolution",
        "executionProvenance",
    }
    assert required.issubset(normalized), f"missing required keys: {sorted(required - set(normalized))}"
    assert normalized["answerability"] in {
        "grounded",
        "limited",
        "insufficient_evidence",
    }
    assert isinstance(normalized["nextActions"], list)
    assert isinstance(normalized["source"], dict)
    assert normalized["searchSessionId"] == "<session-id>"


# ---------------------------------------------------------------------------
# Expert-profile representatives
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_characterization_search_papers_smart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_runtime(monkeypatch)

    payload = _payload(
        await server.call_tool("search_papers_smart", {"query": "transformers"}),
    )
    normalized = _assert_structural_contract("search_papers_smart", payload)

    required = {
        "searchSessionId",
        "results",
        "strategyMetadata",
        "coverageSummary",
        "routingSummary",
        "answerability",
        "structuredSources",
        "candidateLeads",
        "resourceUris",
        "agentHints",
    }
    assert required.issubset(normalized), f"missing required keys: {sorted(required - set(normalized))}"
    assert isinstance(normalized["results"], list)
    assert isinstance(normalized["strategyMetadata"], dict)
    assert isinstance(normalized["coverageSummary"], dict)
    assert isinstance(normalized["agentHints"], dict)
    assert normalized["answerability"] in {
        "grounded",
        "limited",
        "insufficient_evidence",
    }
    assert normalized["searchSessionId"] == "<session-id>"
    # ``leads`` historically mirrors ``candidateLeads``; both shapes should
    # remain list-typed so downstream consumers can iterate uniformly.
    if "leads" in normalized:
        assert isinstance(normalized["leads"], list)


@pytest.mark.asyncio
async def test_characterization_search_papers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider-aggregated expert-profile representative.

    ``search_papers`` is a classic expert-surface tool whose result shape is
    independent of the smart/agentic runtime; pinning it ensures provider
    refactors do not silently alter the lightweight aggregation path.
    """

    _install_deterministic_runtime(monkeypatch)

    payload = _payload(
        await server.call_tool(
            "search_papers",
            {"query": "transformers", "limit": 3},
        ),
    )
    normalized = _assert_structural_contract("search_papers", payload)

    # search_papers keeps a minimal but stable surface.
    assert "data" in normalized, "search_papers must keep a ``data`` result array"
    assert isinstance(normalized["data"], list)
    # Additive metadata is load-bearing for agents picking a next tool.
    assert "agentHints" in normalized
    assert isinstance(normalized["agentHints"], dict)
    # ``searchSessionId`` is optional on raw search_papers but when present
    # must be a string (which normalization collapses to ``<session-id>``).
    if normalized.get("searchSessionId") is not None:
        assert normalized["searchSessionId"] == "<session-id>"
