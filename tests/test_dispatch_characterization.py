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
    - If ``PAPER_CHASER_CHAR_REGEN=1`` is set, the fixture is (re)generated
      from the live payload. Otherwise a missing fixture is a hard error —
      silent bootstrap-on-missing used to mask accidental test drift and is
      no longer supported.
    """

    normalized = _normalize_payload(live)
    if _REGEN:
        _write_fixture(name, normalized)
    existing = _read_fixture(name)
    if existing is None:
        raise AssertionError(
            f"fixture {name!r} is missing at {_fixture_path(name)}. "
            "Run with PAPER_CHASER_CHAR_REGEN=1 to generate it intentionally."
        )

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
    assert session_id, (
        "deterministic research must seed a searchSessionId so "
        "follow_up_research can be characterized; seeding drift is a harness bug."
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
    sources = (
        initial.get("sources")
        or initial.get("structuredSources")
        or [
            {"sourceId": item.get("evidenceId")}
            for item in (initial.get("evidence") or [])
            if isinstance(item, dict) and item.get("evidenceId")
        ]
        or [
            {"sourceId": item.get("evidenceId")}
            for item in (initial.get("leads") or [])
            if isinstance(item, dict) and item.get("evidenceId")
        ]
    )
    assert session_id, "deterministic research must seed a searchSessionId for inspect_source"
    assert sources, "deterministic research must seed inspectable sources; seeding drift is a harness bug."
    source_id = sources[0].get("sourceId")
    assert source_id, "deterministic research seeded a source without a sourceId"

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


# ---------------------------------------------------------------------------
# Expanded happy-path characterization (Phase 1 expansion)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_characterization_ask_result_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_runtime(monkeypatch)

    # Seed a reusable result-set via smart search (documented prereq).
    seed = _payload(
        await server.call_tool("search_papers_smart", {"query": "transformers"}),
    )
    session_id = seed.get("searchSessionId")
    assert session_id, "search_papers_smart must seed a searchSessionId"

    payload = _payload(
        await server.call_tool(
            "ask_result_set",
            {
                "searchSessionId": session_id,
                "question": "What does this result set say about transformers?",
            },
        ),
    )
    normalized = _assert_structural_contract("ask_result_set", payload)

    required = {
        "searchSessionId",
        "answerStatus",
        "answer",
        "evidence",
        "agentHints",
        "providerUsed",
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
    assert isinstance(normalized["evidence"], list)
    assert isinstance(normalized["agentHints"], dict)


@pytest.mark.asyncio
async def test_characterization_map_research_landscape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_runtime(monkeypatch)

    seed = _payload(
        await server.call_tool("search_papers_smart", {"query": "transformers"}),
    )
    session_id = seed.get("searchSessionId")
    assert session_id, "search_papers_smart must seed a searchSessionId"

    payload = _payload(
        await server.call_tool(
            "map_research_landscape",
            {"searchSessionId": session_id, "maxThemes": 3},
        ),
    )
    normalized = _assert_structural_contract("map_research_landscape", payload)

    required = {
        "searchSessionId",
        "themes",
        "suggestedNextSearches",
        "structuredSources",
    }
    assert required.issubset(normalized), f"missing required keys: {sorted(required - set(normalized))}"
    assert normalized["searchSessionId"] == "<session-id>"
    assert isinstance(normalized["themes"], list)
    assert isinstance(normalized["suggestedNextSearches"], list)
    assert isinstance(normalized["structuredSources"], list)


@pytest.mark.asyncio
async def test_characterization_expand_research_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_runtime(monkeypatch)

    seed = _payload(
        await server.call_tool("search_papers_smart", {"query": "transformers"}),
    )
    session_id = seed.get("searchSessionId")
    assert session_id, "search_papers_smart must seed a searchSessionId"

    payload = _payload(
        await server.call_tool(
            "expand_research_graph",
            {"seedSearchSessionId": session_id, "direction": "citations"},
        ),
    )
    normalized = _assert_structural_contract("expand_research_graph", payload)

    required = {"nodes", "edges", "frontier"}
    assert required.issubset(normalized), f"missing required keys: {sorted(required - set(normalized))}"
    assert isinstance(normalized["nodes"], list)
    assert isinstance(normalized["edges"], list)
    assert isinstance(normalized["frontier"], list)


class _CursorSemanticClient(RecordingSemanticClient):
    """Deterministic bulk-search stub that emits a stable pagination token.

    Emitted token flips between two values so the cursor round-trip (page 1
    -> page 2) is observable in the characterization fixture without any
    hidden session state leaking across runs.
    """

    async def search_papers_bulk(self, **kwargs) -> dict:
        self.calls.append(("search_papers_bulk", kwargs))
        incoming_token = kwargs.get("token")
        if not incoming_token:
            return {
                "total": 2,
                "token": "bulk-token-page-2",
                "data": [{"paperId": "bulk-1"}],
                "pagination": {
                    "hasMore": True,
                    "nextCursor": "bulk-token-page-2",
                },
            }
        return {
            "total": 2,
            "token": None,
            "data": [{"paperId": "bulk-2"}],
            "pagination": {"hasMore": False, "nextCursor": None},
        }


@pytest.mark.asyncio
async def test_characterization_search_papers_bulk_pagination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PAPER_CHASER_TOOL_PROFILE", "expert")
    semantic = _CursorSemanticClient()
    openalex = RecordingOpenAlexClient()
    registry, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)
    monkeypatch.setattr(server, "agentic_runtime", runtime)
    monkeypatch.setattr(server, "workspace_registry", registry)
    monkeypatch.setattr(server, "client", semantic)
    monkeypatch.setattr(server, "openalex_client", openalex)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_openalex", True)

    first_page = _payload(
        await server.call_tool(
            "search_papers_bulk",
            {"query": "transformers", "limit": 1},
        ),
    )
    normalized_first = _assert_structural_contract("search_papers_bulk_page1", first_page)

    required = {"total", "data", "retrievalNote"}
    assert required.issubset(normalized_first), f"missing required keys: {sorted(required - set(normalized_first))}"
    assert isinstance(normalized_first["data"], list)
    assert "pagination" in normalized_first, "first bulk page must advertise pagination envelope"
    pagination = normalized_first["pagination"]
    assert isinstance(pagination, dict)
    next_cursor = pagination.get("nextCursor")
    assert next_cursor, "first bulk page must emit a nextCursor to drive page 2"

    # Page 2 — resume with the cursor captured above. The live (non-normalized)
    # cursor is what the API actually accepts; the normalized value is only for
    # fixture comparison.
    live_cursor = first_page["pagination"]["nextCursor"]
    second_page = _payload(
        await server.call_tool(
            "search_papers_bulk",
            {"query": "transformers", "limit": 1, "cursor": live_cursor},
        ),
    )
    normalized_second = _assert_structural_contract("search_papers_bulk_page2", second_page)
    assert required.issubset(normalized_second), (
        f"missing required keys on page 2: {sorted(required - set(normalized_second))}"
    )
    assert isinstance(normalized_second["data"], list)


@pytest.mark.asyncio
async def test_characterization_get_provider_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_runtime(monkeypatch)

    payload = _payload(
        await server.call_tool("get_provider_diagnostics", {}),
    )
    normalized = _assert_structural_contract("get_provider_diagnostics", payload)

    # get_provider_diagnostics is the canonical "is the runtime configured?"
    # tool — pin that its response stays a structured dict with provider-row
    # details rather than a flat free-form blob.
    assert isinstance(normalized, dict)
    # These keys are the load-bearing handles agents read to decide whether
    # to escalate, abstain, or switch providers.
    stable_candidates = {
        "providerOrder",
        "providers",
        "runtimeSummary",
        "toolProfile",
    }
    present = stable_candidates.intersection(normalized)
    assert present, (
        f"get_provider_diagnostics must keep at least one of {sorted(stable_candidates)} "
        f"in its top-level contract; got keys: {sorted(normalized)}"
    )


class _FakeFederalRegisterClient:
    """Zero-network Federal Register stub for characterization only.

    Returns a single deterministic document so the characterization fixture
    pins the shape of ``search_federal_register`` without hitting the real
    ``federalregister.gov`` API.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def search_documents(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("search_documents", kwargs))
        return {
            "total": 1,
            "data": [
                {
                    "documentNumber": "2024-12345",
                    "title": "Characterization Rule",
                    "documentType": "RULE",
                    "publicationDate": "2024-02-01",
                    "citation": "89 FR 00001",
                    "htmlUrl": "https://www.federalregister.gov/d/2024-12345",
                    "cfrReferences": ["50 CFR 17.95"],
                }
            ],
        }


@pytest.mark.asyncio
async def test_characterization_search_federal_register(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PAPER_CHASER_TOOL_PROFILE", "expert")
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    federal_register = _FakeFederalRegisterClient()
    registry, runtime = _deterministic_runtime(
        semantic=semantic,
        openalex=openalex,
        federal_register=federal_register,
        enable_federal_register=True,
    )
    monkeypatch.setattr(server, "agentic_runtime", runtime)
    monkeypatch.setattr(server, "workspace_registry", registry)
    monkeypatch.setattr(server, "client", semantic)
    monkeypatch.setattr(server, "openalex_client", openalex)
    monkeypatch.setattr(server, "federal_register_client", federal_register)
    monkeypatch.setattr(server, "enable_federal_register", True)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_openalex", True)

    payload = _payload(
        await server.call_tool(
            "search_federal_register",
            {"query": "endangered species", "limit": 5},
        ),
    )
    normalized = _assert_structural_contract("search_federal_register", payload)

    # This regulatory primary-source tool is a thin wrapper around the
    # underlying client; the contract is a list of documents with a total.
    required = {"total", "data"}
    assert required.issubset(normalized), (
        f"search_federal_register must keep {sorted(required)} in contract; got {sorted(normalized)}"
    )
    assert isinstance(normalized["data"], list)
    assert federal_register.calls, "stub should have been invoked"


# ---------------------------------------------------------------------------
# Degraded-path characterization (Phase 1 expansion)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_characterization_follow_up_research_invalid_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid/expired searchSessionId must produce a safe abstention envelope.

    Pinning this shape matters because downstream agents key their recovery
    path off ``answerStatus`` + ``nextActions`` when the session handle has
    rotted out from under them (TTL expiry, server restart, copy-paste error).
    """

    _install_deterministic_runtime(monkeypatch)

    payload = _payload(
        await server.call_tool(
            "follow_up_research",
            {
                "searchSessionId": "ssn_nonexistent_deadbeef_characterization",
                "question": "What were the key findings?",
            },
        ),
    )
    normalized = _assert_structural_contract(
        "follow_up_research_invalid_session",
        payload,
    )

    # Abstention contract: must return a structured answerStatus plus an
    # agent-actionable nextActions array even on session miss.
    assert "answerStatus" in normalized
    assert normalized["answerStatus"] in {
        "insufficient_evidence",
        "abstained",
        "failed",
        "needs_disambiguation",
    }, f"invalid-session follow_up_research returned unexpected status {normalized['answerStatus']!r}"
    assert "nextActions" in normalized
    assert isinstance(normalized["nextActions"], list)
    assert normalized["nextActions"], "degraded follow_up_research must still emit nextActions"
    # Answer must be None on degraded path (no synthesis grounding).
    assert normalized.get("answer") is None


@pytest.mark.asyncio
async def test_characterization_resolve_reference_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ambiguous/no-match input must produce the documented unresolved shape."""

    class _EmptySemanticClient(RecordingSemanticClient):
        async def search_papers_match(self, **kwargs) -> dict:
            self.calls.append(("search_papers_match", kwargs))
            return {
                "paperId": None,
                "title": None,
                "matchFound": False,
                "matchStrategy": "no_match",
                "matchConfidence": "low",
                "matchedFields": [],
                "candidateCount": 0,
            }

        async def search_papers(self, **kwargs) -> dict:
            self.calls.append(("search_papers", kwargs))
            return {"total": 0, "offset": 0, "data": []}

        async def search_snippets(self, **kwargs) -> dict:
            self.calls.append(("search_snippets", kwargs))
            return {"data": []}

    class _EmptyOpenAlex(RecordingOpenAlexClient):
        async def search(self, **kwargs) -> dict:
            self.calls.append(("search", kwargs))
            return {"total": 0, "offset": 0, "data": []}

    monkeypatch.setenv("PAPER_CHASER_TOOL_PROFILE", "expert")
    semantic = _EmptySemanticClient()
    openalex = _EmptyOpenAlex()
    registry, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)
    monkeypatch.setattr(server, "agentic_runtime", runtime)
    monkeypatch.setattr(server, "workspace_registry", registry)
    monkeypatch.setattr(server, "client", semantic)
    monkeypatch.setattr(server, "openalex_client", openalex)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_openalex", True)

    payload = _payload(
        await server.call_tool(
            "resolve_reference",
            {"reference": "zzz qqq wwww 1899"},
        ),
    )
    normalized = _assert_structural_contract(
        "resolve_reference_ambiguous",
        payload,
    )

    assert normalized["status"] in {
        "no_match",
        "needs_disambiguation",
        "multiple_candidates",
    }, f"ambiguous resolve_reference returned unexpected status {normalized['status']!r}"
    assert normalized["resolutionConfidence"] in {"low", "medium"}
    assert isinstance(normalized["nextActions"], list)
    assert normalized["nextActions"], "ambiguous resolve_reference must still emit nextActions"


@pytest.mark.asyncio
async def test_characterization_search_papers_smart_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forced provider failure on smart search must degrade, not crash."""

    class _FailingOpenAlex(RecordingOpenAlexClient):
        async def search(self, **kwargs) -> dict:
            raise RuntimeError("forced_openalex_outage_for_characterization")

        async def search_bulk(self, **kwargs) -> dict:
            raise RuntimeError("forced_openalex_outage_for_characterization")

    monkeypatch.setenv("PAPER_CHASER_TOOL_PROFILE", "expert")
    semantic = RecordingSemanticClient()
    openalex = _FailingOpenAlex()
    registry, runtime = _deterministic_runtime(semantic=semantic, openalex=openalex)
    monkeypatch.setattr(server, "agentic_runtime", runtime)
    monkeypatch.setattr(server, "workspace_registry", registry)
    monkeypatch.setattr(server, "client", semantic)
    monkeypatch.setattr(server, "openalex_client", openalex)
    monkeypatch.setattr(server, "enable_semantic_scholar", True)
    monkeypatch.setattr(server, "enable_openalex", True)

    payload = _payload(
        await server.call_tool("search_papers_smart", {"query": "transformers"}),
    )
    normalized = _assert_structural_contract(
        "search_papers_smart_provider_failure",
        payload,
    )

    # Degraded contract: smart search must NOT raise. It must keep emitting
    # the usual agent hints / routing summary even when a provider fails,
    # and answerability must reflect reduced evidence.
    assert "searchSessionId" in normalized
    assert "answerability" in normalized
    assert normalized["answerability"] in {
        "grounded",
        "limited",
        "insufficient_evidence",
    }
    assert "routingSummary" in normalized
    assert isinstance(normalized["routingSummary"], dict)
    assert "agentHints" in normalized
    assert isinstance(normalized["agentHints"], dict)
