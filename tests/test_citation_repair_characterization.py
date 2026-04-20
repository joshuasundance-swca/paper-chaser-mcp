"""Structural characterization for ``citation_repair``.

Phase 9a of the durable refactor will split ``citation_repair.py`` into
smaller modules with a BYTE-EQUIVALENT output claim at the user-visible
boundary (``parse_citation`` + ``resolve_citation``). This file pins the
STRUCTURAL shape of those two seams ahead of that split so regressions
surface as a failing fixture rather than a silent rename/retype.

Design
------

* Deterministic inputs only. ``parse_citation`` is pure; ``resolve_citation``
  is wrapped with the same ``Recording*`` client stubs that
  ``tests/test_citation_repair.py`` already uses so no network calls are made.
* One normalized JSONL fixture per case lives under
  ``tests/fixtures/characterization/citation_repair/``. Fixture layout:

      tests/fixtures/characterization/citation_repair/parse_citation/<case>.jsonl
      tests/fixtures/characterization/citation_repair/resolve_citation/<case>.jsonl

* Volatile fields (session ids, timestamps, UUIDs) are replaced with canonical
  placeholders before comparison. Byte-for-byte equality is intentionally NOT
  asserted — the goal is to catch structural drift (missing key, retyped
  field, list<->dict flip) during the Phase 9a extraction.
* Set ``PAPER_CHASER_CHAR_REGEN=1`` to intentionally (re)generate fixtures.
  Missing fixtures during a normal run are a HARD ERROR.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from paper_chaser_mcp.citation_repair import (
    parse_citation,
    resolve_citation,
)
from tests.helpers import (
    RecordingOpenAlexClient,
    RecordingSemanticClient,
)

FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "characterization" / "citation_repair"
_REGEN = os.environ.get("PAPER_CHASER_CHAR_REGEN") == "1"


# ---------------------------------------------------------------------------
# Normalization + fixture I/O (mirrors tests/test_dispatch_characterization.py)
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


def _normalize(payload: Any) -> Any:
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
                result[key] = _normalize(value)
        return result
    if isinstance(payload, list):
        return [_normalize(item) for item in payload]
    if isinstance(payload, str):
        if _UUID_RE.match(payload):
            return "<uuid>"
        if _SESSION_ID_RE.match(payload):
            return "<session-id>"
        if _TIMESTAMP_RE.match(payload):
            return "<timestamp>"
    return payload


def _fixture_path(*parts: str) -> Path:
    return FIXTURES_ROOT.joinpath(*parts).with_suffix(".jsonl")


def _write_fixture(path: Path, normalized: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normalized, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_fixture(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        line = handle.readline()
    if not line.strip():
        return None
    return json.loads(line)


def _assert_structural(path: Path, live: Any) -> Any:
    """Pin structural contract: key presence + top-level types + stable values.

    When ``PAPER_CHASER_CHAR_REGEN=1`` is set, fixtures are (re)generated.
    Otherwise a missing fixture is a hard error so accidental drift cannot
    silently bootstrap a new contract.
    """

    normalized = _normalize(live)
    if _REGEN:
        _write_fixture(path, normalized)
    existing = _read_fixture(path)
    if existing is None:
        raise AssertionError(
            f"fixture {path!s} is missing. Run with PAPER_CHASER_CHAR_REGEN=1 to generate it intentionally."
        )

    # Top-level key-set comparison (missing-key is always a regression).
    if isinstance(existing, dict) and isinstance(normalized, dict):
        missing = sorted(set(existing) - set(normalized))
        assert not missing, f"{path.name}: pinned top-level keys disappeared: {missing}"
        for key, reference_value in existing.items():
            live_value = normalized[key]
            if reference_value is None or live_value is None:
                continue
            assert type(live_value) is type(reference_value), (
                f"{path.name}: type mismatch on {key!r}: "
                f"{type(live_value).__name__} vs {type(reference_value).__name__}"
            )
            # Stable top-level scalar values (str/int/bool/float) are pinned
            # verbatim — these are what agents key off of.
            if isinstance(reference_value, (str, int, bool, float)):
                assert live_value == reference_value, (
                    f"{path.name}: scalar drift on {key!r}: {live_value!r} vs {reference_value!r}"
                )
    else:
        assert type(normalized) is type(existing), (
            f"{path.name}: top-level type drift: {type(normalized).__name__} vs {type(existing).__name__}"
        )
    return normalized


# ---------------------------------------------------------------------------
# parse_citation characterization
# ---------------------------------------------------------------------------


_PARSE_CASES: list[tuple[str, str]] = [
    (
        "simple_well_formed",
        "Vaswani A, Shazeer N, Parmar N, et al. 2017. Attention Is All You Need. NeurIPS.",
    ),
    (
        "malformed_short_fragment",
        "zzz qqq wwww 1899",
    ),
    (
        "doi_only",
        "10.1038/nrn3241",
    ),
    (
        "arxiv_only",
        "arXiv:1706.03762",
    ),
    (
        "apa_style_complex",
        (
            "Rockstrom, J., Steffen, W., Noone, K., et al. (2009). "
            "A safe operating space for humanity. Nature, 461(7263), 472-475."
        ),
    ),
    (
        "regulatory_reference",
        "Federal Register (2012). Listing of loggerhead sea turtle DPS. 77 FR 4632.",
    ),
]


@pytest.mark.parametrize("case,citation", _PARSE_CASES, ids=[c[0] for c in _PARSE_CASES])
def test_parse_citation_characterization_cases(case: str, citation: str) -> None:
    parsed = parse_citation(citation)
    live = dataclasses.asdict(parsed)
    normalized = _assert_structural(_fixture_path("parse_citation", case), live)

    # ParsedCitation is a dataclass — its shape is the entire public API for
    # this seam. Pin the invariants that callers rely on and that the Phase
    # 9a split must preserve.
    assert "normalized_text" in normalized
    assert isinstance(normalized["normalized_text"], str)
    assert isinstance(normalized["title_candidates"], list)
    assert isinstance(normalized["author_surnames"], list)
    assert isinstance(normalized["venue_hints"], list)
    # year must remain Optional[int]; normalize keeps None as None.
    if normalized["year"] is not None:
        assert isinstance(normalized["year"], int)


# ---------------------------------------------------------------------------
# resolve_citation characterization
# ---------------------------------------------------------------------------


class _EmptySemanticClient(RecordingSemanticClient):
    """Returns no matches for every semantic lookup.

    Used to characterize the unresolved / no-match branch of resolve_citation
    without relying on live data.
    """

    async def search_papers_match(self, **kwargs: Any) -> dict:
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

    async def search_papers(self, **kwargs: Any) -> dict:
        self.calls.append(("search_papers", kwargs))
        return {"total": 0, "offset": 0, "data": []}

    async def search_snippets(self, **kwargs: Any) -> dict:
        self.calls.append(("search_snippets", kwargs))
        return {"data": []}


class _EmptyOpenAlex(RecordingOpenAlexClient):
    async def search(self, **kwargs: Any) -> dict:
        self.calls.append(("search", kwargs))
        return {"total": 0, "offset": 0, "data": []}


async def _call_resolve_citation(
    *,
    citation: str,
    semantic: RecordingSemanticClient,
    openalex: RecordingOpenAlexClient,
    enable_openalex: bool = False,
) -> dict[str, Any]:
    return await resolve_citation(
        citation=citation,
        max_candidates=3,
        client=semantic,
        enable_core=False,
        enable_semantic_scholar=True,
        enable_openalex=enable_openalex,
        enable_arxiv=False,
        enable_serpapi=False,
        core_client=None,
        openalex_client=openalex if enable_openalex else None,
        arxiv_client=None,
        serpapi_client=None,
    )


@pytest.mark.asyncio
async def test_resolve_citation_characterization_doi_identifier() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    payload = await _call_resolve_citation(
        citation="10.1038/nrn3241",
        semantic=semantic,
        openalex=openalex,
    )
    normalized = _assert_structural(
        _fixture_path("resolve_citation", "doi_identifier"),
        payload,
    )
    assert "bestMatch" in normalized
    assert normalized.get("resolutionConfidence") == "high"


@pytest.mark.asyncio
async def test_resolve_citation_characterization_arxiv_identifier() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    payload = await _call_resolve_citation(
        citation="arXiv:1706.03762",
        semantic=semantic,
        openalex=openalex,
    )
    normalized = _assert_structural(
        _fixture_path("resolve_citation", "arxiv_identifier"),
        payload,
    )
    assert "bestMatch" in normalized
    assert normalized.get("resolutionConfidence") == "high"


@pytest.mark.asyncio
async def test_resolve_citation_characterization_regulatory_redirect() -> None:
    semantic = RecordingSemanticClient()
    openalex = RecordingOpenAlexClient()
    payload = await _call_resolve_citation(
        citation="Federal Register (2012). Listing of loggerhead sea turtle DPS. 77 FR 4632.",
        semantic=semantic,
        openalex=openalex,
    )
    normalized = _assert_structural(
        _fixture_path("resolve_citation", "regulatory_redirect"),
        payload,
    )
    # Regulatory redirect must NOT run any paper-search calls.
    assert semantic.calls == []
    assert normalized.get("bestMatch") is None
    assert normalized.get("resolutionConfidence") == "low"


@pytest.mark.asyncio
async def test_resolve_citation_characterization_no_match() -> None:
    semantic = _EmptySemanticClient()
    openalex = _EmptyOpenAlex()
    payload = await _call_resolve_citation(
        citation="zzz qqq wwww 1899",
        semantic=semantic,
        openalex=openalex,
        enable_openalex=True,
    )
    normalized = _assert_structural(
        _fixture_path("resolve_citation", "no_match"),
        payload,
    )
    assert normalized.get("bestMatch") is None
    assert normalized.get("resolutionConfidence") in {"low", "medium"}


@pytest.mark.asyncio
async def test_resolve_citation_characterization_title_match_via_recording_clients() -> None:
    class _MatchingSemantic(RecordingSemanticClient):
        async def search_papers_match(self, **kwargs: Any) -> dict:
            self.calls.append(("search_papers_match", kwargs))
            return {
                "paperId": "transformer-1",
                "title": "Attention Is All You Need",
                "year": 2017,
                "venue": "NeurIPS",
                "authors": [
                    {"name": "Ashish Vaswani"},
                    {"name": "Noam Shazeer"},
                    {"name": "Niki Parmar"},
                ],
                "matchFound": True,
                "matchStrategy": "exact_title",
                "matchConfidence": "high",
                "matchedFields": ["title", "author", "year"],
                "candidateCount": 1,
            }

    semantic = _MatchingSemantic()
    openalex = RecordingOpenAlexClient()
    payload = await _call_resolve_citation(
        citation=("Vaswani A, Shazeer N, Parmar N, et al. 2017. Attention Is All You Need. NeurIPS."),
        semantic=semantic,
        openalex=openalex,
    )
    normalized = _assert_structural(
        _fixture_path("resolve_citation", "title_match"),
        payload,
    )
    assert normalized.get("bestMatch") is not None
    assert normalized.get("resolutionConfidence") in {"medium", "high"}
