"""Phase 3 TDD tests for ``dispatch/guided/findings.py``."""

from __future__ import annotations

import pytest

from paper_chaser_mcp.dispatch.guided import findings as findings_mod
from paper_chaser_mcp.dispatch.guided.findings import (
    _guided_findings_from_sources,
    _guided_unverified_leads_from_sources,
)


def _src(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "sourceId": "s1",
        "title": "A Paper",
        "topicalRelevance": "on_topic",
        "verificationStatus": "verified_primary_source",
    }
    base.update(overrides)
    return base


def test__guided_findings_from_sources_keeps_verified_on_topic() -> None:
    findings = _guided_findings_from_sources([_src()])
    assert len(findings) == 1
    assert findings[0]["claim"] == "A Paper"
    assert findings[0]["supportingSourceIds"] == ["s1"]
    assert findings[0]["trustLevel"] == "verified"


def test__guided_findings_from_sources_skips_off_topic_or_unverified() -> None:
    off_topic = _src(topicalRelevance="weak_match")
    unverified = _src(sourceId="s2", verificationStatus="unverified")
    missing_claim = _src(sourceId="", title="", note="", verificationStatus="verified_metadata")
    out = _guided_findings_from_sources([off_topic, unverified, missing_claim])
    assert out == []


def test__guided_findings_from_sources_falls_back_through_claim_keys() -> None:
    src = _src(title="", note="A note claim")
    out = _guided_findings_from_sources([src])
    assert out and out[0]["claim"] == "A note claim"


def test__guided_unverified_leads_from_sources_keeps_non_verified() -> None:
    verified = _src()
    lead = _src(
        sourceId="s2",
        topicalRelevance="weak_match",
        verificationStatus="unverified",
    )
    out = _guided_unverified_leads_from_sources([verified, lead])
    assert len(out) == 1
    assert out[0]["sourceId"] == "s2"


def test__guided_unverified_leads_from_sources_deduplicates_by_source_id() -> None:
    dup1 = _src(sourceId="s1", verificationStatus="unverified")
    dup2 = _src(sourceId="s1", verificationStatus="unverified")
    new = _src(sourceId="s2", verificationStatus="unverified")
    out = _guided_unverified_leads_from_sources([dup1, dup2, new])
    ids = [item["sourceId"] for item in out]
    assert ids == ["s1", "s2"]


def test__guided_unverified_leads_from_sources_caps_at_six() -> None:
    sources = [_src(sourceId=f"s{idx}", verificationStatus="unverified") for idx in range(10)]
    out = _guided_unverified_leads_from_sources(sources)
    assert len(out) == 6


_EXPECTED_EXPORTS = (
    "_guided_findings_from_sources",
    "_guided_unverified_leads_from_sources",
)


@pytest.mark.parametrize("name", _EXPECTED_EXPORTS)
def test_findings_submodule_exports(name: str) -> None:
    assert hasattr(findings_mod, name)
