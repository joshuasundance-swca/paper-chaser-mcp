"""Phase 3 TDD tests for ``dispatch/guided/sessions.py``."""

from __future__ import annotations

import types

import pytest

from paper_chaser_mcp.dispatch.guided import sessions as sessions_mod


def _make_registry(records: dict[str, object] | None = None) -> types.SimpleNamespace:
    """Build a minimal duck-typed workspace_registry for session helpers."""

    storage = records or {}

    def _get(session_id: str) -> object:
        if session_id not in storage:
            raise KeyError(session_id)
        return storage[session_id]

    return types.SimpleNamespace(_records=storage, get=_get)


def _make_record(
    *,
    session_id: str,
    source_tool: str = "research",
    created_at: float = 1000.0,
    payload: dict[str, object] | None = None,
    papers: list[dict[str, object]] | None = None,
    query: str = "q",
    expired: bool = False,
    metadata: dict[str, object] | None = None,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        search_session_id=session_id,
        source_tool=source_tool,
        created_at=created_at,
        payload=payload or {},
        papers=papers or [],
        query=query,
        metadata=metadata or {},
        is_expired=lambda now, _exp=expired: _exp,
    )


def test__guided_session_exists_returns_false_on_missing() -> None:
    registry = _make_registry({})
    assert sessions_mod._guided_session_exists(workspace_registry=registry, search_session_id="unknown") is False


def test__guided_session_exists_returns_true_when_present() -> None:
    record = _make_record(session_id="s1")
    registry = _make_registry({"s1": record})
    assert sessions_mod._guided_session_exists(workspace_registry=registry, search_session_id="s1") is True


def test__guided_session_exists_rejects_empty_id() -> None:
    registry = _make_registry({"s1": _make_record(session_id="s1")})
    assert sessions_mod._guided_session_exists(workspace_registry=registry, search_session_id="") is False


def test__guided_session_exists_handles_none_registry() -> None:
    assert sessions_mod._guided_session_exists(workspace_registry=None, search_session_id="x") is False


def test__guided_active_session_ids_sorts_by_created_at_desc() -> None:
    r1 = _make_record(session_id="old", created_at=100.0)
    r2 = _make_record(session_id="new", created_at=500.0)
    registry = _make_registry({"old": r1, "new": r2})
    assert sessions_mod._guided_active_session_ids(registry) == ["new", "old"]


def test__guided_active_session_ids_skips_expired() -> None:
    r1 = _make_record(session_id="alive", created_at=200.0)
    r2 = _make_record(session_id="dead", created_at=300.0, expired=True)
    registry = _make_registry({"alive": r1, "dead": r2})
    assert sessions_mod._guided_active_session_ids(registry) == ["alive"]


def test__guided_active_session_ids_returns_empty_for_none() -> None:
    assert sessions_mod._guided_active_session_ids(None) == []


def test__guided_extract_search_session_id_tries_every_alias() -> None:
    assert sessions_mod._guided_extract_search_session_id({"searchSessionId": "a"}) == "a"
    assert sessions_mod._guided_extract_search_session_id({"search_session_id": "b"}) == "b"
    assert sessions_mod._guided_extract_search_session_id({"sessionId": "c"}) == "c"
    assert sessions_mod._guided_extract_search_session_id({"session_id": "d"}) == "d"
    assert sessions_mod._guided_extract_search_session_id({"session": "e"}) == "e"
    assert sessions_mod._guided_extract_search_session_id({}) is None


def test__guided_saved_session_topicality_all_off_topic() -> None:
    has, all_off = sessions_mod._guided_saved_session_topicality(
        [{"topicalRelevance": "off_topic"}, {"topicalRelevance": "off_topic"}]
    )
    assert has is True
    assert all_off is True


def test__guided_saved_session_topicality_mixed() -> None:
    has, all_off = sessions_mod._guided_saved_session_topicality(
        [{"topicalRelevance": "on_topic"}, {"topicalRelevance": "off_topic"}]
    )
    assert has is True
    assert all_off is False


def test__guided_saved_session_topicality_empty() -> None:
    assert sessions_mod._guided_saved_session_topicality(None) == (False, False)
    assert sessions_mod._guided_saved_session_topicality([]) == (False, False)


def test__guided_session_findings_reuses_finding_dicts() -> None:
    payload = {
        "verifiedFindings": [
            {"claim": "A", "supportingSourceIds": ["s1"], "trustLevel": "verified"},
        ]
    }
    out = sessions_mod._guided_session_findings(payload, sources=[])
    assert len(out) == 1
    assert out[0]["claim"] == "A"


def test__guided_session_findings_falls_back_to_source_derivation() -> None:
    sources = [
        {
            "sourceId": "s1",
            "title": "t",
            "topicalRelevance": "on_topic",
            "verificationStatus": "verified_primary_source",
        }
    ]
    out = sessions_mod._guided_session_findings({}, sources)
    assert out and out[0]["claim"] == "t"


def test__guided_infer_single_session_id_returns_none_when_no_records() -> None:
    registry = _make_registry({})
    assert sessions_mod._guided_infer_single_session_id(registry) is None


def test__guided_enrich_records_from_saved_session_merges_by_title() -> None:
    saved = [{"sourceId": "saved1", "title": "Same", "isPrimarySource": True}]
    current = [{"sourceId": "cur1", "title": "Same"}]
    out = sessions_mod._guided_enrich_records_from_saved_session(current, saved)
    assert len(out) == 1
    assert out[0]["isPrimarySource"] is True
    # ``_guided_merge_source_records`` is called with (best_match, record),
    # i.e. saved as primary and current as secondary. Primary's sourceId wins.
    assert out[0]["sourceId"] == "saved1"


_EXPECTED_EXPORTS = (
    "_guided_session_exists",
    "_guided_active_session_ids",
    "_guided_candidate_records",
    "_guided_latest_compatible_session_id",
    "_guided_unique_compatible_session_id",
    "_guided_resolve_session_id_for_source",
    "_guided_infer_single_session_id",
    "_guided_extract_search_session_id",
    "_guided_session_candidates",
    "_guided_follow_up_session_resolution",
    "_guided_inspect_session_resolution",
    "_guided_saved_session_topicality",
    "_guided_session_findings",
    "_guided_session_state",
    "_guided_enrich_records_from_saved_session",
)


@pytest.mark.parametrize("name", _EXPECTED_EXPORTS)
def test_sessions_submodule_exports(name: str) -> None:
    assert hasattr(sessions_mod, name)
