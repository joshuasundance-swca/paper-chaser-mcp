"""Phase 3 TDD tests for ``dispatch/guided/inspect_source.py``."""

from __future__ import annotations

import pytest

from paper_chaser_mcp.dispatch.guided import inspect_source as isrc


def test__guided_extract_question_pulls_first_nonempty_field() -> None:
    assert isrc._guided_extract_question({"question": "q"}) == "q"
    assert isrc._guided_extract_question({"prompt": "p"}) == "p"
    assert isrc._guided_extract_question({"query": "x"}) == "x"
    assert isrc._guided_extract_question({}) is None


def test__guided_compact_source_candidate_returns_dict() -> None:
    out = isrc._guided_compact_source_candidate({"sourceId": "s1", "title": "t", "bogus": "ignore"})
    assert out.get("sourceId") == "s1"


def test__guided_source_resolution_payload_returns_dict() -> None:
    out = isrc._guided_source_resolution_payload(
        requested_source_id="s1",
        resolved_source_id="s1",
        match_type="exact",
    )
    assert isinstance(out, dict)


def test__guided_extract_source_reference_from_question_returns_optional() -> None:
    # explicit_source_reference returns Optional — arbitrary string may return None
    ref = isrc._guided_extract_source_reference_from_question("random text")
    assert ref is None or isinstance(ref, str)


def test__guided_select_follow_up_source_none_when_empty() -> None:
    out = isrc._guided_select_follow_up_source("q", [])
    assert out is None


def test__guided_append_selected_saved_records_returns_list() -> None:
    out = isrc._guided_append_selected_saved_records(
        current_records=[],
        saved_records=[{"sourceId": "s1"}],
        selected_ids=["s1"],
    )
    assert isinstance(out, list)


_EXPECTED_EXPORTS = (
    "_guided_source_resolution_payload",
    "_guided_compact_source_candidate",
    "_guided_extract_question",
    "_guided_append_selected_saved_records",
    "_guided_extract_source_reference_from_question",
    "_guided_select_follow_up_source",
)


@pytest.mark.parametrize("name", _EXPECTED_EXPORTS)
def test_inspect_source_submodule_exports(name: str) -> None:
    assert hasattr(isrc, name)
