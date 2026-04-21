"""Phase 3 TDD tests for ``dispatch/guided/resolve_reference.py``."""

from __future__ import annotations

import pytest

from paper_chaser_mcp.dispatch.guided import resolve_reference as rr


def test__guided_note_repair_noop_when_equal() -> None:
    repairs: list[dict[str, str]] = []
    rr._guided_note_repair(repairs, field="title", original="x", normalized="x", reason="r")
    assert repairs == []


def test__guided_note_repair_records_change() -> None:
    repairs: list[dict[str, str]] = []
    rr._guided_note_repair(repairs, field="title", original="A ", normalized="A", reason="trim")
    assert repairs == [{"field": "title", "from": "A ", "to": "A", "reason": "trim"}]


def test__guided_underspecified_reference_returns_none_for_clear_ident() -> None:
    assert rr._guided_underspecified_reference_clarification(query="10.1234/abcd", focus=None) is None


def test__guided_underspecified_reference_returns_none_for_empty_query() -> None:
    assert rr._guided_underspecified_reference_clarification(query="", focus=None) is None


_EXPECTED_EXPORTS = (
    "_guided_note_repair",
    "_guided_underspecified_reference_clarification",
)


@pytest.mark.parametrize("name", _EXPECTED_EXPORTS)
def test_resolve_reference_submodule_exports(name: str) -> None:
    assert hasattr(rr, name)
