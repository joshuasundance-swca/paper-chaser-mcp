"""Tests for :mod:`paper_chaser_mcp.dispatch.paging`."""

from __future__ import annotations

import pytest

from paper_chaser_mcp.dispatch.paging import _cursor_to_offset, _encode_next_cursor
from paper_chaser_mcp.utils.cursor import PROVIDER, cursor_from_offset


class TestCursorToOffset:
    def test_none_cursor_returns_none(self) -> None:
        assert _cursor_to_offset(None) is None

    def test_legacy_integer_string_returns_offset(self) -> None:
        assert _cursor_to_offset("10") == 10

    def test_negative_legacy_offset_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _cursor_to_offset("-1")

    def test_structured_cursor_roundtrip(self) -> None:
        cursor = cursor_from_offset("search_papers", 20, context_hash="ctx")
        assert _cursor_to_offset(cursor, tool="search_papers", context_hash="ctx") == 20

    def test_structured_cursor_rejects_cross_tool(self) -> None:
        cursor = cursor_from_offset("search_papers", 5, context_hash="ctx")
        with pytest.raises(ValueError, match="tool"):
            _cursor_to_offset(cursor, tool="search_papers_bulk", context_hash="ctx")

    def test_structured_cursor_rejects_cross_context(self) -> None:
        cursor = cursor_from_offset("search_papers", 5, context_hash="a")
        with pytest.raises(ValueError, match="different query context"):
            _cursor_to_offset(cursor, tool="search_papers", context_hash="b")

    def test_malformed_cursor_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be decoded"):
            _cursor_to_offset("notavalidbase64!@#$")


class TestEncodeNextCursor:
    def test_no_pagination_returned_unchanged(self) -> None:
        result: dict[str, object] = {"data": []}
        assert _encode_next_cursor(result, "search_papers") is result

    def test_none_cursor_returned_unchanged(self) -> None:
        result: dict[str, object] = {"pagination": {"nextCursor": None}}
        assert _encode_next_cursor(result, "search_papers") is result

    def test_integer_cursor_is_rewritten_as_structured(self) -> None:
        result: dict[str, object] = {"pagination": {"nextCursor": 20}}
        _encode_next_cursor(result, "search_papers", context_hash="ctx")
        pagination = result["pagination"]
        assert isinstance(pagination, dict)
        new = pagination["nextCursor"]
        assert isinstance(new, str)
        assert new != "20"
        assert _cursor_to_offset(new, tool="search_papers", context_hash="ctx") == 20

    def test_already_structured_cursor_left_alone(self) -> None:
        structured = cursor_from_offset("search_papers", 7, context_hash="ctx")
        result: dict[str, object] = {"pagination": {"nextCursor": structured}}
        _encode_next_cursor(result, "search_papers", context_hash="ctx")
        pagination = result["pagination"]
        assert isinstance(pagination, dict)
        assert pagination["nextCursor"] == structured


class TestReExport:
    def test_core_still_exports_paging(self) -> None:
        import importlib

        core = importlib.import_module("paper_chaser_mcp.dispatch._core")
        assert hasattr(core, "_cursor_to_offset")
        assert hasattr(core, "_encode_next_cursor")
        # Constant must stay accessible — cursor code paths reach for PROVIDER
        # via dispatch._core in several internal helpers.
        assert hasattr(core, "PROVIDER")
        assert core.PROVIDER == PROVIDER


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
