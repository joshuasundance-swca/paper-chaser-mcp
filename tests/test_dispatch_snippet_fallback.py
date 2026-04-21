"""Tests for :mod:`paper_chaser_mcp.dispatch.snippet_fallback`."""

from __future__ import annotations

from typing import Any

import pytest

from paper_chaser_mcp.dispatch.snippet_fallback import (
    _maybe_fallback_snippet_search,
    _snippet_fallback_query,
    _snippet_fallback_results,
)


class TestSnippetFallbackQuery:
    def test_extracts_alphanumeric_tokens(self) -> None:
        assert (
            _snippet_fallback_query('  "retrieval-augmented generation, 2024!"  ')
            == "retrieval augmented generation 2024"
        )

    def test_caps_to_ten_tokens(self) -> None:
        query = " ".join([f"tok{i}" for i in range(20)])
        result = _snippet_fallback_query(query)
        assert len(result.split()) == 10

    def test_empty_input_returns_empty(self) -> None:
        assert _snippet_fallback_query("") == ""

    def test_no_tokens_returns_normalized_text(self) -> None:
        assert _snippet_fallback_query("!!") == "!!"


class TestSnippetFallbackResults:
    def test_abstract_based_fallback(self) -> None:
        degraded = {"degraded": True, "data": []}
        papers = {
            "data": [
                {
                    "paperId": "p1",
                    "title": "A paper",
                    "abstract": "Some abstract content",
                    "year": 2024,
                }
            ]
        }
        out = _snippet_fallback_results(degraded, papers)
        assert out["fallbackUsed"] == "search_papers"
        assert out["data"][0]["snippet"]["section"] == "abstract"
        assert out["data"][0]["snippet"]["text"] == "Some abstract content"

    def test_title_based_fallback_when_no_abstract(self) -> None:
        degraded = {"degraded": True, "data": []}
        papers = {"data": [{"paperId": "p1", "title": "Only a title"}]}
        out = _snippet_fallback_results(degraded, papers)
        assert out["data"][0]["snippet"]["section"] == "title"

    def test_no_usable_papers_returns_degraded_unchanged(self) -> None:
        degraded = {"degraded": True, "data": []}
        papers = {"data": [{"paperId": "p1"}]}  # no title, no abstract
        out = _snippet_fallback_results(degraded, papers)
        assert out is degraded

    def test_non_dict_paper_entries_skipped(self) -> None:
        degraded = {"degraded": True, "data": []}
        papers = {"data": ["not-a-dict", None]}
        out = _snippet_fallback_results(degraded, papers)
        assert out is degraded


class _FakeClient:
    def __init__(self, payload: Any = None, raises: Exception | None = None) -> None:
        self.payload = payload
        self.raises = raises
        self.last_call: dict[str, Any] | None = None

    async def search_papers(self, **kwargs: Any) -> Any:
        self.last_call = kwargs
        if self.raises is not None:
            raise self.raises
        return self.payload


class TestMaybeFallbackSnippetSearch:
    @pytest.mark.asyncio
    async def test_non_degraded_returned_unchanged(self) -> None:
        serialized: dict[str, Any] = {"degraded": False, "data": []}
        client = _FakeClient()
        out = await _maybe_fallback_snippet_search(
            serialized=serialized, args_dict={"query": "anything"}, client=client
        )
        assert out is serialized
        assert client.last_call is None

    @pytest.mark.asyncio
    async def test_degraded_but_has_data_returned_unchanged(self) -> None:
        serialized: dict[str, Any] = {"degraded": True, "data": [{"x": 1}]}
        client = _FakeClient()
        out = await _maybe_fallback_snippet_search(
            serialized=serialized, args_dict={"query": "anything"}, client=client
        )
        assert out is serialized
        assert client.last_call is None

    @pytest.mark.asyncio
    async def test_empty_fallback_query_returns_serialized(self) -> None:
        serialized: dict[str, Any] = {"degraded": True, "data": []}
        client = _FakeClient()
        out = await _maybe_fallback_snippet_search(serialized=serialized, args_dict={"query": ""}, client=client)
        assert out is serialized
        assert client.last_call is None

    @pytest.mark.asyncio
    async def test_client_exception_returns_serialized(self) -> None:
        serialized: dict[str, Any] = {"degraded": True, "data": []}
        client = _FakeClient(raises=RuntimeError("boom"))
        out = await _maybe_fallback_snippet_search(
            serialized=serialized,
            args_dict={"query": "transformers attention"},
            client=client,
        )
        assert out is serialized


class TestReExport:
    def test_core_still_exports_snippet_fallback(self) -> None:
        import importlib

        core = importlib.import_module("paper_chaser_mcp.dispatch._core")
        for name in (
            "_snippet_fallback_query",
            "_snippet_fallback_results",
            "_maybe_fallback_snippet_search",
        ):
            assert hasattr(core, name), name


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
