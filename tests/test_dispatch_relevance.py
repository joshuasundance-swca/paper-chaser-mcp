"""Tests for :mod:`paper_chaser_mcp.dispatch.relevance`.

Phase 2 Step 3a of the dispatch refactor extracts the pure topical-relevance
helpers out of ``paper_chaser_mcp.dispatch._core`` into a focused submodule.
These tests pin the public behavior of that submodule so that the extraction
is a pure move — no behavior change — and so that future tweaks to the
relevance heuristics have an obvious home for regression coverage.
"""

from __future__ import annotations

import pytest

from paper_chaser_mcp.dispatch.relevance import (
    _facet_match,
    _paper_topical_relevance,
    _tokenize_relevance_text,
    _topical_relevance_from_signals,
    compute_topical_relevance,
)


class TestTokenizeRelevanceText:
    def test_lowercases_and_splits_on_non_alnum(self) -> None:
        assert _tokenize_relevance_text("Large Language Models, 2024!") == {
            "large",
            "language",
            "models",
            "2024",
        }

    def test_drops_single_char_tokens(self) -> None:
        assert _tokenize_relevance_text("a b cc dd") == {"cc", "dd"}

    def test_empty_returns_empty_set(self) -> None:
        assert _tokenize_relevance_text("") == set()


class TestFacetMatch:
    def test_multi_token_facet_must_fully_match(self) -> None:
        tokens = _tokenize_relevance_text("retrieval augmented generation systems")
        assert _facet_match(tokens, "retrieval augmented") is True
        assert _facet_match(tokens, "symbolic reasoning") is False

    def test_empty_facet_is_not_a_match(self) -> None:
        assert _facet_match({"a", "b"}, "") is False


class TestTopicalRelevanceFromSignals:
    def test_title_anchor_plus_facet_is_on_topic(self) -> None:
        assert (
            _topical_relevance_from_signals(
                query_similarity=0.3,
                title_facet_coverage=0.5,
                title_anchor_coverage=0.5,
                query_facet_coverage=0.5,
                query_anchor_coverage=0.5,
            )
            == "on_topic"
        )

    def test_low_similarity_without_any_anchor_is_off_topic(self) -> None:
        assert (
            _topical_relevance_from_signals(
                query_similarity=0.0,
                title_facet_coverage=0.0,
                title_anchor_coverage=0.0,
                query_facet_coverage=0.0,
                query_anchor_coverage=0.0,
            )
            == "off_topic"
        )

    def test_weak_match_when_some_anchor_but_no_strong_threshold(self) -> None:
        assert (
            _topical_relevance_from_signals(
                query_similarity=0.2,
                title_facet_coverage=0.0,
                title_anchor_coverage=0.5,
                query_facet_coverage=0.0,
                query_anchor_coverage=0.5,
            )
            == "weak_match"
        )


class TestComputeTopicalRelevance:
    def test_on_topic_when_title_contains_multi_word_facet(self) -> None:
        result = compute_topical_relevance(
            "retrieval augmented generation for scientific papers",
            {"title": "Retrieval Augmented Generation for scientific papers"},
        )
        assert result == "on_topic"

    def test_off_topic_when_nothing_overlaps(self) -> None:
        result = compute_topical_relevance(
            "retrieval augmented generation",
            {"title": "Agricultural yields in rural Kenya"},
        )
        assert result == "off_topic"

    def test_missing_fields_coerce_to_empty_strings(self) -> None:
        result = compute_topical_relevance("anything", {})
        assert result in {"off_topic", "weak_match", "on_topic"}


class TestPaperTopicalRelevanceWrapper:
    def test_wrapper_matches_compute_topical_relevance(self) -> None:
        query = "transformer architecture attention"
        paper = {
            "title": "A Transformer Architecture",
            "abstract": "We study attention heads.",
        }
        assert _paper_topical_relevance(query, paper) == compute_topical_relevance(query, paper)


class TestReExport:
    def test_core_still_exports_relevance_symbols(self) -> None:
        """The ``_core`` module must continue to expose these names so the
        existing internal call sites and test ``monkeypatch.setattr`` targets
        keep working after the extraction.
        """
        import importlib

        core = importlib.import_module("paper_chaser_mcp.dispatch._core")

        for name in (
            "_tokenize_relevance_text",
            "_facet_match",
            "_topical_relevance_from_signals",
            "compute_topical_relevance",
            "_paper_topical_relevance",
        ):
            assert hasattr(core, name), name


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
