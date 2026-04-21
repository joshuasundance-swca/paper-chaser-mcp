"""Phase 6 red-bar: pin the normalization helpers."""

from __future__ import annotations

from paper_chaser_mcp.agentic import planner as legacy_planner
from paper_chaser_mcp.agentic.planner.normalization import (
    looks_like_exact_title,
    looks_like_near_known_item_query,
    looks_like_url,
    normalize_query,
    query_facets,
    query_terms,
)


def test_normalize_query_collapses_whitespace() -> None:
    assert normalize_query("  hello   world  ") == "hello world"
    assert normalize_query("\ta\nb  c") == "a b c"
    assert normalize_query("") == ""


def test_query_facets_splits_on_prepositions() -> None:
    facets = query_facets("bioaccumulation in marine mammals for pollution monitoring")
    assert facets, "Expected at least one facet for a multi-preposition query"
    joined = " ".join(facets)
    assert "marine" in joined or "pollution" in joined


def test_query_terms_returns_unigrams() -> None:
    terms = query_terms("machine learning models for climate prediction")
    # Stopwords dropped, tokens unique, unigrams only
    assert terms, "Expected unigram terms"
    assert all(" " not in term for term in terms)
    assert len(terms) == len(set(terms))


def test_looks_like_url_detects_http_urls() -> None:
    assert looks_like_url("https://example.com/paper")
    assert looks_like_url("http://arxiv.org/abs/1234.56789")
    assert not looks_like_url("just a regular query")
    assert not looks_like_url("")


def test_looks_like_exact_title_positive_and_negative() -> None:
    # Title-ish phrasing with capitalization and stopwords.
    assert looks_like_exact_title("Attention Is All You Need")
    # Question-form should not be a title.
    assert not looks_like_exact_title("what is the transformer architecture?")


def test_looks_like_near_known_item_query_identifies_short_id() -> None:
    assert looks_like_near_known_item_query("Toolformer")
    assert looks_like_near_known_item_query("Med-PaLM")
    # Long conceptual question should not look near-known-item.
    assert not looks_like_near_known_item_query("what is the difference between supervised and unsupervised learning")


def test_legacy_planner_reexports_normalization_helpers() -> None:
    for name in (
        "normalize_query",
        "query_facets",
        "query_terms",
        "looks_like_url",
        "looks_like_exact_title",
        "looks_like_near_known_item_query",
    ):
        assert hasattr(legacy_planner, name), f"legacy planner missing {name}"
