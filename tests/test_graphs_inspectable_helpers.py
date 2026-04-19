"""Regression tests for off-topic-aware inspectability helpers in agentic.graphs."""

from __future__ import annotations

from typing import Literal

from paper_chaser_mcp.agentic.graphs import (
    _has_inspectable_sources,
    _has_on_topic_sources,
)
from paper_chaser_mcp.agentic.models import StructuredSourceRecord


def _make_source(
    *,
    source_id: str,
    topical_relevance: Literal["on_topic", "weak_match", "off_topic"] | None,
    canonical_url: str | None = "https://example.org/a",
) -> StructuredSourceRecord:
    return StructuredSourceRecord(
        sourceId=source_id,
        title=f"Title {source_id}",
        canonicalUrl=canonical_url,
        topicalRelevance=topical_relevance,
    )


def test_has_inspectable_sources_excludes_off_topic_records() -> None:
    records = [
        _make_source(source_id="s1", topical_relevance="off_topic"),
        _make_source(source_id="s2", topical_relevance="off_topic"),
    ]
    assert _has_inspectable_sources(records) is False


def test_has_inspectable_sources_true_when_any_on_topic_has_url() -> None:
    records = [
        _make_source(source_id="s1", topical_relevance="off_topic"),
        _make_source(source_id="s2", topical_relevance="on_topic"),
    ]
    assert _has_inspectable_sources(records) is True


def test_has_inspectable_sources_false_when_on_topic_has_no_url_or_abstract() -> None:
    records = [
        _make_source(source_id="s1", topical_relevance="on_topic", canonical_url=None),
    ]
    assert _has_inspectable_sources(records) is False


def test_has_on_topic_sources_false_when_all_off_topic() -> None:
    records = [
        _make_source(source_id="s1", topical_relevance="off_topic"),
        _make_source(source_id="s2", topical_relevance="off_topic"),
    ]
    assert _has_on_topic_sources(records) is False


def test_has_on_topic_sources_true_when_weak_match_present() -> None:
    records = [
        _make_source(source_id="s1", topical_relevance="off_topic"),
        _make_source(source_id="s2", topical_relevance="weak_match"),
    ]
    assert _has_on_topic_sources(records) is True


def test_has_on_topic_sources_true_when_relevance_unset() -> None:
    records = [_make_source(source_id="s1", topical_relevance=None)]
    assert _has_on_topic_sources(records) is True
