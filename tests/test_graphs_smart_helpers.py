"""Phase 7c-3 — tests for the extracted ``smart_helpers`` submodule.

Pins the six ``_core``-resident helpers that Phase 7c-3 hoisted into
:mod:`paper_chaser_mcp.agentic.graphs.smart_helpers` and guards the identity
preservation that keeps legacy ``from paper_chaser_mcp.agentic.graphs._core
import _X`` imports valid. ``smart_helpers`` is the canonical home; ``_core``
re-imports each name at module top so existing call sites continue to
resolve via the module globals.
"""

from __future__ import annotations

from paper_chaser_mcp.agentic.graphs._core import (
    _best_next_internal_action as core_best_next_internal_action,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _has_inspectable_sources as core_has_inspectable_sources,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _has_on_topic_sources as core_has_on_topic_sources,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _paid_providers_used as core_paid_providers_used,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _smart_coverage_summary as core_smart_coverage_summary,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _smart_provider_fallback_warnings as core_smart_provider_fallback_warnings,
)
from paper_chaser_mcp.agentic.graphs.smart_helpers import (
    _best_next_internal_action,
    _has_inspectable_sources,
    _has_on_topic_sources,
    _paid_providers_used,
    _smart_coverage_summary,
    _smart_provider_fallback_warnings,
)
from paper_chaser_mcp.agentic.models import StructuredSourceRecord


def _make_record(
    *,
    topical_relevance: str = "on_topic",
    canonical_url: str | None = None,
    retrieved_url: str | None = None,
    full_text_url_found: bool | None = None,
    abstract_observed: bool = False,
) -> StructuredSourceRecord:
    return StructuredSourceRecord(
        sourceId="s-1",
        title="Example",
        canonicalUrl=canonical_url,
        retrievedUrl=retrieved_url,
        fullTextUrlFound=full_text_url_found,
        abstractObserved=abstract_observed,
        topicalRelevance=topical_relevance,  # type: ignore[arg-type]
    )


def test_helpers_are_same_object_on_core_and_smart_helpers() -> None:
    """Identity preservation: ``_core`` re-exports are the canonical helpers."""

    assert core_best_next_internal_action is _best_next_internal_action
    assert core_has_inspectable_sources is _has_inspectable_sources
    assert core_has_on_topic_sources is _has_on_topic_sources
    assert core_paid_providers_used is _paid_providers_used
    assert core_smart_coverage_summary is _smart_coverage_summary
    assert core_smart_provider_fallback_warnings is _smart_provider_fallback_warnings


def test_paid_providers_used_filters_and_sorts() -> None:
    result = _paid_providers_used(["openalex", "scholarapi", "arxiv", "scholarapi"])
    # scholarapi is paywalled; openalex / arxiv are free. Output is sorted+deduped.
    assert result == ["scholarapi"]
    assert _paid_providers_used([]) == []
    assert _paid_providers_used(["arxiv", "openalex"]) == []


def test_has_inspectable_sources_requires_on_topic_plus_inspectable_signal() -> None:
    assert _has_inspectable_sources([]) is False
    # Off-topic with a URL is still not inspectable.
    assert (
        _has_inspectable_sources([_make_record(topical_relevance="off_topic", canonical_url="https://example.com")])
        is False
    )
    # On-topic but no URLs / abstract.
    assert _has_inspectable_sources([_make_record()]) is False
    # On-topic with at least one inspectable signal.
    assert _has_inspectable_sources([_make_record(canonical_url="https://example.com")]) is True
    assert _has_inspectable_sources([_make_record(abstract_observed=True)]) is True


def test_has_on_topic_sources_any_non_off_topic_counts() -> None:
    assert _has_on_topic_sources([]) is False
    assert _has_on_topic_sources([_make_record(topical_relevance="off_topic")]) is False
    assert _has_on_topic_sources([_make_record(topical_relevance="on_topic")]) is True
    assert (
        _has_on_topic_sources(
            [_make_record(topical_relevance="off_topic"), _make_record(topical_relevance="on_topic")]
        )
        is True
    )


def test_best_next_internal_action_routes_by_intent() -> None:
    assert _best_next_internal_action(intent="known_item", has_sources=False, result_status="complete") == "get_paper_details"
    assert (
        _best_next_internal_action(intent="regulatory", has_sources=True, result_status="complete") == "inspect_source"
    )
    assert (
        _best_next_internal_action(intent="regulatory", has_sources=False, result_status="complete")
        == "search_papers_smart"
    )
    assert _best_next_internal_action(intent="topic", has_sources=True, result_status="complete") == "ask_result_set"
    assert (
        _best_next_internal_action(intent="topic", has_sources=False, result_status="partial") == "search_papers_smart"
    )
    assert (
        _best_next_internal_action(intent="topic", has_sources=False, result_status="complete") == "resolve_reference"
    )


def test_smart_coverage_summary_partitions_providers_by_status_bucket() -> None:
    summary = _smart_coverage_summary(
        providers_used=["openalex", "core", "arxiv"],
        provider_outcomes=[
            {"provider": "openalex", "statusBucket": "success"},
            {"provider": "core", "statusBucket": "empty"},
            {"provider": "arxiv", "statusBucket": "success"},
            {"provider": "serpapi", "statusBucket": "timeout"},
        ],
        search_mode="smart",
        drift_warnings=["narrowed"],
    )
    assert "openalex" in summary.providers_attempted
    assert "serpapi" in summary.providers_attempted
    assert summary.providers_failed == ["serpapi"]
    assert summary.providers_zero_results == ["core"]
    # core is zero-results, so it must be excluded from succeeded.
    assert "core" not in summary.providers_succeeded
    assert "openalex" in summary.providers_succeeded
    assert summary.likely_completeness == "partial"
    assert summary.search_mode == "smart"
    assert summary.retrieval_notes == ["narrowed"]


def test_smart_provider_fallback_warnings_returns_empty_when_configured_matches_active() -> None:
    assert (
        _smart_provider_fallback_warnings(
            provider_selection={"configuredSmartProvider": "openai", "activeSmartProvider": "openai"},
            provider_outcomes=[],
        )
        == []
    )
    # Missing either field short-circuits.
    assert (
        _smart_provider_fallback_warnings(
            provider_selection={"configuredSmartProvider": "", "activeSmartProvider": "openai"},
            provider_outcomes=[],
        )
        == []
    )


def test_smart_provider_fallback_warnings_lists_endpoints_when_known() -> None:
    warnings = _smart_provider_fallback_warnings(
        provider_selection={"configuredSmartProvider": "openai", "activeSmartProvider": "deterministic"},
        provider_outcomes=[
            {"provider": "openai", "statusBucket": "timeout", "endpoint": "/v1/chat"},
            {"provider": "openai", "statusBucket": "error", "endpoint": "/v1/embeddings"},
            {"provider": "openai", "statusBucket": "success", "endpoint": "/v1/ignored"},
        ],
    )
    assert len(warnings) == 1
    assert "openai" in warnings[0]
    assert "/v1/chat" in warnings[0]
    assert "/v1/embeddings" in warnings[0]
    assert "/v1/ignored" not in warnings[0]


def test_smart_provider_fallback_warnings_generic_when_endpoints_unknown() -> None:
    warnings = _smart_provider_fallback_warnings(
        provider_selection={"configuredSmartProvider": "openai", "activeSmartProvider": "deterministic"},
        provider_outcomes=[],
    )
    assert len(warnings) == 1
    assert "fell back to deterministic mode" in warnings[0]
    assert "openai" in warnings[0]
