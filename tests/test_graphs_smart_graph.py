"""Phase 7c-2 — tests for the extracted ``smart_graph`` submodule.

Pins the seams that were moved out of ``graphs/_core`` in Phase 7c-2 and
guards the identity preservation that keeps legacy call-site imports
valid. The canonical home for these helpers is now
:mod:`paper_chaser_mcp.agentic.graphs.smart_graph`; they continue to be
importable from ``_core`` via the Phase 7c-2 re-import stanza.
"""

from __future__ import annotations

from paper_chaser_mcp.agentic.graphs._core import (
    _dedupe_variants as core_dedupe_variants,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _initial_retrieval_query_text as core_initial_retrieval_query_text,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _result_coverage_label as core_result_coverage_label,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _smart_failure_summary as core_smart_failure_summary,
)
from paper_chaser_mcp.agentic.graphs.smart_graph import (
    _dedupe_variants,
    _initial_retrieval_query_text,
    _result_coverage_label,
    _smart_failure_summary,
    maybe_compile_graphs,
)


def test_helpers_are_same_object_on_core_and_smart_graph() -> None:
    """Identity preservation: ``_core`` re-exports are the canonical helpers."""

    assert core_dedupe_variants is _dedupe_variants
    assert core_initial_retrieval_query_text is _initial_retrieval_query_text
    assert core_result_coverage_label is _result_coverage_label
    assert core_smart_failure_summary is _smart_failure_summary


def test_dedupe_variants_preserves_first_occurrence_ordering() -> None:
    assert _dedupe_variants(["foo", "Foo", "bar", " foo ", "BAR"]) == ["foo", "bar"]
    assert _dedupe_variants([]) == []
    assert _dedupe_variants(["", "   ", "a"]) == ["a"]


def test_result_coverage_label_thresholds() -> None:
    assert _result_coverage_label([{} for _ in range(20)]) == "broad"
    assert _result_coverage_label([{} for _ in range(8)]) == "moderate"
    assert _result_coverage_label([{} for _ in range(19)]) == "moderate"
    assert _result_coverage_label([{} for _ in range(7)]) == "narrow"
    assert _result_coverage_label([]) == "narrow"


def test_initial_retrieval_query_text_returns_normalized_for_known_intents() -> None:
    for intent in ("known_item", "author", "citation", "regulatory"):
        assert (
            _initial_retrieval_query_text(
                normalized_query="cancer screening guidelines",
                focus="pediatric oncology",
                intent=intent,  # type: ignore[arg-type]
            )
            == "cancer screening guidelines"
        )


def test_initial_retrieval_query_text_appends_focus_for_other_intents() -> None:
    result = _initial_retrieval_query_text(
        normalized_query="protein folding",
        focus="alphafold benchmarks",
        intent="topic",  # type: ignore[arg-type]
    )
    assert "protein folding" in result
    assert "alphafold" in result.lower()


def test_initial_retrieval_query_text_returns_normalized_when_focus_empty() -> None:
    assert (
        _initial_retrieval_query_text(
            normalized_query="quantum error correction",
            focus="",
            intent="topic",  # type: ignore[arg-type]
        )
        == "quantum error correction"
    )
    assert (
        _initial_retrieval_query_text(
            normalized_query="quantum error correction",
            focus=None,
            intent="topic",  # type: ignore[arg-type]
        )
        == "quantum error correction"
    )


def test_smart_failure_summary_none_when_all_outcomes_succeed() -> None:
    outcomes = [
        {"provider": "openalex", "statusBucket": "success"},
        {"provider": "core", "statusBucket": "empty"},
        {"provider": "arxiv", "statusBucket": "skipped"},
    ]
    assert _smart_failure_summary(provider_outcomes=outcomes, fallback_attempted=False) is None


def test_smart_failure_summary_reports_failed_providers_sorted() -> None:
    outcomes = [
        {"provider": "serpapi", "statusBucket": "timeout"},
        {"provider": "openalex", "statusBucket": "success"},
        {"provider": "core", "statusBucket": "error"},
    ]
    summary = _smart_failure_summary(provider_outcomes=outcomes, fallback_attempted=True)
    assert summary is not None
    assert summary.outcome == "fallback_success"
    assert summary.fallback_attempted is True
    assert summary.fallback_mode == "smart_provider_fallback"
    assert summary.primary_path_failure_reason == "core, serpapi"
    assert summary.recommended_next_action == "review_partial_results"


def test_maybe_compile_graphs_returns_four_graph_names_when_langgraph_available() -> None:
    compiled = maybe_compile_graphs(runtime=None)  # type: ignore[arg-type]
    # LangGraph is an optional dependency. When present, we get four compiled
    # graph placeholders; when absent, an empty dict. Either shape is valid
    # for this seam — assert both arms explicitly.
    if compiled:
        assert set(compiled.keys()) == {
            "smart_search",
            "grounded_answer",
            "landscape_map",
            "graph_expand",
        }
    else:
        assert compiled == {}
