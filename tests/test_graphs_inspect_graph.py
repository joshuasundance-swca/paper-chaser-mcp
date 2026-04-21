"""Phase 7b: ``agentic.graphs.inspect_graph`` submodule (landscape helpers)."""

from __future__ import annotations

import pytest

from paper_chaser_mcp.agentic.graphs import _core as core_module
from paper_chaser_mcp.agentic.graphs import inspect_graph


@pytest.mark.parametrize(
    "name",
    [
        "_cluster_papers",
        "_compute_disagreements",
        "_compute_gaps",
        "_finalize_theme_label",
        "_label_tokens",
        "_normalized_theme_label",
        "_suggest_next_searches",
        "_theme_terms_from_papers",
        "_top_terms_for_cluster",
    ],
)
def test_inspect_graph_exports_helper(name: str) -> None:
    assert getattr(inspect_graph, name) is not None


def test_inspect_graph_helpers_match_core_legacy_bindings() -> None:
    assert inspect_graph._finalize_theme_label is core_module._finalize_theme_label
    assert inspect_graph._theme_terms_from_papers is core_module._theme_terms_from_papers
    assert inspect_graph._cluster_papers is core_module._cluster_papers
    assert inspect_graph._suggest_next_searches is core_module._suggest_next_searches
    assert inspect_graph._compute_gaps is core_module._compute_gaps
    assert inspect_graph._compute_disagreements is core_module._compute_disagreements


def test_label_tokens_tokenizes_alphanumeric() -> None:
    assert inspect_graph._label_tokens("Large-Scale Pretraining for NLP") == [
        "large",
        "scale",
        "pretraining",
        "for",
        "nlp",
    ]


def test_theme_terms_prefers_title_weight() -> None:
    papers = [
        {"title": "microplastic toxicity fish", "abstract": "ocean study"},
    ]
    terms = inspect_graph._theme_terms_from_papers([], papers)
    assert "microplastic" in terms[:3] or "toxicity" in terms[:3]


def test_compute_gaps_empty_result_set() -> None:
    assert inspect_graph._compute_gaps([]) == [
        "No papers were available to analyze for gaps.",
    ]


def test_facade_reexports_finalize_theme_label() -> None:
    from paper_chaser_mcp.agentic.graphs import _finalize_theme_label as facade_fn

    assert facade_fn is inspect_graph._finalize_theme_label
