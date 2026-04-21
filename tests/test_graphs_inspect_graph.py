"""Phase 7b: ``agentic.graphs.inspect_graph`` submodule (landscape helpers).

Every private helper is reached via an explicit ``from`` import so the
``tests/test_test_seam_inventory.py`` firewall can see the seam — attribute
access on a module-handle import would silently bypass that guard.
"""

from __future__ import annotations

from paper_chaser_mcp.agentic.graphs import (
    _finalize_theme_label as facade_finalize_theme_label,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _cluster_papers as core_cluster_papers,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _compute_disagreements as core_compute_disagreements,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _compute_gaps as core_compute_gaps,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _finalize_theme_label as core_finalize_theme_label,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _label_tokens as core_label_tokens,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _normalized_theme_label as core_normalized_theme_label,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _suggest_next_searches as core_suggest_next_searches,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _theme_terms_from_papers as core_theme_terms_from_papers,
)
from paper_chaser_mcp.agentic.graphs._core import (
    _top_terms_for_cluster as core_top_terms_for_cluster,
)
from paper_chaser_mcp.agentic.graphs.inspect_graph import (
    _cluster_papers,
    _compute_disagreements,
    _compute_gaps,
    _finalize_theme_label,
    _label_tokens,
    _normalized_theme_label,
    _suggest_next_searches,
    _theme_terms_from_papers,
    _top_terms_for_cluster,
)


def test_inspect_graph_exports_cluster_papers() -> None:
    assert callable(_cluster_papers)


def test_inspect_graph_exports_compute_disagreements() -> None:
    assert callable(_compute_disagreements)


def test_inspect_graph_exports_compute_gaps() -> None:
    assert callable(_compute_gaps)


def test_inspect_graph_exports_finalize_theme_label() -> None:
    assert callable(_finalize_theme_label)


def test_inspect_graph_exports_label_tokens() -> None:
    assert callable(_label_tokens)


def test_inspect_graph_exports_normalized_theme_label() -> None:
    assert callable(_normalized_theme_label)


def test_inspect_graph_exports_suggest_next_searches() -> None:
    assert callable(_suggest_next_searches)


def test_inspect_graph_exports_theme_terms_from_papers() -> None:
    assert callable(_theme_terms_from_papers)


def test_inspect_graph_exports_top_terms_for_cluster() -> None:
    assert callable(_top_terms_for_cluster)


def test_inspect_graph_cluster_papers_matches_core() -> None:
    assert _cluster_papers is core_cluster_papers


def test_inspect_graph_compute_disagreements_matches_core() -> None:
    assert _compute_disagreements is core_compute_disagreements


def test_inspect_graph_compute_gaps_matches_core() -> None:
    assert _compute_gaps is core_compute_gaps


def test_inspect_graph_finalize_theme_label_matches_core() -> None:
    assert _finalize_theme_label is core_finalize_theme_label


def test_inspect_graph_label_tokens_matches_core() -> None:
    assert _label_tokens is core_label_tokens


def test_inspect_graph_normalized_theme_label_matches_core() -> None:
    assert _normalized_theme_label is core_normalized_theme_label


def test_inspect_graph_suggest_next_searches_matches_core() -> None:
    assert _suggest_next_searches is core_suggest_next_searches


def test_inspect_graph_theme_terms_from_papers_matches_core() -> None:
    assert _theme_terms_from_papers is core_theme_terms_from_papers


def test_inspect_graph_top_terms_for_cluster_matches_core() -> None:
    assert _top_terms_for_cluster is core_top_terms_for_cluster


def test_label_tokens_tokenizes_alphanumeric() -> None:
    assert _label_tokens("Large-Scale Pretraining for NLP") == [
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
    terms = _theme_terms_from_papers([], papers)
    assert "microplastic" in terms[:3] or "toxicity" in terms[:3]


def test_compute_gaps_empty_result_set() -> None:
    assert _compute_gaps([]) == [
        "No papers were available to analyze for gaps.",
    ]


def test_facade_reexports_finalize_theme_label() -> None:
    assert facade_finalize_theme_label is _finalize_theme_label
