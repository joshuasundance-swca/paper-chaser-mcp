"""Phase 7a: shared_state submodule identity and contract tests.

The submodule owns the module-level constants and LangGraph optional-dep stubs
that used to live at the top of the flat ``graphs.py``. These tests ensure:

* The constants can be imported directly from
  ``paper_chaser_mcp.agentic.graphs.shared_state`` (no hidden coupling to
  ``_core``).
* The facade and the submodule expose the *exact same object* so monkeypatch
  sites and runtime behaviour stay consistent.
* The LangGraph optional-dep sentinel values remain sane when the dependency
  is missing.
"""

from __future__ import annotations

from paper_chaser_mcp.agentic import graphs as facade
from paper_chaser_mcp.agentic.graphs import shared_state


def test_constants_are_identical_between_facade_and_submodule() -> None:
    for name in (
        "SMART_SEARCH_PROGRESS_TOTAL",
        "_GRAPH_GENERIC_TERMS",
        "_COMPARISON_MARKERS",
        "_THEME_LABEL_STOPWORDS",
        "_COMPARISON_FOCUS_STOPWORDS",
        "_REGULATORY_SUBJECT_STOPWORDS",
        "_AGENCY_GUIDANCE_TERMS",
        "_AGENCY_AUTHORITY_TERMS",
        "_AGENCY_GUIDANCE_QUERY_NOISE_TERMS",
        "_AGENCY_GUIDANCE_DOCUMENT_TERMS",
        "_AGENCY_GUIDANCE_DISCUSSION_TERMS",
        "_CULTURAL_RESOURCE_DOCUMENT_TERMS",
        "_REGULATORY_QUERY_NOISE_TERMS",
        "_SPECIES_QUERY_NOISE_TERMS",
        "_CFR_DOC_TYPE_GENERIC",
    ):
        submodule_value = getattr(shared_state, name)
        facade_value = getattr(facade, name)
        assert submodule_value is facade_value, (
            f"{name}: facade and submodule must share the same object "
            "so monkeypatching / identity checks stay consistent"
        )


def test_shared_state_exposes_langgraph_stubs() -> None:
    # When langgraph is installed these are real classes/objects; when not
    # they are sentinels. Either way the attributes must be defined so
    # _core.py and future graphs submodules can rely on them.
    assert hasattr(shared_state, "InMemorySaver")
    assert hasattr(shared_state, "StateGraph")
    assert shared_state.START == "__start__" or shared_state.START is not None
    assert shared_state.END == "__end__" or shared_state.END is not None


def test_graph_generic_terms_includes_common_query_words() -> None:
    from paper_chaser_mcp.agentic.providers import COMMON_QUERY_WORDS

    assert COMMON_QUERY_WORDS.issubset(shared_state._GRAPH_GENERIC_TERMS), (
        "_GRAPH_GENERIC_TERMS must remain a superset of COMMON_QUERY_WORDS "
        "so graph-topic scoring keeps filtering generic query vocabulary"
    )


def test_theme_label_stopwords_builds_on_graph_generic_terms() -> None:
    assert shared_state._GRAPH_GENERIC_TERMS.issubset(shared_state._THEME_LABEL_STOPWORDS)


def test_comparison_focus_stopwords_builds_on_theme_label_stopwords() -> None:
    assert shared_state._THEME_LABEL_STOPWORDS.issubset(shared_state._COMPARISON_FOCUS_STOPWORDS)
