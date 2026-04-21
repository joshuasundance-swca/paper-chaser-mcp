"""Phase 6 red-bar: pin constants re-exported from planner/constants.py.

Every constant listed in the Phase 6 seam map as low-risk extraction fodder
must be importable from its new direct path AND remain re-exported from the
legacy ``paper_chaser_mcp.agentic.planner`` namespace.
"""

from __future__ import annotations

import re

from paper_chaser_mcp.agentic import planner as legacy_planner
from paper_chaser_mcp.agentic.planner.constants import (
    AGENCY_REGULATORY_MARKERS,
    ARXIV_RE,
    DOI_RE,
    FACET_SPLIT_RE,
    GENERIC_EVIDENCE_WORDS,
    HYPOTHESIS_QUERY_STOPWORDS,
    LITERATURE_QUERY_TERMS,
    QUERY_FACET_TOKEN_ALLOWLIST,
    QUERYISH_TITLE_BLOCKERS,
    REGULATORY_QUERY_TERMS,
    STRONG_REGULATORY_TITLE_BLOCKERS,
    TITLE_STOPWORDS,
    VARIANT_DEDUPE_STOPWORDS,
    _CULTURAL_RESOURCE_MARKERS,
    _DEFINITIONAL_PATTERNS,
)


def test_regex_constants_compile() -> None:
    assert isinstance(DOI_RE, re.Pattern)
    assert isinstance(ARXIV_RE, re.Pattern)
    assert isinstance(FACET_SPLIT_RE, re.Pattern)
    # Definitional patterns is a tuple of compiled patterns.
    assert isinstance(_DEFINITIONAL_PATTERNS, tuple)
    assert _DEFINITIONAL_PATTERNS
    for pattern in _DEFINITIONAL_PATTERNS:
        assert isinstance(pattern, re.Pattern)


def test_regex_constants_match_expected_examples() -> None:
    assert DOI_RE.search("see 10.1234/abc.def for details")
    assert ARXIV_RE.search("arxiv:1234.56789")
    assert FACET_SPLIT_RE.search("analysis for climate change")


def test_term_sets_are_frozen_style_sets() -> None:
    for label, value in (
        ("GENERIC_EVIDENCE_WORDS", GENERIC_EVIDENCE_WORDS),
        ("QUERY_FACET_TOKEN_ALLOWLIST", QUERY_FACET_TOKEN_ALLOWLIST),
        ("HYPOTHESIS_QUERY_STOPWORDS", HYPOTHESIS_QUERY_STOPWORDS),
        ("LITERATURE_QUERY_TERMS", LITERATURE_QUERY_TERMS),
        ("TITLE_STOPWORDS", TITLE_STOPWORDS),
        ("QUERYISH_TITLE_BLOCKERS", QUERYISH_TITLE_BLOCKERS),
        ("STRONG_REGULATORY_TITLE_BLOCKERS", STRONG_REGULATORY_TITLE_BLOCKERS),
        ("AGENCY_REGULATORY_MARKERS", AGENCY_REGULATORY_MARKERS),
        ("REGULATORY_QUERY_TERMS", REGULATORY_QUERY_TERMS),
        ("_CULTURAL_RESOURCE_MARKERS", _CULTURAL_RESOURCE_MARKERS),
        ("VARIANT_DEDUPE_STOPWORDS", VARIANT_DEDUPE_STOPWORDS),
    ):
        assert isinstance(value, (set, frozenset)), f"{label} is not a set/frozenset"
        assert value, f"{label} is empty"


def test_regulatory_query_terms_contain_key_keywords() -> None:
    for marker in ("cfr", "esa", "federal register", "section 106", "nhpa"):
        assert marker in REGULATORY_QUERY_TERMS


def test_legacy_planner_module_reexports_constants() -> None:
    # Every constant must remain accessible via the legacy dotted path so that
    # ``paper_chaser_mcp.agentic.planner.DOI_RE`` keeps resolving for callers
    # that did not migrate to the new ``planner.constants`` submodule yet.
    for name in (
        "DOI_RE",
        "ARXIV_RE",
        "FACET_SPLIT_RE",
        "GENERIC_EVIDENCE_WORDS",
        "QUERY_FACET_TOKEN_ALLOWLIST",
        "HYPOTHESIS_QUERY_STOPWORDS",
        "LITERATURE_QUERY_TERMS",
        "TITLE_STOPWORDS",
        "QUERYISH_TITLE_BLOCKERS",
        "STRONG_REGULATORY_TITLE_BLOCKERS",
        "AGENCY_REGULATORY_MARKERS",
        "REGULATORY_QUERY_TERMS",
        "_CULTURAL_RESOURCE_MARKERS",
        "VARIANT_DEDUPE_STOPWORDS",
        "_DEFINITIONAL_PATTERNS",
    ):
        assert hasattr(legacy_planner, name), f"legacy planner missing {name}"
