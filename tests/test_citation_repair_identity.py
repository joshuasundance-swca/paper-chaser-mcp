"""Identity checks pinning the citation_repair subpackage layout.

These tests guard the Phase 9a structural split: ``citation_repair/_core.py``
is reduced to a thin compat shim that re-exports the legacy surface, while the
implementation lives in three cohesive siblings:

- ``normalization.py`` owns regex patterns, vocab constants, and stateless
  text/identifier normalizers.
- ``candidates.py`` owns ``ParsedCitation``, the famous-citation registry,
  feature extractors, ``_why_selected``, and ``build_match_metadata``.
- ``api.py`` owns the async ``resolve_citation`` orchestrator plus every
  provider-layered ``_resolve_*`` helper and response serialization.

The shim must re-export each symbol via ``from .X import Y`` so callers keep
seeing the same object (not a redefinition). ``_rank_candidate`` continues to
live in ``ranking.py``; we pin its identity too to guard the acyclic import
chain ``normalization <- candidates (+lazy ranking) <- ranking <- api``.
"""

from paper_chaser_mcp.citation_repair import _core as _shim
from paper_chaser_mcp.citation_repair import api as _api
from paper_chaser_mcp.citation_repair import candidates as _candidates
from paper_chaser_mcp.citation_repair import normalization as _normalization
from paper_chaser_mcp.citation_repair import ranking as _ranking


def test_normalization_module_owns_regex_and_vocab() -> None:
    for name in (
        "DOI_RE",
        "ARXIV_RE",
        "URL_RE",
        "YEAR_RE",
        "PAGES_RE",
        "QUOTED_RE",
        "WORD_RE",
        "REGULATORY_CITATION_RE",
        "VENUE_HINTS",
        "NON_PAPER_TERMS",
        "REGULATORY_TERMS",
        "GENERIC_TITLE_WORDS",
        "normalize_citation_text",
        "looks_like_paper_identifier",
    ):
        value = getattr(_normalization, name)
        assert getattr(_shim, name) is value, name


def test_candidates_module_owns_parser_and_extractors() -> None:
    for name in (
        "ParsedCitation",
        "parse_citation",
        "looks_like_citation_query",
        "_FAMOUS_CITATION_ENTRIES",
        "_lookup_famous_citation",
        "_why_selected",
        "classify_known_item_resolution_state",
        "_classify_resolution_confidence",
        "build_match_metadata",
        "_sparse_search_queries",
    ):
        value = getattr(_candidates, name)
        assert getattr(_shim, name) is value, name

    parsed = getattr(_candidates, "ParsedCitation")
    assert parsed.__module__ == "paper_chaser_mcp.citation_repair.candidates"


def test_api_module_owns_resolve_citation_and_helpers() -> None:
    for name in (
        "resolve_citation",
        "_build_famous_citation_candidate",
    ):
        value = getattr(_api, name)
        assert getattr(_shim, name) is value, name

    resolve = getattr(_api, "resolve_citation")
    assert resolve.__module__ == "paper_chaser_mcp.citation_repair.api"


def test_ranking_module_owns_rank_candidate() -> None:
    value = getattr(_ranking, "_rank_candidate")
    assert getattr(_shim, "_rank_candidate") is value
    assert value.__module__ == "paper_chaser_mcp.citation_repair.ranking"


def test_shim_public_surface_matches_subpackage() -> None:
    assert _shim.ParsedCitation is _candidates.ParsedCitation
    assert _shim.parse_citation is _candidates.parse_citation
    assert _shim.resolve_citation is _api.resolve_citation
    assert _shim.looks_like_citation_query is _candidates.looks_like_citation_query
    assert _shim.looks_like_paper_identifier is _normalization.looks_like_paper_identifier
    assert _shim._rank_candidate is _ranking._rank_candidate
