from types import SimpleNamespace

from paper_chaser_mcp import compat


def _parsed_citation(**overrides: object) -> SimpleNamespace:
    payload: dict[str, object] = {
        "identifier": None,
        "looks_like_non_paper": False,
        "quoted_fragments": [],
        "title_candidates": ["Attention Is All You Need"],
        "year": 2017,
        "author_surnames": ["Vaswani"],
        "looks_like_regulatory": False,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def test_build_clarification_flags_low_specificity_search_query() -> None:
    clarification = compat.build_clarification(
        "search_papers",
        {"query": "AI"},
        {},
    )

    assert clarification is not None
    assert clarification.reason == "low_specificity_query"
    assert clarification.can_proceed_without_answer is True


def test_build_clarification_prefers_identifier_for_title_match_miss(monkeypatch) -> None:
    monkeypatch.setattr(
        compat,
        "parse_citation",
        lambda *_args, **_kwargs: _parsed_citation(identifier="10.1234/example-doi"),
    )

    clarification = compat.build_clarification(
        "search_papers_match",
        {"query": "attention is all you need"},
        {"matchFound": False},
    )

    assert clarification is not None
    assert clarification.reason == "identifier_available"
    assert "DOI" in clarification.question


def test_build_clarification_requests_year_for_ambiguous_title_match(monkeypatch) -> None:
    monkeypatch.setattr(
        compat,
        "parse_citation",
        lambda *_args, **_kwargs: _parsed_citation(year=None),
    )

    clarification = compat.build_clarification(
        "search_papers_match",
        {"query": "attention is all you need"},
        {"matchFound": False},
    )

    assert clarification is not None
    assert clarification.reason == "missing_year"
    assert clarification.options == ["add year", "use resolve_citation", "broaden to search_papers"]


def test_build_clarification_flags_non_paper_title_match_inputs(monkeypatch) -> None:
    monkeypatch.setattr(
        compat,
        "parse_citation",
        lambda *_args, **_kwargs: _parsed_citation(looks_like_non_paper=True, title_candidates=[]),
    )

    clarification = compat.build_clarification(
        "search_papers_match",
        {"query": "Caltrans technical guidance for highway noise on birds"},
        {"matchFound": False},
    )

    assert clarification is not None
    assert clarification.reason == "likely_non_paper_output"
    assert clarification.options == ["search_papers", "search_authors", "external verification"]


def test_build_clarification_detects_quote_fragments_for_title_match(monkeypatch) -> None:
    monkeypatch.setattr(
        compat,
        "parse_citation",
        lambda *_args, **_kwargs: _parsed_citation(quoted_fragments=["quoted fragment"], title_candidates=[]),
    )

    clarification = compat.build_clarification(
        "search_papers_match",
        {"query": '"quoted fragment"'},
        {"matchFound": False},
    )

    assert clarification is not None
    assert clarification.reason == "quote_fragment_only"
    assert clarification.options == ["use search_snippets", "use resolve_citation", "broaden to search_papers"]


def test_build_clarification_requests_author_for_title_match_when_year_is_present(monkeypatch) -> None:
    monkeypatch.setattr(
        compat,
        "parse_citation",
        lambda *_args, **_kwargs: _parsed_citation(author_surnames=[]),
    )

    clarification = compat.build_clarification(
        "search_papers_match",
        {"query": "Attention Is All You Need 2017"},
        {"matchFound": False},
    )

    assert clarification is not None
    assert clarification.reason == "missing_author"
    assert clarification.options == ["add author", "use resolve_citation", "broaden to search_papers"]


def test_build_clarification_returns_fallback_prompt_for_near_tied_title_match(monkeypatch) -> None:
    monkeypatch.setattr(
        compat,
        "parse_citation",
        lambda *_args, **_kwargs: _parsed_citation(),
    )

    clarification = compat.build_clarification(
        "search_papers_match",
        {"query": "Attention Is All You Need 2017 Vaswani"},
        {"matchFound": False},
    )

    assert clarification is not None
    assert clarification.reason == "near_tied_or_missing_title_match"
    assert clarification.options == ["use DOI", "use arXiv ID", "broaden to search_papers"]


def test_build_clarification_redirects_regulatory_resolve_citation_requests(monkeypatch) -> None:
    monkeypatch.setattr(
        compat,
        "parse_citation",
        lambda *_args, **_kwargs: _parsed_citation(looks_like_regulatory=True, title_candidates=[]),
    )

    clarification = compat.build_clarification(
        "resolve_citation",
        {"citation": "77 FR 4632"},
        {"bestMatch": None},
    )

    assert clarification is not None
    assert clarification.reason == "likely_regulatory_primary_source"
    assert clarification.options == [
        "search_federal_register",
        "get_federal_register_document",
        "get_cfr_text",
    ]


def test_build_clarification_suppresses_resolve_citation_prompt_on_high_confidence(monkeypatch) -> None:
    monkeypatch.setattr(
        compat,
        "parse_citation",
        lambda *_args, **_kwargs: _parsed_citation(),
    )

    clarification = compat.build_clarification(
        "resolve_citation",
        {"citation": "Attention Is All You Need"},
        {"resolutionConfidence": "high", "bestMatch": {"paperId": "paper-1"}},
    )

    assert clarification is None


def test_build_clarification_flags_non_paper_resolve_citation_inputs(monkeypatch) -> None:
    monkeypatch.setattr(
        compat,
        "parse_citation",
        lambda *_args, **_kwargs: _parsed_citation(looks_like_non_paper=True, title_candidates=[]),
    )

    clarification = compat.build_clarification(
        "resolve_citation",
        {"citation": "Caltrans technical guidance for highway noise on birds"},
        {"bestMatch": None},
    )

    assert clarification is not None
    assert clarification.reason == "likely_non_paper_output"
    assert clarification.options == ["search_papers", "search_authors", "external verification"]


def test_build_clarification_requests_author_for_resolve_citation_when_year_is_present(monkeypatch) -> None:
    monkeypatch.setattr(
        compat,
        "parse_citation",
        lambda *_args, **_kwargs: _parsed_citation(author_surnames=[]),
    )

    clarification = compat.build_clarification(
        "resolve_citation",
        {"citation": "Attention Is All You Need 2017"},
        {"resolutionConfidence": "medium", "bestMatch": None},
    )

    assert clarification is not None
    assert clarification.reason == "missing_author"
    assert clarification.options == ["add author", "add venue", "broaden to search_papers"]


def test_build_clarification_detects_quote_fragments_for_resolve_citation(monkeypatch) -> None:
    monkeypatch.setattr(
        compat,
        "parse_citation",
        lambda *_args, **_kwargs: _parsed_citation(quoted_fragments=["quoted fragment"], title_candidates=[]),
    )

    clarification = compat.build_clarification(
        "resolve_citation",
        {"citation": '"quoted fragment"'},
        {"resolutionConfidence": "medium", "bestMatch": None},
    )

    assert clarification is not None
    assert clarification.reason == "quote_fragment_only"
    assert clarification.options == ["use search_snippets", "search_papers", "search_authors"]


def test_build_clarification_returns_ambiguous_citation_prompt_when_partial_data_remains(monkeypatch) -> None:
    monkeypatch.setattr(
        compat,
        "parse_citation",
        lambda *_args, **_kwargs: _parsed_citation(title_candidates=[], quoted_fragments=[]),
    )

    clarification = compat.build_clarification(
        "resolve_citation",
        {"citation": "Attention Is All You Need, 2017"},
        {"resolutionConfidence": "medium", "bestMatch": None},
    )

    assert clarification is not None
    assert clarification.reason == "ambiguous_citation"
    assert clarification.options == ["year", "author", "venue"]


def test_build_clarification_detects_ambiguous_author_identity() -> None:
    clarification = compat.build_clarification(
        "search_authors",
        {"query": "John Smith"},
        {"data": [{"authorId": "a-1"}, {"authorId": "a-2"}]},
    )

    assert clarification is not None
    assert clarification.reason == "ambiguous_author_identity"
    assert clarification.options == ["affiliation", "coauthor", "topic clue"]
