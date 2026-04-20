"""Tests for :mod:`paper_chaser_mcp.dispatch.normalization`."""

from __future__ import annotations

import pytest

from paper_chaser_mcp.dispatch.normalization import (
    _guided_normalize_citation_surface,
    _guided_normalize_source_locator,
    _guided_normalize_whitespace,
    _guided_normalize_year_hint,
    _guided_strip_research_prefix,
)


class TestGuidedNormalizeWhitespace:
    def test_collapses_internal_runs(self) -> None:
        assert _guided_normalize_whitespace("  foo    bar\tbaz\n\n qux ") == "foo bar baz qux"

    def test_none_returns_empty(self) -> None:
        assert _guided_normalize_whitespace(None) == ""

    def test_non_string_coerced(self) -> None:
        assert _guided_normalize_whitespace(42) == "42"


class TestGuidedNormalizeSourceLocator:
    def test_strips_scheme_www_and_trailing_slash(self) -> None:
        assert _guided_normalize_source_locator("HTTPS://www.Example.com/foo/") == "example.com/foo"

    def test_http_is_also_stripped(self) -> None:
        assert _guided_normalize_source_locator("http://arxiv.org/abs/2401.1") == "arxiv.org/abs/2401.1"

    def test_empty_returns_empty(self) -> None:
        assert _guided_normalize_source_locator("  ") == ""


class TestGuidedStripResearchPrefix:
    def test_strips_find_papers_about(self) -> None:
        assert _guided_strip_research_prefix("please find papers about transformers") == "transformers"

    def test_strips_help_me_research(self) -> None:
        assert _guided_strip_research_prefix("help me research attention mechanisms") == "attention mechanisms"

    def test_returns_unchanged_when_no_prefix(self) -> None:
        assert _guided_strip_research_prefix("transformer architectures") == "transformer architectures"


class TestGuidedNormalizeCitationSurface:
    def test_cfr_section_is_normalized(self) -> None:
        assert _guided_normalize_citation_surface("40 C.F.R. 261.4") == "40 CFR 261.4"

    def test_cfr_part_is_normalized(self) -> None:
        assert (
            _guided_normalize_citation_surface("see 40 C.F.R. Part 261 for details")
            == "see 40 CFR Part 261 for details"
        )

    def test_fr_citation_is_normalized(self) -> None:
        assert _guided_normalize_citation_surface("published at 89 F.R. 12345") == "published at 89 FR 12345"


class TestGuidedNormalizeYearHint:
    def test_year_range_returns_start_end(self) -> None:
        assert _guided_normalize_year_hint("papers from 2019-2022") == "2019:2022"

    def test_reversed_range_is_reordered(self) -> None:
        assert _guided_normalize_year_hint("2022-2019") == "2019:2022"

    def test_two_separate_years(self) -> None:
        assert _guided_normalize_year_hint("compare 2018 and 2020 reviews") == "2018:2020"

    def test_single_year_returned(self) -> None:
        assert _guided_normalize_year_hint("from 2021") == "2021"

    def test_empty_returns_none(self) -> None:
        assert _guided_normalize_year_hint("") is None

    def test_non_year_text_returned_as_is(self) -> None:
        assert _guided_normalize_year_hint("recent") == "recent"


class TestReExport:
    def test_core_re_exports_normalization(self) -> None:
        import importlib

        core = importlib.import_module("paper_chaser_mcp.dispatch._core")
        for name in (
            "_guided_normalize_whitespace",
            "_guided_normalize_source_locator",
            "_guided_strip_research_prefix",
            "_guided_normalize_citation_surface",
            "_guided_normalize_year_hint",
        ):
            assert hasattr(core, name), name


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
