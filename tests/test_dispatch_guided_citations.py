"""RED/green TDD tests for ``dispatch/guided/citations.py``.

Phase 3 extraction: these tests import the moved symbols from their new
home (``paper_chaser_mcp.dispatch.guided.citations``). Before the extraction
lands the import fails (RED). After the extraction + _core re-imports land,
the same tests pass (GREEN) and characterize the moved behavior.

A subset of these assertions also pass against the old location because
the facade re-exports the same symbol — per Phase 3 plan rule 1, that is a
GREEN-before-move observation; the tests intentionally target the new home
to pin the seam.
"""

from __future__ import annotations

import pytest

from paper_chaser_mcp.dispatch.guided import citations as citations_mod
from paper_chaser_mcp.dispatch.guided.citations import (
    _assign_verification_status,
    _guided_citation_from_paper,
    _guided_citation_from_structured_source,
    _guided_journal_or_publisher,
    _guided_normalize_access_axes,
    _guided_normalize_verification_status,
    _guided_open_access_route,
    _guided_year_text,
)


def test__assign_verification_status_regulatory_needs_body_text() -> None:
    assert (
        _assign_verification_status(
            source_type="federal_register_rule",
            body_text_embedded=True,
        )
        == "verified_primary_source"
    )
    # URL-only regulatory hit falls back to verified_metadata.
    assert (
        _assign_verification_status(
            source_type="federal_register_rule",
            full_text_url_found=True,
            body_text_embedded=False,
        )
        == "verified_metadata"
    )


def test__assign_verification_status_scholarly_with_doi() -> None:
    assert (
        _assign_verification_status(
            source_type="scholarly_article",
            has_doi=True,
        )
        == "verified_metadata"
    )
    assert (
        _assign_verification_status(
            source_type="scholarly_article",
            has_doi=False,
        )
        == "unverified"
    )


def test__guided_normalize_access_axes_body_text_implies_full_text_observed() -> None:
    access_status, url_found, observed, body, qa = _guided_normalize_access_axes({"accessStatus": "body_text_embedded"})
    assert access_status == "body_text_embedded"
    assert body is True
    assert observed is True
    assert url_found is False
    assert qa is False


def test__guided_normalize_access_axes_qa_readable_upgrades_body() -> None:
    access_status, _url_found, observed, body, qa = _guided_normalize_access_axes({"qaReadableText": True})
    assert access_status == "qa_readable_text"
    assert qa is True
    assert body is True
    assert observed is True


def test__guided_normalize_access_axes_url_only() -> None:
    access_status, url_found, observed, body, qa = _guided_normalize_access_axes(
        {"fullTextUrlFound": True, "canonicalUrl": "https://example.com"}
    )
    assert access_status == "url_verified"
    assert url_found is True
    assert body is False
    assert qa is False
    assert observed is False


def test__guided_normalize_verification_status_downgrades_regulatory_without_body() -> None:
    status = _guided_normalize_verification_status(
        {"verificationStatus": "verified_primary_source"},
        source_type="federal_register_rule",
        full_text_url_found=True,
        body_text_embedded=False,
    )
    assert status == "verified_metadata"


def test__guided_open_access_route_prefers_explicit_value() -> None:
    assert _guided_open_access_route({"openAccessRoute": "repository_open_access"}) == "repository_open_access"


def test__guided_open_access_route_detects_mirror_only() -> None:
    assert _guided_open_access_route({"retrievedUrl": "https://sci-hub.example/abc"}) == "mirror_only"


def test__guided_open_access_route_repository_provider() -> None:
    assert _guided_open_access_route({"provider": "arxiv"}) == "repository_open_access"


def test__guided_open_access_route_unknown_default() -> None:
    assert _guided_open_access_route({}) == "unknown"


def test__guided_citation_from_structured_source_returns_none_when_no_fields() -> None:
    assert _guided_citation_from_structured_source({}) is None


def test__guided_citation_from_structured_source_passthrough_when_dict() -> None:
    existing = {"title": "X", "year": "2020"}
    assert _guided_citation_from_structured_source({"citation": existing}) is existing


def test__guided_citation_from_structured_source_builds_from_surface() -> None:
    result = _guided_citation_from_structured_source(
        {"title": "A title", "date": "March 2021", "sourceType": "scholarly_article"}
    )
    assert result is not None
    assert result["title"] == "A title"
    assert result["year"] == "2021"
    assert result["sourceType"] == "scholarly_article"
    assert result["authors"] == []


def test__guided_citation_from_paper_none_when_empty() -> None:
    assert _guided_citation_from_paper({}, None) is None


def test__guided_citation_from_paper_dedupes_authors() -> None:
    result = _guided_citation_from_paper(
        {
            "title": "Example",
            "authors": [{"name": "Ada Lovelace"}, {"name": "A Lovelace"}, {"name": "Ada Lovelace"}],
            "year": 2010,
        },
        "https://example.com/abc",
    )
    assert result is not None
    assert result["authors"] == ["Ada Lovelace"]
    assert result["year"] == "2010"


def test__guided_year_text_extracts_four_digit_year() -> None:
    assert _guided_year_text("Published March 2021") == "2021"
    assert _guided_year_text("1999-07-01") == "1999"
    assert _guided_year_text("") is None
    assert _guided_year_text(None) is None
    assert _guided_year_text("not a year") is None


def test__guided_journal_or_publisher_prefers_crossref_publisher() -> None:
    assert _guided_journal_or_publisher({"enrichments": {"crossref": {"publisher": "Nature"}}}) == "Nature"


def test__guided_journal_or_publisher_falls_back_to_venue_then_provider() -> None:
    assert _guided_journal_or_publisher({"venue": "Cell"}) == "Cell"
    assert _guided_journal_or_publisher({"provider": "arxiv"}) == "arxiv"
    assert _guided_journal_or_publisher({}) is None


# Prevent unused-import false positives if citations_mod's exported API changes.
_EXPECTED_EXPORTS = (
    "_assign_verification_status",
    "_guided_normalize_access_axes",
    "_guided_normalize_verification_status",
    "_guided_open_access_route",
    "_guided_citation_from_structured_source",
    "_guided_citation_from_paper",
    "_guided_year_text",
    "_guided_journal_or_publisher",
)


@pytest.mark.parametrize("name", _EXPECTED_EXPORTS)
def test_citations_submodule_exports(name: str) -> None:
    assert hasattr(citations_mod, name), f"citations submodule missing {name}"
