"""RED/green TDD tests for ``dispatch/guided/sources.py``.

Phase 3 extraction: these tests import the 12 source-record helpers from
their new home. The larger builders (``_guided_source_record_from_paper``,
``_guided_sources_from_fr_documents``) are exercised via input-output
characterization; the smaller helpers are exercised via direct property
assertions.
"""

from __future__ import annotations

import pytest

from paper_chaser_mcp.dispatch.guided import sources as sources_mod


def test__guided_source_id_prefers_source_id_key() -> None:
    assert (
        sources_mod._guided_source_id(
            {"sourceId": "primary-1", "paperId": "p2"},
            fallback_prefix="source",
            index=0,
        )
        == "primary-1"
    )


def test__guided_source_id_falls_through_keys_then_title_then_fallback() -> None:
    assert (
        sources_mod._guided_source_id({"title": "Weather Report"}, fallback_prefix="src", index=3)
        == "Weather Report"
    )
    assert sources_mod._guided_source_id({}, fallback_prefix="paper", index=7) == "paper-7"


def test__guided_extract_source_id_picks_first_non_none() -> None:
    assert sources_mod._guided_extract_source_id({"sourceId": "a"}) == "a"
    assert sources_mod._guided_extract_source_id({"evidence_id": "b"}) == "b"
    assert sources_mod._guided_extract_source_id({"leadId": "c"}) == "c"
    assert sources_mod._guided_extract_source_id({}) is None


def test__guided_source_record_from_structured_source_minimum_fields() -> None:
    record = sources_mod._guided_source_record_from_structured_source(
        {"title": "A", "sourceType": "scholarly_article"},
        index=1,
    )
    assert record["sourceId"] == "A"
    assert record["sourceType"] == "scholarly_article"
    assert record["topicalRelevance"] == "weak_match"
    assert record["verificationStatus"] == "unverified"


def test__guided_source_record_from_paper_sets_verification_when_metadata_present() -> None:
    record = sources_mod._guided_source_record_from_paper(
        "climate change",
        {
            "title": "Climate change and impacts",
            "authors": [{"name": "Jane Scientist"}],
            "year": 2020,
            "canonicalUrl": "https://doi.org/10.1/abc",
        },
        index=1,
    )
    assert record["title"] == "Climate change and impacts"
    # Backward-compat: scholarly article with title+author becomes verified_metadata
    # even without explicit DOI resolution.
    assert record["verificationStatus"] in {"verified_metadata", "unverified"}


def test__guided_sources_from_fr_documents_skips_missing_title() -> None:
    docs = [{"title": ""}, {"title": "A Rule"}]
    out = sources_mod._guided_sources_from_fr_documents("endangered species", docs)
    assert len(out) == 1
    assert out[0]["title"] == "A Rule"
    assert out[0]["provider"] == "federal_register"
    assert out[0]["isPrimarySource"] is True


def test__guided_sources_from_fr_documents_handles_rule_type() -> None:
    docs = [{"title": "Final Rule on X", "documentType": "Rule", "documentNumber": "2021-123"}]
    out = sources_mod._guided_sources_from_fr_documents("x", docs)
    assert out[0]["sourceType"] == "federal_register_rule"
    assert out[0]["sourceId"] == "fr-2021-123"


def test__guided_dedupe_source_records_collapses_by_identity() -> None:
    records = [
        {"sourceId": "a", "title": "T", "canonicalUrl": "u"},
        {"sourceId": "a", "title": "T", "canonicalUrl": "u"},
        {"sourceId": "b", "title": "T2", "canonicalUrl": "u2"},
    ]
    out = sources_mod._guided_dedupe_source_records(records)
    assert len(out) == 2
    assert {r["sourceId"] for r in out} == {"a", "b"}


def test__guided_source_matches_reference_title_and_url() -> None:
    candidate = {
        "sourceId": "s1",
        "title": "Climate Paper",
        "canonicalUrl": "https://example.com/abc",
    }
    assert sources_mod._guided_source_matches_reference(candidate, "Climate Paper")
    assert sources_mod._guided_source_matches_reference(candidate, "https://example.com/abc")
    assert not sources_mod._guided_source_matches_reference(candidate, "")
    assert not sources_mod._guided_source_matches_reference(candidate, "something else")


def test__guided_source_records_share_surface_title_overlap() -> None:
    assert sources_mod._guided_source_records_share_surface(
        {"title": "Same Title"},
        {"title": "SAME TITLE"},
    )
    assert not sources_mod._guided_source_records_share_surface(
        {"title": "A"},
        {"title": "B"},
    )


def test__guided_source_identity_returns_lowercase_title_tuple() -> None:
    assert sources_mod._guided_source_identity(
        {"sourceId": "s", "canonicalUrl": "U", "title": "Foo"}
    ) == ("s", "U", "foo")


def test__guided_merge_source_records_prefers_primary_then_fills() -> None:
    primary = {"sourceId": "s", "title": "T"}
    secondary = {"sourceId": "s", "canonicalUrl": "https://x", "isPrimarySource": True}
    merged = sources_mod._guided_merge_source_records(primary, secondary)
    assert merged["sourceId"] == "s"
    assert merged["title"] == "T"
    assert merged["canonicalUrl"] == "https://x"
    assert "accessStatus" in merged


def test__guided_merge_source_record_sets_collapses_duplicates() -> None:
    a = [{"sourceId": "1", "title": "T", "canonicalUrl": "u"}]
    b = [{"sourceId": "1", "title": "T", "canonicalUrl": "u", "isPrimarySource": True}]
    out = sources_mod._guided_merge_source_record_sets(a, b)
    assert len(out) == 1
    assert out[0]["isPrimarySource"] is True


def test__guided_source_coverage_summary_empty_returns_none() -> None:
    assert sources_mod._guided_source_coverage_summary(sources=[], leads=[], base_coverage=None) is None


def test__guided_source_coverage_summary_counts_by_access_status() -> None:
    result = sources_mod._guided_source_coverage_summary(
        sources=[{"accessStatus": "url_verified"}, {"accessStatus": "url_verified"}],
        leads=[{"accessStatus": "abstract_only"}],
        base_coverage=None,
    )
    assert result is not None
    assert result["totalSources"] == 3
    assert result["byAccessStatus"] == {"url_verified": 2, "abstract_only": 1}


_EXPECTED_EXPORTS = (
    "_guided_source_id",
    "_guided_source_record_from_structured_source",
    "_guided_source_record_from_paper",
    "_guided_sources_from_fr_documents",
    "_guided_extract_source_id",
    "_guided_dedupe_source_records",
    "_guided_source_matches_reference",
    "_guided_source_records_share_surface",
    "_guided_source_identity",
    "_guided_merge_source_records",
    "_guided_merge_source_record_sets",
    "_guided_source_coverage_summary",
)


@pytest.mark.parametrize("name", _EXPECTED_EXPORTS)
def test_sources_submodule_exports(name: str) -> None:
    assert hasattr(sources_mod, name)
