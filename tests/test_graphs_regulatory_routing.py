"""Phase 7a: regulatory_routing submodule identity and behavioural contract tests."""

from __future__ import annotations

from paper_chaser_mcp.agentic import graphs as facade
from paper_chaser_mcp.agentic.graphs import regulatory_routing

_EXTRACTED = (
    "_agency_guidance_facet_terms",
    "_agency_guidance_priority_terms",
    "_agency_guidance_subject_terms",
    "_cfr_tokens",
    "_derive_regulatory_query_flags",
    "_ecos_query_variants",
    "_extract_common_name_candidate",
    "_extract_scientific_name_candidate",
    "_extract_subject_terms",
    "_format_cfr_citation",
    "_guidance_query_prefers_recency",
    "_is_agency_guidance_query",
    "_is_current_cfr_text_request",
    "_is_opaque_query",
    "_is_species_regulatory_query",
    "_parse_cfr_request",
    "_query_requests_regulatory_history",
    "_rank_ecos_variant_hits",
    "_rank_regulatory_documents",
    "_regulatory_document_matches_subject",
    "_regulatory_query_priority_terms",
    "_regulatory_query_subject_terms",
    "_regulatory_retrieval_hypotheses",
)

_EXTRACTED_CONSTANTS = (
    "_OPAQUE_ARXIV_RE",
    "_OPAQUE_DOI_RE",
)


def test_facade_and_submodule_expose_the_same_callables() -> None:
    for name in _EXTRACTED:
        submodule_value = getattr(regulatory_routing, name)
        facade_value = getattr(facade, name)
        assert submodule_value is facade_value, (
            f"{name}: facade and submodule must share the same object so "
            "legacy monkeypatch and call sites keep working after Phase 7a"
        )


def test_facade_and_submodule_expose_the_same_constants() -> None:
    for name in _EXTRACTED_CONSTANTS:
        submodule_value = getattr(regulatory_routing, name)
        facade_value = getattr(facade, name)
        assert submodule_value is facade_value, (
            f"{name}: facade and submodule must share the same regex so legacy call sites keep working after Phase 7a"
        )


def test_is_opaque_query_matches_doi_and_arxiv_tokens() -> None:
    assert regulatory_routing._is_opaque_query("10.1234/abcd.5678") is True
    assert regulatory_routing._is_opaque_query("arXiv:2401.12345") is True
    assert regulatory_routing._is_opaque_query("polar bear listing status") is False


def test_is_agency_guidance_query_requires_authority_and_guidance_signals() -> None:
    assert (
        regulatory_routing._is_agency_guidance_query(
            "FDA guidance on medical device approvals",
        )
        is True
    )
    assert regulatory_routing._is_agency_guidance_query("bacteria growth rates") is False


def test_parse_and_format_cfr_citation_round_trip() -> None:
    parsed = regulatory_routing._parse_cfr_request("50 CFR 17.12")
    assert parsed is not None
    assert parsed["title_number"] == 50
    assert parsed["part_number"] == 17
    assert regulatory_routing._format_cfr_citation(parsed) == "50 CFR 17.12"


def test_ecos_query_variants_fall_back_to_original_for_prose() -> None:
    variants = regulatory_routing._ecos_query_variants("polar bear critical habitat")
    assert variants, "non-opaque prose queries should yield at least one variant"
    opaque_variants = regulatory_routing._ecos_query_variants("10.1234/abcd.5678")
    # Opaque inputs should not produce scientific/common-name variants.
    assert opaque_variants == [] or all("10.1234" not in variant for variant in opaque_variants)


def test_is_current_cfr_text_request_detects_keywords() -> None:
    assert (
        regulatory_routing._is_current_cfr_text_request(
            "give me the current text of 50 CFR 17.12",
        )
        is True
    )
    assert (
        regulatory_routing._is_current_cfr_text_request(
            "history of amendments to CFR part",
        )
        is False
    )
