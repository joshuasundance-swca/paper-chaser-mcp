import pytest

from scholar_search_mcp.identifiers import (
    normalize_doi,
    resolve_doi_from_paper_payload,
    resolve_doi_inputs,
)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("10.1234/Example.DoI", "10.1234/example.doi"),
        ("doi:10.1234/example-doi", "10.1234/example-doi"),
        ("https://doi.org/10.1234/example-doi", "10.1234/example-doi"),
        ("https://dx.doi.org/10.1234/example-doi).", "10.1234/example-doi"),
    ],
)
def test_normalize_doi_handles_raw_and_url_forms(
    raw_value: str,
    expected: str,
) -> None:
    assert normalize_doi(raw_value) == expected


def test_resolve_doi_from_paper_payload_checks_multiple_identity_fields() -> None:
    doi, source = resolve_doi_from_paper_payload(
        {
            "paperId": "paper-1",
            "canonicalId": "doi:10.7777/from-canonical",
            "externalIds": {"DOI": "10.8888/from-external"},
        }
    )

    assert doi == "10.7777/from-canonical"
    assert source == "canonical_id"


def test_resolve_doi_inputs_prefers_explicit_then_identifier_then_payload() -> None:
    explicit_doi, explicit_source = resolve_doi_inputs(
        doi="10.1111/explicit",
        paper_id="doi:10.2222/paper-id",
        paper={"canonicalId": "doi:10.3333/payload"},
    )
    identifier_doi, identifier_source = resolve_doi_inputs(
        paper_id="https://doi.org/10.2222/paper-id",
        paper={"canonicalId": "doi:10.3333/payload"},
    )
    payload_doi, payload_source = resolve_doi_inputs(
        paper={"canonicalId": "doi:10.3333/payload"},
    )

    assert (explicit_doi, explicit_source) == ("10.1111/explicit", "doi")
    assert (identifier_doi, identifier_source) == ("10.2222/paper-id", "paper_id")
    assert (payload_doi, payload_source) == ("10.3333/payload", "canonical_id")
